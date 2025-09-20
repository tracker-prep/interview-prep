# URL Shortener – Production-Grade System Design (Principal/Senior Architect View)

## Goals & Constraints

**Primary objective:** ultra-reliable, low-latency redirects.

**SLOs:**
- Redirect (GET) availability ≥ 99.99%, p95 latency < 50 ms (edge) / < 120 ms (origin).
- Shorten (POST) availability ≥ 99.9%, p95 latency < 200 ms.

**Scale assumptions (order of magnitude):**
- Writes: 1–10k QPS (campaign spikes).
- Reads: 100–300x the write rate (viral links), long tail distribution.
- Storage: 10–100B URLs, cold data retained for years.
- Traffic pattern: highly skewed (hot keys). Heavy bot traffic. Global user base.

**Regulatory:** GDPR/CCPA deletions; link expiration; regional data residency for some tenants.

---

## Product/Feature Requirements

- Create short link from a long URL. Optional **custom alias** and **expiration**.  
- Redirect with **301 (permanent)** or **302 (temporary)** based on publisher needs (can be a per-link flag).  
- Optional **link metadata**: title, UTM, tags, owner, policy flags.  
- **Analytics** (async): clicks, uniques, geo, device; bot filtration; privacy preserving.  
- **Admin & Abuse:** domain allow/deny lists, malware/phishing checks, rate limits, takedown pipeline.  
- **Enterprise:** custom domains, SSO, RBAC, audit trails, data export, SLAs.

---

## High-Level Architecture

- **Edge:** CDN (CloudFront/Akamai/Fastly) + edge compute (Lambda@Edge/Workers) for ultra-fast lookup and redirect, WAF, DDoS protection.  
- **Caching tier:** Redis/Memcached cluster (regional), hot key protection.  
- **Primary KV store:** DynamoDB/Cassandra/Bigtable (global table strategy) for `code → target`.  
- **ID/Code service:** Base62/ULID generator with collision-safe writes.  
- **Control plane:** REST/GraphQL APIs for create/update/rotate; Admin console; AuthZ via OIDC.  
- **Async plane:** Stream (Kafka/Kinesis/PubSub) for click events → analytics store (S3/Parquet + Spark/EMR/Databricks), real-time metrics in Druid/Pinot.  
- **Security & Abuse:** URL scanners (Google Safe Browsing, internal ML), rate limiters, anomaly detectors.  
- **Observability:** Tracing (OpenTelemetry), metrics (RED/USE), logs, synthetic probes.

---

## API Design (minimal but expressive)

- `POST /v1/links`  
  Body: `{ longUrl, customAlias?, expiresAt?, redirectType? (301|302), domain? }`  
  Returns: `{ shortCode, shortUrl, ... }`  

- `GET /{shortCode}` → 301/302 redirect to `longUrl`.

- Admin: `GET/PUT /v1/links/{shortCode}` (rotate URL, pause, delete/takedown).  
- Tenant: `GET /v1/links?owner=...` (paginated list, RBAC enforced).  

**Idempotency:** `Idempotency-Key` header on create to prevent dupes in retries.

---

## Data Model

**Primary table (KV):** `Links`  
- PK: `code` (string, 6–9 chars Base62)  
- Attrs: `longUrl, createdAt, ownerId, expiresAt?, status (active/paused/deleted), redirectType, domain, checksum, flags (e.g., nsfw, blocked), version`.

**Secondary index:** `OwnerIndex` (ownerId → list for UI), optionally `CreatedAtIndex`.

**Analytics (append-only):** Stream to S3/Parquet with schema: `timestamp, code, ipHash, ua, geo, referrer, tenantId, botScore`.

---

## Code Generation Strategies (trade-offs)

### Monotonic Counter + Base62
- Pros: dense codes (shorter), predictable capacity; great for read cache locality.  
- Cons: guessable; requires strong write serialization or allocation blocks.

### Random Base62 (6–9 chars)
- Pros: non-guessable; horizontally scalable; simple.  
- Cons: collision handling needed (but with 8+ chars, collision probability negligible).  
- Implementation: generate `n` chars; conditional put (if absent). On conflict, retry with backoff.

### ULID/KSUID-derived (time-sortable, randomness included)
- Pros: lexicographic time clustering, good for analytics; easy sharding.  
- Cons: longer codes unless base-62 compacted; still need collision checks.

**Recommendation:** For public internet service: **Random Base62 length 8** default; **length 6** allowed for premium tenants with collision budget + throttled retries; **custom alias** path bypasses generator with uniqueness checks.

**Capacity math:** Base62^8 ≈ 2.18e14 combinations—enough for decades. Even with 10^11 links, collision odds remain tiny; conditional write ensures correctness.

---

## Write Path (Create Short Link)

1. **AuthN/AuthZ** (tenant RBAC), rate limit per user/tenant/IP.  
2. **Validation**: URL syntax, allowed scheme (http/https), resolve domain allowlist, no private IPs (prevent SSRF), size limits.  
3. **Safety check**: async scanner enqueues the URL; if high-risk, mark `status=paused`. For enterprise, synchronous risk scoring path can be toggleable.  
4. **Code generation**: Random Base62(8).  
5. **Conditional write** into KV: `PutItem if not exists` else retry (bounded retries).  
6. **Warm caches**: write-through to Redis (code → target + metadata) with TTL (e.g., 24h) and to **edge key-value** store if available.  
7. **Emit audit** event to stream (for analytics & billing).

**Consistency:** Redirect path requires **read-after-write** for the creator. Solve with:  
- Returning the **short URL** in the response (client uses it), and  
- **Sticking** the code in L1 (Redis) immediately.  
- If multi-region, create and first read should route to same region (sticky session or region-aware API).

---

## Read Path (Redirect)

**Critical path:** ultra-fast, minimal dependencies.

1. **Edge (CDN/WAF):** route `/{code}` to edge function:  
   - Validate code shape; drop obvious garbage.  
   - **L0 cache** lookup (edge KV/edge cache). If hit → redirect (no origin).  
2. **L1 cache**: regional Redis (hotset).  
3. **Origin fallback**: KV store (Dynamo/Cassandra), 1-row get by `code`.  
4. **Policy checks**: TTL/expiration, paused/deleted, geo/domain restrictions.  
5. **Redirect**:  
   - Default **302**; allow per-link override to **301** (beware caching implications).  
   - Include security headers (no-store for 302 if desired).  
6. **Async click log**: fire-and-forget to stream with sampling; don’t block redirect.

**Hot key protection:**  
- **Local per-node in-process cache** (e.g., Caffeine with tiny TTL).  
- **Request coalescing** to prevent cache stampedes (singleflight).  
- **Probabilistic early refresh**/background refresh.

---

## Storage Choices & Partitioning

- **Primary KV:**  
  - **DynamoDB (global tables)** or **Cassandra**: O(1) key lookups, automatic partitioning by `code`.  
  - Item size tiny (< 1KB), RCU/WCU predictable.  
- **Cold archive:** snapshot to **S3** periodically for durability & cost.  
- **Multi-tenant isolation:** per-tenant prefixes or separate tables (depending on blast radius/rate limits).  
- **Partition health:** watch skew from **hot codes**; use **adaptive capacity** (Dynamo) or virtual nodes (Cassandra).

---

## Caching Strategy (3 tiers)

- **L0 Edge:** fastest; TTL 5–60 min; invalidation on link mutation (publish purge message) or use short TTL for safety.  
- **L1 Redis:** TTL 1–24h; size bound; LFU eviction; store `{longUrl, status, expiresAt, redirectType, checksum}`.  
- **L2 App Cache:** tiny per-process cache with ~1–5 min TTL; protects Redis from spikes.

**Cache invalidation:**  
- On update/delete/pause: publish **invalidation event** → edge + Redis purge.  
- Fallback: short TTL + versioning (include version in cache key).

---

## Consistency, Availability, and Failure Modes

- **Redirects favor availability** over strict consistency: stale cache briefly returning old target is acceptable if policy allows.  
- **Create/update** needs stronger correctness: conditional writes ensure uniqueness; read-after-write via cache warm.  
- **Partial outages:**  
  - If Redis down → fall back to KV.  
  - If KV region down → failover to secondary region with **global table** replication; if replication lags, degrade to “temporary unavailable” for mutated links only.  
- **Circuit breakers & timeouts** on each hop; **hedged reads** (two reads to different replicas on slow tail).

---

## Multi-Region Strategy

- **Active-Active** for reads using **global tables** (DynamoDB) or multi-region Cassandra.  
- **Writes**: write to local region + async replicate; expose **eventual** cross-region read consistency.  
- **Creator locality**: API gateway routes the creator’s first reads to the write region for consistency.  
- **DR:** regional RTO < 15 min; continuous backups; infra as code (Terraform/CloudFormation) for rebuilds.

---

## Abuse, Security, and Compliance

- **Input hardening:** no `file://`, no private CIDRs; DNS resolution checks to avoid SSRF.  
- **Malware/phishing:** URL scanning on create + periodic re-scan; blocklists; **quarantine mode** with interstitial warning.  
- **Rate limiting & quotas:** token buckets at edge by IP/user/tenant; adaptive limits during spikes.  
- **Bot detection:** UA heuristics + ML score; **do not** count bot clicks; optionally serve **410 Gone** to bad bots.  
- **GDPR/CCPA:** per-tenant data location; right-to-erasure deletes link + downstream analytics keyed by a salted hash of PII (no raw IP storage).  
- **Tenant isolation:** per-tenant keys, custom domains, RBAC scopes; audit logs.

---

## Observability & Operations

- **Golden signals:**  
  - GET p50/p95 latency, success rate, cache hit ratios (edge/L1), origin read QPS.  
  - POST p95 latency, write success, code collision rate.  
  - Invalidations processed, lag to consistency.  
- **Tracing:** add `X-Request-Id`, propagate trace headers; sample GET at low rate, POST at higher.  
- **Synthetic monitors:** global checks for representative links (active/expired/paused).  
- **Cost dashboards:** Dynamo RCU/WCU, Redis memory, egress at CDN, stream/analytics spend.  
- **Runbooks:** hot key storm, cache stampede, KV partition hotspot, mass invalidations, edge config rollback.

---

## Analytics Pipeline (async)

- **Edge/logger → Stream (Kafka/Kinesis)** with backpressure friendly producers.  
- **Consumer**: enrich (geoIP, UA parsing), bot filtering, PII hashing.  
- **Storage**: S3 → Parquet partitioned by `dt=YYYY-MM-DD/tenantId`; compact via Iceberg/Delta.  
- **Serving**: batch queries via Spark/Trino; real-time via **Druid/Pinot** for dashboards (last 7–30 days).  
- **Privacy**: configurable retention; IP salted hashing; per-tenant export.

---

## Cost & Efficiency

- **Edge hits** are “free latency & cheaper origin”—maximize hit ratio (short TTLs balanced with correctness).  
- **Redis** sized for **top 1–5%** hot links; everything else falls through.  
- **KV**: pay per read/write; cache miss rate directly drives cost → make it a KPI.  
- **Storage tiering**: cold links archived to S3; keep only metadata in KV (store longUrl in S3 when very long/rare, but beware extra hop—usually unnecessary).

---

## Extensibility

- **Custom domains**: `https://go.example.com/abc123`. Keep `domain` in PK if you want tenant-scoped namespaces.  
- **Branded/custom aliases**: reserve names, block profanity, enforce uniqueness by `(domain, code)`.  
- **QR codes**: pre-render and store in object storage.  
- **Per-region routing rules**: e.g., different target by geo or platform (mobile deep links).  
- **Link policies**: max clicks, time-boxed, A/B redirect rules (feature flagged).

---

## Edge Logic (pseudo)

- Validate code shape: `^[A-Za-z0-9]{6,12}$`.  
- L0 lookup; if miss: call origin `/v1/resolve/{code}` (fast path returning minimal JSON).  
- Enforce `expiresAt`/`status`; compute redirect type.  
- Set cache headers appropriately (avoid over-caching 302 if targets can change).  
- Emit minimal click beacon (non-blocking).

---

## Failure Scenarios to Call Out

- **Massive spike on single code (celebrity tweet):** edge + Redis absorb; singleflight + TTL jitter avoid stampede; rate limit analytics writes, not redirects.  
- **KV partial outage:** circuit break to “temporary unavailable” for new/updated links; serve stale for existing ones with a **stale-while-revalidate** policy.  
- **Bad config deploy at edge:** staged rollout + canary + instant rollback (versioned edge configs).  
- **Cache invalidation storm:** batch/pipeline invalidations; backoff; prioritize high-traffic keys.

---

## What I’d Say in a Bar-Raiser/Principal Review

- I’m optimizing for **availability and latency on GET**; writes can be eventual-consistent cross-region.  
- **Three-tier caching + edge compute** removes origin from hot path for >95% of traffic.  
- **Random Base62 with conditional writes** keeps code gen horizontally scalable and secure; custom aliases supported.  
- Read path is **idempotent**, write path protected by **idempotency keys**.  
- Abuse prevention and **privacy controls** are first-class, not bolt-ons.  
- Observability is designed into the system (SLIs/SLOs defined, error budgets tied to release gates).  
- The system is **multi-region active/active**, with clear consistency semantics and DR posture.  
- Cost drivers are measured; **cache hit ratio** and **KV read rate** are owned KPIs.

---
