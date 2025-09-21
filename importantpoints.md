DynamoDB Partitioning (Hash Function)

DynamoDB uses a partition key hash to determine which physical partition a record belongs to.

The hash function is MD5 (Message Digest 5).

AWS docs: “The system uses an MD5 hash function to map the partition key value to a partition.”

The result is a 128-bit hash, which DynamoDB uses to place the item into a partition range.

So, if you have code = "abc123", DynamoDB does:

partition = MD5("abc123") → value in [0 … 2^128)


Then maps that number into a physical partition (each ~10GB, split dynamically as they grow).

Cassandra Partitioning (Hash Function)

Cassandra also assigns rows to nodes via partitioning + consistent hashing.

Earlier versions used MD5 (via RandomPartitioner).

Later versions (default since Cassandra 1.2) use Murmur3Partitioner with MurmurHash3 (64-bit).

So Cassandra does:

token = MurmurHash3(partition_key)


This token is a 64-bit number in the range [-2^63 … 2^63-1].

Nodes in the cluster are assigned token ranges.

Each row is stored on the node(s) responsible for the token’s range.

✅ Why MurmurHash3?

Extremely fast, non-cryptographic hash.

Good uniform distribution.
| System        | Hash Function                       | Output size | Purpose                                      |
| ------------- | ----------------------------------- | ----------- | -------------------------------------------- |
| **DynamoDB**  | **MD5** (cryptographic)             | 128-bit     | Uniformly distribute keys across partitions. |
| **Cassandra** | **MurmurHash3** (non-cryptographic) | 64-bit      | Consistent hashing for token ranges.         |

Low CPU overhead → important for large clusters.

