Hotel booking LLD

```
import java.math.BigDecimal;
import java.time.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.concurrent.locks.*;

/**
 * Movie Ticket Booking System - Single-file plain Java LLD implementation
 *
 * Features:
 * - Domain: Theater, Screen, Seat, Movie, Show, Booking
 * - In-memory thread-safe repositories
 * - Seat hold mechanism (holds expire after HOLD_EXPIRY_SECONDS)
 * - Pessimistic locking per show-seat using ReentrantLock managed by SeatLockManager
 * - Booking flow:
 *     1. holdSeats(userId, showId, seatIds) -> returns holdId
 *     2. confirmBooking(holdId, paymentToken) -> creates CONFIRMED booking
 *     3. cancelBooking(bookingRef) -> cancels and releases seats (refund stub)
 * - Search available seats (excludes CONFIRMED and current HOLDS)
 *
 * Compile & run:
 *   javac MovieBookingLLD.java
 *   java MovieBookingLLD
 *
 * Note: This is a demo LLD for interviews; in production you'd split classes into files,
 *       use a DB for persistence, and avoid holding locks during external calls.
 */
public class MovieBookingLLD {

    /* -----------------------------
     * Domain Models & Enums
     * ----------------------------- */

    enum SeatType { REGULAR, PREMIUM, VIP }
    enum BookingStatus { HOLD, CONFIRMED, CANCELLED, FAILED }

    static class Theater {
        final long id;
        final String name;
        final String city;
        Theater(long id, String name, String city) { this.id = id; this.name = name; this.city = city; }
    }

    static class Screen {
        final long id;
        final long theaterId;
        final String name;
        final int totalRows;
        final int seatsPerRow;
        Screen(long id, long theaterId, String name, int totalRows, int seatsPerRow) {
            this.id = id; this.theaterId = theaterId; this.name = name; this.totalRows = totalRows; this.seatsPerRow = seatsPerRow;
        }
    }

    static class Seat {
        final long id;
        final long screenId;
        final String seatCode; // e.g., "A1"
        final SeatType type;
        final BigDecimal price;
        Seat(long id, long screenId, String seatCode, SeatType type, BigDecimal price) {
            this.id = id; this.screenId = screenId; this.seatCode = seatCode; this.type = type; this.price = price;
        }
    }

    static class Movie {
        final long id;
        final String title;
        final int durationMinutes;
        Movie(long id, String title, int durationMinutes) { this.id = id; this.title = title; this.durationMinutes = durationMinutes; }
    }

    static class Show {
        final long id;
        final long movieId;
        final long screenId;
        final LocalDateTime startTime;
        final LocalDateTime endTime;
        Show(long id, long movieId, long screenId, LocalDateTime startTime, LocalDateTime endTime) {
            this.id = id; this.movieId = movieId; this.screenId = screenId; this.startTime = startTime; this.endTime = endTime;
        }
    }

    static class Booking {
        final long id;
        final String bookingRef;
        final long userId;
        final long showId;
        final Set<Long> seatIds; // seats reserved
        final BigDecimal amount;
        BookingStatus status;
        final Instant createdAt;
        Instant updatedAt;
        String paymentTxnId; // nullable

        Booking(long id, String bookingRef, long userId, long showId, Set<Long> seatIds, BigDecimal amount, BookingStatus status) {
            this.id = id; this.bookingRef = bookingRef; this.userId = userId; this.showId = showId;
            this.seatIds = new HashSet<>(seatIds);
            this.amount = amount;
            this.status = status;
            this.createdAt = Instant.now();
            this.updatedAt = this.createdAt;
        }

        void setStatus(BookingStatus s) { this.status = s; this.updatedAt = Instant.now(); }
        void setPaymentTxnId(String txn) { this.paymentTxnId = txn; this.updatedAt = Instant.now(); }
    }

    /**
     * SeatHold represents a temporary hold over seats for a user.
     * Holds expire after a configured TTL and are automatically released.
     */
    static class SeatHold {
        final String holdId; // unique
        final long userId;
        final long showId;
        final Set<Long> seatIds;
        final BigDecimal amount;
        final Instant createdAt;
        volatile boolean active = true; // false if released/expired/confirmed
        SeatHold(String holdId, long userId, long showId, Set<Long> seatIds, BigDecimal amount) {
            this.holdId = holdId; this.userId = userId; this.showId = showId; this.seatIds = new HashSet<>(seatIds);
            this.amount = amount; this.createdAt = Instant.now();
        }
        void deactivate() { this.active = false; }
    }

    /* -----------------------------
     * Exceptions
     * ----------------------------- */

    static class NotFoundException extends RuntimeException { NotFoundException(String m){ super(m); } }
    static class InvalidRequestException extends RuntimeException { InvalidRequestException(String m){ super(m); } }
    static class SeatUnavailableException extends RuntimeException { SeatUnavailableException(String m){ super(m); } }
    static class PaymentFailedException extends RuntimeException { PaymentFailedException(String m){ super(m); } }

    /* -----------------------------
     * In-memory Repositories (thread-safe)
     * ----------------------------- */

    interface TheaterRepo {
        Theater save(Theater t);
        Optional<Theater> findById(long id);
        List<Theater> findByCity(String city);
    }
    static class InMemoryTheaterRepo implements TheaterRepo {

        private final ConcurrentMap<Long, Theater> map = new ConcurrentHashMap<>();
        public Theater save(Theater t){ map.put(t.id, t); return t; }
        public Optional<Theater> findById(long id){ return Optional.ofNullable(map.get(id)); }
        public List<Theater> findByCity(String city){
            // List<Theater> out = new ArrayList<>();
            List<Theater> res= map.values().stream().filter(x->x.city.equalsIgnoreCase(city)).toList();
            // for (Theater t : map.values()) if (t.city.equalsIgnoreCase(city)) out.add(t);
            // return out;
            return res;
        }
    }

    interface ScreenRepo {
        Screen save(Screen s);
        Optional<Screen> findById(long id);
        List<Screen> findByTheaterId(long theaterId);
    }
    static class InMemoryScreenRepo implements ScreenRepo {
        private final ConcurrentMap<Long, Screen> map = new ConcurrentHashMap<>();
        public Screen save(Screen s){ map.put(s.id, s); return s; }
        public Optional<Screen> findById(long id){ return Optional.ofNullable(map.get(id)); }
        public List<Screen> findByTheaterId(long theaterId){
            List<Screen> out = new ArrayList<>();
            for (Screen s : map.values()) if (s.theaterId == theaterId) out.add(s);
            return out;
        }
    }

    interface SeatRepo {
        Seat save(Seat s);
        Optional<Seat> findById(long id);
        List<Seat> findByScreenId(long screenId);
        Optional<Seat> findByScreenIdAndSeatCode(long screenId, String code);
    }
    static class InMemorySeatRepo implements SeatRepo {
        private final ConcurrentMap<Long, Seat> map = new ConcurrentHashMap<>();
        public Seat save(Seat s){ map.put(s.id, s); return s; }
        public Optional<Seat> findById(long id){ return Optional.ofNullable(map.get(id)); }
        public List<Seat> findByScreenId(long screenId){
            List<Seat> out = new ArrayList<>();
            for (Seat s : map.values()) if (s.screenId == screenId) out.add(s);
            return out;
        }
        public Optional<Seat> findByScreenIdAndSeatCode(long screenId, String code){
            Optional<Seat> res=map.values().stream().filter(x->x.screenId==screenId && x.seatCode.equalsIgnoreCase(code)).findFirst();
            // for (Seat s : map.values()) if (s.screenId == screenId && s.seatCode.equalsIgnoreCase(code)) return Optional.of(s);
            // return Optional.empty();
            return res;
        }
    }

    interface MovieRepo {
        Movie save(Movie m);
        Optional<Movie> findById(long id);
    }
    static class InMemoryMovieRepo implements MovieRepo {
        private final ConcurrentMap<Long, Movie> map = new ConcurrentHashMap<>();
        public Movie save(Movie m){ map.put(m.id, m); return m; }
        public Optional<Movie> findById(long id){ return Optional.ofNullable(map.get(id)); }
    }

    interface ShowRepo {
        Show save(Show s);
        Optional<Show> findById(long id);
        List<Show> findByTheaterAndDate(long theaterId, LocalDate date);
        List<Show> findByMovieAndDate(long movieId, LocalDate date);
    }
    static class InMemoryShowRepo implements ShowRepo {
        private final ConcurrentMap<Long, Show> map = new ConcurrentHashMap<>();
        public Show save(Show s){ map.put(s.id, s); return s; }
        public Optional<Show> findById(long id){ return Optional.ofNullable(map.get(id)); }
        public List<Show> findByTheaterAndDate(long theaterId, LocalDate date){
            List<Show> out = new ArrayList<>();
            for (Show sh : map.values()) {
                // need screen -> theater; keep simple: caller filters
                out.add(sh);
            }
            return out;
        }
        public List<Show> findByMovieAndDate(long movieId, LocalDate date){
            List<Show> out = new ArrayList<>();
            for (Show sh : map.values()) if (sh.movieId == movieId && sh.startTime.toLocalDate().equals(date)) out.add(sh);
            return out;
        }
    }

    interface BookingRepo {
        Booking save(Booking b);
        Optional<Booking> findByRef(String ref);
        List<Booking> findConfirmedByShow(long showId);
        List<Booking> findByShow(long showId);
        List<Booking> findByUser(long userId);
    }
    static class InMemoryBookingRepo implements BookingRepo {
        private final ConcurrentMap<Long, Booking> map = new ConcurrentHashMap<>();
        private final ConcurrentMap<String, Long> refIndex = new ConcurrentHashMap<>();
        private final AtomicLong idGen = new AtomicLong(1);

        public Booking createAndSave(long userId, long showId, Set<Long> seatIds, BigDecimal amount, BookingStatus status) {
            long id = idGen.getAndIncrement();
            String ref = generateRef();
            Booking b = new Booking(id, ref, userId, showId, seatIds, amount, status);
            map.put(id, b);
            refIndex.put(ref, id);
            return b;
        }

        @Override public Booking save(Booking b){ map.put(b.id, b); refIndex.put(b.bookingRef, b.id); return b; }
        @Override public Optional<Booking> findByRef(String ref){ Long id = refIndex.get(ref); return id == null ? Optional.empty() : Optional.ofNullable(map.get(id)); }
        @Override public List<Booking> findConfirmedByShow(long showId){
            List<Booking> out = new ArrayList<>();
            for (Booking b : map.values()) if (b.showId == showId && b.status == BookingStatus.CONFIRMED) out.add(b);
            return out;
        }
        @Override public List<Booking> findByShow(long showId){
            List<Booking> out = new ArrayList<>();
            for (Booking b : map.values()) if (b.showId == showId) out.add(b);
            return out;
        }
        @Override public List<Booking> findByUser(long userId){
            List<Booking> out = new ArrayList<>();
            for (Booking b : map.values()) if (b.userId == userId) out.add(b);
            return out;
        }

        private static String generateRef(){ return UUID.randomUUID().toString().replace("-", "").substring(0, 10).toUpperCase(); }
    }

    interface HoldRepo {
        SeatHold save(SeatHold h);
        Optional<SeatHold> findById(String holdId);
        void remove(String holdId);
    }
    static class InMemoryHoldRepo implements HoldRepo {
        private final ConcurrentMap<String, SeatHold> map = new ConcurrentHashMap<>();
        public SeatHold save(SeatHold h){ map.put(h.holdId, h); return h; }
        public Optional<SeatHold> findById(String holdId){ return Optional.ofNullable(map.get(holdId)); }
        public void remove(String holdId){ map.remove(holdId); }
    }

    /* -------------------------------
     * Seat Lock Manager (simulate row-level lock for show-seat)
     * ----------------------------- */

    /**
     * We manage a lock per (showId, seatId) pair using a nested map.
     * This allows concurrent bookings on different seats or shows.
     */
    static class SeatLockManager {
        private final ConcurrentMap<Long, ConcurrentMap<Long, ReentrantLock>> locks = new ConcurrentHashMap<>();

        private ReentrantLock getLock(long showId, long seatId) {
            locks.computeIfAbsent(showId, k -> new ConcurrentHashMap<>());
            ConcurrentMap<Long, ReentrantLock> showLocks = locks.get(showId);
            return showLocks.computeIfAbsent(seatId, k -> new ReentrantLock());
        }

        /**
         * Acquire locks for the given seatIds for a show in ascending seatId order to avoid deadlocks.
         */
        public List<ReentrantLock> acquireLocks(long showId, Collection<Long> seatIds) {
            List<Long> ids = new ArrayList<>(seatIds);
            Collections.sort(ids);
            List<ReentrantLock> acquired = new ArrayList<>();
            for (long sid : ids) {
                ReentrantLock lock = getLock(showId, sid);
                lock.lock();
                acquired.add(lock);
            }
            return acquired;
        }

        public void releaseLocks(Collection<ReentrantLock> locks) {
            for (ReentrantLock l : locks) {
                if (l.isHeldByCurrentThread()) l.unlock();
            }
        }
    }

    /* -----------------------------
     * Payment Service (stub)
     * ----------------------------- */

    static class PaymentResult {
        final boolean success;
        final String txnId;
        final String message;
        PaymentResult(boolean success, String txnId, String message){ this.success = success; this.txnId = txnId; this.message = message; }
        static PaymentResult success(String txnId){ return new PaymentResult(true, txnId, "OK"); }
        static PaymentResult failed(String msg){ return new PaymentResult(false, null, msg); }
    }
    static class PaymentService {
        PaymentResult charge(BigDecimal amount, String paymentToken) {
            try { Thread.sleep(50); } catch (InterruptedException ignored) {}
            // For demo: treat token "fail" as failure
            if ("fail".equalsIgnoreCase(paymentToken)) return PaymentResult.failed("card declined");
            return PaymentResult.success("TXN-" + UUID.randomUUID().toString().substring(0, 8).toUpperCase());
        }
        PaymentResult refund(String txnId, BigDecimal amount) {
            return PaymentResult.success("REF-" + UUID.randomUUID().toString().substring(0, 8).toUpperCase());
        }
    }

    /* -----------------------------
     * Seat Hold Manager (expiry scheduler)
     * ----------------------------- */

    /**
     * Manages holds and automatic expiry.
     * When creating a hold, we schedule an expiry task to release held seats if not confirmed.
     */
    static class SeatHoldManager {
        private final HoldRepo holdRepo;
        private final SeatLockManager lockManager;
        private final ScheduledExecutorService scheduler;
        private final BookingService bookingService; // to call release when expired
        private final int HOLD_EXPIRY_SECONDS;

        // track scheduled futures to cancel if hold confirmed/removed early
        private final ConcurrentMap<String, ScheduledFuture<?>> scheduledTasks = new ConcurrentHashMap<>();

        SeatHoldManager(HoldRepo holdRepo, SeatLockManager lockManager, BookingService bookingService, int holdExpirySeconds) {
            this.holdRepo = holdRepo;
            this.lockManager = lockManager;
            this.bookingService = bookingService;
            this.scheduler = Executors.newScheduledThreadPool(2);
            this.HOLD_EXPIRY_SECONDS = holdExpirySeconds;
        }

        SeatHold createHold(long userId, long showId, Set<Long> seatIds, BigDecimal amount) {
            String holdId = "HOLD-" + UUID.randomUUID().toString().replace("-", "").substring(0, 8).toUpperCase();
            SeatHold h = new SeatHold(holdId, userId, showId, seatIds, amount);
            holdRepo.save(h);
            // schedule expiry
            ScheduledFuture<?> future = scheduler.schedule(() -> {
                // expiry callback
                try {
                    bookingService.expireHold(holdId);
                } catch (Exception ex) {
                    // log in real system
                } finally {
                    scheduledTasks.remove(holdId);
                }
            }, HOLD_EXPIRY_SECONDS, TimeUnit.SECONDS);
            scheduledTasks.put(holdId, future);
            return h;
        }

        void cancelScheduled(String holdId) {
            ScheduledFuture<?> f = scheduledTasks.remove(holdId);
            if (f != null) f.cancel(false);
        }

        void shutdown() { scheduler.shutdown(); }
    }

    /* -----------------------------
     * Booking Service (core)
     * ----------------------------- */

    /**
     * BookingService responsibilities:
     * - Search available seats for a show
     * - Hold seats (locks seats, record hold, schedule expiry)
     * - Confirm hold (charge payment -> create CONFIRMED booking)
     * - Cancel booking (release seats, partial/full refund)
     * - Expire holds (releases seats automatically)
     *
     * Note: For demo, holds are stored in-memory. For real system, we'd persist holds and use DB transactions.
     */
    static class BookingService {
        private final SeatRepo seatRepo;
        private final ShowRepo showRepo;
        private final ScreenRepo screenRepo;
        private final BookingRepo bookingRepo;
        private final HoldRepo holdRepo;
        private final SeatLockManager lockManager;
        private final SeatHoldManager holdManager;
        private final PaymentService paymentService;
        private final InMemoryBookingRepo inMemoryBookingRepo; // to create id'ed bookings

        BookingService(SeatRepo seatRepo, ShowRepo showRepo, ScreenRepo screenRepo,
                       BookingRepo bookingRepo, HoldRepo holdRepo, SeatLockManager lockManager,
                       SeatHoldManager holdManager, PaymentService paymentService, InMemoryBookingRepo inMemoryBookingRepo) {
            this.seatRepo = seatRepo; this.showRepo = showRepo; this.screenRepo = screenRepo;
            this.bookingRepo = bookingRepo; this.holdRepo = holdRepo; this.lockManager = lockManager;
            this.holdManager = holdManager; this.paymentService = paymentService; this.inMemoryBookingRepo = inMemoryBookingRepo;
        }

        /**
         * Returns available seatIds for a show by filtering out seats already CONFIRMED in bookings and currently held seats.
         */
        public List<Seat> getAvailableSeats(long showId) {
            Show show = showRepo.findById(showId).orElseThrow(() -> new NotFoundException("Show not found"));
            Screen screen = screenRepo.findById(show.screenId).orElseThrow(() -> new NotFoundException("Screen not found"));
            List<Seat> seats = seatRepo.findByScreenId(screen.id);

            // collect confirmed seats
            Set<Long> confirmed = new HashSet<>();
            for (Booking b : bookingRepo.findConfirmedByShow(showId)) confirmed.addAll(b.seatIds);

            // collect active holds
            Set<Long> held = new HashSet<>();
            // iterate holds in repo (in-memory): we don't have a direct index, so check all holds
            if (holdRepo instanceof InMemoryHoldRepo) {
                InMemoryHoldRepo hrepo = (InMemoryHoldRepo) holdRepo;
                for (SeatHold h : hrepo.map.values()) {
                    if (h.showId == showId && h.active) held.addAll(h.seatIds);
                }
            }

            List<Seat> out = new ArrayList<>();
            for (Seat s : seats) if (!confirmed.contains(s.id) && !held.contains(s.id)) out.add(s);
            return out;
        }

        /**
         * Hold seats for a user for a show. This method:
         *  - acquires locks for seatIds (to avoid races)
         *  - double-checks seats are still free (no confirmed booking)
         *  - creates a SeatHold and schedules expiry
         *
         * returns holdId
         */
        public String holdSeats(long userId, long showId, Set<Long> seatIds) {
            if (seatIds == null || seatIds.isEmpty()) throw new InvalidRequestException("No seats selected");
            Show show = showRepo.findById(showId).orElseThrow(() -> new NotFoundException("Show not found"));
            // Acquire locks
            List<ReentrantLock> locks = lockManager.acquireLocks(showId, seatIds);
            try {
                // check seats exist & are in same screen
                for (long sid : seatIds) {
                    Seat s = seatRepo.findById(sid).orElseThrow(() -> new NotFoundException("Seat not found: " + sid));
                    if (s.screenId != show.screenId) throw new InvalidRequestException("Seat " + sid + " not in show screen");
                }
                // check CONFIRMED bookings for overlap (seat cannot be in any confirmed booking for this show)
                Set<Long> confirmed = new HashSet<>();
                for (Booking b : bookingRepo.findConfirmedByShow(showId)) confirmed.addAll(b.seatIds);
                for (long sid : seatIds) {
                    if (confirmed.contains(sid)) throw new SeatUnavailableException("Seat already booked: " + sid);
                }
                // check existing holds overlap
                if (holdRepo instanceof InMemoryHoldRepo) {
                    InMemoryHoldRepo hrepo = (InMemoryHoldRepo) holdRepo;
                    for (SeatHold h : hrepo.map.values()) {
                        if (!h.active) continue;
                        if (h.showId != showId) continue;
                        for (long sid : seatIds) if (h.seatIds.contains(sid)) throw new SeatUnavailableException("Seat currently held: " + sid);
                    }
                }
                // compute amount
                BigDecimal amount = BigDecimal.ZERO;
                for (long sid : seatIds) {
                    Seat s = seatRepo.findById(sid).get();
                    amount = amount.add(s.price);
                }
                SeatHold hold = holdManager.createHold(userId, showId, seatIds, amount);
                return hold.holdId;
            } finally {
                lockManager.releaseLocks(locks);
            }
        }

        /**
         * Confirm a hold: charge payment and create a CONFIRMED booking atomically
         * For demo we do payment while holding locks briefly to avoid races.
         */
        public Booking confirmHold(String holdId, String paymentToken) {
            SeatHold hold = holdRepo.findById(holdId).orElseThrow(() -> new NotFoundException("Hold not found"));
            if (!hold.active) throw new InvalidRequestException("Hold not active");
            long showId = hold.showId;
            Set<Long> seatIds = new HashSet<>(hold.seatIds);

            // Acquire locks for seats
            List<ReentrantLock> locks = lockManager.acquireLocks(showId, seatIds);
            try {
                // double check seats not CONFIRMED in meantime
                Set<Long> confirmed = new HashSet<>();
                for (Booking b : bookingRepo.findConfirmedByShow(showId)) confirmed.addAll(b.seatIds);
                for (long sid : seatIds) if (confirmed.contains(sid)) throw new SeatUnavailableException("Seat already booked: " + sid);

                // perform payment
                PaymentResult pr = paymentService.charge(hold.amount, paymentToken);
                if (!pr.success) {
                    // mark hold inactive, remove
                    hold.deactivate();
                    holdRepo.remove(holdId);
                    throw new PaymentFailedException(pr.message);
                }

                // create booking
                Booking booking = inMemoryBookingRepo.createAndSave(hold.userId, showId, seatIds, hold.amount, BookingStatus.CONFIRMED);
                booking.setPaymentTxnId(pr.txnId);

                // mark hold inactive and cancel scheduled expiry
                hold.deactivate();
                holdRepo.remove(holdId);
                holdManager.cancelScheduled(holdId);

                return booking;
            } finally {
                lockManager.releaseLocks(locks);
            }
        }

        /**
         * Cancel a confirmed booking (refund stub) or release a hold.
         */
        public Booking cancelBooking(String bookingRef) {
            Booking b = bookingRepo.findByRef(bookingRef).orElseThrow(() -> new NotFoundException("Booking not found"));
            synchronized (b) {
                if (b.status == BookingStatus.CANCELLED) return b;
                if (b.status == BookingStatus.CONFIRMED) {
                    // Acquire locks to release seats safely
                    List<ReentrantLock> locks = lockManager.acquireLocks(b.showId, b.seatIds);
                    try {
                        b.setStatus(BookingStatus.CANCELLED);
                        bookingRepo.save(b);
                        if (b.paymentTxnId != null) paymentService.refund(b.paymentTxnId, b.amount);
                        return b;
                    } finally {
                        lockManager.releaseLocks(locks);
                    }
                } else {
                    b.setStatus(BookingStatus.CANCELLED);
                    bookingRepo.save(b);
                    return b;
                }
            }
        }

        /**
         * Called by SeatHoldManager when hold expires.
         */
        public void expireHold(String holdId) {
            Optional<SeatHold> opt = holdRepo.findById(holdId);
            if (opt.isEmpty()) return;
            SeatHold h = opt.get();
            if (!h.active) {
                holdRepo.remove(holdId);
                return;
            }
            // Acquire locks and then deactivate
            List<ReentrantLock> locks = lockManager.acquireLocks(h.showId, h.seatIds);
            try {
                h.deactivate();
                holdRepo.remove(holdId);
            } finally {
                lockManager.releaseLocks(locks);
            }
        }
    }

    /* -----------------------------
     * Demo main: seed data and simulate scenarios including concurrent holds
     * ----------------------------- */

    public static void main(String[] args) throws Exception {
        // Repos & services
        InMemoryTheaterRepo theaterRepo = new InMemoryTheaterRepo();
        InMemoryScreenRepo screenRepo = new InMemoryScreenRepo();
        InMemorySeatRepo seatRepo = new InMemorySeatRepo();
        InMemoryMovieRepo movieRepo = new InMemoryMovieRepo();
        InMemoryShowRepo showRepo = new InMemoryShowRepo();
        InMemoryBookingRepo bookingRepo = new InMemoryBookingRepo();
        InMemoryHoldRepo holdRepo = new InMemoryHoldRepo();

        SeatLockManager lockManager = new SeatLockManager();
        PaymentService paymentService = new PaymentService();
        BookingService bookingService = null; // initialize after holdManager
        SeatHoldManager holdManager = null;

        // seed theaters, screens, seats
        Theater t1 = new Theater(1L, "PVR Mall", "Bengaluru");
        theaterRepo.save(t1);
        Screen s1 = new Screen(1L, t1.id, "Screen 1", 5, 10); // 5 rows x 10 seats
        screenRepo.save(s1);

        // create seats A1..A10, B1..B10... with simple pricing
        AtomicLong seatIdGen = new AtomicLong(100);
        for (int r = 0; r < s1.totalRows; r++) {
            char row = (char)('A' + r);
            for (int c = 1; c <= s1.seatsPerRow; c++) {
                String code = row + String.valueOf(c);
                SeatType st = (r < 1) ? SeatType.VIP : (r < 3) ? SeatType.PREMIUM : SeatType.REGULAR;
                BigDecimal price = st == SeatType.VIP ? BigDecimal.valueOf(500) : st == SeatType.PREMIUM ? BigDecimal.valueOf(350) : BigDecimal.valueOf(200);
                Seat seat = new Seat(seatIdGen.getAndIncrement(), s1.id, code, st, price);
                seatRepo.save(seat);
            }
        }

        // seed movie and show
        Movie m1 = new Movie(1L, "Interstellar", 170);
        movieRepo.save(m1);
        LocalDateTime showStart = LocalDateTime.of(LocalDate.now().plusDays(1), LocalTime.of(18, 0));
        Show sh1 = new Show(1L, m1.id, s1.id, showStart, showStart.plusMinutes(m1.durationMinutes));
        showRepo.save(sh1);

        // initialize holdManager and bookingService now
        InMemoryBookingRepo inMemBookingRepo = bookingRepo;
        holdManager = new SeatHoldManager(holdRepo, lockManager, null, 10); // temp null bookingService
        bookingService = new BookingService(seatRepo, showRepo, screenRepo, bookingRepo, holdRepo, lockManager, holdManager, paymentService, inMemBookingRepo);
        // inject bookingService into holdManager
        holdManager.bookingService = bookingService;

        // List available seats before booking
        System.out.println("Available seats initially: " + bookingService.getAvailableSeats(sh1.id).size());

        // Single-user happy path: hold seats and confirm
        Set<Long> chosenSeats = new HashSet<>();
        // pick first two seats
        List<Seat> allSeats = seatRepo.findByScreenId(s1.id);
        chosenSeats.add(allSeats.get(0).id);
        chosenSeats.add(allSeats.get(1).id);

        String holdId = bookingService.holdSeats(501L, sh1.id, chosenSeats);
        System.out.println("Hold created: " + holdId);

        Booking confirmed = null;
        try {
            confirmed = bookingService.confirmHold(holdId, "tok_ok");
            System.out.println("Booking confirmed: ref=" + confirmed.bookingRef + " seats=" + confirmed.seatIds + " amount=" + confirmed.amount);
        } catch (Exception ex) {
            System.err.println("Confirm failed: " + ex.getMessage());
        }

        // Attempt concurrent holds on same seat by two users
        Set<Long> seatToFight = new HashSet<>();
        seatToFight.add(allSeats.get(2).id); // seat index 2

        ExecutorService es = Executors.newFixedThreadPool(2);
        Callable<String> user1 = () -> {
            try {
                String h = bookingService.holdSeats(601L, sh1.id, seatToFight);
                Thread.sleep(100); // simulate some delay before confirm
                Booking b = bookingService.confirmHold(h, "tok_ok");
                return "User1 success bookingRef=" + b.bookingRef;
            } catch (Exception ex) {
                return "User1 failed: " + ex.getClass().getSimpleName() + " - " + ex.getMessage();
            }
        };
        Callable<String> user2 = () -> {
            try {
                // small stagger to increase race
                Thread.sleep(10);
                String h = bookingService.holdSeats(602L, sh1.id, seatToFight);
                Thread.sleep(100);
                Booking b = bookingService.confirmHold(h, "tok_ok");
                return "User2 success bookingRef=" + b.bookingRef;
            } catch (Exception ex) {
                return "User2 failed: " + ex.getClass().getSimpleName() + " - " + ex.getMessage();
            }
        };

        Future<String> f1 = es.submit(user1);
        Future<String> f2 = es.submit(user2);
        System.out.println(f1.get());
        System.out.println(f2.get());
        es.shutdown();

        // Wait a bit to let auto-expire holds (we set expiry to 10s)
        System.out.println("Available seats after operations: " + bookingService.getAvailableSeats(sh1.id).size());

        // Demo cancel booking
        if (confirmed != null) {
            bookingService.cancelBooking(confirmed.bookingRef);
            System.out.println("Cancelled booking: " + confirmed.bookingRef);
            System.out.println("Available seats after cancellation: " + bookingService.getAvailableSeats(sh1.id).size());
        }

        // Shutdown hold manager scheduler
        holdManager.shutdown();
    }
}

```
