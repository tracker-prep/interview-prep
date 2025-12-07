Hotel booking LLD
```
import java.math.BigDecimal;
import java.time.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.concurrent.locks.*;

/**
 * Movie Ticket Booking System - Single-file plain Java LLD implementation (fully corrected)
 *
 * Fixes:
 * - SeatHoldManager no longer requires BookingService in constructor.
 * - bookingService is created once (effectively final) after holdManager and then injected via setter.
 * - Lambdas capture an effectively final bookingService variable (no compile errors).
 *
 * Compile & run:
 *   javac MovieBookingLLD.java
 *   java MovieBookingLLD
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
            this.id = id;
            this.bookingRef = bookingRef;
            this.userId = userId;
            this.showId = showId;
            this.seatIds = new HashSet<>(seatIds);
            this.amount = amount;
            this.status = status;
            this.createdAt = Instant.now();
            this.updatedAt = this.createdAt;
        }

        void setStatus(BookingStatus s) { this.status = s; this.updatedAt = Instant.now(); }
        void setPaymentTxnId(String txn) { this.paymentTxnId = txn; this.updatedAt = Instant.now(); }
    }

    static class SeatHold {
        final String holdId;
        final long userId;
        final long showId;
        final Set<Long> seatIds;
        final BigDecimal amount;
        final Instant createdAt;
        volatile boolean active = true;
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
     * In-memory Repositories
     * ----------------------------- */

    interface TheaterRepo { Theater save(Theater t); Optional<Theater> findById(long id); List<Theater> findByCity(String city); }
    static class InMemoryTheaterRepo implements TheaterRepo {
        private final ConcurrentMap<Long, Theater> map = new ConcurrentHashMap<>();
        public Theater save(Theater t){ map.put(t.id, t); return t; }
        public Optional<Theater> findById(long id){ return Optional.ofNullable(map.get(id)); }
        public List<Theater> findByCity(String city){ return map.values().stream().filter(x->x.city.equalsIgnoreCase(city)).toList(); }
    }

    interface ScreenRepo { Screen save(Screen s); Optional<Screen> findById(long id); List<Screen> findByTheaterId(long theaterId); }
    static class InMemoryScreenRepo implements ScreenRepo {
        private final ConcurrentMap<Long, Screen> map = new ConcurrentHashMap<>();
        public Screen save(Screen s){ map.put(s.id, s); return s; }
        public Optional<Screen> findById(long id){ return Optional.ofNullable(map.get(id)); }
        public List<Screen> findByTheaterId(long theaterId){ List<Screen> out = new ArrayList<>(); for (Screen s : map.values()) if (s.theaterId == theaterId) out.add(s); return out; }
    }

    interface SeatRepo { Seat save(Seat s); Optional<Seat> findById(long id); List<Seat> findByScreenId(long screenId); Optional<Seat> findByScreenIdAndSeatCode(long screenId, String code); }
    static class InMemorySeatRepo implements SeatRepo {
        private final ConcurrentMap<Long, Seat> map = new ConcurrentHashMap<>();
        public Seat save(Seat s){ map.put(s.id, s); return s; }
        public Optional<Seat> findById(long id){ return Optional.ofNullable(map.get(id)); }
        public List<Seat> findByScreenId(long screenId){ List<Seat> out = new ArrayList<>(); for (Seat s : map.values()) if (s.screenId == screenId) out.add(s); return out; }
        public Optional<Seat> findByScreenIdAndSeatCode(long screenId, String code){ return map.values().stream().filter(x->x.screenId==screenId && x.seatCode.equalsIgnoreCase(code)).findFirst(); }
    }

    interface MovieRepo { Movie save(Movie m); Optional<Movie> findById(long id); }
    static class InMemoryMovieRepo implements MovieRepo {
        private final ConcurrentMap<Long, Movie> map = new ConcurrentHashMap<>();
        public Movie save(Movie m){ map.put(m.id, m); return m; }
        public Optional<Movie> findById(long id){ return Optional.ofNullable(map.get(id)); }
    }

    interface ShowRepo { Show save(Show s); Optional<Show> findById(long id); List<Show> findByTheaterAndDate(long theaterId, LocalDate date); List<Show> findByMovieAndDate(long movieId, LocalDate date); }
    static class InMemoryShowRepo implements ShowRepo {
        private final ConcurrentMap<Long, Show> map = new ConcurrentHashMap<>();
        public Show save(Show s){ map.put(s.id, s); return s; }
        public Optional<Show> findById(long id){ return Optional.ofNullable(map.get(id)); }
        public List<Show> findByTheaterAndDate(long theaterId, LocalDate date){ List<Show> out = new ArrayList<>(); for (Show sh : map.values()) out.add(sh); return out; }
        public List<Show> findByMovieAndDate(long movieId, LocalDate date){ List<Show> out = new ArrayList<>(); for (Show sh : map.values()) if (sh.movieId == movieId && sh.startTime.toLocalDate().equals(date)) out.add(sh); return out; }
    }

    interface BookingRepo { Booking save(Booking b); Optional<Booking> findByRef(String ref); List<Booking> findConfirmedByShow(long showId); List<Booking> findByShow(long showId); List<Booking> findByUser(long userId); }
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
        @Override public List<Booking> findConfirmedByShow(long showId){ List<Booking> out = new ArrayList<>(); for (Booking b : map.values()) if (b.showId == showId && b.status == BookingStatus.CONFIRMED) out.add(b); return out; }
        @Override public List<Booking> findByShow(long showId){ List<Booking> out = new ArrayList<>(); for (Booking b : map.values()) if (b.showId == showId) out.add(b); return out; }
        @Override public List<Booking> findByUser(long userId){ List<Booking> out = new ArrayList<>(); for (Booking b : map.values()) if (b.userId == userId) out.add(b); return out; }
        private static String generateRef(){ return UUID.randomUUID().toString().replace("-", "").substring(0, 10).toUpperCase(); }
    }

    interface HoldRepo { SeatHold save(SeatHold h); Optional<SeatHold> findById(String holdId); void remove(String holdId); }
    static class InMemoryHoldRepo implements HoldRepo {
        private final ConcurrentMap<String, SeatHold> map = new ConcurrentHashMap<>();
        public SeatHold save(SeatHold h){ map.put(h.holdId, h); return h; }
        public Optional<SeatHold> findById(String holdId){ return Optional.ofNullable(map.get(holdId)); }
        public void remove(String holdId){ map.remove(holdId); }
    }

    /* -------------------------------
     * Seat Lock Manager
     * ----------------------------- */

    static class SeatLockManager {
        private final ConcurrentMap<Long, ConcurrentMap<Long, ReentrantLock>> locks = new ConcurrentHashMap<>();
        private ReentrantLock getLock(long showId, long seatId) {
            locks.computeIfAbsent(showId, k -> new ConcurrentHashMap<>());
            ConcurrentMap<Long, ReentrantLock> showLocks = locks.get(showId);
            return showLocks.computeIfAbsent(seatId, k -> new ReentrantLock());
        }
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
        public void releaseLocks(Collection<ReentrantLock> locks) { for (ReentrantLock l : locks) if (l.isHeldByCurrentThread()) l.unlock(); }
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
            if ("fail".equalsIgnoreCase(paymentToken)) return PaymentResult.failed("card declined");
            return PaymentResult.success("TXN-" + UUID.randomUUID().toString().substring(0, 8).toUpperCase());
        }
        PaymentResult refund(String txnId, BigDecimal amount) { return PaymentResult.success("REF-" + UUID.randomUUID().toString().substring(0, 8).toUpperCase()); }
    }

    /* -----------------------------
     * Seat Hold Manager (expiry) - setter injection
     * ----------------------------- */

    static class SeatHoldManager {
        private final HoldRepo holdRepo;
        private final SeatLockManager lockManager;
        private final ScheduledExecutorService scheduler;
        private BookingService bookingService; // injected later
        private final int HOLD_EXPIRY_SECONDS;
        private final ConcurrentMap<String, ScheduledFuture<?>> scheduledTasks = new ConcurrentHashMap<>();

        SeatHoldManager(HoldRepo holdRepo, SeatLockManager lockManager, int holdExpirySeconds) {
            this.holdRepo = holdRepo;
            this.lockManager = lockManager;
            this.scheduler = Executors.newScheduledThreadPool(2);
            this.HOLD_EXPIRY_SECONDS = holdExpirySeconds;
        }

        public void setBookingService(BookingService bookingService) { this.bookingService = bookingService; }

        SeatHold createHold(long userId, long showId, Set<Long> seatIds, BigDecimal amount) {
            String holdId = "HOLD-" + UUID.randomUUID().toString().replace("-", "").substring(0, 8).toUpperCase();
            SeatHold h = new SeatHold(holdId, userId, showId, seatIds, amount);
            holdRepo.save(h);
            ScheduledFuture<?> future = scheduler.schedule(() -> {
                try {
                    BookingService bs = this.bookingService;
                    if (bs != null) bs.expireHold(holdId);
                } catch (Exception ex) {
                    // ignore
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

    static class BookingService {
        private final SeatRepo seatRepo;
        private final ShowRepo showRepo;
        private final ScreenRepo screenRepo;
        private final BookingRepo bookingRepo;
        private final HoldRepo holdRepo;
        private final SeatLockManager lockManager;
        private final SeatHoldManager holdManager;
        private final PaymentService paymentService;
        private final InMemoryBookingRepo inMemoryBookingRepo;

        BookingService(SeatRepo seatRepo, ShowRepo showRepo, ScreenRepo screenRepo,
                       BookingRepo bookingRepo, HoldRepo holdRepo, SeatLockManager lockManager,
                       SeatHoldManager holdManager, PaymentService paymentService, InMemoryBookingRepo inMemoryBookingRepo) {
            this.seatRepo = seatRepo; this.showRepo = showRepo; this.screenRepo = screenRepo;
            this.bookingRepo = bookingRepo; this.holdRepo = holdRepo; this.lockManager = lockManager;
            this.holdManager = holdManager; this.paymentService = paymentService; this.inMemoryBookingRepo = inMemoryBookingRepo;
        }

        public List<Seat> getAvailableSeats(long showId) {
            Show show = showRepo.findById(showId).orElseThrow(() -> new NotFoundException("Show not found"));
            Screen screen = screenRepo.findById(show.screenId).orElseThrow(() -> new NotFoundException("Screen not found"));
            List<Seat> seats = seatRepo.findByScreenId(screen.id);
            Set<Long> confirmed = new HashSet<>();
            for (Booking b : bookingRepo.findConfirmedByShow(showId)) confirmed.addAll(b.seatIds);
            Set<Long> held = new HashSet<>();
            if (holdRepo instanceof InMemoryHoldRepo) {
                InMemoryHoldRepo hrepo = (InMemoryHoldRepo) holdRepo;
                for (SeatHold h : hrepo.map.values()) if (h.showId == showId && h.active) held.addAll(h.seatIds);
            }
            List<Seat> out = new ArrayList<>();
            for (Seat s : seats) if (!confirmed.contains(s.id) && !held.contains(s.id)) out.add(s);
            return out;
        }

        public String holdSeats(long userId, long showId, Set<Long> seatIds) {
            if (seatIds == null || seatIds.isEmpty()) throw new InvalidRequestException("No seats selected");
            Show show = showRepo.findById(showId).orElseThrow(() -> new NotFoundException("Show not found"));
            List<ReentrantLock> locks = lockManager.acquireLocks(showId, seatIds);
            try {
                for (long sid : seatIds) {
                    Seat s = seatRepo.findById(sid).orElseThrow(() -> new NotFoundException("Seat not found: " + sid));
                    if (s.screenId != show.screenId) throw new InvalidRequestException("Seat " + sid + " not in show screen");
                }
                Set<Long> confirmed = new HashSet<>();
                for (Booking b : bookingRepo.findConfirmedByShow(showId)) confirmed.addAll(b.seatIds);
                for (long sid : seatIds) if (confirmed.contains(sid)) throw new SeatUnavailableException("Seat already booked: " + sid);
                if (holdRepo instanceof InMemoryHoldRepo) {
                    InMemoryHoldRepo hrepo = (InMemoryHoldRepo) holdRepo;
                    for (SeatHold h : hrepo.map.values()) {
                        if (!h.active) continue;
                        if (h.showId != showId) continue;
                        for (long sid : seatIds) if (h.seatIds.contains(sid)) throw new SeatUnavailableException("Seat currently held: " + sid);
                    }
                }
                BigDecimal amount = BigDecimal.ZERO;
                for (long sid : seatIds) amount = amount.add(seatRepo.findById(sid).get().price);
                SeatHold hold = holdManager.createHold(userId, showId, seatIds, amount);
                return hold.holdId;
            } finally {
                lockManager.releaseLocks(locks);
            }
        }

        public Booking confirmHold(String holdId, String paymentToken) {
            SeatHold hold = holdRepo.findById(holdId).orElseThrow(() -> new NotFoundException("Hold not found"));
            if (!hold.active) throw new InvalidRequestException("Hold not active");
            long showId = hold.showId;
            Set<Long> seatIds = new HashSet<>(hold.seatIds);
            List<ReentrantLock> locks = lockManager.acquireLocks(showId, seatIds);
            try {
                Set<Long> confirmed = new HashSet<>();
                for (Booking b : bookingRepo.findConfirmedByShow(showId)) confirmed.addAll(b.seatIds);
                for (long sid : seatIds) if (confirmed.contains(sid)) throw new SeatUnavailableException("Seat already booked: " + sid);
                PaymentResult pr = paymentService.charge(hold.amount, paymentToken);
                if (!pr.success) { hold.deactivate(); holdRepo.remove(holdId); throw new PaymentFailedException(pr.message); }
                Booking booking = inMemoryBookingRepo.createAndSave(hold.userId, showId, seatIds, hold.amount, BookingStatus.CONFIRMED);
                booking.setPaymentTxnId(pr.txnId);
                hold.deactivate();
                holdRepo.remove(holdId);
                holdManager.cancelScheduled(holdId);
                return booking;
            } finally {
                lockManager.releaseLocks(locks);
            }
        }

        public Booking cancelBooking(String bookingRef) {
            Booking b = bookingRepo.findByRef(bookingRef).orElseThrow(() -> new NotFoundException("Booking not found"));
            synchronized (b) {
                if (b.status == BookingStatus.CANCELLED) return b;
                if (b.status == BookingStatus.CONFIRMED) {
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

        public void expireHold(String holdId) {
            Optional<SeatHold> opt = holdRepo.findById(holdId);
            if (opt.isEmpty()) return;
            SeatHold h = opt.get();
            if (!h.active) { holdRepo.remove(holdId); return; }
            List<ReentrantLock> locks = lockManager.acquireLocks(h.showId, h.seatIds);
            try { h.deactivate(); holdRepo.remove(holdId); } finally { lockManager.releaseLocks(locks); }
        }
    }

    /* -----------------------------
     * Demo main
     * ----------------------------- */

    public static void main(String[] args) throws Exception {
        InMemoryTheaterRepo theaterRepo = new InMemoryTheaterRepo();
        InMemoryScreenRepo screenRepo = new InMemoryScreenRepo();
        InMemorySeatRepo seatRepo = new InMemorySeatRepo();
        InMemoryMovieRepo movieRepo = new InMemoryMovieRepo();
        InMemoryShowRepo showRepo = new InMemoryShowRepo();
        InMemoryBookingRepo bookingRepo = new InMemoryBookingRepo();
        InMemoryHoldRepo holdRepo = new InMemoryHoldRepo();

        SeatLockManager lockManager = new SeatLockManager();
        PaymentService paymentService = new PaymentService();

        // seed data
        Theater t1 = new Theater(1L, "PVR Mall", "Bengaluru");
        theaterRepo.save(t1);
        Screen s1 = new Screen(1L, t1.id, "Screen 1", 5, 10);
        screenRepo.save(s1);

        AtomicLong seatIdGen = new AtomicLong(100);
        for (int r = 0; r < s1.totalRows; r++) {
            for (int c = 1; c <= s1.seatsPerRow; c++) {
                String code = (char)('A'+r) + String.valueOf(c);
                SeatType st = (r < 1) ? SeatType.VIP : (r < 3) ? SeatType.PREMIUM : SeatType.REGULAR;
                BigDecimal price = st == SeatType.VIP ? BigDecimal.valueOf(500) : st == SeatType.PREMIUM ? BigDecimal.valueOf(350) : BigDecimal.valueOf(200);
                seatRepo.save(new Seat(seatIdGen.getAndIncrement(), s1.id, code, st, price));
            }
        }

        Movie m1 = new Movie(1L, "Interstellar", 170); movieRepo.save(m1);
        LocalDateTime showStart = LocalDateTime.of(LocalDate.now().plusDays(1), LocalTime.of(18, 0));
        Show sh1 = new Show(1L, m1.id, s1.id, showStart, showStart.plusMinutes(m1.durationMinutes));
        showRepo.save(sh1);

        // Create holdManager first (no BookingService)
        SeatHoldManager holdManager = new SeatHoldManager(holdRepo, lockManager, 10);

        // Create bookingService (assigned once) and inject holdManager
        final BookingService bookingService = new BookingService(seatRepo, showRepo, screenRepo, bookingRepo, holdRepo, lockManager, holdManager, paymentService, bookingRepo);

        // Inject bookingService into holdManager via setter
        holdManager.setBookingService(bookingService);

        System.out.println("Available seats initially: " + bookingService.getAvailableSeats(sh1.id).size());

        // Single-user happy path
        List<Seat> allSeats = seatRepo.findByScreenId(s1.id);
        Set<Long> chosenSeats = new HashSet<>(Arrays.asList(allSeats.get(0).id, allSeats.get(1).id));

        String holdId = bookingService.holdSeats(501L, sh1.id, chosenSeats);
        System.out.println("Hold created: " + holdId);

        Booking confirmed = null;
        try {
            confirmed = bookingService.confirmHold(holdId, "tok_ok");
            System.out.println("Booking confirmed: ref=" + confirmed.bookingRef + " seats=" + confirmed.seatIds + " amount=" + confirmed.amount);
        } catch (Exception ex) {
            System.err.println("Confirm failed: " + ex.getMessage());
        }

        // Concurrent attempt
        Set<Long> seatToFight = Collections.singleton(allSeats.get(2).id);

        ExecutorService es = Executors.newFixedThreadPool(2);
        Callable<String> user1 = () -> {
            try {
                String h = bookingService.holdSeats(601L, sh1.id, seatToFight);
                Thread.sleep(100);
                Booking b = bookingService.confirmHold(h, "tok_ok");
                return "User1 success bookingRef=" + b.bookingRef;
            } catch (Exception ex) { return "User1 failed: " + ex.getMessage(); }
        };
        Callable<String> user2 = () -> {
            try {
                Thread.sleep(10);
                String h = bookingService.holdSeats(602L, sh1.id, seatToFight);
                Thread.sleep(100);
                Booking b = bookingService.confirmHold(h, "tok_ok");
                return "User2 success bookingRef=" + b.bookingRef;
            } catch (Exception ex) { return "User2 failed: " + ex.getMessage(); }
        };

        Future<String> f1 = es.submit(user1);
        Future<String> f2 = es.submit(user2);
        System.out.println(f1.get());
        System.out.println(f2.get());
        es.shutdown();

        System.out.println("Available seats after operations: " + bookingService.getAvailableSeats(sh1.id).size());

        if (confirmed != null) {
            bookingService.cancelBooking(confirmed.bookingRef);
            System.out.println("Cancelled booking: " + confirmed.bookingRef);
            System.out.println("Available seats after cancellation: " + bookingService.getAvailableSeats(sh1.id).size());
        }

        holdManager.shutdown();
    }
}

```
