//! R4 (docs/proposals/2026-11-15-r4-rust-serving-loop.md): the GPU process's native-front connections in Rust.
//!
//! `Loop` follows serving/r4.py's contract (the Python `PyLoop` is its stand-in, and serving/tests/test_r4.py holds
//! both to it). Per connection, a reader thread reads the binproto frames (u32 length, payload): a search frame
//! (type 1) goes on the batch queue with its id, index name and enqueue time, any other frame on the calls queue. A
//! writer thread sends what `send` queues, in order. `next_batch` and `next_call` wait with the GIL released.
//!
//! The GIL rule: nothing here holds a Python object. The queues hold Rust bytes and the batch is turned into Python
//! objects only after the wait, under the GIL. The load slot (serving/gpu.py `LoadTable`, `load.bin`) is written with
//! `pwrite` under the queue's lock, so its value is never stale against a later one.
//!
//! Shutdown (`close`): no new connection or frame is taken; every connection's read side is shut, so its reader ends
//! and its writer sends what is already queued, then closes the socket. `close` waits (without the GIL) up to its
//! timeout for that and returns how many connections are still open. A connection whose threads have both ended is
//! reported once by `closed_conns`, so the caller can forget it.
//!
//! CPU (`stats()["cpu_s"]`): each Loop thread's time on a CPU, from Linux's /proc/<pid>/task/<tid>/schedstat (its first
//! field, nanoseconds), the live threads read when asked and the ended ones summed as they end; 0 where /proc has no
//! schedstat (not Linux).

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};
use std::collections::{HashMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::net::Shutdown;
use std::os::unix::fs::FileExt;
use std::os::unix::io::{FromRawFd, IntoRawFd, RawFd};
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crate::reply;

const READ_BUF: usize = 1 << 16;

/// What a connection's writer sends: bytes already made, or (R4d) a batch's replies collected under the GIL, whose
/// kind 4 frame the writer lays out without it.
enum OutItem {
    Bytes(Vec<u8>),
    Batch4(reply::Collected),
}

enum Entry {
    Search { conn: u32, id: u64, name: String, payload: Vec<u8>, t: Instant },
    Py { token: u64, t: Instant },
}

struct Queue {
    d: VecDeque<Entry>,
    in_batch: usize,
}

struct Out {
    q: VecDeque<OutItem>,
    reader_done: bool,
    dead: bool,
}

struct Conn {
    stream: UnixStream,
    out: Mutex<Out>,
    cv: Condvar,
    reader_tid: AtomicU64,   // the reader's Linux thread id while it runs (0 otherwise), for its CPU
    writer_tid: AtomicU64,
    ended: AtomicU64,        // threads of this connection that have ended (2: both)
}

const STATS: [&str; 17] = [
    "conns_opened", "conns_closed", "frames", "searches", "calls", "bad_heads", "pushed", "batches", "batch_items",
    "sends", "send_bytes", "send_dropped", "send_errors", "max_out_depth", "batch4_sends", "batch4_frame_ns",
    "batch4_collect_ns",
];
const S_OPENED: usize = 0;
const S_CLOSED: usize = 1;
const S_FRAMES: usize = 2;
const S_SEARCHES: usize = 3;
const S_CALLS: usize = 4;
const S_BAD: usize = 5;
const S_PUSHED: usize = 6;
const S_BATCHES: usize = 7;
const S_ITEMS: usize = 8;
const S_SENDS: usize = 9;
const S_BYTES: usize = 10;
const S_DROPPED: usize = 11;
const S_ERRORS: usize = 12;
const S_MAXDEPTH: usize = 13;
const S_B4_SENDS: usize = 14;
const S_B4_FRAME_NS: usize = 15;    // the writers' frame layout (no GIL)
const S_B4_COLLECT_NS: usize = 16;  // send_batch4's collect (the GIL held)

struct Shared {
    q: Mutex<Queue>,
    cv: Condvar,
    calls: Mutex<VecDeque<(u32, Vec<u8>)>>,
    calls_cv: Condvar,
    conns: Mutex<HashMap<u32, Arc<Conn>>>,
    load: Option<(File, u64)>,
    st: [AtomicU64; 17],
    closing: AtomicBool,
    closed: Mutex<Vec<u32>>,         // connections both of whose threads have ended, not yet reported
    cpu_done_ns: [AtomicU64; 2],     // CPU of ended reader and writer threads
    live: AtomicU64,                 // reader and writer threads running (close waits for 0)
}

impl Shared {
    fn add(&self, i: usize, n: u64) {
        self.st[i].fetch_add(n, Ordering::Relaxed);
    }

    fn publish(&self, q: &Queue) {
        // under the queue's lock: this process's load, LoadTable.set's int64 (little-endian) in its slot
        if let Some((f, off)) = &self.load {
            let v = (q.d.len() + q.in_batch) as i64;
            let _ = f.write_at(&v.to_le_bytes(), *off);
        }
    }

    fn ended(&self, cid: u32, c: &Conn) {
        // a connection's reader or writer has ended; the second to end reports the connection closed
        if c.ended.fetch_add(1, Ordering::AcqRel) == 1 {
            self.closed.lock().unwrap().push(cid);
        }
    }

    fn push(&self, e: Entry) {
        let mut q = self.q.lock().unwrap();
        q.d.push_back(e);
        self.publish(&q);
        drop(q);
        self.cv.notify_one();
    }

    /// serving/gpu.py `_BatchQueue.take`: the first item (waiting up to `first`), then what is queued up to `mb`,
    /// then, with a `wait`, arrivals until `wait` after the first item or the batch is full.
    fn take(&self, mb: usize, wait: Duration, first: Duration) -> Vec<Entry> {
        let mut q = self.q.lock().unwrap();
        if q.d.is_empty() {
            let end = Instant::now() + first;
            while q.d.is_empty() {
                let now = Instant::now();
                if now >= end {
                    return Vec::new();
                }
                q = self.cv.wait_timeout(q, end - now).unwrap().0;
            }
        }
        let deadline = Instant::now() + wait;
        let k = mb.min(q.d.len());
        let mut batch: Vec<Entry> = q.d.drain(..k).collect();
        while batch.len() < mb && !wait.is_zero() {
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            if q.d.is_empty() {
                q = self.cv.wait_timeout(q, deadline - now).unwrap().0;
            }
            while batch.len() < mb {
                match q.d.pop_front() {
                    Some(e) => batch.push(e),
                    None => break,
                }
            }
        }
        drop(q);
        self.add(S_BATCHES, 1);
        self.add(S_ITEMS, batch.len() as u64);
        batch
    }
}

/// binproto.search_head's (id, name): None where Python raises (a head under 11 bytes, a name that is not UTF-8). A
/// name cut short by the payload's end is cut, as Python's slice is.
fn search_head(p: &[u8]) -> Option<(u64, String)> {
    if p.len() < 11 {
        return None;
    }
    let id = u64::from_le_bytes(p[1..9].try_into().ok()?);
    let n = u16::from_le_bytes(p[9..11].try_into().ok()?) as usize;
    let end = (11 + n).min(p.len());
    String::from_utf8(p[11..end].to_vec()).ok().map(|name| (id, name))
}

/// This thread's Linux thread id (from /proc/thread-self), or 0.
fn own_tid() -> u64 {
    std::fs::read_link("/proc/thread-self").ok()
        .and_then(|p| p.file_name().and_then(|n| n.to_str()).and_then(|n| n.parse().ok()))
        .unwrap_or(0)
}

/// Nanoseconds on a CPU of thread `tid` of this process (schedstat's first field), or 0.
fn thread_cpu_ns(tid: u64) -> u64 {
    if tid == 0 {
        return 0;
    }
    std::fs::read_to_string(format!("/proc/self/task/{tid}/schedstat")).ok()
        .and_then(|s| s.split_whitespace().next().and_then(|x| x.parse().ok()))
        .unwrap_or(0)
}

fn reader(sh: Arc<Shared>, cid: u32, c: Arc<Conn>) {
    let tid = own_tid();
    c.reader_tid.store(tid, Ordering::Relaxed);
    reader_loop(&sh, cid, &c);
    sh.cpu_done_ns[0].fetch_add(thread_cpu_ns(tid), Ordering::Relaxed);
    c.reader_tid.store(0, Ordering::Relaxed);
    sh.ended(cid, &c);
    sh.live.fetch_sub(1, Ordering::AcqRel);
}

fn reader_loop(sh: &Shared, cid: u32, c: &Conn) {
    let mut r = BufReader::with_capacity(READ_BUF, &c.stream);
    let mut head = [0u8; 4];
    loop {
        if r.read_exact(&mut head).is_err() {
            break;
        }
        let n = u32::from_le_bytes(head) as usize;
        let mut p = vec![0u8; n];
        if r.read_exact(&mut p).is_err() || p.is_empty() {
            break; // EOF, or an empty frame (a Python reader's IndexError closed it too)
        }
        if sh.closing.load(Ordering::Relaxed) {
            break; // closing: no new frame is taken
        }
        sh.add(S_FRAMES, 1);
        if p[0] == 1 {
            match search_head(&p) {
                Some((id, name)) => {
                    sh.add(S_SEARCHES, 1);
                    sh.push(Entry::Search { conn: cid, id, name, payload: p, t: Instant::now() });
                }
                None => {
                    sh.add(S_BAD, 1);
                    break;
                }
            }
        } else {
            sh.add(S_CALLS, 1);
            sh.calls.lock().unwrap().push_back((cid, p));
            sh.calls_cv.notify_one();
        }
    }
    let mut o = c.out.lock().unwrap();
    o.reader_done = true;
    drop(o);
    c.cv.notify_one();
}

fn writer(sh: Arc<Shared>, cid: u32, c: Arc<Conn>) {
    let tid = own_tid();
    c.writer_tid.store(tid, Ordering::Relaxed);
    writer_loop(&sh, cid, &c);
    sh.cpu_done_ns[1].fetch_add(thread_cpu_ns(tid), Ordering::Relaxed);
    c.writer_tid.store(0, Ordering::Relaxed);
    sh.ended(cid, &c);
    sh.live.fetch_sub(1, Ordering::AcqRel);
}

fn writer_loop(sh: &Shared, cid: u32, c: &Conn) {
    loop {
        let bufs: Vec<OutItem> = {
            let mut o = c.out.lock().unwrap();
            while o.q.is_empty() && !o.reader_done && !o.dead {
                o = c.cv.wait(o).unwrap();
            }
            if o.dead || o.q.is_empty() {
                break;
            }
            o.q.drain(..).collect()
        };
        let mut w: &UnixStream = &c.stream;
        let mut ok = true;
        for item in &bufs {
            let made;
            let b: &[u8] = match item {
                OutItem::Bytes(b) => b,
                OutItem::Batch4(c) => {
                    let t = Instant::now();
                    made = reply::frame(c);
                    sh.add(S_B4_FRAME_NS, t.elapsed().as_nanos() as u64);
                    sh.add(S_BYTES, made.len() as u64);
                    &made
                }
            };
            if w.write_all(b).is_err() {
                ok = false;
                break;
            }
        }
        if !ok {
            c.out.lock().unwrap().dead = true;
            sh.add(S_ERRORS, 1);
            break;
        }
    }
    sh.conns.lock().unwrap().remove(&cid);
    let _ = c.stream.shutdown(Shutdown::Both);
    sh.add(S_CLOSED, 1);
}

/// R4's serving loop: see the module docs and serving/r4.py.
#[pyclass(module = "s3_hostpath._native")]
pub struct Loop {
    sh: Arc<Shared>,
    base: Instant,
    base_pc: f64,
}

impl Loop {
    fn pc(&self, t: Instant) -> f64 {
        match t.checked_duration_since(self.base) {
            Some(d) => self.base_pc + d.as_secs_f64(),
            None => self.base_pc - self.base.duration_since(t).as_secs_f64(),
        }
    }
}

fn secs(x: f64) -> Duration {
    if x.is_finite() && x > 0.0 { Duration::from_secs_f64(x) } else { Duration::ZERO }
}

#[pymethods]
impl Loop {
    #[new]
    #[pyo3(signature = (perf_now, load_path=None, load_slot=0))]
    fn new(perf_now: f64, load_path: Option<String>, load_slot: u64) -> PyResult<Self> {
        let base = Instant::now();
        let load = match load_path {
            None => None,
            Some(p) => {
                let f = OpenOptions::new().read(true).write(true).open(&p)
                    .map_err(|e| PyValueError::new_err(format!("Loop: cannot open {p}: {e}")))?;
                Some((f, 8 * load_slot))
            }
        };
        let sh = Shared {
            q: Mutex::new(Queue { d: VecDeque::new(), in_batch: 0 }),
            cv: Condvar::new(),
            calls: Mutex::new(VecDeque::new()),
            calls_cv: Condvar::new(),
            conns: Mutex::new(HashMap::new()),
            load,
            st: Default::default(),
            closing: AtomicBool::new(false),
            closed: Mutex::new(Vec::new()),
            cpu_done_ns: Default::default(),
            live: AtomicU64::new(0),
        };
        Ok(Loop { sh: Arc::new(sh), base, base_pc: perf_now })
    }

    #[getter]
    fn name(&self) -> &'static str {
        "s3_hostpath.Loop"
    }

    /// Take the connected socket `fd` (after its magic) as connection `cid`; start its reader and writer threads.
    /// A ValueError (the Loop closing, a cid in use, a socket it cannot set blocking) leaves `fd` the caller's to
    /// close; a RuntimeError (its threads could not start) means the Loop took `fd` and has shut it.
    fn add_conn(&self, cid: u32, fd: RawFd, proto: u8) -> PyResult<()> {
        let _ = proto; // the wire format of the replies is the caller's (it encodes them); kept for the stats' sake
        if self.sh.closing.load(Ordering::Relaxed) {
            return Err(PyValueError::new_err("Loop: closed"));
        }
        let stream = unsafe { UnixStream::from_raw_fd(fd) };
        if let Err(e) = stream.set_nonblocking(false) {
            let _ = stream.into_raw_fd(); // back to the caller, not closed here
            return Err(PyValueError::new_err(format!("Loop: fd {fd}: {e}")));
        }
        let c = {
            let mut m = self.sh.conns.lock().unwrap();
            if m.contains_key(&cid) {
                drop(m);
                let _ = stream.into_raw_fd(); // back to the caller, not closed here
                return Err(PyValueError::new_err(format!("Loop: connection {cid} exists")));
            }
            let c = Arc::new(Conn { stream, out: Mutex::new(Out { q: VecDeque::new(), reader_done: false, dead: false }),
                                    cv: Condvar::new(), reader_tid: AtomicU64::new(0), writer_tid: AtomicU64::new(0),
                                    ended: AtomicU64::new(0) });
            m.insert(cid, c.clone());
            c
        };
        self.sh.add(S_OPENED, 1);
        self.sh.live.fetch_add(2, Ordering::AcqRel); // each thread takes one off as it ends
        let (sh, c2) = (self.sh.clone(), c.clone());
        let r = thread::Builder::new().name(format!("r4-read-{cid}")).spawn(move || reader(sh, cid, c2));
        let (sh, c3) = (self.sh.clone(), c.clone());
        let w = thread::Builder::new().name(format!("r4-write-{cid}")).spawn(move || writer(sh, cid, c3));
        if r.is_err() || w.is_err() {
            let _ = c.stream.shutdown(Shutdown::Both);
            if r.is_err() {
                c.out.lock().unwrap().reader_done = true; // the writer, if it started, ends after its queue
                c.cv.notify_one();
                self.sh.ended(cid, &c);
                self.sh.live.fetch_sub(1, Ordering::AcqRel);
            }
            if w.is_err() {
                self.sh.conns.lock().unwrap().remove(&cid);
                self.sh.ended(cid, &c);
                self.sh.live.fetch_sub(1, Ordering::AcqRel);
            }
            return Err(PyRuntimeError::new_err("Loop: could not start the connection's threads (it is shut)"));
        }
        Ok(())
    }

    /// Shut down: no new connection or frame; every connection's reader ends and its writer sends what is queued,
    /// then closes. Waits up to `timeout` seconds (without the GIL); returns the connections still open.
    #[pyo3(signature = (timeout=2.0))]
    fn close(&self, py: Python<'_>, timeout: f64) -> usize {
        self.sh.closing.store(true, Ordering::Relaxed);
        let conns: Vec<Arc<Conn>> = self.sh.conns.lock().unwrap().values().cloned().collect();
        for c in &conns {
            let _ = c.stream.shutdown(Shutdown::Read);
            c.out.lock().unwrap().reader_done = true;
            c.cv.notify_one();
        }
        let sh = self.sh.clone();
        let t = secs(timeout);
        py.allow_threads(move || {
            let end = Instant::now() + t;
            loop {
                let n = sh.conns.lock().unwrap().len();
                if (n == 0 && sh.live.load(Ordering::Acquire) == 0) || Instant::now() >= end {
                    return n; // every thread ended, so closed_conns now lists every connection
                }
                thread::sleep(Duration::from_millis(5));
            }
        })
    }

    /// Connections both of whose threads have ended since the last call, each reported once.
    fn closed_conns(&self) -> Vec<u32> {
        std::mem::take(&mut *self.sh.closed.lock().unwrap())
    }

    /// A pickle connection's request, by its token, into the same queue.
    fn push_py(&self, token: u64) {
        self.sh.add(S_PUSHED, 1);
        self.sh.push(Entry::Py { token, t: Instant::now() });
    }

    /// The next batch, waiting without the GIL: [(cid, id, name, payload, t)] and (-1, token, None, None, t).
    #[pyo3(signature = (mb, wait, first_timeout=0.2))]
    fn next_batch<'py>(&self, py: Python<'py>, mb: usize, wait: f64, first_timeout: f64) -> PyResult<Bound<'py, PyList>> {
        let sh = self.sh.clone();
        let (w, f) = (secs(wait), secs(first_timeout));
        let got = py.allow_threads(move || sh.take(mb.max(1), w, f));
        let out = PyList::empty_bound(py);
        for e in got {
            match e {
                Entry::Search { conn, id, name, payload, t } => {
                    out.append((conn as i64, id, name, PyBytes::new_bound(py, &payload), self.pc(t)))?;
                }
                Entry::Py { token, t } => {
                    out.append((-1i64, token, py.None(), py.None(), self.pc(t)))?;
                }
            }
        }
        Ok(out)
    }

    /// The batch in hand's size, for the load slot.
    fn set_in_batch(&self, n: usize) {
        let mut q = self.sh.q.lock().unwrap();
        q.in_batch = n;
        self.sh.publish(&q);
    }

    fn qsize(&self) -> usize {
        self.sh.q.lock().unwrap().d.len()
    }

    /// The next call frame, (cid, payload), waiting up to `timeout` without the GIL; None if there is none.
    fn next_call<'py>(&self, py: Python<'py>, timeout: f64) -> Option<(u32, Bound<'py, PyBytes>)> {
        let sh = self.sh.clone();
        let t = secs(timeout);
        let got = py.allow_threads(move || {
            let mut q = sh.calls.lock().unwrap();
            if q.is_empty() {
                q = sh.calls_cv.wait_timeout(q, t).unwrap().0;
            }
            q.pop_front()
        });
        got.map(|(cid, p)| (cid, PyBytes::new_bound(py, &p)))
    }

    /// Queue `data` for connection `cid`'s writer; False when the connection is gone.
    fn send(&self, cid: u32, data: &[u8]) -> bool {
        let c = self.sh.conns.lock().unwrap().get(&cid).cloned();
        let c = match c {
            Some(c) => c,
            None => {
                self.sh.add(S_DROPPED, 1);
                return false;
            }
        };
        let mut o = c.out.lock().unwrap();
        if o.dead {
            drop(o);
            self.sh.add(S_DROPPED, 1);
            return false;
        }
        o.q.push_back(OutItem::Bytes(data.to_vec()));
        let depth = o.q.len() as u64;
        drop(o);
        c.cv.notify_one();
        self.sh.add(S_SENDS, 1);
        self.sh.add(S_BYTES, data.len() as u64);
        self.sh.st[S_MAXDEPTH].fetch_max(depth, Ordering::Relaxed);
        true
    }

    /// R4d: one batch's reply messages for connection `cid` (what s3_hostpath.batch4 takes, with its ctx): collected
    /// here under the GIL (meta, columns copied, shell), the kind 4 frame laid out by the writer without it; the bytes
    /// are batch4's. False when the connection is gone (nothing collected). Raises what batch4 raises, before queuing
    /// anything, so the caller can fall back to the Python writer.
    fn send_batch4<'py>(&self, py: Python<'py>, cid: u32, msgs: &Bound<'py, PyAny>, ctx: &Bound<'py, PyTuple>)
                         -> PyResult<bool> {
        let c = self.sh.conns.lock().unwrap().get(&cid).cloned();
        let c = match c {
            Some(c) => c,
            None => {
                self.sh.add(S_DROPPED, 1);
                return Ok(false);
            }
        };
        let t = Instant::now();
        let got = reply::collect(py, msgs, ctx)?;
        self.sh.add(S_B4_COLLECT_NS, t.elapsed().as_nanos() as u64);
        let mut o = c.out.lock().unwrap();
        if o.dead {
            drop(o);
            self.sh.add(S_DROPPED, 1);
            return Ok(false);
        }
        o.q.push_back(OutItem::Batch4(got));
        let depth = o.q.len() as u64;
        drop(o);
        c.cv.notify_one();
        self.sh.add(S_SENDS, 1);
        self.sh.add(S_B4_SENDS, 1);
        self.sh.st[S_MAXDEPTH].fetch_max(depth, Ordering::Relaxed);
        Ok(true)
    }

    /// Shut connection `cid` down; its reader and writer then end.
    fn close_conn(&self, cid: u32) {
        if let Some(c) = self.sh.conns.lock().unwrap().get(&cid).cloned() {
            let _ = c.stream.shutdown(Shutdown::Both);
        }
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        for (i, k) in STATS.iter().enumerate() {
            d.set_item(*k, self.sh.st[i].load(Ordering::Relaxed))?;
        }
        d.set_item("queued", self.sh.q.lock().unwrap().d.len())?;
        let (mut rd, mut wr) = (self.sh.cpu_done_ns[0].load(Ordering::Relaxed), self.sh.cpu_done_ns[1].load(Ordering::Relaxed));
        for c in self.sh.conns.lock().unwrap().values() {
            rd += thread_cpu_ns(c.reader_tid.load(Ordering::Relaxed));
            wr += thread_cpu_ns(c.writer_tid.load(Ordering::Relaxed));
        }
        let cpu = PyDict::new_bound(py);
        cpu.set_item("reader", rd as f64 * 1e-9)?;
        cpu.set_item("writer", wr as f64 * 1e-9)?;
        d.set_item("cpu_s", cpu)?;
        d.set_item("conns", self.sh.conns.lock().unwrap().len())?;
        Ok(d)
    }
}
