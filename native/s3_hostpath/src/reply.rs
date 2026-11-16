//! R3c: serving/binproto.py's `batch4` (S3B3, kind 4: one frame for a batch's replies) in Rust.
//!
//! `batch4(msgs, ctx)` returns the same bytes as `binproto.batch4(msgs)`. The platform's golden4 frames and
//! serving/tests hold it to them. `ctx` holds the Python objects it needs, passed in once by s3_hostpath:
//! `(numpy.ndarray, dtype float32, dtype float64, numpy.asarray, dtype int64, serving.encode.encode, round)`.
//!
//! The Python reference spends its time in `json.dumps` (the response shell and the record metas), in `_ids_ok`'s
//! per-id loop and in many small numpy calls. Here:
//! - the shell and the metas are written by a JSON writer with `json.dumps`'s rules: `separators=(",", ":")`,
//!   `ensure_ascii`, NaN as NaN;
//! - float text is `float.__repr__`'s, from ryu's shortest digits laid out by CPython's rule: `push_pyrepr`, the
//!   platform's function from serving/native_front/src/encode.rs, which is held there to the pyrepr vectors. A big
//!   int's text is Python's own `int.__repr__`;
//! - `round(x, 3)` is the correctly rounded 3-decimal value. An exact tie, where Python rounds half to even, is left
//!   to Python's round;
//! - the ids are checked on their raw bytes, and the arrays are read through the buffer protocol. An array the fast
//!   path does not take (not C-contiguous, a byte order or format outside it) goes through `tobytes()` and numpy as
//!   before.
//! - no flush here: the GPU process flushes subnormals once, upstream of every writer (serving.encode.flush_response
//!   in gpu.py's _answer, PR #45, lab f134183), so the writers see none.
//! Measured on a Colab CPU (arch-hp10, 96a8a2a), before these: about 1 us per Python call made from here. So the
//! writer avoids them: tuple items, dict lookups with interned keys, and borrowed UTF-8 (no abi3).
//!
//! Anything the writer does not take (a type `json.dumps` would refuse, a string with a lone surrogate) raises, so the
//! guarded caller falls back to the Python writer for that batch.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::{ffi, intern};
use std::ffi::CStr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use pyo3::types::{PyBool, PyByteArray, PyBytes, PyDict, PyFloat, PyList, PyLong, PyString, PyTuple};

const PLACEHOLDER: &str = "\u{0}s3-hits\u{0}";
const MARKER: &[u8] = b"\"hits\":\"\\u0000s3-hits\\u0000\"";
const MAX_DEPTH: usize = 512;

// Development timing (batch4_profile): nanoseconds spent per phase, summed over calls.
// The phases are: meta, columns, shell, frame, and the whole call.
static PROF: [AtomicU64; 5] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];

fn lap(i: usize, t: &mut Instant) {
    let now = Instant::now();
    PROF[i].fetch_add((now - *t).as_nanos() as u64, Ordering::Relaxed);
    *t = now;
}

/// The phase totals in seconds since the last call, as {meta, columns, shell, frame, total}; resets them.
#[pyfunction]
pub fn batch4_profile() -> Vec<(String, f64)> {
    ["meta", "columns", "shell", "frame", "total"]
        .iter()
        .enumerate()
        .map(|(i, n)| (n.to_string(), PROF[i].swap(0, Ordering::Relaxed) as f64 * 1e-9))
        .collect()
}

#[cfg(not(target_endian = "little"))]
compile_error!("batch4 copies the host's bytes as little-endian");

struct Ctx<'py> {
    ndarray: Bound<'py, PyAny>,
    f32: Bound<'py, PyAny>,
    f64: Bound<'py, PyAny>,
    asarray: Bound<'py, PyAny>,
    i64: Bound<'py, PyAny>,
    encode: Bound<'py, PyAny>,
    round: Bound<'py, PyAny>,
    int_repr: Bound<'py, PyAny>,
    bool_type: Bound<'py, PyAny>,
}

// ---- the JSON writer (json.dumps with separators=(",", ":"), ensure_ascii, allow_nan) ----------------------------

fn esc(out: &mut Vec<u8>, s: &str) {
    out.push(b'"');
    for ch in s.chars() {
        match ch {
            '"' => out.extend_from_slice(b"\\\""),
            '\\' => out.extend_from_slice(b"\\\\"),
            '\n' => out.extend_from_slice(b"\\n"),
            '\r' => out.extend_from_slice(b"\\r"),
            '\t' => out.extend_from_slice(b"\\t"),
            '\u{8}' => out.extend_from_slice(b"\\b"),
            '\u{c}' => out.extend_from_slice(b"\\f"),
            ' '..='~' => out.push(ch as u8),
            _ => {
                let mut buf = [0u16; 2];
                for u in ch.encode_utf16(&mut buf) {
                    out.extend_from_slice(format!("\\u{:04x}", u).as_bytes());
                }
            }
        }
    }
    out.push(b'"');
}

fn float_text(_ctx: &Ctx, out: &mut Vec<u8>, _v: &Bound<PyAny>, f: f64) -> PyResult<()> {
    // json.dumps calls float.__repr__ (a float subclass's own repr is not used), which depends on the value alone
    push_pyrepr(out, f, &mut ryu::Buffer::new());
    Ok(())
}

/// `json.dumps(x)` for any float64, byte for byte: the platform's function (serving/native_front/src/encode.rs,
/// S3B2), copied unchanged. Python's `float.__repr__`: the shortest round-trip digits (ryu's, the same digits as
/// CPython's dtoa shortest mode), laid out by CPython's 'r' rule. Positional when the decimal point position decpt
/// (x = 0.d1d2... x 10^decpt) has -4 < decpt <= 16, else d[.ddd]e+XX / e-XX (at least two exponent digits); and
/// NaN, Infinity, -Infinity as json.dumps writes them.
pub fn push_pyrepr(out: &mut Vec<u8>, x: f64, buf: &mut ryu::Buffer) {
    let m = x.abs();
    if x.is_finite() && x != 0.0 && m >= 1e-4 && m < 1e16 {
        out.extend_from_slice(buf.format_finite(x).as_bytes());
        return;
    }
    if x.is_nan() {
        out.extend_from_slice(b"NaN");
        return;
    }
    if x.is_infinite() {
        out.extend_from_slice(if x > 0.0 { b"Infinity" } else { b"-Infinity" });
        return;
    }
    if x == 0.0 {
        out.extend_from_slice(if x.is_sign_negative() { b"-0.0" } else { b"0.0" });
        return;
    }
    let s = buf.format_finite(x.abs());
    let (mant, exp) = match s.find('e') {
        Some(p) => (&s[..p], s[p + 1..].parse::<i32>().unwrap_or(0)),
        None => (s, 0),
    };
    let (ip, fp) = match mant.find('.') {
        Some(p) => (&mant[..p], &mant[p + 1..]),
        None => (mant, ""),
    };
    let n = ip.len() + fp.len();
    let mut dbuf = [0u8; 64];
    let heap: Vec<u8>;
    let digits: &[u8] = if n <= dbuf.len() {
        dbuf[..ip.len()].copy_from_slice(ip.as_bytes());
        dbuf[ip.len()..n].copy_from_slice(fp.as_bytes());
        &dbuf[..n]
    } else {
        heap = [ip.as_bytes(), fp.as_bytes()].concat();
        &heap
    };
    let z = digits.iter().take_while(|&&c| c == b'0').count();
    let decpt = ip.len() as i32 - z as i32 + exp;
    let mut d: &[u8] = &digits[z..];
    while d.len() > 1 && d[d.len() - 1] == b'0' {
        d = &d[..d.len() - 1];
    }
    if x < 0.0 {
        out.push(b'-');
    }
    if decpt > -4 && decpt <= 16 {
        if decpt <= 0 {
            out.extend_from_slice(b"0.");
            for _ in 0..(-decpt) {
                out.push(b'0');
            }
            out.extend_from_slice(d);
        } else if decpt as usize >= d.len() {
            out.extend_from_slice(d);
            for _ in 0..(decpt as usize - d.len()) {
                out.push(b'0');
            }
            out.extend_from_slice(b".0");
        } else {
            out.extend_from_slice(&d[..decpt as usize]);
            out.push(b'.');
            out.extend_from_slice(&d[decpt as usize..]);
        }
    } else {
        out.push(d[0]);
        if d.len() > 1 {
            out.push(b'.');
            out.extend_from_slice(&d[1..]);
        }
        let e = decpt - 1;
        out.push(b'e');
        out.push(if e < 0 { b'-' } else { b'+' });
        let ea = e.unsigned_abs();
        if ea < 10 {
            out.push(b'0');
        }
        out.extend_from_slice(ea.to_string().as_bytes());
    }
}

/// `round(x, 3)` for a float: the correctly rounded 3-decimal value (Rust's exact formatting, then a correctly
/// rounded parse, as CPython's dtoa and strtod). A possible exact tie (x * 2000 an odd integer) goes to Python's round.
fn round3(ctx: &Ctx, x: f64) -> PyResult<f64> {
    if !x.is_finite() {
        return Ok(x);
    }
    let y = x * 2000.0;
    if y.is_finite() && y.fract() == 0.0 && (y.abs() >= 9.0e15 || (y as i64) % 2 != 0) {
        return ctx.round.call1((x, 3))?.extract::<f64>();
    }
    format!("{:.3}", x).parse::<f64>().map_err(|_| PyValueError::new_err("round"))
}

/// `float(v)`: a float's value, a small int's, or Python's float() of anything else.
fn as_float(v: &Bound<PyAny>) -> PyResult<f64> {
    if let Ok(f) = v.downcast::<PyFloat>() {
        return Ok(f.value());
    }
    if v.downcast::<PyLong>().is_ok() && v.downcast::<PyBool>().is_err() {
        if let Ok(i) = v.extract::<i64>() {
            if i.unsigned_abs() <= (1u64 << 53) {
                return Ok(i as f64);
            }
        }
    }
    v.py().get_type_bound::<PyFloat>().call1((v,))?.downcast::<PyFloat>().map(|f| f.value()).map_err(Into::into)
}

/// `f"{v}"` for a str or a small int directly, else `v.__format__("")`.
fn fmt_text(v: &Bound<PyAny>) -> PyResult<String> {
    if v.get_type().is(&v.py().get_type_bound::<PyString>()) {
        return Ok(v.downcast::<PyString>()?.to_cow()?.into_owned());
    }
    if v.get_type().is(&v.py().get_type_bound::<PyLong>()) {
        if let Ok(i) = v.extract::<i64>() {
            return Ok(i.to_string());
        }
    }
    let r = v.call_method1(intern!(v.py(), "__format__"), ("",))?;
    Ok(r.downcast::<PyString>()?.to_cow()?.into_owned())
}

/// A C-contiguous buffer view of an array (numpy's buffer protocol); released on drop.
struct Buf {
    view: ffi::Py_buffer,
}

impl Buf {
    fn get(obj: &Bound<PyAny>) -> Option<Buf> {
        let mut view = std::mem::MaybeUninit::<ffi::Py_buffer>::uninit();
        let rc = unsafe { ffi::PyObject_GetBuffer(obj.as_ptr(), view.as_mut_ptr(), ffi::PyBUF_C_CONTIGUOUS | ffi::PyBUF_FORMAT) };
        if rc != 0 {
            unsafe { ffi::PyErr_Clear() };
            return None;
        }
        let b = Buf { view: unsafe { view.assume_init() } };
        if b.view.ndim != 1 || b.view.shape.is_null() || b.view.format.is_null() {
            return None;
        }
        Some(b)
    }
    fn len(&self) -> usize {
        unsafe { *self.view.shape as usize }
    }
    fn itemsize(&self) -> usize {
        self.view.itemsize as usize
    }
    fn format(&self) -> &[u8] {
        unsafe { CStr::from_ptr(self.view.format).to_bytes() }
    }
    fn bytes(&self) -> &[u8] {
        if self.view.len == 0 || self.view.buf.is_null() {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.view.buf as *const u8, self.view.len as usize) }
    }
}

impl Drop for Buf {
    fn drop(&mut self) {
        unsafe { ffi::PyBuffer_Release(&mut self.view) };
    }
}

/// numpy's buffer format of a native-order integer array -> (itemsize, signed); None for anything else.
fn int_format(f: &[u8], itemsize: usize) -> Option<(usize, bool)> {
    if f.len() != 1 {
        return None;
    }
    let signed = match f[0] {
        b'b' | b'h' | b'i' | b'l' | b'q' | b'n' => true,
        b'B' | b'H' | b'I' | b'L' | b'Q' | b'N' => false,
        _ => return None,
    };
    if matches!(itemsize, 1 | 2 | 4 | 8) {
        Some((itemsize, signed))
    } else {
        None
    }
}

/// `values.astype(np.int64)` of a native integer array's bytes.
fn ints_to_i64(b: &[u8], size: usize, signed: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(b.len() / size * 8);
    for c in b.chunks(size) {
        let v: i64 = match (size, signed) {
            (1, true) => c[0] as i8 as i64,
            (1, false) => c[0] as i64,
            (2, true) => i16::from_le_bytes([c[0], c[1]]) as i64,
            (2, false) => u16::from_le_bytes([c[0], c[1]]) as i64,
            (4, true) => i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64,
            (4, false) => u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64,
            (_, true) => i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]),
            (_, false) => u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as i64,
        };
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// The id bytes of an S array through the buffer protocol: (raw, itemsize, h); None to use the slow path.
fn ids_buf(ids: &Bound<PyAny>) -> Option<(Vec<u8>, usize, usize)> {
    let b = Buf::get(ids)?;
    let f = b.format();
    if f.is_empty() || f[f.len() - 1] != b's' || !f[..f.len() - 1].iter().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let (h, w) = (b.len(), b.itemsize());
    if b.bytes().len() != h * w {
        return None;
    }
    Some((b.bytes().to_vec(), w, h))
}

fn int_text(ctx: &Ctx, out: &mut Vec<u8>, v: &Bound<PyAny>) -> PyResult<()> {
    match v.extract::<i64>() {
        Ok(i) => out.extend_from_slice(i.to_string().as_bytes()),
        Err(_) => {
            let r = ctx.int_repr.call1((v.clone(),))?;
            out.extend_from_slice(r.downcast::<PyString>()?.to_cow()?.as_bytes());
        }
    }
    Ok(())
}

fn key_text(ctx: &Ctx, out: &mut Vec<u8>, k: &Bound<PyAny>) -> PyResult<()> {
    if let Ok(s) = k.downcast::<PyString>() {
        esc(out, &s.to_cow()?);
    } else if let Ok(f) = k.downcast::<PyFloat>() {
        let mut t = Vec::new();
        float_text(ctx, &mut t, k, f.value())?;
        out.push(b'"');
        out.extend_from_slice(&t);
        out.push(b'"');
    } else if let Ok(b) = k.downcast::<PyBool>() {
        out.extend_from_slice(if b.is_true() { b"\"true\"" as &[u8] } else { b"\"false\"" });
    } else if k.is_none() {
        out.extend_from_slice(b"\"null\"");
    } else if k.downcast::<PyLong>().is_ok() {
        out.push(b'"');
        int_text(ctx, out, k)?;
        out.push(b'"');
    } else {
        return Err(PyTypeError::new_err("keys must be str, int, float, bool or None"));
    }
    Ok(())
}

fn dump(ctx: &Ctx, out: &mut Vec<u8>, v: &Bound<PyAny>, depth: usize) -> PyResult<()> {
    if depth > MAX_DEPTH {
        return Err(PyValueError::new_err("too deep"));
    }
    if v.is_none() {
        out.extend_from_slice(b"null");
    } else if let Ok(b) = v.downcast::<PyBool>() {
        out.extend_from_slice(if b.is_true() { b"true" as &[u8] } else { b"false" });
    } else if let Ok(s) = v.downcast::<PyString>() {
        esc(out, &s.to_cow()?);
    } else if v.downcast::<PyLong>().is_ok() {
        int_text(ctx, out, v)?;
    } else if let Ok(f) = v.downcast::<PyFloat>() {
        float_text(ctx, out, v, f.value())?;
    } else if v.downcast::<PyList>().is_ok() || v.downcast::<PyTuple>().is_ok() {
        out.push(b'[');
        for (i, x) in v.iter()?.enumerate() {
            if i > 0 {
                out.push(b',');
            }
            dump(ctx, out, &x?, depth + 1)?;
        }
        out.push(b']');
    } else if let Ok(d) = v.downcast::<PyDict>() {
        out.push(b'{');
        for (i, (k, x)) in d.iter().enumerate() {
            if i > 0 {
                out.push(b',');
            }
            key_text(ctx, out, &k)?;
            out.push(b':');
            dump(ctx, out, &x, depth + 1)?;
        }
        out.push(b'}');
    } else {
        return Err(PyTypeError::new_err("not JSON serializable"));
    }
    Ok(())
}

fn is_str(k: &Bound<PyAny>, s: &str) -> PyResult<bool> {
    match k.downcast::<PyString>() {
        Ok(t) => Ok(t.to_cow()? == s),
        Err(_) => Ok(false),
    }
}

/// `json.dumps(encode._shell(resp, PLACEHOLDER), separators=(",", ":"))`: "hits" first (its keys but "columns", then
/// "hits", replaced in place when it is already there), then resp's other keys.
fn shell(ctx: &Ctx, out: &mut Vec<u8>, resp: &Bound<PyDict>, hits: &Bound<PyDict>) -> PyResult<()> {
    let py = resp.py();
    let ph = PyString::new_bound(py, PLACEHOLDER).into_any();
    out.extend_from_slice(b"{\"hits\":{");
    let mut n = 0;
    let mut seen = false;
    for (k, x) in hits.iter() {
        if is_str(&k, "columns")? {
            continue;
        }
        if n > 0 {
            out.push(b',');
        }
        n += 1;
        key_text(ctx, out, &k)?;
        out.push(b':');
        if is_str(&k, "hits")? {
            seen = true;
            dump(ctx, out, &ph, 1)?;
        } else {
            dump(ctx, out, &x, 2)?;
        }
    }
    if !seen {
        if n > 0 {
            out.push(b',');
        }
        out.extend_from_slice(b"\"hits\":");
        dump(ctx, out, &ph, 1)?;
    }
    out.push(b'}');
    for (k, x) in resp.iter() {
        if is_str(&k, "hits")? {
            continue;
        }
        out.push(b',');
        key_text(ctx, out, &k)?;
        out.push(b':');
        dump(ctx, out, &x, 1)?;
    }
    out.push(b'}');
    Ok(())
}

// ---- one reply's columns (binproto._batch_columns) ---------------------------------------------------------------

struct Col {
    kind: u8,                 // 0: the score widened; 1: float64; 2: int64
    vals: Vec<u8>,            // h * 8 bytes, little-endian
    miss: Option<Vec<bool>>,  // None when no value is missing
}

struct Row {
    ids: Vec<u8>, // h * w0 bytes, NUL-padded
    w0: usize,
    h: usize,
    s32: Vec<u8>, // h * 4 bytes
    cols: Vec<Col>,
}

fn tobytes(a: &Bound<PyAny>) -> PyResult<Vec<u8>> {
    let b = a.call_method0(intern!(a.py(), "tobytes"))?;
    Ok(b.downcast::<PyBytes>()?.as_bytes().to_vec())
}

fn ids_ok(raw: &[u8], w: usize) -> bool {
    for x in raw.chunks(w) {
        let mut e = x.len();
        while e > 0 && x[e - 1] == 0 {
            e -= 1;
        }
        let x = &x[..e];
        if x.contains(&0) || std::str::from_utf8(x).is_err() {
            return false;
        }
    }
    true
}

fn columns(ctx: &Ctx, resp: &Bound<PyAny>) -> PyResult<Option<Row>> {
    // encode._is_columnar
    let Ok(rd) = resp.downcast::<PyDict>() else { return Ok(None) };
    let Some(hits) = rd.get_item(intern!(resp.py(), "hits"))? else { return Ok(None) };
    let Ok(hd) = hits.downcast::<PyDict>() else { return Ok(None) };
    if !hd.contains(intern!(resp.py(), "columns"))? {
        return Ok(None);
    }
    let py = resp.py();
    let c = hd.get_item(intern!(py, "columns"))?.ok_or_else(|| PyValueError::new_err("columns"))?;
    let ids = c.get_item(intern!(py, "_id"))?;
    if !ids.is_instance(&ctx.ndarray)? {
        return Ok(None);
    }
    let (raw, w0, h) = match ids_buf(&ids) {
        Some(x) => x,
        None => {
            let dt = ids.getattr(intern!(py, "dtype"))?;
            if dt.getattr(intern!(py, "kind"))?.extract::<String>()? != "S" {
                return Ok(None);
            }
            let h = ids.len()?;
            if h == 0 {
                return Ok(None);
            }
            if ids.getattr(intern!(py, "ndim"))?.extract::<usize>()? != 1 {
                return Err(PyValueError::new_err("ids not 1-D"));
            }
            (tobytes(&ids)?, dt.getattr(intern!(py, "itemsize"))?.extract::<usize>()?, h)
        }
    };
    if h == 0 || w0 == 0 || raw.len() != h * w0 || !ids_ok(&raw, w0) {
        return Ok(None);
    }
    let score = c.get_item(intern!(py, "_score"))?;
    let fast = if score.is_instance(&ctx.ndarray)? {
        Buf::get(&score).and_then(|b| if b.format() == b"f" { Some((b.len(), b.bytes().to_vec())) } else { None })
    } else {
        None
    };
    let (sh, s32) = match fast {
        Some(x) => x,
        None => {
            let s = ctx.asarray.call1((score, ctx.f32.clone()))?;
            (s.len()?, tobytes(&s)?)
        }
    };
    if sh != h {
        return Ok(None);
    }
    if s32.len() != 4 * h {
        return Err(PyValueError::new_err("scores not 1-D"));
    }
    let s64: Vec<f64> = s32.chunks(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64).collect();
    let mut cols = Vec::new();
    let sort = c.call_method1(intern!(py, "get"), (intern!(py, "sort").clone(),))?;
    if sort.is_truthy()? {
        for col in sort.iter()? {
            let col = col?;
            let v = col.call_method1(intern!(py, "get"), (intern!(py, "values").clone(),))?;
            let miss = col.call_method1(intern!(py, "get"), (intern!(py, "missing").clone(),))?;
            if !v.is_instance(&ctx.ndarray)? || v.len()? != h {
                return Ok(None);
            }
            let m: Option<Vec<bool>> = if miss.is_none() {
                None
            } else {
                let fastm = if miss.is_instance(&ctx.ndarray)? {
                    Buf::get(&miss).and_then(|b| if b.format() == b"?" { Some((b.len(), b.bytes().to_vec())) } else { None })
                } else {
                    None
                };
                let (mh, mb) = match fastm {
                    Some(x) => x,
                    None => {
                        let ma = ctx.asarray.call1((miss, ctx.bool_type.clone()))?;
                        (ma.len()?, tobytes(&ma)?)
                    }
                };
                if mh != h {
                    return Ok(None);
                }
                if mb.len() != h {
                    return Err(PyValueError::new_err("missing not 1-D"));
                }
                if mb.iter().any(|&x| x != 0) {
                    Some(mb.iter().map(|&x| x != 0).collect())
                } else {
                    None
                }
            };
            let vb = Buf::get(&v);
            let vfmt: Option<(Vec<u8>, usize)> = vb.as_ref().map(|b| (b.format().to_vec(), b.itemsize()));
            if let (Some(b), Some((f, isz))) = (vb.as_ref(), vfmt.as_ref()) {
                if f.as_slice() == b"d" || int_format(f, *isz).is_some() {
                    let vals = if f.as_slice() == b"d" {
                        b.bytes().to_vec()
                    } else {
                        let (size, signed) = int_format(f, *isz).unwrap();
                        ints_to_i64(b.bytes(), size, signed)
                    };
                    if vals.len() != 8 * h {
                        return Err(PyValueError::new_err("values not 1-D"));
                    }
                    if f.as_slice() == b"d" {
                        let same = m.is_none()
                            && vals.chunks(8).zip(s64.iter()).all(|(b, &x)| {
                                f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) == x
                            });
                        cols.push(if same { Col { kind: 0, vals: Vec::new(), miss: None } } else { Col { kind: 1, vals, miss: m } });
                    } else {
                        cols.push(Col { kind: 2, vals, miss: m });
                    }
                    continue;
                }
            }
            drop(vb);
            let vdt = v.getattr(intern!(py, "dtype"))?;
            if vdt.eq(&ctx.f64)? {
                let vals = tobytes(&v)?;
                if vals.len() != 8 * h {
                    return Err(PyValueError::new_err("values not 1-D"));
                }
                let same = m.is_none()
                    && vals.chunks(8).zip(s64.iter()).all(|(b, &x)| {
                        f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) == x
                    });
                if same {
                    cols.push(Col { kind: 0, vals: Vec::new(), miss: None });
                } else {
                    cols.push(Col { kind: 1, vals, miss: m });
                }
            } else {
                let k = vdt.getattr(intern!(py, "kind"))?.extract::<String>()?;
                if k == "i" || k == "u" {
                    let vals = tobytes(&v.call_method1(intern!(py, "astype"), (ctx.i64.clone(),))?)?;
                    if vals.len() != 8 * h {
                        return Err(PyValueError::new_err("values not 1-D"));
                    }
                    cols.push(Col { kind: 2, vals, miss: m });
                } else {
                    return Ok(None);
                }
            }
        }
    }
    Ok(Some(Row { ids: raw, w0, h, s32, cols }))
}

// ---- the frame ---------------------------------------------------------------------------------------------------

enum Rec {
    Hits { id: u64, meta: Vec<u8>, row: usize },
    Whole { id: u64, meta: Vec<u8>, status: u16, body: Vec<u8> },
}

fn body_bytes(b: &Bound<PyAny>) -> PyResult<Vec<u8>> {
    if let Ok(x) = b.downcast::<PyBytes>() {
        Ok(x.as_bytes().to_vec())
    } else if let Ok(x) = b.downcast::<PyByteArray>() {
        Ok(x.to_vec())
    } else {
        Err(PyTypeError::new_err("body is not bytes"))
    }
}

fn s_eq(v: &Bound<PyAny>, s: &str) -> PyResult<bool> {
    is_str(v, s)
}

fn ok_reply(ctx: &Ctx, msg: &Bound<PyAny>, rows: &mut Vec<Row>) -> PyResult<Rec> {
    let mut t = Instant::now();
    let n = msg.len()?;
    if n < 6 {
        return Err(PyValueError::new_err("a search reply needs 6 items"));
    }
    let id: u64 = msg.get_item(1)?.extract()?;
    let resp = msg.get_item(3)?;
    let version = msg.get_item(4)?;
    let mut meta = Vec::with_capacity(256);
    // {"version": f"{v[0]}:{v[1]}", "queue_ms": round(float(q), 3), "batch": rest[0] or 1, ["batch_ms": ...]}
    let (v0, v1) = match version.downcast::<PyTuple>() {
        Ok(t) => (t.get_item(0)?, t.get_item(1)?),
        Err(_) => (version.get_item(0)?, version.get_item(1)?),
    };
    let vs = format!("{}:{}", fmt_text(&v0)?, fmt_text(&v1)?);
    meta.extend_from_slice(b"{\"version\":");
    esc(&mut meta, &vs);
    let mut fb = ryu::Buffer::new();
    let q = round3(ctx, as_float(&msg.get_item(5)?)?)?;
    meta.extend_from_slice(b",\"queue_ms\":");
    push_pyrepr(&mut meta, q, &mut fb);
    meta.extend_from_slice(b",\"batch\":");
    if n > 6 {
        dump(ctx, &mut meta, &msg.get_item(6)?, 1)?;
    } else {
        meta.push(b'1');
    }
    if n > 7 {
        let bm = msg.get_item(7)?;
        if !bm.is_none() {
            let b = round3(ctx, as_float(&bm)?)?;
            meta.extend_from_slice(b",\"batch_ms\":");
            push_pyrepr(&mut meta, b, &mut fb);
        }
    }
    lap(0, &mut t);
    let got = columns(ctx, &resp)?;
    lap(1, &mut t);
    if let Some(row) = got {
        let rd = resp.downcast::<PyDict>()?;
        let hits = rd.get_item(intern!(rd.py(), "hits"))?.ok_or_else(|| PyValueError::new_err("hits"))?;
        let mut text = Vec::with_capacity(256);
        shell(ctx, &mut text, rd, hits.downcast::<PyDict>()?)?;
        lap(2, &mut t);
        let mut count = 0;
        let mut at = 0;
        let mut i = 0;
        while i + MARKER.len() <= text.len() {
            if text[i] == b'"' && text[i + 1] == b'h' && &text[i..i + MARKER.len()] == MARKER {
                count += 1;
                at = i;
                i += MARKER.len();
            } else {
                i += 1;
            }
        }
        if count == 1 {
            let pre = format!("{}\"hits\":", std::str::from_utf8(&text[..at]).map_err(|_| PyValueError::new_err("utf8"))?);
            let post = std::str::from_utf8(&text[at + MARKER.len()..]).map_err(|_| PyValueError::new_err("utf8"))?;
            meta.extend_from_slice(b",\"pre\":");
            esc(&mut meta, &pre);
            meta.extend_from_slice(b",\"post\":");
            esc(&mut meta, post);
            meta.push(b'}');
            rows.push(row);
            lap(0, &mut t);
            return Ok(Rec::Hits { id, meta, row: rows.len() - 1 });
        }
    }
    meta.push(b'}');
    let body = body_bytes(&ctx.encode.call1((resp,))?)?;
    Ok(Rec::Whole { id, meta, status: 200, body })
}


// ---- R4d: the GIL-held part (collect) apart from the frame layout (frame) -----------------------------------------
//
// `batch4` is `collect` then `frame`. `collect` reads the Python objects: each reply's meta, its columns (copied out of
// the numpy buffers) and its response shell, and checks every limit the layout has, so `frame` cannot fail. `frame`
// touches no Python object: the Loop's writer thread runs it without the GIL (`Loop.send_batch4`, R4d).

/// One batch's replies, collected under the GIL: what `frame` lays out.
pub struct Collected {
    recs: Vec<Rec>,
    rows: Vec<Row>,
}

fn make_ctx<'py>(py: Python<'py>, ctx: &Bound<'py, PyTuple>) -> PyResult<Ctx<'py>> {
    Ok(Ctx {
        ndarray: ctx.get_item(0)?,
        f32: ctx.get_item(1)?,
        f64: ctx.get_item(2)?,
        asarray: ctx.get_item(3)?,
        i64: ctx.get_item(4)?,
        encode: ctx.get_item(5)?,
        round: ctx.get_item(6)?,
        int_repr: py.get_type_bound::<PyLong>().getattr("__repr__")?,
        bool_type: py.get_type_bound::<PyBool>().into_any(),
    })
}

fn shared_keys(rows: &[Row]) -> Vec<(usize, u8)> {
    // shared columns: one per (sort position, kind) any reply uses, in order of first use
    let mut keys: Vec<(usize, u8)> = Vec::new();
    for r in rows {
        for (k, c) in r.cols.iter().enumerate() {
            if c.kind != 0 && !keys.contains(&(k, c.kind)) {
                keys.push((k, c.kind));
            }
        }
    }
    keys
}

/// The GIL-held part of `batch4`: every reply's meta, columns and shell, and the layout's limits checked.
pub fn collect<'py>(py: Python<'py>, msgs: &Bound<'py, PyAny>, ctx: &Bound<'py, PyTuple>) -> PyResult<Collected> {
    let ctx = make_ctx(py, ctx)?;
    let mut recs: Vec<Rec> = Vec::new();
    let mut rows: Vec<Row> = Vec::new();
    for msg in msgs.iter()? {
        let msg = msg?;
        let tag = msg.get_item(0)?;
        if s_eq(&tag, "search")? && s_eq(&msg.get_item(2)?, "ok")? {
            recs.push(ok_reply(&ctx, &msg, &mut rows)?);
        } else if s_eq(&tag, "search")? {
            let mut body = Vec::new();
            dump(&ctx, &mut body, &msg.get_item(4)?, 0)?;
            recs.push(Rec::Whole {
                id: msg.get_item(1)?.extract()?,
                meta: b"{}".to_vec(),
                status: msg.get_item(3)?.extract()?,
                body,
            });
        } else {
            if msg.len()? != 6 {
                return Err(PyValueError::new_err("a call reply has 6 items"));
            }
            let mut meta = b"{\"type\":".to_vec();
            dump(&ctx, &mut meta, &msg.get_item(4)?, 1)?;
            meta.push(b'}');
            recs.push(Rec::Whole {
                id: msg.get_item(1)?.extract()?,
                meta,
                status: msg.get_item(2)?.extract()?,
                body: body_bytes(&msg.get_item(3)?)?,
            });
        }
    }
    let w = rows.iter().map(|r| r.w0).max().unwrap_or(1).max(1);
    if w > u16::MAX as usize {
        return Err(PyValueError::new_err("ids too wide"));
    }
    if shared_keys(&rows).len() > 254 {
        return Err(PyValueError::new_err("too many shared columns"));
    }
    if rows.iter().any(|r| r.cols.len() > 255) {
        return Err(PyValueError::new_err("too many sort columns"));
    }
    Ok(Collected { recs, rows })
}

/// The kind 4 frame (its u32 length, then the payload) for a collected batch: no Python object, no failure. Laid out
/// in one buffer; every 8-byte alignment is of the payload's offset, as binproto.batch4's pad8.
pub fn frame(c: &Collected) -> Vec<u8> {
    let (recs, rows) = (&c.recs, &c.rows);
    let w = rows.iter().map(|r| r.w0).max().unwrap_or(1).max(1);
    let h_total: usize = rows.iter().map(|r| r.h).sum();
    let keys = shared_keys(rows);
    let meta_len: usize = recs.iter().map(|r| match r {
        Rec::Hits { meta, .. } => meta.len() + 32,
        Rec::Whole { meta, body, .. } => meta.len() + body.len() + 32,
    }).sum();
    let mut out: Vec<u8> = Vec::with_capacity(4 + 32 + h_total * (w + 4) + keys.len() * (h_total * 8 + h_total / 8 + 16)
                                              + meta_len + 16);
    out.extend_from_slice(&[0u8; 4]); // the frame's length, written at the end
    let pad = |out: &mut Vec<u8>| {
        while (out.len() - 4) % 8 != 0 {
            out.push(0);
        }
    };
    out.push(4);
    out.extend_from_slice(&0u64.to_le_bytes());
    out.extend_from_slice(&(recs.len() as u32).to_le_bytes());
    out.extend_from_slice(&(w as u16).to_le_bytes());
    out.extend_from_slice(&(h_total as u32).to_le_bytes());
    for r in rows {
        if r.w0 == w {
            out.extend_from_slice(&r.ids);
        } else {
            for x in r.ids.chunks(r.w0) {
                out.extend_from_slice(x);
                out.resize(out.len() + (w - r.w0), 0);
            }
        }
    }
    pad(&mut out);
    for r in rows {
        out.extend_from_slice(&r.s32);
    }
    out.push(keys.len() as u8);
    let mut offs = vec![0usize];
    for r in rows {
        offs.push(offs.last().unwrap() + r.h);
    }
    for &(k, kind) in &keys {
        let flags = if rows.iter().any(|r| r.cols.get(k).map_or(false, |c| c.kind == kind && c.miss.is_some())) {
            1u8
        } else {
            0
        };
        out.push(kind);
        out.push(flags);
        pad(&mut out);
        let at = out.len();
        out.resize(at + 8 * h_total, 0);
        for (ri, r) in rows.iter().enumerate() {
            if let Some(c) = r.cols.get(k) {
                if c.kind == kind {
                    out[at + 8 * offs[ri]..at + 8 * offs[ri + 1]].copy_from_slice(&c.vals);
                }
            }
        }
        if flags == 1 {
            let at = out.len();
            out.resize(at + (h_total + 7) / 8, 0);
            for (ri, r) in rows.iter().enumerate() {
                if let Some(c) = r.cols.get(k) {
                    if c.kind == kind {
                        if let Some(m) = &c.miss {
                            for (j, &x) in m.iter().enumerate() {
                                if x {
                                    let i = offs[ri] + j;
                                    out[at + i / 8] |= 1 << (i % 8);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    for rec in recs {
        match rec {
            Rec::Hits { id, meta, row } => {
                let r = &rows[*row];
                out.extend_from_slice(&id.to_le_bytes());
                out.push(3);
                out.extend_from_slice(&(offs[*row] as u32).to_le_bytes());
                out.extend_from_slice(&(r.h as u32).to_le_bytes());
                out.push(r.cols.len() as u8);
                for (k, c) in r.cols.iter().enumerate() {
                    out.push(if c.kind == 0 {
                        0
                    } else {
                        1 + keys.iter().position(|&x| x == (k, c.kind)).unwrap() as u8
                    });
                }
                out.extend_from_slice(&(meta.len() as u32).to_le_bytes());
                out.extend_from_slice(meta);
            }
            Rec::Whole { id, meta, status, body } => {
                out.extend_from_slice(&id.to_le_bytes());
                out.push(2);
                out.extend_from_slice(&(meta.len() as u32).to_le_bytes());
                out.extend_from_slice(meta);
                out.extend_from_slice(&status.to_le_bytes());
                out.extend_from_slice(&(body.len() as u32).to_le_bytes());
                out.extend_from_slice(body);
            }
        }
    }
    let n = (out.len() - 4) as u32;
    out[..4].copy_from_slice(&n.to_le_bytes());
    out
}

#[pyfunction]
pub fn batch4<'py>(py: Python<'py>, msgs: &Bound<'py, PyAny>, ctx: &Bound<'py, PyTuple>) -> PyResult<Bound<'py, PyBytes>> {
    let t_all = Instant::now();
    let c = collect(py, msgs, ctx)?;
    let mut t = Instant::now();
    let framed = frame(&c);
    let r = PyBytes::new_bound(py, &framed);
    lap(3, &mut t);
    PROF[4].fetch_add((Instant::now() - t_all).as_nanos() as u64, Ordering::Relaxed);
    Ok(r)
}

/// R4d's split, for the tests and the phase bench: `(batch4's bytes, ns in collect (GIL held), ns in frame (no GIL
/// needed))`. The frame is laid out here with the GIL released, as the Loop's writer does.
#[pyfunction]
pub fn batch4_split<'py>(py: Python<'py>, msgs: &Bound<'py, PyAny>, ctx: &Bound<'py, PyTuple>)
                         -> PyResult<(Bound<'py, PyBytes>, u64, u64)> {
    let t0 = Instant::now();
    let c = collect(py, msgs, ctx)?;
    let collect_ns = t0.elapsed().as_nanos() as u64;
    let (framed, frame_ns) = py.allow_threads(move || {
        let t1 = Instant::now();
        let f = frame(&c);
        (f, t1.elapsed().as_nanos() as u64)
    });
    Ok((PyBytes::new_bound(py, &framed), collect_ns, frame_ns))
}
