//! R3c: serving/binproto.py's `batch4` (S3B3, kind 4: one frame for a batch's replies) in Rust.
//!
//! `batch4(msgs, ctx)` returns the same bytes as `binproto.batch4(msgs)`. The platform's golden4 frames and
//! serving/tests hold it to them. `ctx` holds the Python objects it needs, passed in once by s3_hostpath:
//! `(numpy.ndarray, dtype float32, dtype float64, numpy.asarray, dtype int64, serving.encode.encode, round)`.
//!
//! The Python reference spends its time in `json.dumps` (the response shell and the record metas), in `_ids_ok`'s
//! per-id loop and in many small numpy calls. Here:
//! - the shell and the metas are written by a JSON writer with `json.dumps`'s rules: `separators=(",", ":")`,
//!   `ensure_ascii`, NaN as NaN, and float and big-int text from Python's own `float.__repr__` and `int.__repr__`;
//! - the ids are checked on their raw bytes;
//! - the columns are copied from each array's `tobytes()`.
//!
//! Anything the writer does not take (a type `json.dumps` would refuse, a string with a lone surrogate) raises, so the
//! guarded caller falls back to the Python writer for that batch.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
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

fn float_text(_ctx: &Ctx, out: &mut Vec<u8>, v: &Bound<PyAny>, f: f64) -> PyResult<()> {
    if f.is_nan() {
        out.extend_from_slice(b"NaN");
    } else if f.is_infinite() {
        out.extend_from_slice(if f > 0.0 { b"Infinity" as &[u8] } else { b"-Infinity" });
    } else {
        // float.__repr__, as json.dumps calls it (a float subclass's own repr is not used)
        let py = v.py();
        let r = if v.get_type().is(&py.get_type_bound::<PyFloat>()) {
            v.repr()?
        } else {
            PyFloat::new_bound(py, f).repr()?
        };
        out.extend_from_slice(r.to_cow()?.as_bytes());
    }
    Ok(())
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
    let b = a.call_method0("tobytes")?;
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
    let Some(hits) = rd.get_item("hits")? else { return Ok(None) };
    let Ok(hd) = hits.downcast::<PyDict>() else { return Ok(None) };
    if !hd.contains("columns")? {
        return Ok(None);
    }
    let c = hd.get_item("columns")?.ok_or_else(|| PyValueError::new_err("columns"))?;
    let ids = c.get_item("_id")?;
    if !ids.is_instance(&ctx.ndarray)? {
        return Ok(None);
    }
    let dt = ids.getattr("dtype")?;
    if dt.getattr("kind")?.extract::<String>()? != "S" {
        return Ok(None);
    }
    let h = ids.len()?;
    if h == 0 {
        return Ok(None);
    }
    if ids.getattr("ndim")?.extract::<usize>()? != 1 {
        return Err(PyValueError::new_err("ids not 1-D"));
    }
    let w0 = dt.getattr("itemsize")?.extract::<usize>()?;
    let raw = tobytes(&ids)?;
    if w0 == 0 || raw.len() != h * w0 || !ids_ok(&raw, w0) {
        return Ok(None);
    }
    let s = ctx.asarray.call1((c.get_item("_score")?, ctx.f32.clone()))?;
    if s.len()? != h {
        return Ok(None);
    }
    let s32 = tobytes(&s)?;
    if s32.len() != 4 * h {
        return Err(PyValueError::new_err("scores not 1-D"));
    }
    let s64: Vec<f64> = s32.chunks(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64).collect();
    let mut cols = Vec::new();
    let sort = c.call_method1("get", ("sort",))?;
    if sort.is_truthy()? {
        for col in sort.iter()? {
            let col = col?;
            let v = col.call_method1("get", ("values",))?;
            let miss = col.call_method1("get", ("missing",))?;
            if !v.is_instance(&ctx.ndarray)? || v.len()? != h {
                return Ok(None);
            }
            let m: Option<Vec<bool>> = if miss.is_none() {
                None
            } else {
                let ma = ctx.asarray.call1((miss, ctx.bool_type.clone()))?;
                if ma.len()? != h {
                    return Ok(None);
                }
                let mb = tobytes(&ma)?;
                if mb.len() != h {
                    return Err(PyValueError::new_err("missing not 1-D"));
                }
                if mb.iter().any(|&x| x != 0) {
                    Some(mb.iter().map(|&x| x != 0).collect())
                } else {
                    None
                }
            };
            let vdt = v.getattr("dtype")?;
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
                let k = vdt.getattr("kind")?.extract::<String>()?;
                if k == "i" || k == "u" {
                    let vals = tobytes(&v.call_method1("astype", (ctx.i64.clone(),))?)?;
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
    let py = msg.py();
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
    let v0 = version.get_item(0)?.call_method1("__format__", ("",))?;
    let v1 = version.get_item(1)?.call_method1("__format__", ("",))?;
    let vs = format!("{}:{}", v0.downcast::<PyString>()?.to_cow()?, v1.downcast::<PyString>()?.to_cow()?);
    meta.extend_from_slice(b"{\"version\":");
    esc(&mut meta, &vs);
    let float_t = py.get_type_bound::<PyFloat>();
    let q = ctx.round.call1((float_t.call1((msg.get_item(5)?,))?, 3))?;
    meta.extend_from_slice(b",\"queue_ms\":");
    dump(ctx, &mut meta, &q, 1)?;
    meta.extend_from_slice(b",\"batch\":");
    if n > 6 {
        dump(ctx, &mut meta, &msg.get_item(6)?, 1)?;
    } else {
        meta.push(b'1');
    }
    if n > 7 {
        let bm = msg.get_item(7)?;
        if !bm.is_none() {
            let b = ctx.round.call1((float_t.call1((bm,))?, 3))?;
            meta.extend_from_slice(b",\"batch_ms\":");
            dump(ctx, &mut meta, &b, 1)?;
        }
    }
    lap(0, &mut t);
    let got = columns(ctx, &resp)?;
    lap(1, &mut t);
    if let Some(row) = got {
        let rd = resp.downcast::<PyDict>()?;
        let hits = rd.get_item("hits")?.ok_or_else(|| PyValueError::new_err("hits"))?;
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

fn pad8(out: &mut Vec<u8>) {
    while out.len() % 8 != 0 {
        out.push(0);
    }
}

#[pyfunction]
pub fn batch4<'py>(py: Python<'py>, msgs: &Bound<'py, PyAny>, ctx: &Bound<'py, PyTuple>) -> PyResult<Bound<'py, PyBytes>> {
    let ctx = Ctx {
        ndarray: ctx.get_item(0)?,
        f32: ctx.get_item(1)?,
        f64: ctx.get_item(2)?,
        asarray: ctx.get_item(3)?,
        i64: ctx.get_item(4)?,
        encode: ctx.get_item(5)?,
        round: ctx.get_item(6)?,
        int_repr: py.get_type_bound::<PyLong>().getattr("__repr__")?,
        bool_type: py.get_type_bound::<PyBool>().into_any(),
    };
    let t_all = Instant::now();
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
    let mut t = Instant::now();
    let w = rows.iter().map(|r| r.w0).max().unwrap_or(1).max(1);
    if w > u16::MAX as usize {
        return Err(PyValueError::new_err("ids too wide"));
    }
    let h_total: usize = rows.iter().map(|r| r.h).sum();
    let mut out: Vec<u8> = Vec::with_capacity(64 + h_total * (w + 4 + 16) + recs.len() * 256);
    out.push(4);
    out.extend_from_slice(&0u64.to_le_bytes());
    out.extend_from_slice(&(recs.len() as u32).to_le_bytes());
    out.extend_from_slice(&(w as u16).to_le_bytes());
    out.extend_from_slice(&(h_total as u32).to_le_bytes());
    for r in &rows {
        for x in r.ids.chunks(r.w0) {
            out.extend_from_slice(x);
            out.resize(out.len() + (w - r.w0), 0);
        }
    }
    pad8(&mut out);
    for r in &rows {
        out.extend_from_slice(&r.s32);
    }
    // shared columns: one per (sort position, kind) any reply uses, in order of first use
    let mut keys: Vec<(usize, u8)> = Vec::new();
    for r in &rows {
        for (k, c) in r.cols.iter().enumerate() {
            if c.kind != 0 && !keys.contains(&(k, c.kind)) {
                keys.push((k, c.kind));
            }
        }
    }
    if keys.len() > 254 {
        return Err(PyValueError::new_err("too many shared columns"));
    }
    out.push(keys.len() as u8);
    let mut offs = vec![0usize];
    for r in &rows {
        offs.push(offs.last().unwrap() + r.h);
    }
    for &(k, kind) in &keys {
        let mut vals = vec![0u8; 8 * h_total];
        let mut miss = vec![false; h_total];
        for (ri, r) in rows.iter().enumerate() {
            if let Some(c) = r.cols.get(k) {
                if c.kind == kind {
                    vals[8 * offs[ri]..8 * offs[ri + 1]].copy_from_slice(&c.vals);
                    if let Some(m) = &c.miss {
                        miss[offs[ri]..offs[ri + 1]].copy_from_slice(m);
                    }
                }
            }
        }
        let flags = if miss.iter().any(|&x| x) { 1u8 } else { 0 };
        out.push(kind);
        out.push(flags);
        pad8(&mut out);
        out.extend_from_slice(&vals);
        if flags == 1 {
            let mut bits = vec![0u8; (h_total + 7) / 8];
            for (i, &x) in miss.iter().enumerate() {
                if x {
                    bits[i / 8] |= 1 << (i % 8);
                }
            }
            out.extend_from_slice(&bits);
        }
    }
    for rec in &recs {
        match rec {
            Rec::Hits { id, meta, row } => {
                let r = &rows[*row];
                out.extend_from_slice(&id.to_le_bytes());
                out.push(3);
                out.extend_from_slice(&(offs[*row] as u32).to_le_bytes());
                out.extend_from_slice(&(r.h as u32).to_le_bytes());
                if r.cols.len() > 255 {
                    return Err(PyValueError::new_err("too many sort columns"));
                }
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
    let mut framed = Vec::with_capacity(out.len() + 4);
    framed.extend_from_slice(&(out.len() as u32).to_le_bytes());
    framed.extend_from_slice(&out);
    let r = PyBytes::new_bound(py, &framed);
    lap(3, &mut t);
    PROF[4].fetch_add((Instant::now() - t_all).as_nanos() as u64, Ordering::Relaxed);
    Ok(r)
}
