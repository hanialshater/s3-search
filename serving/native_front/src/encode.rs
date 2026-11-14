//! The hits list as JSON: the same bytes as `serving/encode.py`'s `encode_hits` for what the GPU process sends as
//! kind 1 (`serving/binproto.py`).
//!
//! The GPU process sends kind 1 only when every id is printable ASCII other than `"` and `\`, and every float is
//! finite and either 0 or in 1e-4 <= |x| < 1e16. In that range `ryu`'s shortest round-trip text is what Python's
//! `float.__repr__` writes: positional, with `.0` on integral values. `tests` checks this against vectors that Python
//! wrote.

pub struct Block<'a> {
    pub h: usize,
    pub w: usize,
    pub ids: &'a [u8],
    pub scores: &'a [u8],       // h little-endian f32
    pub sort: Vec<Col<'a>>,
}

pub enum Col<'a> {
    SameAsScore,
    F64(&'a [u8]),              // h little-endian f64
    I64(&'a [u8]),              // h little-endian i64
    F64M(&'a [u8], &'a [u8]),   // S3B3 (kind 4, split): f64 values and a missing bitmap (bit i set: hit i is null)
    I64M(&'a [u8], &'a [u8]),
}

#[inline]
fn missing(m: &[u8], i: usize) -> bool {
    m[i / 8] >> (i % 8) & 1 == 1
}

#[inline]
pub fn push_f64(out: &mut Vec<u8>, x: f64, buf: &mut ryu::Buffer) {
    out.extend_from_slice(buf.format_finite(x).as_bytes());
}

#[inline]
fn f32_at(b: &[u8], i: usize) -> f32 {
    f32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]])
}

#[inline]
fn f64_at(b: &[u8], i: usize) -> f64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[8 * i..8 * i + 8]);
    f64::from_le_bytes(a)
}

#[inline]
fn i64_at(b: &[u8], i: usize) -> i64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[8 * i..8 * i + 8]);
    i64::from_le_bytes(a)
}

/// Parse a hits block: `u32 h, u16 w, ids (h*w), scores (h*4), u8 n, per column u8 code [values]`.
pub fn parse_block(b: &[u8]) -> Option<Block<'_>> {
    if b.len() < 6 {
        return None;
    }
    let h = u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize;
    let w = u16::from_le_bytes([b[4], b[5]]) as usize;
    let mut o = 6;
    let ids = b.get(o..o + h * w)?;
    o += h * w;
    let scores = b.get(o..o + 4 * h)?;
    o += 4 * h;
    let n = *b.get(o)? as usize;
    o += 1;
    let mut sort = Vec::with_capacity(n);
    for _ in 0..n {
        let code = *b.get(o)?;
        o += 1;
        match code {
            0 => sort.push(Col::SameAsScore),
            1 => {
                sort.push(Col::F64(b.get(o..o + 8 * h)?));
                o += 8 * h;
            }
            2 => {
                sort.push(Col::I64(b.get(o..o + 8 * h)?));
                o += 8 * h;
            }
            3 | 4 => {                                          // only in a kind 4 frame's split replies
                let v = b.get(o..o + 8 * h)?;
                let m = b.get(o + 8 * h..o + 8 * h + (h + 7) / 8)?;
                o += 8 * h + (h + 7) / 8;
                sort.push(if code == 3 { Col::F64M(v, m) } else { Col::I64M(v, m) });
            }
            _ => return None,
        }
    }
    Some(Block { h, w, ids, scores, sort })
}

/// `[{"_id":"..","_score":..,"sort":[..]},...]`
pub fn encode_hits(blk: &Block, out: &mut Vec<u8>) {
    let mut fb = ryu::Buffer::new();
    out.push(b'[');
    let mut score_txt: Vec<u8> = Vec::with_capacity(24);
    for i in 0..blk.h {
        if i > 0 {
            out.push(b',');
        }
        out.extend_from_slice(b"{\"_id\":\"");
        let id = &blk.ids[i * blk.w..(i + 1) * blk.w];
        let end = id.iter().rposition(|&c| c != 0).map(|p| p + 1).unwrap_or(0);
        out.extend_from_slice(&id[..end]);
        out.extend_from_slice(b"\",\"_score\":");
        score_txt.clear();
        push_f64(&mut score_txt, f32_at(blk.scores, i) as f64, &mut fb);
        out.extend_from_slice(&score_txt);
        out.extend_from_slice(b",\"sort\":[");
        for (k, c) in blk.sort.iter().enumerate() {
            if k > 0 {
                out.push(b',');
            }
            match c {
                Col::SameAsScore => out.extend_from_slice(&score_txt),
                Col::F64(v) => push_f64(out, f64_at(v, i), &mut fb),
                Col::I64(v) => {
                    let mut ib = itoa_buf(i64_at(v, i));
                    out.append(&mut ib);
                }
                Col::F64M(v, m) => if missing(m, i) { out.extend_from_slice(b"null") } else { push_pyrepr(out, f64_at(v, i), &mut fb) },
                Col::I64M(v, m) => if missing(m, i) { out.extend_from_slice(b"null") } else { out.append(&mut itoa_buf(i64_at(v, i))) },
            }
        }
        out.extend_from_slice(b"]}");
    }
    out.push(b']');
}

#[inline]
fn itoa_buf(v: i64) -> Vec<u8> {
    v.to_string().into_bytes()
}

/// `json.dumps(x)` for any float64, byte for byte (S3B2, kind 3): Python's `float.__repr__`, the shortest round-trip
/// digits (ryu's, the same digits as CPython's dtoa shortest mode) laid out by CPython's 'r' rule: positional when the
/// decimal point position decpt (x = 0.d1d2... x 10^decpt) has -4 < decpt <= 16, else d[.ddd]e+XX / e-XX (at least
/// two exponent digits); and NaN, Infinity, -Infinity as json.dumps writes them.
pub fn push_pyrepr(out: &mut Vec<u8>, x: f64, buf: &mut ryu::Buffer) {
    // The common case first (plat-m26: render 0.66 against 0.30 ms on kind 1): in repr's positional range ryu's text is
    // already Python's (kind 1's whole premise, held by tests/float_vectors.txt), so no layout work and no allocation.
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
    let mut dbuf = [0u8; 64];                                  // no allocation on the slow path (ryu's text is short)
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

/// `json.encoder.encode_basestring_ascii(s)`, byte for byte: quoted; `"` `\` and \b \f \n \r \t as their short
/// escapes; every other character outside space..~ as \uXXXX (lower-case hex; a surrogate pair above U+FFFF).
pub fn push_json_ascii(out: &mut Vec<u8>, s: &str) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let b = s.as_bytes();
    if b.iter().all(|&c| (0x20..0x7f).contains(&c) && c != b'"' && c != b'\\') {
        out.push(b'"');                                         // printable ASCII, nothing to escape: copy it
        out.extend_from_slice(b);
        out.push(b'"');
        return;
    }
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
            c => {
                let mut units = [0u16; 2];
                for u in c.encode_utf16(&mut units).iter() {
                    out.extend_from_slice(b"\\u");
                    for shift in [12, 8, 4, 0] {
                        out.push(HEX[((u >> shift) & 0xf) as usize]);
                    }
                }
            }
        }
    }
    out.push(b'"');
}

/// Kind 3 (S3B2): the hits list as `serving/encode.py`'s `encode_hits` writes it for any ids and floats: each id
/// (UTF-8, its trailing NUL padding dropped) escaped by `push_json_ascii`, each float by `push_pyrepr`. None when an
/// id is not UTF-8 (the interface says it always is).
pub fn encode_hits3(blk: &Block, out: &mut Vec<u8>) -> Option<()> {
    let mut fb = ryu::Buffer::new();
    out.push(b'[');
    let mut score_txt: Vec<u8> = Vec::with_capacity(32);
    for i in 0..blk.h {
        if i > 0 {
            out.push(b',');
        }
        out.extend_from_slice(b"{\"_id\":");
        let id = &blk.ids[i * blk.w..(i + 1) * blk.w];
        let end = id.iter().rposition(|&c| c != 0).map(|p| p + 1).unwrap_or(0);
        push_json_ascii(out, std::str::from_utf8(&id[..end]).ok()?);
        out.extend_from_slice(b",\"_score\":");
        score_txt.clear();
        push_pyrepr(&mut score_txt, f32_at(blk.scores, i) as f64, &mut fb);
        out.extend_from_slice(&score_txt);
        out.extend_from_slice(b",\"sort\":[");
        for (k, c) in blk.sort.iter().enumerate() {
            if k > 0 {
                out.push(b',');
            }
            match c {
                Col::SameAsScore => out.extend_from_slice(&score_txt),
                Col::F64(v) => push_pyrepr(out, f64_at(v, i), &mut fb),
                Col::I64(v) => {
                    let mut ib = itoa_buf(i64_at(v, i));
                    out.append(&mut ib);
                }
                Col::F64M(v, m) => if missing(m, i) { out.extend_from_slice(b"null") } else { push_pyrepr(out, f64_at(v, i), &mut fb) },
                Col::I64M(v, m) => if missing(m, i) { out.extend_from_slice(b"null") } else { out.append(&mut itoa_buf(i64_at(v, i))) },
            }
        }
        out.extend_from_slice(b"]}");
    }
    out.push(b']');
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Python wrote `repr(x)` for each value (tests/float_vectors.txt: one `hexbits repr` pair per line).
    #[test]
    fn floats_match_python_repr() {
        let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/float_vectors.txt"))
            .expect("float vectors");
        let mut fb = ryu::Buffer::new();
        let mut n = 0;
        for line in text.lines() {
            let mut it = line.split(' ');
            let bits = u64::from_str_radix(it.next().unwrap(), 16).unwrap();
            let want = it.next().unwrap();
            let mut out = Vec::new();
            push_f64(&mut out, f64::from_bits(bits), &mut fb);
            assert_eq!(std::str::from_utf8(&out).unwrap(), want, "bits {:016x}", bits);
            n += 1;
        }
        assert!(n > 10_000);
    }

    fn vectors(name: &str) -> Vec<(String, String)> {
        let path = format!("{}/tests/{}", env!("CARGO_MANIFEST_DIR"), name);
        std::fs::read_to_string(&path).expect(name).lines().filter(|l| !l.is_empty())
            .map(|l| { let mut it = l.splitn(2, ' '); (it.next().unwrap().to_string(), it.next().unwrap_or("").to_string()) })
            .collect()
    }

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
    }

    /// S3B2: json.dumps(x) for every float64 class (tests/pyrepr_vectors.txt, written by tests/make_vectors.py).
    #[test]
    fn pyrepr_matches_python_json_dumps() {
        let mut fb = ryu::Buffer::new();
        let v = vectors("pyrepr_vectors.txt");
        for (bits, want) in &v {
            let x = f64::from_bits(u64::from_str_radix(bits, 16).unwrap());
            let mut out = Vec::new();
            push_pyrepr(&mut out, x, &mut fb);
            assert_eq!(std::str::from_utf8(&out).unwrap(), want, "bits {}", bits);
        }
        assert!(v.len() > 30_000);
    }

    /// S3B2: encode_basestring_ascii for fuzzed UTF-8 (tests/escape_vectors.txt).
    #[test]
    fn escape_matches_python_encode_basestring_ascii() {
        let v = vectors("escape_vectors.txt");
        for (src, want) in &v {
            let s = String::from_utf8(unhex(src)).unwrap();
            let mut out = Vec::new();
            push_json_ascii(&mut out, &s);
            assert_eq!(out, unhex(want), "source {}", src);
        }
        assert!(v.len() > 2_000);
    }

    /// S3B2: a whole kind 3 reply payload rendered is serving.encode.encode's body (tests/golden3.txt).
    #[test]
    fn kind3_frames_render_python_bytes() {
        let v = vectors("golden3.txt");
        for (payload, body) in &v {
            let p = unhex(payload);
            let (st, out, _) = crate::render(&p).expect("render");
            assert_eq!(st, 200);
            assert_eq!(out, unhex(body), "payload {}", &payload[..40]);
        }
        assert!(v.len() >= 60);
    }
}
