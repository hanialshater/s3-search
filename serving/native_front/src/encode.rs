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
}
