//! S3B3 (kind 4): one frame per batch, split here into one payload per reply, each in the form the front already
//! renders: kind 3 (the reply's slices of the shared arrays: its ids, scores and sort values, a missing value as
//! sort code 3 or 4 with a bitmap) or kind 2 (a whole response). The layout is `serving/binproto.py`'s (`batch4`).

fn u16_at(b: &[u8], o: usize) -> Option<u16> {
    Some(u16::from_le_bytes(b.get(o..o + 2)?.try_into().ok()?))
}
fn u32_at(b: &[u8], o: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(o..o + 4)?.try_into().ok()?))
}
fn u64_at(b: &[u8], o: usize) -> Option<u64> {
    Some(u64::from_le_bytes(b.get(o..o + 8)?.try_into().ok()?))
}
fn pad8(o: usize) -> usize {
    o + ((8 - o % 8) % 8)
}

struct Col<'a> {
    kind: u8,                  // 1 float64, 2 int64
    values: &'a [u8],          // 8 * h_total
    missing: Option<&'a [u8]>, // ceil(h_total / 8), LSB first
}

/// `[(id, payload)]`, each payload kind 3 or kind 2 as `render` takes; None when the frame is malformed.
pub fn split(p: &[u8]) -> Option<Vec<(u64, Vec<u8>)>> {
    let mut o = 9;                                              // u8 kind, u64 0
    let n = u32_at(p, o)? as usize;
    let w = u16_at(p, o + 4)? as usize;
    let ht = u32_at(p, o + 6)? as usize;
    o += 10;
    let ids = p.get(o..o + ht * w)?;
    o = pad8(o + ht * w);
    let scores = p.get(o..o + 4 * ht)?;
    o += 4 * ht;
    let nc = *p.get(o)? as usize;
    o += 1;
    let mut cols = Vec::with_capacity(nc);
    for _ in 0..nc {
        let kind = *p.get(o)?;
        let flags = *p.get(o + 1)?;
        o = pad8(o + 2);
        let values = p.get(o..o + 8 * ht)?;
        o += 8 * ht;
        let missing = if flags & 1 == 1 {
            let nb = (ht + 7) / 8;
            let m = p.get(o..o + nb)?;
            o += nb;
            Some(m)
        } else {
            None
        };
        if kind != 1 && kind != 2 {
            return None;
        }
        cols.push(Col { kind, values, missing });
    }
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let id = u64_at(p, o)?;
        let rk = *p.get(o + 8)?;
        o += 9;
        let mut q: Vec<u8> = Vec::new();
        if rk == 3 {
            let off = u32_at(p, o)? as usize;
            let h = u32_at(p, o + 4)? as usize;
            let ns = *p.get(o + 8)? as usize;
            o += 9;
            let codes = p.get(o..o + ns)?.to_vec();
            o += ns;
            let ml = u32_at(p, o)? as usize;
            let meta = p.get(o + 4..o + 4 + ml)?;
            o += 4 + ml;
            if off + h > ht {
                return None;
            }
            q.push(3);
            q.extend_from_slice(&id.to_le_bytes());
            q.extend_from_slice(&(ml as u32).to_le_bytes());
            q.extend_from_slice(meta);
            q.extend_from_slice(&(h as u32).to_le_bytes());     // the kind 3 block: u32 h, u16 w, ids, scores, cols
            q.extend_from_slice(&(w as u16).to_le_bytes());
            q.extend_from_slice(&ids[off * w..(off + h) * w]);
            q.extend_from_slice(&scores[off * 4..(off + h) * 4]);
            q.push(ns as u8);
            for &code in &codes {
                if code == 0 {
                    q.push(0);
                    continue;
                }
                let c = cols.get(code as usize - 1)?;
                let vals = &c.values[off * 8..(off + h) * 8];
                match c.missing {
                    None => {
                        q.push(c.kind);                         // 1 float64, 2 int64
                        q.extend_from_slice(vals);
                    }
                    Some(m) => {
                        q.push(c.kind + 2);                     // 3 float64, 4 int64, each with this reply's bitmap
                        q.extend_from_slice(vals);
                        let mut bits = vec![0u8; (h + 7) / 8];
                        for i in 0..h {
                            let g = off + i;
                            if m[g / 8] >> (g % 8) & 1 == 1 {
                                bits[i / 8] |= 1 << (i % 8);
                            }
                        }
                        q.extend_from_slice(&bits);
                    }
                }
            }
        } else if rk == 2 {
            let ml = u32_at(p, o)? as usize;
            let meta = p.get(o + 4..o + 4 + ml)?;
            o += 4 + ml;
            let st = u16_at(p, o)?;
            let bl = u32_at(p, o + 2)? as usize;
            let body = p.get(o + 6..o + 6 + bl)?;
            o += 6 + bl;
            q.push(2);
            q.extend_from_slice(&id.to_le_bytes());
            q.extend_from_slice(&(ml as u32).to_le_bytes());
            q.extend_from_slice(meta);
            q.extend_from_slice(&st.to_le_bytes());
            q.extend_from_slice(&(bl as u32).to_le_bytes());
            q.extend_from_slice(body);
        } else {
            return None;
        }
        out.push((id, q));
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    /// S3B3: every record of each kind 4 frame (tests/golden4.txt, written by tests/make_vectors.py) split and rendered
    /// is the body serving.encode.encode writes for that reply.
    #[test]
    fn kind4_frames_split_and_render_python_bytes() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden4.txt");
        let text = std::fs::read_to_string(path).expect("golden4");
        let unhex = |s: &str| (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect::<Vec<u8>>();
        let mut n = 0;
        for line in text.lines().filter(|l| !l.is_empty()) {
            let mut it = line.split(' ');
            let frame = unhex(it.next().unwrap());
            let want: Vec<(u64, u16, Vec<u8>)> = it.map(|r| {
                let mut f = r.split(':');
                (f.next().unwrap().parse().unwrap(), f.next().unwrap().parse().unwrap(), unhex(f.next().unwrap()))
            }).collect();
            let got = super::split(&frame).expect("split");
            assert_eq!(got.len(), want.len());
            for ((gid, payload), (wid, wst, wbody)) in got.iter().zip(want.iter()) {
                assert_eq!(gid, wid);
                let (st, body, _) = crate::render(payload).expect("render");
                assert_eq!((st, &body), (*wst, wbody), "id {}", gid);
                n += 1;
            }
        }
        assert!(n > 100);
    }
}
