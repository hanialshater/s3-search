//! R1 of the founder's CPU-headroom plan (architect): the fused lexical path's per-request host work in Rust.
//!
//! `HostIndex.prepare(payloads)` takes a batch of request JSON bodies. For each request whose plan the fused path
//! (engine/backend/triton_fused.py) serves, it does what engine.dsl, engine.plan and the executor's `_fused_lexical`
//! do in Python, and returns the kernel's inputs for the whole batch as two byte blocks, one int64 and one float32,
//! laid out per chunk exactly as `triton_fused._run_kernel` builds them. The steps are:
//!
//! - parse the request;
//! - analyse the query text;
//! - plan: terms, msm, field weights, idf mode, boost, from, size and terms aggregations;
//! - group the requests by weights and idf mode, in chunks of 64.
//!
//! Every other request is returned in `fallback` and goes through today's Python path, which also gives every error
//! response. Being exact comes first: a request is served here only when its plan is certain to equal the Python
//! planner's. That rules out, for example:
//! - a non-ASCII query text;
//! - a word the index's analysis table does not hold;
//! - a key, type or value outside the accepted shapes;
//! - a field that is not a text field (keyword fields for aggregations);
//! - a synonym, or fuzziness;
//! - msm 0;
//! - a query over the postings chunk limit.
//!
//! The analysis table (word -> token, or none for a stopword) comes from engine/analysis.py at index load, so no
//! tokeniser or stemmer logic is duplicated: the regex's words are found here, and their tokens are Python's own.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use serde_json::{Map, Value};
use std::collections::{HashMap, HashSet};

const MAX_WINDOW: u64 = 10_000;
const BATCH_ROWS: usize = 64;

#[pyclass]
struct HostIndex {
    word_token: HashMap<String, Option<String>>,
    vocab: HashMap<String, i64>,
    term_ptr: Vec<i64>,
    text_fields: Vec<String>,
    field_index: HashMap<String, usize>,
    field_b: Vec<f32>,                   // the mapping's b per text field (float32, as the executor's bb)
    keyword_sizes: HashMap<String, i64>, // keyword field -> its number of codes (len of its values)
    synonyms: HashSet<String>,
    max_postings: i64,
}

struct Agg {
    name: String,
    field: String,
    size: i64,
}

struct Planned {
    pos: usize,
    terms: Vec<i64>, // term ids of the known terms, in query order
    msm: i64,
    boost: f64,
    from: i64,
    size: i64,
    w: Vec<f32>,     // per text field, in the index's field order (0 where not queried)
    idf_field: i64,  // the one queried field's index for "field" idf, else -1 ("global")
    aggs: Vec<Agg>,
}

fn is_int(v: &Value) -> Option<i64> {
    // Python's _int: an int (not a bool, not a float), >= 0
    match v {
        Value::Number(n) if n.is_u64() => n.as_u64().and_then(|x| i64::try_from(x).ok()),
        _ => None,
    }
}

fn finite_num(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64().filter(|x| x.is_finite()),
        _ => None,
    }
}

fn msm_ok(v: &Value) -> Option<Option<String>> {
    // Python's _msm: None, an int (not a bool), or a string matching \s*-?\d+%?\s*; its text form is returned.
    // Only spaces, digits, one leading '-' and one trailing '%' are accepted here: Python's \s, \d and str.strip
    // are Unicode-aware (and treat \x1c-\x1f as whitespace), so anything else goes to Python.
    match v {
        Value::Null => Some(None),
        Value::Number(n) if n.is_i64() || n.is_u64() => Some(Some(n.to_string())),
        Value::String(s) => {
            if !s.bytes().all(|c| c == b' ' || c == b'-' || c == b'%' || c.is_ascii_digit()) {
                return None;
            }
            let t = s.trim_matches(' ');
            let b = t.as_bytes();
            let mut i = 0;
            if i < b.len() && b[i] == b'-' {
                i += 1;
            }
            let d0 = i;
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
            if i == d0 {
                return None;
            }
            if i < b.len() && b[i] == b'%' {
                i += 1;
            }
            if i != b.len() {
                return None;
            }
            Some(Some(t.to_string()))
        }
        _ => None,
    }
}

fn resolve_msm(spec: &Option<String>, n: i64) -> Option<i64> {
    // engine.dsl.parse.resolve_msm
    let k = match spec {
        None => return Some(1),
        Some(s) => {
            if let Some(p) = s.strip_suffix('%') {
                let p: i64 = p.parse().ok()?;
                if p >= 0 {
                    ((n * p) as f64 / 100.0).floor() as i64
                } else {
                    n - ((n * -p) as f64 / 100.0).floor() as i64
                }
            } else {
                let v: i64 = s.parse().ok()?;
                if v >= 0 { v } else { n + v }
            }
        }
    };
    Some(k.clamp(0, n))
}

fn words(text: &str) -> Vec<&str> {
    // [a-z0-9]+(?:'[a-z0-9]+)? over an ASCII, already lowercased text, as re.findall
    let b = text.as_bytes();
    let al = |c: u8| c.is_ascii_lowercase() || c.is_ascii_digit();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if !al(b[i]) {
            i += 1;
            continue;
        }
        let s = i;
        while i < b.len() && al(b[i]) {
            i += 1;
        }
        if i + 1 < b.len() && b[i] == b'\'' && al(b[i + 1]) {
            i += 1;
            while i < b.len() && al(b[i]) {
                i += 1;
            }
        }
        out.push(&text[s..i]);
    }
    out
}

fn parse_fields_entry(f: &str) -> Option<(String, Option<f64>)> {
    // ([^\^*?]+)(?:\^([0-9]*\.?[0-9]+))?  (fullmatch)
    let (name, w) = match f.find('^') {
        Some(k) => (&f[..k], Some(&f[k + 1..])),
        None => (f, None),
    };
    if name.is_empty() || name.contains('*') || name.contains('?') {
        return None;
    }
    let wv = match w {
        None => None,
        Some(s) => {
            let b = s.as_bytes();
            let mut i = 0;
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
            if i < b.len() && b[i] == b'.' {
                i += 1;
            }
            let d = i;
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
            if i == d || i != b.len() {
                return None;
            }
            Some(s.parse::<f64>().ok()?)
        }
    };
    Some((name.to_string(), wv))
}

impl HostIndex {
    fn plan_one(&self, pos: usize, body: &[u8]) -> Option<Planned> {
        let v: Value = serde_json::from_slice(body).ok()?;
        let req = v.as_object()?;
        for k in req.keys() {
            if !matches!(k.as_str(), "query" | "size" | "from" | "aggs" | "aggregations" | "track_total_hits") {
                return None;
            }
        }
        if req.contains_key("aggs") && req.contains_key("aggregations") {
            return None;
        }
        if let Some(t) = req.get("track_total_hits") {
            match t {
                Value::Bool(_) => {}
                Value::Number(n) if n.is_i64() || n.is_u64() => {}
                _ => return None,
            }
        }
        let from = match req.get("from") { None => 0, Some(x) => is_int(x)? };
        let size = match req.get("size") { None => 10, Some(x) => is_int(x)? };
        if (from as u64) + (size as u64) > MAX_WINDOW || from + size == 0 {
            return None;
        }
        let q = req.get("query")?.as_object()?;
        if q.len() != 1 {
            return None;
        }
        let (kind, body) = q.iter().next()?;
        let (text, weights, operator, msm, boost) = match kind.as_str() {
            "match" => self.match_body(body.as_object()?)?,
            "multi_match" => self.multi_match_body(body.as_object()?)?,
            _ => return None,
        };
        if !text.is_ascii() {
            return None;
        }
        let lower = text.to_ascii_lowercase();
        let mut terms: Vec<String> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for w in words(&lower) {
            match self.word_token.get(w)? {
                None => {}
                Some(t) => {
                    if seen.insert(t.clone()) {
                        terms.push(t.clone());
                    }
                }
            }
        }
        if terms.iter().any(|t| self.synonyms.contains(t)) {
            return None;
        }
        let n = terms.len() as i64;
        let need = if operator == "and" { n } else { std::cmp::max(1, resolve_msm(&msm, n)?) };
        let msm_final = std::cmp::min(need, n);
        if msm_final < 1 {
            return None;
        }
        let f = self.text_fields.len();
        let mut wv = vec![0f32; f];
        let mut positive = Vec::new();
        for (name, wt) in &weights {
            let i = *self.field_index.get(name)?;
            wv[i] = *wt as f32;
        }
        for i in 0..f {
            let name = &self.text_fields[i];
            if let Some(wt) = weights.iter().find(|(nm, _)| nm == name).map(|(_, w)| *w) {
                if wt > 0.0 {
                    positive.push(i);
                }
            }
        }
        let idf_field = if positive.len() == 1 { positive[0] as i64 } else { -1 };
        let mut ids = Vec::new();
        let mut n_post: i64 = 0;
        for t in &terms {
            if let Some(&id) = self.vocab.get(t) {
                ids.push(id);
                n_post += self.term_ptr[(id + 1) as usize] - self.term_ptr[id as usize];
            }
        }
        if n_post > self.max_postings {
            return None;
        }
        let akey = if req.contains_key("aggs") { "aggs" } else { "aggregations" };
        let mut aggs = Vec::new();
        if let Some(a) = req.get(akey) {
            for (name, spec) in a.as_object()? {
                let spec = spec.as_object()?;
                if spec.len() != 1 {
                    return None;
                }
                let t = spec.get("terms")?.as_object()?;
                for k in t.keys() {
                    if !matches!(k.as_str(), "field" | "size") {
                        return None;
                    }
                }
                let field = t.get("field")?.as_str()?.to_string();
                if !self.keyword_sizes.contains_key(&field) {
                    return None;
                }
                let sz = match t.get("size") { None => 10, Some(x) => is_int(x)? };
                if sz < 1 {
                    return None;
                }
                aggs.push(Agg { name: name.clone(), field, size: sz });
            }
        }
        Some(Planned { pos, terms: ids, msm: msm_final, boost, from, size, w: wv, idf_field, aggs })
    }

    fn match_body(&self, b: &Map<String, Value>) -> Option<(String, Vec<(String, f64)>, String, Option<String>, f64)> {
        if b.len() != 1 {
            return None;
        }
        let (field, spec) = b.iter().next()?;
        if field.is_empty() || !self.field_index.contains_key(field) {
            return None;
        }
        let w = vec![(field.clone(), 1.0)];
        match spec {
            Value::String(s) => Some((s.clone(), w, "or".into(), None, 1.0)),
            Value::Object(o) => {
                for k in o.keys() {
                    if !matches!(k.as_str(), "query" | "operator" | "minimum_should_match" | "boost") {
                        return None;
                    }
                }
                let text = o.get("query")?.as_str()?.to_string();
                let op = self.operator(o.get("operator"))?;
                let msm = msm_ok(o.get("minimum_should_match").unwrap_or(&Value::Null))?;
                let boost = match o.get("boost") { None => 1.0, Some(x) => finite_num(x)? };
                Some((text, w, op, msm, boost))
            }
            _ => None,
        }
    }

    fn multi_match_body(&self, o: &Map<String, Value>) -> Option<(String, Vec<(String, f64)>, String, Option<String>, f64)> {
        for k in o.keys() {
            if !matches!(k.as_str(), "query" | "fields" | "type" | "operator" | "minimum_should_match" | "tie_breaker" | "boost") {
                return None;
            }
        }
        match o.get("type") {
            None => {}
            Some(Value::String(t)) if matches!(t.as_str(), "best_fields" | "most_fields" | "cross_fields") => {}
            _ => return None,
        }
        let text = o.get("query")?.as_str()?.to_string();
        let mut weights: Vec<(String, f64)> = Vec::new();
        match o.get("fields") {
            None => {
                for f in &self.text_fields {
                    weights.push((f.clone(), 1.0));
                }
            }
            Some(fl) => {
                let list: Vec<&Value> = match fl {
                    Value::String(_) => vec![fl],
                    Value::Array(a) if !a.is_empty() => a.iter().collect(),
                    _ => return None,
                };
                for f in list {
                    let (name, w) = parse_fields_entry(f.as_str()?)?;
                    if !self.field_index.contains_key(&name) {
                        return None;
                    }
                    let w = w.unwrap_or(1.0);
                    if let Some(e) = weights.iter_mut().find(|(n, _)| *n == name) {
                        e.1 = w; // a dict: the later weight wins
                    } else {
                        weights.push((name, w));
                    }
                }
            }
        }
        if let Some(tb) = o.get("tie_breaker") {
            if !tb.is_null() {
                finite_num(tb)?;
            }
        }
        let op = self.operator(o.get("operator"))?;
        let msm = msm_ok(o.get("minimum_should_match").unwrap_or(&Value::Null))?;
        let boost = match o.get("boost") { None => 1.0, Some(x) => finite_num(x)? };
        Some((text, weights, op, msm, boost))
    }

    fn operator(&self, v: Option<&Value>) -> Option<String> {
        match v {
            None => Some("or".into()),
            Some(Value::String(s)) => {
                let l = s.to_ascii_lowercase();
                if s.is_ascii() && (l == "or" || l == "and") { Some(l) } else { None }
            }
            _ => None,
        }
    }
}

#[pymethods]
impl HostIndex {
    #[new]
    #[pyo3(signature = (words, tokens, vocab, term_ptr, text_fields, field_b, keyword_sizes, synonyms, max_postings))]
    #[allow(clippy::too_many_arguments)]
    fn new(words: Vec<String>, tokens: Vec<Option<String>>, vocab: Vec<String>, term_ptr: &[u8],
           text_fields: Vec<String>, field_b: Vec<f64>, keyword_sizes: HashMap<String, i64>, synonyms: Vec<String>,
           max_postings: i64) -> PyResult<Self> {
        if words.len() != tokens.len() || term_ptr.len() % 8 != 0 || term_ptr.len() / 8 != vocab.len() + 1
            || field_b.len() != text_fields.len() {
            return Err(PyValueError::new_err("HostIndex: mismatched inputs"));
        }
        let term_ptr: Vec<i64> = term_ptr.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect();
        let word_token = words.into_iter().zip(tokens).collect();
        let vocab = vocab.into_iter().enumerate().map(|(i, t)| (t, i as i64)).collect();
        let field_index = text_fields.iter().enumerate().map(|(i, f)| (f.clone(), i)).collect();
        let field_b = field_b.into_iter().map(|x| x as f32).collect();
        Ok(HostIndex { word_token, vocab, term_ptr, text_fields, field_index, field_b, keyword_sizes,
                       synonyms: synonyms.into_iter().collect(), max_postings })
    }

    /// The batch's kernel inputs (see the module docs). Returns a dict:
    /// - `fallback`: the positions Python must serve; `eligible`: the served ones, in chunk order;
    /// - `ints` and `floats`: little-endian int64 and float32 byte blocks;
    /// - `chunks`: one per kernel call, each with `positions`, `int_at`, `int_len` (row_start, row_len, q_lo, q_hi,
    ///   msm, facet offsets, facet sizes, row_term), `float_at` (the offsets of w, b and 1 - b per text field, and a
    ///   boost per query; every segment 16-byte aligned),
    ///   `idf_field` and `facets` (field names), plus per
    ///   position `from`, `size` and `aggs` (name, field, size).
    fn prepare<'py>(&self, py: Python<'py>, payloads: Vec<Vec<u8>>) -> PyResult<Bound<'py, PyDict>> {
        let mut fallback = Vec::new();
        let mut planned = Vec::new();
        for (pos, body) in payloads.iter().enumerate() {
            match self.plan_one(pos, body) {
                Some(p) => planned.push(p),
                None => fallback.push(pos),
            }
        }
        // group by (weights' bits, idf field), in order of first appearance; chunks of BATCH_ROWS
        let mut keys: Vec<(Vec<u32>, i64)> = Vec::new();
        let mut members: Vec<Vec<usize>> = Vec::new();
        for (k, p) in planned.iter().enumerate() {
            let key = (p.w.iter().map(|x| x.to_bits()).collect::<Vec<u32>>(), p.idf_field);
            match keys.iter().position(|x| *x == key) {
                Some(g) => members[g].push(k),
                None => {
                    keys.push(key);
                    members.push(vec![k]);
                }
            }
        }
        let mut ints: Vec<i64> = Vec::new();
        let mut floats: Vec<f32> = Vec::new();
        let mut eligible: Vec<usize> = Vec::new(); // the served positions in chunk order: search_prepared's order
        let chunks = PyList::empty_bound(py);
        for (g, mem) in members.iter().enumerate() {
            for ch in mem.chunks(BATCH_ROWS) {
                let (mut rs, mut rl, mut qlo, mut qhi, mut msm, mut rt) = (vec![], vec![], vec![], vec![], vec![], vec![]);
                let mut facets: Vec<String> = Vec::new();
                for &k in ch {
                    let p = &planned[k];
                    qlo.push(rs.len() as i64);
                    for &t in &p.terms {
                        let s = self.term_ptr[t as usize];
                        rs.push(s);
                        rl.push(self.term_ptr[(t + 1) as usize] - s);
                        rt.push(t);
                    }
                    qhi.push(rs.len() as i64);
                    msm.push(p.msm);
                    for a in &p.aggs {
                        if !facets.contains(&a.field) {
                            facets.push(a.field.clone());
                        }
                    }
                }
                let q = ch.len() as i64;
                let sizes: Vec<i64> = facets.iter().map(|f| self.keyword_sizes[f]).collect();
                let mut offs = Vec::new();
                let mut acc = 0i64;
                for s in &sizes {
                    offs.push(acc);
                    acc += q * s;
                }
                let parts: [Vec<i64>; 8] = [
                    if rs.is_empty() { vec![0] } else { rs },
                    if rl.is_empty() { vec![0] } else { rl },
                    qlo, qhi, msm,
                    if offs.is_empty() { vec![0] } else { offs },
                    if sizes.is_empty() { vec![1] } else { sizes },
                    if rt.is_empty() { vec![0] } else { rt },
                ];
                let mut int_at = Vec::new();
                let mut int_len = Vec::new();
                for part in parts.iter() {
                    while ints.len() % 2 != 0 {
                        ints.push(0); // each segment 16-byte aligned (triton_fused.pack)
                    }
                    int_at.push(ints.len());
                    int_len.push(part.len());
                    ints.extend_from_slice(part);
                }
                let w = &planned[ch[0]].w;
                let omb: Vec<f32> = self.field_b.iter().map(|b| 1.0f32 - *b).collect(); // 1 - b, in float32
                let boosts: Vec<f32> = ch.iter().map(|&k| planned[k].boost as f32).collect();
                let mut float_at = Vec::new();
                for seg in [w.as_slice(), self.field_b.as_slice(), omb.as_slice(), boosts.as_slice()] {
                    while floats.len() % 4 != 0 {
                        floats.push(0.0); // 16-byte aligned
                    }
                    float_at.push(floats.len());
                    floats.extend_from_slice(seg);
                }
                let d = PyDict::new_bound(py);
                d.set_item("group", g)?;
                d.set_item("positions", ch.iter().map(|&k| planned[k].pos).collect::<Vec<_>>())?;
                eligible.extend(ch.iter().map(|&k| planned[k].pos));
                d.set_item("int_at", int_at)?;
                d.set_item("int_len", int_len)?;
                d.set_item("float_at", float_at)?;
                d.set_item("idf_field", keys[g].1)?;
                d.set_item("facets", facets)?;
                d.set_item("from", ch.iter().map(|&k| planned[k].from).collect::<Vec<_>>())?;
                d.set_item("size", ch.iter().map(|&k| planned[k].size).collect::<Vec<_>>())?;
                let aggs = PyList::empty_bound(py);
                for &k in ch {
                    let l: Vec<(String, String, i64)> =
                        planned[k].aggs.iter().map(|a| (a.name.clone(), a.field.clone(), a.size)).collect();
                    aggs.append(l)?;
                }
                d.set_item("aggs", aggs)?;
                d.set_item("n_terms", ch.iter().map(|&k| planned[k].terms.len()).collect::<Vec<_>>())?;
                chunks.append(d)?;
            }
        }
        let out = PyDict::new_bound(py);
        out.set_item("fallback", fallback)?;
        let ib: Vec<u8> = ints.iter().flat_map(|x| x.to_le_bytes()).collect();
        let fb: Vec<u8> = floats.iter().flat_map(|x| x.to_le_bytes()).collect();
        out.set_item("ints", PyBytes::new_bound(py, &ib))?;
        out.set_item("floats", PyBytes::new_bound(py, &fb))?;
        out.set_item("eligible", &eligible)?;
        out.set_item("chunks", chunks)?;
        out.set_item("n_text_fields", self.text_fields.len())?;
        Ok(out)
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<HostIndex>()?;
    Ok(())
}
