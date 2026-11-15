//! R3a: engine/analysis.py's `s3_english` analyser in Rust, token for token. The analyser oracle
//! (engine/tests/test_hostpath_analysis.py) holds it to the Python one on every catalogue and query text.
//!
//! The Python analyser:
//! 1. lowercases the text (`str.lower`);
//! 2. finds the words with `[a-z0-9]+(?:'[a-z0-9]+)?`;
//! 3. strips a trailing `'s`, then removes every `'`;
//! 4. drops Lucene's English stopwords;
//! 5. stems with snowballstemmer's `porter` (porter.sbl, Snowball 3.1.1).
//!
//! Lowercasing: only ASCII letters and digits can ever be in a word. A character lowercases per char, as `str.lower`
//! does outside Greek final sigma (which yields no ASCII). Its lowercase contributes its ASCII `[a-z0-9']` characters
//! (U+0130 gives "i" then U+0307, a separator; the Kelvin sign gives "k"), and anything else is a separator.
//!
//! The stemmer is a transcription of the generated porter_stemmer.py: the same regions (p1, p2), the same steps in the
//! same order, and the same longest-suffix choice of each `among`, with no fallback to a shorter suffix when a region
//! test fails.

const STOP: [&str; 33] = ["a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it",
    "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these", "they", "this", "to",
    "was", "will", "with"];

fn is_v(c: u8) -> bool {
    matches!(c, b'a' | b'e' | b'i' | b'o' | b'u' | b'y')
}

fn is_v_wxy(c: u8) -> bool {
    is_v(c) || matches!(c, b'w' | b'x' | b'Y')
}

/// The lowercased text as ASCII word characters (`[a-z0-9']`), with 0 for every separator.
fn lowered(text: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(text.len());
    for ch in text.chars() {
        if ch.is_ascii() {
            let c = (ch as u8).to_ascii_lowercase();
            out.push(if c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'\'' { c } else { 0 });
        } else {
            for l in ch.to_lowercase() {
                let c = if l.is_ascii() { l as u8 } else { 0 };
                out.push(if c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'\'' { c } else { 0 });
            }
        }
    }
    out
}

/// `re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())`
pub fn words(text: &str) -> Vec<String> {
    let b = lowered(text);
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
        out.push(String::from_utf8(b[s..i].to_vec()).unwrap());
    }
    out
}

/// engine/analysis.py's `_token`: the token of one regex word, or None.
pub fn token(w: &str) -> Option<String> {
    let x = if w.ends_with("'s") { &w[..w.len() - 2] } else { w };
    let x: String = x.chars().filter(|&c| c != '\'').collect();
    if x.is_empty() || STOP.contains(&x.as_str()) {
        return None;
    }
    Some(stem(&x))
}

/// engine/analysis.py's `analyze` for `s3_english`.
pub fn analyze(text: &str) -> Vec<String> {
    words(text).iter().filter_map(|w| token(w)).collect()
}

struct S {
    w: Vec<u8>,
    p1: usize,
    p2: usize,
}

impl S {
    fn ends(&self, suf: &str) -> bool {
        self.w.ends_with(suf.as_bytes())
    }
    /// The longest of `list` the word ends with.
    fn longest<'a>(&self, list: &[&'a str]) -> Option<&'a str> {
        list.iter().filter(|s| self.ends(s)).max_by_key(|s| s.len()).copied()
    }
    /// Is there a vowel in w[..at]? (go_out_grouping_b(g_v) from `at`)
    fn vowel_before(&self, at: usize) -> bool {
        self.w[..at].iter().any(|&c| is_v(c))
    }
    /// shortv at cursor `at`: w[at-1] not in v_WXY, w[at-2] in v, w[at-3] not in v.
    fn shortv(&self, at: usize) -> bool {
        at >= 3 && !is_v_wxy(self.w[at - 1]) && is_v(self.w[at - 2]) && !is_v(self.w[at - 3])
    }
    fn replace_suffix(&mut self, n: usize, with: &str) {
        let l = self.w.len();
        self.w.truncate(l - n);
        self.w.extend_from_slice(with.as_bytes());
    }

    fn step_1a(&mut self) {
        match self.longest(&["s", "ies", "sses", "ss"]) {
            Some("sses") => self.replace_suffix(4, "ss"),
            Some("ies") => self.replace_suffix(3, "i"),
            Some("ss") => {}
            Some("s") => self.replace_suffix(1, ""),
            _ => {}
        }
    }

    fn step_1b(&mut self) {
        let suf = match self.longest(&["ed", "eed", "ing"]) {
            Some(s) => s,
            None => return,
        };
        let start = self.w.len() - suf.len();
        if suf == "eed" {
            if self.p1 <= start {
                self.replace_suffix(3, "ee");
            }
            return;
        }
        if !self.vowel_before(start) {
            return;
        }
        self.w.truncate(start);
        let a1 = ["", "bb", "dd", "ff", "gg", "bl", "mm", "nn", "pp", "rr", "at", "tt", "iz"];
        match self.longest(&a1) {
            Some("bl") | Some("at") | Some("iz") => self.w.push(b'e'),
            Some("") | None => {
                let end = self.w.len();
                if end == self.p1 && self.shortv(end) {
                    self.w.push(b'e');
                }
            }
            Some(_) => {
                self.w.pop();
            }
        }
    }

    fn step_1c(&mut self) {
        let n = self.w.len();
        if n == 0 || !(self.w[n - 1] == b'y' || self.w[n - 1] == b'Y') {
            return;
        }
        if !self.vowel_before(n - 1) {
            return;
        }
        self.w[n - 1] = b'i';
    }

    fn step_2(&mut self) {
        let table: [(&str, &str); 20] = [("anci", "ance"), ("enci", "ence"), ("abli", "able"), ("eli", "e"),
            ("alli", "al"), ("ousli", "ous"), ("entli", "ent"), ("aliti", "al"), ("biliti", "ble"), ("iviti", "ive"),
            ("tional", "tion"), ("ational", "ate"), ("alism", "al"), ("ation", "ate"), ("ization", "ize"),
            ("izer", "ize"), ("ator", "ate"), ("iveness", "ive"), ("fulness", "ful"), ("ousness", "ous")];
        self.table_step(&table, true);
    }

    fn step_3(&mut self) {
        let table: [(&str, &str); 7] = [("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
            ("ical", "ic"), ("ful", ""), ("ness", "")];
        self.table_step(&table, true);
    }

    fn table_step(&mut self, table: &[(&str, &str)], r1: bool) {
        let keys: Vec<&str> = table.iter().map(|(k, _)| *k).collect();
        if let Some(suf) = self.longest(&keys) {
            let start = self.w.len() - suf.len();
            let ok = if r1 { self.p1 <= start } else { self.p2 <= start };
            if ok {
                let with = table.iter().find(|(k, _)| *k == suf).unwrap().1;
                self.replace_suffix(suf.len(), with);
            }
        }
    }

    fn step_4(&mut self) {
        let list = ["ic", "ance", "ence", "able", "ible", "ate", "ive", "ize", "iti", "al", "ism", "ion", "er", "ous",
            "ant", "ent", "ment", "ement", "ou"];
        let suf = match self.longest(&list) {
            Some(s) => s,
            None => return,
        };
        let start = self.w.len() - suf.len();
        if self.p2 > start {
            return;
        }
        if suf == "ion" {
            if start >= 1 && (self.w[start - 1] == b's' || self.w[start - 1] == b't') {
                self.w.truncate(start);
            }
            return;
        }
        self.w.truncate(start);
    }

    fn step_5a(&mut self) {
        let n = self.w.len();
        if n == 0 || self.w[n - 1] != b'e' {
            return;
        }
        let at = n - 1;
        if self.p2 <= at || (self.p1 <= at && !self.shortv(at)) {
            self.w.truncate(at);
        }
    }

    fn step_5b(&mut self) {
        let n = self.w.len();
        if n == 0 || self.w[n - 1] != b'l' {
            return;
        }
        let at = n - 1;
        if self.p2 <= at && at >= 1 && self.w[at - 1] == b'l' {
            self.w.truncate(at);
        }
    }
}

/// snowballstemmer's `porter` on a lowercase ASCII word.
pub fn stem(word: &str) -> String {
    let mut w: Vec<u8> = word.as_bytes().to_vec();
    let mut y_found = false;
    if !w.is_empty() && w[0] == b'y' {
        w[0] = b'Y';
        y_found = true;
    }
    let mut i = 0;
    while i + 1 < w.len() {
        if is_v(w[i]) && w[i + 1] == b'y' {
            w[i + 1] = b'Y';
            y_found = true;
        }
        i += 1;
    }
    let n = w.len();
    let (mut p1, mut p2) = (n, n);
    // gopast v, gopast non-v: p1; again: p2
    let region = |from: usize| -> Option<usize> {
        let mut c = from;
        while c < n && !is_v(w[c]) {
            c += 1;
        }
        if c >= n {
            return None;
        }
        c += 1;
        while c < n && is_v(w[c]) {
            c += 1;
        }
        if c >= n {
            return None;
        }
        Some(c + 1)
    };
    if let Some(a) = region(0) {
        p1 = a;
        if let Some(b) = region(a) {
            p2 = b;
        }
    }
    let mut s = S { w, p1, p2 };
    s.step_1a();
    s.step_1b();
    s.step_1c();
    s.step_2();
    s.step_3();
    s.step_4();
    s.step_5a();
    s.step_5b();
    if y_found {
        for c in s.w.iter_mut() {
            if *c == b'Y' {
                *c = b'y';
            }
        }
    }
    String::from_utf8(s.w).unwrap()
}
