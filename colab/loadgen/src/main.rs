//! S6-22s: fair_bench's open-loop load client, in Rust, equivalent to its aiohttp client (colab/fair_bench.py
//! `_client_proc`), so the client is not the limit above 2,400 QPS. Development only.
//!
//!     s3-loadgen --url URL --bodies FILE --rate R --duration D --t0 EPOCH --threads N --out FILE --records FILE
//!                [--sample-every K] [--sample-bodies a,b] [--samples-dir DIR] [--timeout S] [--accept-encoding E]
//!
//! As the aiohttp client: request i (0 .. rate*duration) carries body i % len(bodies) and is due at t0 + i/rate (wall
//! clock, epoch seconds); it is sent when due without waiting for any reply (open loop); its latency is from its due
//! time to the end of reading its whole body (so a late send counts); its send lag is send time minus due time. A
//! non-200 status or a connection error is an error, counted by kind, without a latency; no answer within the timeout
//! (10 s, from the send) is a timeout. A sampled request (i % K == 0, or its body index in --sample-bodies) has its
//! raw body written to DIR/{int(rate)}-{i}.json and its X-S3-Gpu-Process, X-S3-Index-Version and X-S3-Batch-Size
//! kept; fair_bench computes the hit count and the digest from those files with its own result_digest.
//! --out is the JSON fair_bench's load() merges (lat, errors, timeouts, hits, lag, sent, samples, why); --records is
//! one CSV line per request: i, latency ms, lag ms, status (0 timeout, 1 error), gpu, batch size, queue ms, serve ms,
//! batch ms (the server's X-S3-* timing headers; empty when absent).

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::Request;
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

fn arg(name: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == name).and_then(|i| a.get(i + 1).cloned())
}

fn need(name: &str) -> String {
    arg(name).unwrap_or_else(|| panic!("{} is required", name))
}

fn read_bodies(path: &str) -> Vec<Bytes> {
    let d = std::fs::read(path).expect("bodies file");
    let n = u32::from_le_bytes(d[0..4].try_into().unwrap()) as usize;
    let mut out = Vec::with_capacity(n);
    let mut o = 4;
    for _ in 0..n {
        let len = u32::from_le_bytes(d[o..o + 4].try_into().unwrap()) as usize;
        o += 4;
        out.push(Bytes::copy_from_slice(&d[o..o + len]));
        o += len;
    }
    out
}

/// Wall clock as epoch seconds, kept monotonic: the epoch at start plus an Instant's elapsed time.
#[derive(Clone, Copy)]
struct Clock {
    epoch0: f64,
    inst0: Instant,
}

impl Clock {
    fn new() -> Clock {
        let epoch0 = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64();
        Clock { epoch0, inst0: Instant::now() }
    }
    fn now(&self) -> f64 {
        self.epoch0 + self.inst0.elapsed().as_secs_f64()
    }
}

struct Rec {
    i: u64,
    lat_ms: Option<f64>,
    lag_ms: f64,
    status: u16, // 0 timeout, 1 error, else the HTTP status
    why: Option<String>,
    gpu: String,
    version: String,
    batch: String,
    queue: String,
    serve: String,
    batch_ms: String,
    sampled: bool,
}

fn header(h: &hyper::HeaderMap, k: &str) -> String {
    h.get(k).and_then(|v| v.to_str().ok()).unwrap_or("").to_string()
}

fn main() {
    let url: hyper::Uri = need("--url").parse().expect("--url");
    let bodies = Arc::new(read_bodies(&need("--bodies")));
    let rate: f64 = need("--rate").parse().unwrap();
    let duration: f64 = need("--duration").parse().unwrap();
    let t0: f64 = need("--t0").parse().unwrap();
    let threads: usize = arg("--threads").and_then(|x| x.parse().ok()).unwrap_or(2);
    let out_path = need("--out");
    let rec_path = need("--records");
    let every: u64 = arg("--sample-every").and_then(|x| x.parse().ok()).unwrap_or(97);
    let sample_bodies: Arc<HashSet<usize>> = Arc::new(
        arg("--sample-bodies").unwrap_or_default().split(',').filter_map(|x| x.trim().parse().ok()).collect());
    let samples_dir = Arc::new(arg("--samples-dir").unwrap_or_else(|| "/content/fair-samples".into()));
    let timeout = Duration::from_secs_f64(arg("--timeout").and_then(|x| x.parse().ok()).unwrap_or(10.0));
    let accept_encoding = Arc::new(arg("--accept-encoding").unwrap_or_else(|| "gzip, deflate".into()));
    let n_total = (rate * duration) as u64;
    let rate_tag = rate as u64;

    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(threads).enable_all().build().unwrap();
    let recs: Vec<Rec> = rt.block_on(async move {
        let mut conn = HttpConnector::new();
        conn.set_nodelay(true);                                 // as aiohttp
        let client: Client<HttpConnector, Full<Bytes>> = Client::builder(TokioExecutor::new()).build(conn);
        let clock = Clock::new();
        let mut handles = Vec::with_capacity(n_total as usize);
        for i in 0..n_total {
            let due = t0 + i as f64 / rate;
            let wait = due - clock.now();
            if wait > 0.0 {
                tokio::time::sleep(Duration::from_secs_f64(wait)).await;
            }
            let sent = clock.now();
            let lag_ms = (sent - due) * 1e3;
            let bi = (i as usize) % bodies.len();
            let sampled = i % every == 0 || sample_bodies.contains(&bi);
            let body = bodies[bi].clone();
            let client = client.clone();
            let url = url.clone();
            let dir = samples_dir.clone();
            let enc = accept_encoding.clone();
            handles.push(tokio::spawn(async move {
                // aiohttp's request headers but for the client name (the steward's condition): Host and
                // Content-Length are set by hyper as by aiohttp; Accept is aiohttp's default and Accept-Encoding the
                // value the installed aiohttp sends (fair_bench passes it: gzip, deflate, br with brotli installed).
                let req = Request::post(url).header("Content-Type", "application/json").header("Accept", "*/*")
                    .header("Accept-Encoding", enc.as_str()).header("User-Agent", "s3-loadgen/0.1")
                    .body(Full::new(body)).unwrap();
                let fut = async {
                    let resp = client.request(req).await?;
                    let status = resp.status().as_u16();
                    let h = resp.headers().clone();
                    let raw = resp.into_body().collect().await?.to_bytes();
                    Ok::<_, Box<dyn std::error::Error + Send + Sync>>((status, h, raw))
                };
                let mut r = Rec { i, lat_ms: None, lag_ms, status: 1, why: None, gpu: String::new(),
                                  version: String::new(), batch: String::new(), queue: String::new(),
                                  serve: String::new(), batch_ms: String::new(), sampled: false };
                match tokio::time::timeout(timeout, fut).await {
                    Err(_) => r.status = 0,
                    Ok(Err(e)) => {
                        let kind = if e.downcast_ref::<hyper_util::client::legacy::Error>().map(|x| x.is_connect())
                            .unwrap_or(false) { "ClientConnectorError" } else { "ClientError" };
                        r.why = Some(kind.into());
                    }
                    Ok(Ok((status, h, raw))) => {
                        r.status = status;
                        if status == 200 {
                            r.lat_ms = Some((clock.now() - due) * 1e3);
                            r.gpu = header(&h, "x-s3-gpu-process");
                            r.version = header(&h, "x-s3-index-version");
                            r.batch = header(&h, "x-s3-batch-size");
                            r.queue = header(&h, "x-s3-queue-ms");
                            r.serve = header(&h, "x-s3-serve-ms");
                            r.batch_ms = header(&h, "x-s3-batch-ms");
                            if sampled {
                                let p = format!("{}/{}-{}.json", dir, rate_tag, i);
                                if std::fs::write(&p, &raw).is_ok() {
                                    r.sampled = true;
                                }
                            }
                        } else {
                            r.why = Some(status.to_string());
                        }
                    }
                }
                r
            }));
        }
        let mut recs = Vec::with_capacity(handles.len());
        for (i, h) in handles.into_iter().enumerate() {
            match h.await {
                Ok(r) => recs.push(r),
                // A request task that panicked is an error, never dropped (the steward, on 82c0ebd): sent stays
                // int(rate * duration).
                Err(_) => recs.push(Rec { i: i as u64, lat_ms: None, lag_ms: f64::NAN, status: 1,
                                          why: Some("JoinError".into()), gpu: String::new(), version: String::new(),
                                          batch: String::new(), queue: String::new(), serve: String::new(),
                                          batch_ms: String::new(), sampled: false }),
            }
        }
        recs
    });

    let (mut lat, mut lag, mut samples) = (Vec::new(), Vec::new(), Vec::new());
    let (mut errors, mut timeouts) = (0u64, 0u64);
    let mut why: HashMap<String, u64> = HashMap::new();
    let mut csv = std::io::BufWriter::new(std::fs::File::create(&rec_path).expect("--records"));
    writeln!(csv, "i,lat_ms,lag_ms,status,gpu,batch,queue_ms,serve_ms,batch_ms").unwrap();
    for r in &recs {
        if !r.lag_ms.is_nan() {
            lag.push(r.lag_ms);
        }
        match r.status {
            0 => timeouts += 1,
            200 => lat.push(r.lat_ms.unwrap()),
            _ => {
                errors += 1;
                *why.entry(r.why.clone().unwrap_or_else(|| "error".into())).or_insert(0) += 1;
            }
        }
        if r.sampled {
            samples.push(serde_json::json!([r.i, null, null, r.gpu, r.version, r.batch]));
        }
        writeln!(csv, "{},{},{:.3},{},{},{},{},{},{}", r.i, r.lat_ms.map(|x| format!("{:.3}", x)).unwrap_or_default(),
                 r.lag_ms, r.status, r.gpu, r.batch, r.queue, r.serve, r.batch_ms).unwrap();
    }
    csv.flush().unwrap();
    let out = serde_json::json!({"lat": lat, "errors": errors, "timeouts": timeouts, "hits": [], "lag": lag,
                                 "sent": recs.len(), "samples": samples, "why": why});
    std::fs::write(&out_path, out.to_string()).expect("--out");
}
