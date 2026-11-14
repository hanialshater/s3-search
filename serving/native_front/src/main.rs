//! S6-22s: a native HTTP front for the GPU processes. Development only; the Python front stays the default.
//!
//!     s3-native-front --port 8080 --gpu-sockets /state/gpu-0.sock,/state/gpu-1.sock [--threads N] [--worker 0]
//!
//! Each search is sent to the GPU process with the fewest requests in flight (or the one `X-S3-Route-Gpu` names), over the binary protocol
//! (`serving/binproto.py`). The hits come back as columns, and this process writes them as JSON (`encode.rs`), the same
//! bytes as the Python front. Every other route goes to GPU process 0 as a call and is answered with what it returns.
//! TCP_NODELAY is set on every accepted connection.

mod encode;

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, UnixStream};
use tokio::sync::{mpsc, oneshot};

struct Gpu {
    tx: mpsc::UnboundedSender<Vec<u8>>,
    pending: Mutex<HashMap<u64, oneshot::Sender<Vec<u8>>>>,
    inflight: AtomicUsize,
}

struct Front {
    gpus: Vec<Arc<Gpu>>,
    ids: AtomicU64,
    worker: String,
    threads: usize,
}

async fn connect(path: &str) -> std::io::Result<Arc<Gpu>> {
    let mut s = UnixStream::connect(path).await?;
    // S3_SERVING_BINPROTO=2 (opt-in, S6-22s): open S3B2, so the GPU process sends kind 3 replies and each reply alone
    let magic: &[u8; 4] = if std::env::var("S3_SERVING_BINPROTO").map(|v| v.trim() == "2").unwrap_or(false) {
        b"S3B2"
    } else {
        b"S3B1"
    };
    s.write_all(magic).await?;
    let (mut rd, mut wr) = s.into_split();
    let (tx, mut rx) = mpsc::unbounded_channel::<Vec<u8>>();
    let gpu = Arc::new(Gpu { tx, pending: Mutex::new(HashMap::new()), inflight: AtomicUsize::new(0) });
    tokio::spawn(async move {
        while let Some(frame) = rx.recv().await {
            if wr.write_all(&frame).await.is_err() {
                break;
            }
        }
    });
    let g = gpu.clone();
    tokio::spawn(async move {
        loop {
            let mut len = [0u8; 4];
            if rd.read_exact(&mut len).await.is_err() {
                break;
            }
            let mut payload = vec![0u8; u32::from_le_bytes(len) as usize];
            if rd.read_exact(&mut payload).await.is_err() {
                break;
            }
            if payload.len() < 9 {
                continue;
            }
            let id = u64::from_le_bytes(payload[1..9].try_into().unwrap());
            if let Some(tx) = g.pending.lock().unwrap().remove(&id) {
                let _ = tx.send(payload);
            }
        }
        g.pending.lock().unwrap().clear(); // the GPU process went away: waiting requests fail
    });
    Ok(gpu)
}

impl Front {
    fn pick(&self) -> (usize, Arc<Gpu>) {
        let (i, g) = self.gpus.iter().enumerate().min_by_key(|(_, g)| g.inflight.load(Ordering::Relaxed)).unwrap();
        (i, g.clone())
    }

    /// `X-S3-Route-Gpu: j` (for diagnosis, as the Python front): GPU process j when it exists, else the least loaded.
    fn route(&self, j: Option<usize>) -> (usize, Arc<Gpu>) {
        match j {
            Some(j) if j < self.gpus.len() => (j, self.gpus[j].clone()),
            _ => self.pick(),
        }
    }

    async fn send(&self, gpu: &Arc<Gpu>, frame: Vec<u8>, id: u64) -> Option<Vec<u8>> {
        let (tx, rx) = oneshot::channel();
        gpu.pending.lock().unwrap().insert(id, tx);
        gpu.inflight.fetch_add(1, Ordering::Relaxed);
        let ok = gpu.tx.send(frame).is_ok();
        let out = if ok {
            tokio::time::timeout(std::time::Duration::from_secs(30), rx).await.ok().and_then(|r| r.ok())
        } else {
            None
        };
        gpu.inflight.fetch_sub(1, Ordering::Relaxed);
        if out.is_none() {
            gpu.pending.lock().unwrap().remove(&id);
        }
        out
    }

    fn search_frame(id: u64, name: &[u8], body: &[u8]) -> Vec<u8> {
        let mut p = Vec::with_capacity(4 + 9 + 2 + name.len() + 4 + body.len());
        let len = (9 + 2 + name.len() + 4 + body.len()) as u32;
        p.extend_from_slice(&len.to_le_bytes());
        p.push(1);
        p.extend_from_slice(&id.to_le_bytes());
        p.extend_from_slice(&(name.len() as u16).to_le_bytes());
        p.extend_from_slice(name);
        p.extend_from_slice(&(body.len() as u32).to_le_bytes());
        p.extend_from_slice(body);
        p
    }

    fn call_frame(id: u64, method: &[u8], path: &[u8], body: &[u8]) -> Vec<u8> {
        let mut p = Vec::new();
        let len = (9 + 2 + method.len() + 2 + path.len() + 4 + body.len()) as u32;
        p.extend_from_slice(&len.to_le_bytes());
        p.push(2);
        p.extend_from_slice(&id.to_le_bytes());
        p.extend_from_slice(&(method.len() as u16).to_le_bytes());
        p.extend_from_slice(method);
        p.extend_from_slice(&(path.len() as u16).to_le_bytes());
        p.extend_from_slice(path);
        p.extend_from_slice(&(body.len() as u32).to_le_bytes());
        p.extend_from_slice(body);
        p
    }
}

/// A reply frame -> (status, body, meta JSON, content type)
fn render(payload: &[u8]) -> Option<(u16, Vec<u8>, serde_json::Value)> {
    let kind = payload[0];
    let mut o = 9;
    let mlen = u32::from_le_bytes(payload.get(o..o + 4)?.try_into().ok()?) as usize;
    o += 4;
    let meta: serde_json::Value = serde_json::from_slice(payload.get(o..o + mlen)?).ok()?;
    o += mlen;
    if kind == 2 {
        let st = u16::from_le_bytes(payload.get(o..o + 2)?.try_into().ok()?);
        o += 2;
        let n = u32::from_le_bytes(payload.get(o..o + 4)?.try_into().ok()?) as usize;
        o += 4;
        return Some((st, payload.get(o..o + n)?.to_vec(), meta));
    }
    let blk = encode::parse_block(&payload[o..])?;
    let pre = meta.get("pre")?.as_str()?.as_bytes();
    let post = meta.get("post")?.as_str()?.as_bytes();
    let mut out = Vec::with_capacity(pre.len() + post.len() + blk.h * 64 + 2);
    out.extend_from_slice(pre);
    if kind == 3 {                                              // S3B2: unchecked ids and floats, written as Python does
        encode::encode_hits3(&blk, &mut out)?;
    } else {
        encode::encode_hits(&blk, &mut out);
    }
    out.extend_from_slice(post);
    Some((200, out, meta))
}

/// GET /readyz: every GPU process asked, as the Python front does (serving/front.py ``_readyz``): ready only when all
/// are; the same keys in the same order (plat-m21: asking GPU process 0 alone let a routed warm-up reach a process
/// still loading its index).
async fn readyz(front: &Arc<Front>) -> Response<Full<Bytes>> {
    let mut ready = 0usize;
    let mut indexes = String::from("[]");
    for g in &front.gpus {
        let id = front.ids.fetch_add(1, Ordering::Relaxed);
        let frame = Front::call_frame(id, b"GET", b"/readyz", b"");
        if let Some(reply) = front.send(g, frame, id).await {
            if let Some((st, body, _)) = render(&reply) {
                if st == 200 {
                    ready += 1;
                }
                if indexes == "[]" {
                    if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&body) {
                        if let Some(ix) = v.get("indexes") {
                            indexes = ix.to_string();
                        }
                    }
                }
            }
        }
    }
    let ok = ready == front.gpus.len();
    let worker = front.worker.parse::<i64>().map(|w| w.to_string()).unwrap_or_else(|_| format!("{:?}", front.worker));
    let body = format!("{{\"ready\":{},\"draining\":false,\"indexes\":{},\"fronts\":{},\"gpu_processes\":{},\"gpu_ready\":{},\"worker\":{}}}",
                       ok, indexes, front.threads, front.gpus.len(), ready, worker);
    Response::builder().status(if ok { 200 } else { 503 }).header("Content-Type", "application/json")
        .body(Full::new(Bytes::from(body))).unwrap()
}

fn json(status: u16, body: &'static str) -> Response<Full<Bytes>> {
    Response::builder().status(status).header("Content-Type", "application/json")
        .body(Full::new(Bytes::from_static(body.as_bytes()))).unwrap()
}

async fn handle(front: Arc<Front>, req: Request<Incoming>) -> Result<Response<Full<Bytes>>, Infallible> {
    let t0 = Instant::now();
    let method = req.method().as_str().to_string();
    let path = req.uri().path_and_query().map(|p| p.as_str().to_string()).unwrap_or_else(|| "/".into());
    let bare = req.uri().path().to_string();
    let route: Option<usize> = req.headers().get("x-s3-route-gpu").and_then(|v| v.to_str().ok())
        .and_then(|v| v.trim().parse().ok());
    let body = match req.into_body().collect().await {
        Ok(b) => b.to_bytes(),
        Err(_) => return Ok(json(400, "{\"error\":{\"type\":\"invalid_request\",\"path\":\"\",\"reason\":\"bad body\"}}")),
    };
    let t_read = t0.elapsed();                                  // S6-22s timers: the request body read
    let rid = format!("{:016x}", front.ids.fetch_add(1, Ordering::Relaxed) ^ 0x9e3779b97f4a7c15);
    if bare == "/healthz" && method == "GET" {
        return Ok(json(200, "{\"status\":\"ok\"}"));
    }
    if bare == "/readyz" && method == "GET" {
        return Ok(readyz(&front).await);
    }
    let id = front.ids.fetch_add(1, Ordering::Relaxed);
    let parts: Vec<&str> = bare.trim_start_matches('/').split('/').collect();
    let is_search = method == "POST" && parts.len() == 3 && parts[0] == "indexes" && parts[2] == "_search";
    let (gi, gpu) = if is_search { front.route(route) } else { (0, front.gpus[0].clone()) };
    let frame = if is_search {
        Front::search_frame(id, parts[1].as_bytes(), &body)
    } else {
        Front::call_frame(id, method.as_bytes(), path.as_bytes(), &body)
    };
    let t_sent = t0.elapsed();
    let reply = match front.send(&gpu, frame, id).await {
        Some(r) => r,
        None => return Ok(json(503, "{\"error\":{\"type\":\"not_ready\",\"path\":\"\",\"reason\":\"the GPU process did not answer\"}}")),
    };
    let t_reply = t0.elapsed();                                 // the round trip to the GPU process ends
    let (status, out, meta) = match render(&reply) {
        Some(x) => x,
        None => return Ok(json(500, "{\"error\":{\"type\":\"internal_error\",\"path\":\"\",\"reason\":\"bad reply frame\"}}")),
    };
    let t_render = t0.elapsed();
    let ctype = meta.get("type").and_then(|v| v.as_str()).unwrap_or("application/json").to_string();
    let mut rb = Response::builder().status(StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR))
        .header("Content-Type", ctype).header("X-Request-Id", rid).header("X-S3-Worker", front.worker.as_str());
    if is_search && status == 200 {
        let q = meta.get("queue_ms").and_then(|v| v.as_f64()).unwrap_or(0.0);
        rb = rb.header("X-S3-Cache", "miss").header("X-S3-Queue-Ms", format!("{:.3}", q))
            .header("X-S3-Serve-Ms", format!("{:.3}", t0.elapsed().as_secs_f64() * 1000.0))
            .header("X-S3-Gpu-Process", gi.to_string())
            .header("X-S3-Index-Version", meta.get("version").and_then(|v| v.as_str()).unwrap_or("").to_string())
            .header("X-S3-Batch-Size", meta.get("batch").map(|v| v.to_string()).unwrap_or_else(|| "1".into()));
        if let Some(b) = meta.get("batch_ms").and_then(|v| v.as_f64()) {   // S6-22s: the GPU process's batch time
            rb = rb.header("X-S3-Batch-Ms", format!("{:.3}", b));
        }
        // S6-22s timers: where the front's share goes. Read: arrival to body read; Build: frame built and queued;
        // Rtt: frame sent to reply received; Render: the hits written as JSON. From the GPU process: Parse (its
        // reader's parse of the frame) and Post (a reply's wait from its batch's end to the writer's encode).
        let ms = |d: std::time::Duration| format!("{:.3}", d.as_secs_f64() * 1000.0);
        rb = rb.header("X-S3-Front-Read-Ms", ms(t_read)).header("X-S3-Front-Build-Ms", ms(t_sent - t_read))
            .header("X-S3-Front-Rtt-Ms", ms(t_reply - t_sent)).header("X-S3-Front-Render-Ms", ms(t_render - t_reply));
        for (k, h) in [("parse_ms", "X-S3-Gpu-Parse-Ms"), ("post_ms", "X-S3-Gpu-Post-Ms")] {
            if let Some(v) = meta.get(k).and_then(|v| v.as_f64()) {
                rb = rb.header(h, format!("{:.3}", v));
            }
        }
    }
    Ok(rb.body(Full::new(Bytes::from(out))).unwrap())
}

fn arg(name: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == name).and_then(|i| a.get(i + 1).cloned())
}

fn main() {
    let port: u16 = arg("--port").and_then(|p| p.parse().ok()).unwrap_or(8080);
    let host = arg("--host").unwrap_or_else(|| "0.0.0.0".into());
    let sockets: Vec<String> = arg("--gpu-sockets").expect("--gpu-sockets").split(',').map(String::from).collect();
    let threads: usize = arg("--threads").and_then(|p| p.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4));
    let worker = arg("--worker").unwrap_or_else(|| "0".into());
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(threads).enable_all().build().unwrap();
    rt.block_on(async move {
        let mut gpus = Vec::new();
        for s in &sockets {
            loop {                                              // the GPU processes may still be binding their sockets
                match connect(s).await {
                    Ok(g) => {
                        gpus.push(g);
                        break;
                    }
                    Err(_) => tokio::time::sleep(std::time::Duration::from_millis(500)).await,
                }
            }
        }
        let front = Arc::new(Front { gpus, ids: AtomicU64::new(1), worker, threads });
        let listener = TcpListener::bind((host.as_str(), port)).await.expect("bind");
        eprintln!("s3-native-front on {}:{} ({} threads, {} GPU processes)", host, port, threads, sockets.len());
        loop {
            let (stream, _) = match listener.accept().await {
                Ok(x) => x,
                Err(_) => continue,
            };
            let _ = stream.set_nodelay(true);
            let f = front.clone();
            tokio::spawn(async move {
                let svc = service_fn(move |req| handle(f.clone(), req));
                let _ = http1::Builder::new().keep_alive(true).serve_connection(TokioIo::new(stream), svc).await;
            });
        }
    });
}
