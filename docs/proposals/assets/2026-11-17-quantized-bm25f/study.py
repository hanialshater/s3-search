"""The binary-first / RaBitQ-style BM25F feasibility study (CPU, development only, never a score).

Per query (multi_match over the tuned BM25F fields, size 2,000), from the engine's own postings:
- the exact score of every matched doc, with the engine's float32 operations and its float64 posting-order sum
  (held to the engine's answers on 50 queries: engine_check.pkl);
- per bit width b (0 = presence only, then 1, 2, 4, 8): per posting a code q = min(floor(c / D_t), 2^b - 1) of its
  contribution c = idf * g(ptf), D_t = cmax_t / 2^b, the reconstruction (q + 1/2) D_t, and the error half-step D_t/2.
  All in integer units u (D_t / 2 rounded up to a whole number of u), so Shat and E are integer sums. Then:
  L = Shat - E, U = Shat + E (with a relative float margin), theta = the 2,000th largest L, candidates {U >= theta}.
- what stage 1 and 2 would move, and the 1,024-doc tiles the postings and the candidates touch (as built, and after two
  cheap reorderings);
- the approximate-only ranking (top 2,000 by Shat, ties by doc slot): overlap with the exact top 2,000 and top 100.
On the fit sample (train-split judgments, which members may read): nDCG@10, recall@100, recall@1000 of the exact and
of the approximate rankings, with referee/metrics.py.
"""
import json
import pickle
import sys
import time

import numpy as np

D = sys.argv[1]
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[4]))
from referee import metrics  # noqa: E402

z = np.load(D + "/index.npz")
term_ptr, doc_ids, tf, rel_len, idf_all, live = (z["term_ptr"], z["doc_ids"], z["tf"], z["rel_len"], z["idf"],
                                                 z["live"])
bb, k1 = z["bb"], np.float32(z["k1"])
P = pickle.load(open(D + "/plans.pkl", "rb"))
ids = np.asarray(P["ids"], dtype=object)
N = rel_len.shape[0]
TILE = 1024
BITS = (0, 1, 2, 4, 8)
K = 2000
REL = 1e-6                 # float margin: |exact f32 score - sum of the f32 contributions| <= ~2^-23 relative; 1e-6 covers it
norm_all = None


def minhash_order():
    """A cheap locality reordering: docs sorted by two min-hashes of their term sets (similar docs, near slots)."""
    lens = np.diff(term_ptr)
    term_of = np.repeat(np.arange(len(lens), dtype=np.int64), lens)
    rank = np.empty(N, np.int64)
    keys = []
    for seed in (0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F):
        h = (np.arange(len(lens), dtype=np.uint64) * np.uint64(seed)) >> np.uint64(20)
        mh = np.full(N, np.iinfo(np.uint64).max, np.uint64)
        np.minimum.at(mh, doc_ids, h[term_of])
        keys.append(mh)
        del h
    order = np.lexsort((keys[1], keys[0]))
    rank[order] = np.arange(N)
    return rank


RANK = None


def exact_parts(terms, w):
    """Per query term: (docs, c f32, present) with the engine's float32 operations (ops.lexical.bm25f)."""
    out = []
    for t in terms:
        a, e = int(term_ptr[t]), int(term_ptr[t + 1])
        docs = doc_ids[a:e]
        tfa = tf[a:e].astype(np.float32)
        norm = (np.float32(1.0) - bb) + bb * rel_len[docs]
        norm = np.where(norm > np.float32(1e-6), norm, np.float32(1e-6)).astype(np.float32)
        x = (w * tfa) / norm
        ptf = x[:, 0]
        for i in range(1, x.shape[1]):
            ptf = ptf + x[:, i]
        c = idf_all[t] * ptf * (k1 + np.float32(1.0)) / (k1 + ptf)
        out.append((docs, c.astype(np.float32), ptf > 0))
    return out


def run_query(p, want_ranked=False):
    terms, w, msm, boost = p["terms"], p["w"], p["msm"], p["boost"]
    if not terms or msm < 1:
        return None
    parts = exact_parts(terms, w)
    alld = np.concatenate([d for d, _, _ in parts])
    allc = np.concatenate([c for _, c, _ in parts]).astype(np.float64)
    allp = np.concatenate([pr for _, _, pr in parts])
    s64 = np.bincount(alld, weights=allc, minlength=N)          # float64, posting order: the engine's sum
    nm = np.bincount(alld[allp], minlength=N)
    match = (nm >= msm) & live
    score = (s64.astype(np.float32) * np.float32(boost))
    md = np.flatnonzero(match)
    total = len(md)
    if total == 0:
        return None
    ms = score[md]
    order = np.lexsort((md, -ms.astype(np.float64)))           # score desc, then slot asc
    top = md[order[:K]]
    theta = ms[order[min(K, total) - 1]]
    res = {"total": total, "postings": int(len(alld)), "n_terms": len(terms),
           "tiles_post": int(len(np.unique(alld // TILE))), "top": top if want_ranked else None,
           "tiles_post_mh": int(len(np.unique(RANK[alld] // TILE))) if RANK is not None else None,
           "bits": {}}
    # the bound check, presence (b = 0): U0 = (k1 + 1) * sum idf over present terms * boost
    B = np.bincount(alld[allp], weights=np.repeat([float(idf_all[t]) for t in terms], [len(d) for d, _, _ in parts])[allp],
                    minlength=N)
    U0 = (float(k1) + 1.0) * B[md] * boost
    res["bound_max_ratio"] = float(np.max(ms.astype(np.float64) / U0))
    for b in BITS:
        sh_parts, e_parts = [], []
        for t, (docs, c, pr) in zip(terms, parts):
            cmax_t = float(idf_all[t]) * (float(k1) + 1.0)          # the analytic bound: weight-independent
            if b > 0:
                cmax_t = max(float(c.max()) if len(c) else 0.0, 1e-30) * (1 + 1e-7)
            levels = 1 << b
            half = cmax_t / levels / 2.0
            q = np.minimum(np.floor(c.astype(np.float64) / (2 * half)), levels - 1) if b > 0 else np.zeros(len(c))
            sh_parts.append((2 * q + 1) * half * pr)
            e_parts.append(half * pr)
        sh = np.bincount(alld, weights=np.concatenate(sh_parts), minlength=N)[md] * boost
        ee = np.bincount(alld, weights=np.concatenate(e_parts), minlength=N)[md] * boost
        L, U = sh - ee, (sh + ee) * (1 + REL)
        th = np.sort(L)[::-1][min(K, total) - 1] if total >= 1 else 0.0
        aorder = np.lexsort((md, -sh))
        # the two-step threshold: the exact scores of the approximate top 2,000 (stage 2a), whose 2,000th is <= theta
        th2 = float(ms[aorder[:K]].astype(np.float64).min())
        n_c2 = int(((U >= max(th, th2) * (1 - REL))).sum())
        th_lo = th * (1 - REL)
        cand = U >= th_lo
        n_c = int(cand.sum())
        ok = bool(np.all(cand[order[:K]]))                         # the exact top 2,000 are all candidates
        valid = bool(np.all(ms.astype(np.float64) <= U) and np.all(ms.astype(np.float64) >= (L * (1 - REL) - 1e-30)))
        atop = md[aorder[:K]]
        rec2000 = len(np.intersect1d(atop, top)) / max(1, min(K, total))
        rec100 = len(np.intersect1d(md[aorder[:100]], top[:100])) / max(1, min(100, total))
        ties = int((cand & (L < th) & (U >= th_lo)).sum())         # candidates whose interval only straddles theta
        ct = np.unique(md[cand] // TILE)
        ct_mh = np.unique(RANK[md[cand]] // TILE) if RANK is not None else ct
        post_in_ct = int(np.isin(alld // TILE, ct).sum())
        post_in_ct_mh = int(np.isin(RANK[alld] // TILE, ct_mh).sum()) if RANK is not None else post_in_ct
        res["bits"][b] = {"cand": n_c, "cand2": n_c2, "post_in_cand_tiles": post_in_ct,
                          "post_in_cand_tiles_mh": post_in_ct_mh, "cand_tiles_mh": int(len(ct_mh)), "covers_top": ok, "bounds_valid": valid, "rec2000": rec2000, "rec100": rec100,
                          "straddle": ties, "cand_tiles": int(len(np.unique(md[cand] // TILE))),
                          "atop": atop if want_ranked else None}
    return res


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else None


def summarise(rows):
    out = {"queries": len(rows), "total_postings": int(term_ptr[-1]), "N": int(N),
           "postings_per_query": {k: pct([r["postings"] for r in rows], k) for k in (50, 90, 99)},
           "matched_per_query": {k: pct([r["total"] for r in rows], k) for k in (50, 90, 99)},
           "tiles_touched": {k: pct([r["tiles_post"] for r in rows], k) for k in (50, 90, 99, 100)},
           "tiles_touched_minhash": {k: pct([r["tiles_post_mh"] for r in rows], k) for k in (50, 90, 99, 100)},
           "n_tiles": int((N + TILE - 1) // TILE),
           "bound_max_ratio": max(r["bound_max_ratio"] for r in rows), "bits": {}}
    for b in BITS:
        c = [r["bits"][b]["cand"] for r in rows]
        c2 = [r["bits"][b]["cand2"] for r in rows]
        out["bits"][b] = {"cand": {"p50": pct(c, 50), "p90": pct(c, 90), "p99": pct(c, 99), "max": max(c),
                                   "mean": float(np.mean(c))},
                          "cand_two_step": {"p50": pct(c2, 50), "p90": pct(c2, 90), "p99": pct(c2, 99), "max": max(c2),
                                            "mean": float(np.mean(c2))},
                          "cand_over_K_p50": pct([r["bits"][b]["cand"] / min(K, r["total"]) for r in rows], 50),
                          "straddle_share_p50": pct([r["bits"][b]["straddle"] / max(1, r["bits"][b]["cand"]) for r in rows], 50),
                          "covers_top_all": all(r["bits"][b]["covers_top"] for r in rows),
                          "bounds_valid_all": all(r["bits"][b]["bounds_valid"] for r in rows),
                          "rec2000_mean": float(np.mean([r["bits"][b]["rec2000"] for r in rows])),
                          "rec100_mean": float(np.mean([r["bits"][b]["rec100"] for r in rows])),
                          "cand_tiles_p50": pct([r["bits"][b]["cand_tiles"] for r in rows], 50),
                          "cand_tiles_mh_p50": pct([r["bits"][b]["cand_tiles_mh"] for r in rows], 50),
                          "post_in_cand_tiles_share_p50": pct([r["bits"][b]["post_in_cand_tiles"] / r["postings"] for r in rows], 50),
                          "post_in_cand_tiles_mh_share_p50": pct([r["bits"][b]["post_in_cand_tiles_mh"] / r["postings"] for r in rows], 50),
                          "post_in_cand_tiles_share_mean": float(np.mean([r["bits"][b]["post_in_cand_tiles"] / r["postings"] for r in rows])),
                          "post_in_cand_tiles_mh_share_mean": float(np.mean([r["bits"][b]["post_in_cand_tiles_mh"] / r["postings"] for r in rows])),
                          "cand_tiles_p99": pct([r["bits"][b]["cand_tiles"] for r in rows], 99)}
    return out


def main():
    global RANK
    t0 = time.time()
    RANK = minhash_order()
    print(f"minhash order in {time.time() - t0:.0f} s", flush=True)
    chk = pickle.load(open(D + "/engine_check.pkl", "rb"))
    agree = 0
    for i, (tot, eids, esc) in chk.items():
        r = run_query(P["test"][i], want_ranked=True)
        if r is None:
            agree += tot == 0
            continue
        mine = [ids[d] for d in r["top"]]
        s = None
        if mine == eids and r["total"] == tot:
            agree += 1
    print(f"engine check: {agree} of {len(chk)} queries give the engine's ids, order and total", flush=True)
    rows = []
    for k, p in enumerate(P["test"][:LIMIT]):
        r = run_query(p)
        if r is not None:
            rows.append(r)
        if k % 1000 == 0:
            print(f"test {k} {time.time() - t0:.0f} s", flush=True)
    summ = {"engine_check": f"{agree}/{len(chk)}", "test": summarise(rows), "skipped_empty": len(P["test"][:LIMIT]) - len(rows)}
    # fit sample quality: exact and approximate rankings, top 1,000
    runs = {"exact": {}}
    for b in BITS:
        runs[f"b{b}"] = {}
    for (qid, _), p in zip(P["fit_queries"], P["fit"]):
        r = run_query(p, want_ranked=True)
        if r is None:
            for k in runs:
                runs[k][qid] = []
            continue
        runs["exact"][qid] = [ids[d] for d in r["top"][:1000]]
        for b in BITS:
            runs[f"b{b}"][qid] = [ids[d] for d in r["bits"][b]["atop"][:1000]]
    judg = P["fit_judg"]
    summ["fit_quality"] = {k: metrics.score_run(v, judg)["mean"] for k, v in runs.items()}
    summ["fit_queries"] = len(P["fit_queries"])
    summ["took_s"] = round(time.time() - t0)
    json.dump(summ, open(D + "/study.json", "w"), indent=1, default=str)
    print(json.dumps(summ, indent=1, default=str))


if __name__ == "__main__":
    main()
