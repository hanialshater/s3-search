# The baseline: the Colab prototype

`esci_gpu_vs_opensearch.ipynb` is the founder's Colab notebook, copied with its outputs. It was created on 2026-11-04
and last run on 2026-11-05 on one A100-SXM4-80GB. It covers ESCI US: 1.22M products and 8,956 test queries (reduced
set). S1 reproduces it under CI. It is the reference, not production code.

| Measurement | GPU engine | OpenSearch 2.19 (+ the same GPU ranker) |
|---|---|---|
| Lexical nDCG@10, tuned | 0.304 | 0.270 (`cross_fields`) |
| Title-only parity | within 0.0002 | (same) |
| Page 1, batch 1, p50 / p99 | 20.6 / 33.8 ms | 268 / 379 ms |
| Pipeline throughput under a 100 ms p99 | 275 QPS (347 peak) | never met |
| Exact recall of the 20,000 candidates | 0.926 | 0.923 |
| GBDT vs RRF order, nDCG@10 | 0.167 vs 0.319 | 0.160 vs 0.316 |
| Peak GPU memory | 4.6 GB | n/a |

Known problems before any work:

- **The GBDT recipe hurts ranking.** It trained the top unjudged candidates as negatives.
- **Query encoding dominates latency.** It takes 11 of the 20.6 ms median.
