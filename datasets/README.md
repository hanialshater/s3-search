# datasets/

owner: evaluation-steward · B1 (2026-11-09)

The datasets are built and served only by the referee's loader (`referee/loader.py`): `loader.load_dataset(name)`
returns the catalogue, its mapping, the fit judgments and the test queries (no judgments). The files live under
`data/`, which git ignores. This folder holds only what proves a build is the right one.

| Name | What | Built | Size |
|---|---|---|---|
| `toy` | synthetic, deterministic (`referee/toy.py`), for the contract and tests | in memory, about 20 ms | 200 products; 30 queries (20 fit, 10 test); 1,098 fit judgments |
| `esci-sample` | ESCI US, reduced set: 2,000 train and 500 test queries drawn with seed 20261109, every judgment of them and every product they judge | `loader.build_esci_sample()` from `data/esci`, about 5 s; cached in `data/esci-sample` on first `load_dataset` | 48,639 products; 40,714 fit judgments (E 17,562, S 14,253, C 1,693, I 7,206); 9,701 test judgments |
| `esci-us` | ESCI US, reduced set, whole | `loader.fetch()` (1.1 GB) | 1.22M products; 8,956 test queries |

- `esci-sample.manifest.json`: counts and content hashes of the draw (independent of the parquet writer). The tests
  check a local build against it, so everyone develops on the same sample. It carries no test-label counts.
- **esci-sample's catalogue holds only judged products,** so its whole-catalogue nDCG is much higher than ESCI-US's:
  BM25F with weight 1 on every field gives 0.464 on its fit split, against the prototype's 0.304 on ESCI-US test. Use
  it to compare changes with each other, never with ESCI-US numbers.
- The 500 test queries' judgments are read only by the scorer (`referee full`). The founder's held-out draw is separate
  and is scored only in Colab.
