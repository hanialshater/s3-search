# Kickoff: running the search floor's crew from Claude Code

owner: founder · written by onboarding, 2026-11-09 · design 2 (`shakabuku-labs`
`company/decisions/2026-11-09-search-floor-v2.md`; approved when its pull request merges)

## 1. Set up once

```bash
mkdir -p ~/s3 && cd ~/s3
git clone https://github.com/hanialshater/s3-search
git clone https://github.com/hanialshater/shakabuku-labs   # the records: milestones, asks, lessons, roles
cd s3-search
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m referee fast                                      # green before anything changes
python scripts/check_roles.py                               # the subagents match the lab's roles
claude                                                      # Claude Code, opened in s3-search
```

The crew's roles are in `.claude/agents/` as subagents: `architect`, `dataplane-engineer`, `platform-engineer`,
`tools-engineer` and `evaluation-steward`. They are copies; the lab's `.claude/agents/` is the source. Records go in
`../shakabuku-labs`.

## 2. How a session runs

You are the owner. The session you talk to is the coordinator: it routes work and writes records, and does not build
(P11). Each piece of work goes to the role that owns its area, as a subagent. A subagent runs warm in its own context
and lands its work only past the referee.

1. **Start of session.** "Read `../shakabuku-labs/company/milestones.json`, `asks.json` and `docs/BOARD.md`. Run
   `python -m referee fast`. Tell me where each milestone stands."
2. **Give one milestone to its owner.**
   > Use the architect subagent for S1. It claims the S1 items in `docs/BOARD.md`, writes the interfaces' sections 4 to
   > 6 first, then ports the prototype's operators from `baseline/esci_gpu_vs_opensearch.ipynb` into `engine/ops/`, with
   > `engine/api.py` per the interfaces. It works on a branch, keeps `python -m referee fast` green and under 1 minute,
   > and opens a pull request.
3. **Run areas in parallel only once their interfaces are written.** For example, the dataplane-engineer on S2 and
   the platform-engineer on S4's API, once the architect has written sections 3 to 7 of the interfaces.
4. **A score of record.** When a pull request claims a quality, latency or freshness number, open
   `referee/colab_referee.ipynb` in Colab on a GPU and run it on that branch. Its `results.json` is the number. The CPU
   numbers are for development only.
5. **End of session.** "Write each milestone's result against its predictions, wrong ones included, in the lab's
   `company/milestones.json`, add the lessons to `knowledge/entries.jsonl`, and update `docs/BOARD.md`."

## 3. Rules the crew must keep

These are from `CLAUDE.md` here and in the lab.

- **Clean room.** Public data and your own work only.
- **Data.** ESCI only through `referee/loader.py`. Members never see the test judgments or your held-out draw.
- **The referee.** Only the evaluation steward edits `referee/` and `datasets/`. That is you, or the
  evaluation-steward subagent with you in the loop. Never a maker.
- **Predictions before runs.** A run that is not in `company/milestones.json` adds its predictions first.
- **Time budgets.** `python -m referee fast` stays under 1 minute, and the CPU benchmark under 5 minutes, on 2 cores.
- **Small models on CPU.** For example TF-IDF reduced to 128 dimensions with SVD, trained on the sample. bge-small is
  fine where the network allows it.
- **No paid spend without a budget line you approve (L-9).** No deploy without your approval, and nothing public
  before L-10.

## 4. What is waiting on you first

- **L-7, the held-out draw.** Run https://colab.research.google.com/github/hanialshater/shakabuku-labs/blob/main/lab/held-out/draw_search_esci.ipynb
- **Merge design 2's pull request** in the lab. Merging approves design 2.
- **Then start the crew on B1 and S1.** They can run in parallel, because B1 is the steward's and S1 the architect's.
