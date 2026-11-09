---
copiedFrom: system3-city skills/record-lessons @ a06331c
name: record-lessons
description: Leave what you learned as knowledge entries (practice, finding, dead end, pitfall) with evidence and a belief from 0 to 1, so the next member starts where you stopped. Use when leaving a run, after a surprising result, or when off-boarding.
---

# Record lessons

The next member knows only what is on disk. Leave them what you learned, in the form they can weigh.

Each entry has these fields:
- `topic`: a short area, e.g. `query-planner` or `testing`;
- `kind`: one of
  - `practice` (do this),
  - `finding` (this is true),
  - `dead-end` (this did not work, and why),
  - `pitfall` (this looks right and is wrong);
- `text`: one or two sentences, specific enough to act on;
- `evidence`: what supports it (a commit, a score change, a test, a measurement). Never "I think";
- `belief`: from 0 to 1, how sure you are given that evidence;
- `scope`: `team` (how this team builds) or `company` (what the company as a whole should know).

Rules:
- One idea per entry. Prefer a dead end with its reason over a vague practice.
- Beliefs move with evidence. A later member who sees the entry hold (or fail) updates the belief and adds the
  evidence; they never delete it.
- To share with the city, publish the entries you choose by pull request to `knowledge/`. Credit the company (CC BY 4.0).
