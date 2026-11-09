# S3 Search: rules for every session

The engine of **Floor 2 · Search engine** in Shakabuku Labs (System 3 city). The floor's records live in
`hanialshater/shakabuku-labs`: the brief (`docs/search-floor/BRIEF.md`), the interfaces
(`docs/search-floor/INTERFACES.md`), the milestones (`company/milestones.json`), the asks (`company/asks.json`), the
roles (`.claude/agents/`) and the lessons (`company/knowledge/entries.jsonl`). The lab's `CLAUDE.md` applies here too.
**The disk is the memory.**

## Start of session

1. Read the floor's milestones and asks in the lab repository. If you were given a role, read its file there.
2. Claim an item before touching an area that is not yours (the areas and owners are in `README.md`).
3. Run `python -m referee fast` before you change anything, and record the result.

## Hard rules

- **Clean room.** Public data and the founder's own work only. Nothing from any employer or client: no code, data,
  documents or numbers.
- **Data access.** Read ESCI only through `referee/loader.py`. Never load the test judgments yourself, and never look
  for the founder's held-out draw.
- **The referee.** Only the evaluation steward changes `referee/` (P6). A maker who thinks a check is wrong raises it
  with the steward and does not edit it.
- **Land past the referee.** A change lands on `main` only by pull request with `python -m referee fast` green. A land
  that claims a quality, latency or freshness number needs a Colab score of record, which the founder runs.
- **Predictions before runs.** Every milestone's predictions are in the lab's `company/milestones.json`. A new run adds
  its own predictions there first, with a wrong-if, and they are never edited after the run starts.
- **No spending without approval.** Spending needs a budget line the founder has approved. GPU work runs in the
  founder's Colab.
- **No deploying without approval.** Never deploy the demo, and never put `[deploy]` in a commit message, without the
  founder's approval of that deploy in the conversation.
- **Private.** Nothing from this repository is published without the founder's decision (ask L-10).
- **No secrets and no personal data.**

## Before you finish

- Run `python -m referee fast` and record the result in the pull request.
- Write the result against its predictions, wrong ones included, in the lab's `company/milestones.json`.
- Add a lesson to the lab's `company/knowledge/entries.jsonl` if you learned something the next session should know.
