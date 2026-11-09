---
copiedFrom: system3-city skills/decision-record @ a06331c
name: decision-record
description: Record a decision with its owner, date, status, the options, the reason and the objections that travel with it. Use for any decision another role will rely on (architecture, org, budget, product).
---

# Decision record

Write a decision down where the next session will find it, with the objections still attached.

```markdown
# <the decision, as a sentence>

owner: <role> · date: YYYY-MM-DD · status: proposed | decided | superseded · class: product | engineering | org | budget

**Decision.** What was decided, in one paragraph.

**Why.** The evidence and the reasoning. Link the data.

**Alternatives.** What else was considered, and why not.

## Objections

- <who>: <the objection>. Answer it, or say why it stands. Write "none raised" if there were none. Never just "noted".
```

Rules:
- A decision that changes a protected rule (a gate, a constitution) goes through that rule's amendment route.
- The owner's approval is quoted verbatim, with its date.
- Superseding a decision is a new record that links the old one. The old one stays.
- The documents tool indexes these files, so they show in the HQ's Knowledge.
