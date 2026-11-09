---
copiedFrom: system3-city skills/predictions-first @ 05ee6a0
name: predictions-first
description: Commit predictions before a run, an experiment or a launch, then judge the result against them (wrong ones included). Use whenever work will produce a number someone will read as success or failure.
---

# Predictions first

Write down what you expect before you run anything, commit it, and only then run. A result read without a prior
prediction tends to look like a success.

1. Write the design: what runs, the arms (what is compared with what), the caps (money, time), the measure, and how it
   is scored.
2. For each prediction, write a row:
   - an id;
   - the prediction itself, in numbers;
   - what would show it wrong;
   - your confidence, as a percentage.
3. Commit the design and the predictions **before** the run starts. The commit time is the evidence.
4. If something changes before the run (a cap, a gate), adjust the prediction in a separate, dated commit, with
   the reason. Never change one after the run has started.
5. After the run, give every prediction a verdict: right, wrong or partly. Report the wrong ones as prominently as
   the right ones, and say what you now believe instead.
6. A deviation during the run (a rule bent for both arms alike) is written next to the result. It is not hidden in a
   footnote.

Origin: System 3 Meta's benchmarks (B6 to B13). Each one's design and predictions were committed before it ran.
