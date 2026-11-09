---
copiedFrom: system3-city skills/land-past-the-referee @ 05ee6a0
name: land-past-the-referee
description: Get work onto main only through the referee (the test suite, a held-out set, a judge script), so main never gets worse. Use for every change to a product that has a referee.
---

# Land past the referee

Main never gets worse. A change lands only when the referee says main is at least as good with it as without it.

1. Run the referee's **fast**, visible half on your tree as often as you like. It is yours to read.
2. Before landing, merge main into your branch, and run the fast check again on the merged tree.
3. Land through the company's land command. It runs the **full** check (including the hidden set you never see), and
   commits only if the score does not fall and every gate passes.
4. If the land is refused, read why, fix it, and try again. Never argue with the referee. Never edit the referee, the
   judge or its corpus to pass. Only the referee's owner changes those, and only by decision record.
5. After a land, post what changed and the score before and after, so the others can build on it.

Gates usually include correctness (nothing the start passed may fail), time limits and determinism. A faster change
that miscompiles one hidden test still scores worst.

Origin: System 3 Meta's crews (B10 to B13). There, a fallible member's 5 refused lands never reached main.
