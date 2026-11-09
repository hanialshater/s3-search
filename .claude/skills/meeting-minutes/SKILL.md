---
copiedFrom: system3-city skills/meeting-minutes @ a06331c
name: meeting-minutes
description: Record a meeting's minutes (title, room, chair, attendees, decisions, objections, a summary and the record it wrote) as an event the HQ shows in Meetings and Rooms. Use whenever two or more roles meet to decide something.
---

# Meeting minutes

A meeting that leaves no minutes did not happen, as far as the next session can tell.

Record one event per meeting in the company's event log (v1):

```json
{ "v": 1, "t": "<ISO time>", "type": "meeting", "by": "<chair>", "chair": "<chair>",
  "title": "<what it was for>", "room": "<a meeting room's id>",
  "attendees": ["<role>", "..."], "summary": "<two or three sentences>",
  "decisions": ["<each decision, as a sentence>"], "objections": ["<who: the objection, and its answer>"],
  "record": "<path to the full record, if one was written>" }
```

Rules:
- List every role that took part, including the owner if they were there. The HQ draws the attendees in the room
  while the meeting is on.
- A decision that others will rely on also gets a decision record (see `decision-record`). Link it from `record`.
- Write objections as they were raised, with their answers.
- Crews record minutes with their crew tool (`crew minutes`). Other companies append the event with their own
  event writer.
