"""Check that the crew's roles here match the lab's (the source), so the subagents never drift from the records.

    python scripts/check_roles.py [../shakabuku-labs]     exit 1 and name the roles that differ

Compares each `.claude/agents/<role>.md` with the lab's copy, ignoring the copy note this repository adds.
"""
import re
import sys
from pathlib import Path

LAB = Path(sys.argv[1] if len(sys.argv) > 1 else "../shakabuku-labs")
NOTE = re.compile(r"\n> Copy of `hanialshater/shakabuku-labs`.*?\n(?=\n|#)", re.S)


def body(text: str) -> str:
    return NOTE.sub("", text).strip()


def main() -> int:
    if not (LAB / ".claude" / "agents").is_dir():
        print(f"check_roles: no lab checkout at {LAB}; clone hanialshater/shakabuku-labs beside this repository")
        return 1
    bad = [p.name for p in sorted(Path(".claude/agents").glob("*.md"))
           if not (LAB / ".claude/agents" / p.name).exists()
           or body(p.read_text()) != body((LAB / ".claude/agents" / p.name).read_text())]
    if bad:
        print("check_roles: differs from the lab, update from the lab's copy:", ", ".join(bad))
        return 1
    print("check_roles: the crew's roles match the lab's")
    return 0


if __name__ == "__main__":
    sys.exit(main())
