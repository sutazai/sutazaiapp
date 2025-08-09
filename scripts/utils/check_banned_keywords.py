#!/usr/bin/env python3
"""
check_banned_keywords.py

Rule 1 enforcement: fail if banned fantasy terms appear in code or docs.

Exit codes:
  0 -> OK
  1 -> Found banned keywords
  2 -> Error
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable


REPO = Path(__file__).resolve().parents[1]
IGNORE_DIRS = {".git", "node_modules", "venv", "__pycache__", "htmlcov"}
TEXT_EXT = {".py", ".ts", ".tsx", ".js", ".json", ".md", ".yml", ".yaml", ".sh", ".toml"}

BANNED = [
    r"\bmagic\b",
    r"\bwizard\w*\b",
    r"\bteleport\w*\b",
    r"black[- ]?box",
    r"superIntuitiveAI",
    r"telekinesis",
]

PATTERN = re.compile("|".join(BANNED), re.IGNORECASE)


def iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in IGNORE_DIRS for part in p.parts):
            continue
        if p.suffix.lower() in TEXT_EXT:
            yield p


def main() -> int:
    try:
        violations = []
        for path in iter_files(REPO):
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for i, line in enumerate(content.splitlines(), 1):
                if PATTERN.search(line):
                    violations.append((str(path), i, line.strip()))

        if violations:
            print("Banned keywords detected (Rule 1):", file=sys.stderr)
            for f, ln, line in violations[:200]:
                print(f" - {f}:L{ln}: {line}", file=sys.stderr)
            if len(violations) > 200:
                print(f" ... and {len(violations)-200} more", file=sys.stderr)
            return 1

        print("OK: No banned fantasy terms found.")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


