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
IGNORE_DIRS = {".git", "node_modules", "venv", "__pycache__", "htmlcov", "compliance-reports", "reports"}
TEXT_EXT = {".py", ".ts", ".tsx", ".js", ".json", ".md", ".yml", ".yaml", ".sh", ".toml"}

BANNED = [
    r"\bwizard(?:Service|Handler|Manager|Module|Function)\b",
    r"\bmagic(?:Mailer|Handler|Function|Method|Service)\b",
    r"\bteleport(?:Data|Function|Service)\b",
    r"\bsuperIntuitiveAI\b",
    r"\bmystical(?:Connection|Service|Manager)\b",
    r"\bblack[- ]?box(?!-exporter)\b",
    r"\benchanted\w*\b",
    r"\bsupernatural\w*\b",
    r"\bethereal\w*\b",
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
            # Skip fantasy element detection tools themselves
            if "fantasy" in path.name or "banned" in path.name or "compliance" in path.name:
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for i, line in enumerate(content.splitlines(), 1):
                # Skip lines that are just mapping/configuration examples
                if ":" in line and ("configService" in line or "emailSender" in line or "transferData" in line):
                    continue
                # Skip Prometheus blackbox exporter references
                if "blackbox" in line.lower() and ("exporter" in line.lower() or "prometheus" in line.lower() or "groupadd" in line or "useradd" in line):
                    continue
                # Skip documentation that lists fantasy terms as examples
                if "No fantasy terms" in line or "magic, wizard, teleport" in line:
                    continue
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


