#!/usr/bin/env python3
"""
audit_docs.py

Rule 6, 15, 17 enforcement (docs-as-code, centralization, IMPORTANT precedence):
- Ensure Markdown docs are centralized under /docs or /IMPORTANT
- Flag Markdown files outside allowed locations
- Detect duplicate filenames across docs (potential dedup needed)

Exit codes:
  0 -> OK
  1 -> Issues found
  2 -> Error
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parents[1]
ALLOWED_ROOTS = {REPO / "docs", REPO / "IMPORTANT"}
IGNORE_DIRS = {".git", "node_modules", "venv", "__pycache__", "htmlcov"}


def main() -> int:
    try:
        issues: List[str] = []
        name_to_paths: Dict[str, List[Path]] = defaultdict(list)

        # Ensure /docs exists
        if not (REPO / "docs").exists():
            issues.append("Missing /docs directory (Rule 6)")

        for p in REPO.rglob("*.md"):
            if any(part in IGNORE_DIRS for part in p.parts):
                continue
            # Allow important and docs
            if any(str(p).startswith(str(root) + "/") for root in ALLOWED_ROOTS):
                pass
            else:
                # Allow README.md at root
                if p == REPO / "README.md":
                    pass
                else:
                    rel = p.relative_to(REPO)
                    issues.append(f"Markdown outside /docs or /IMPORTANT: {rel}")

            name_to_paths[p.name].append(p)

        # Flag duplicates across different directories (excluding identical paths)
        for name, paths in name_to_paths.items():
            uniq_dirs = {str(p.parent) for p in paths}
            if len(uniq_dirs) > 1 and name != "README.md":
                rels = ", ".join(str(p.relative_to(REPO)) for p in paths)
                issues.append(f"Duplicate doc filename '{name}' across: {rels}")

        if issues:
            print("Documentation audit found issues:")
            for msg in issues:
                print(f" - {msg}")
            return 1

        print("OK: Documentation centralized and no duplicates detected.")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


