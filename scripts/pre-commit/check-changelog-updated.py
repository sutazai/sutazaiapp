#!/usr/bin/env python3
"""
Rule 19: Require docs/CHANGELOG.md update in every commit.

- Fails commit if there are staged changes but docs/CHANGELOG.md is not staged
  with an added entry line matching the required format:
  [Time] - [Date] - [Version] - [Component] - [Change Type] - [Description]

- Allows bypass via SKIP=require-changelog-update (logged by pre-commit)
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path("/opt/sutazaiapp")
CHANGELOG_PATH = REPO_ROOT / "docs" / "CHANGELOG.md"

ENTRY_REGEX = re.compile(
    r"^##?\s*\[?\d{4}-\d{2}-\d{2}(?:[ T].*?)?\]?\s*-\s*\[v?[0-9A-Za-z_.-]+\]\s*-\s*\[[^\]]+\]\s*-\s*\[[^\]]+\]\s*-\s*\[[^\]]+\]",
)


def run_git(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, text=True, capture_output=True)


def get_staged_files() -> list[str]:
    res = run_git(["diff", "--cached", "--name-only"])
    if res.returncode != 0:
        return []
    return [l for l in res.stdout.strip().split("\n") if l]


def file_staged(path: Path) -> bool:
    res = run_git(["diff", "--cached", "--name-only", "--", str(path)])
    return res.returncode == 0 and bool(res.stdout.strip())


def changelog_has_new_entry() -> bool:
    # Check staged diff for added lines matching ENTRY_REGEX
    res = run_git(["diff", "--cached", "-U0", "--", str(CHANGELOG_PATH)])
    if res.returncode != 0:
        return False
    for line in res.stdout.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            if ENTRY_REGEX.search(line[1:]):
                return True
    return False


def main() -> int:
    # Allow explicit bypass for emergencies
    if os.getenv("SKIP", "") == "require-changelog-update":
        print("SKIP=require-changelog-update detected. Bypassing Rule 19 check.")
        return 0

    staged = get_staged_files()
    if not staged:
        print("No staged changes; skipping CHANGELOG requirement.")
        return 0

    # Ignore commits that only touch generated reports or coverage artifacts
    non_trivial = [
        f for f in staged
        if not (
            f.startswith("reports/") or f.startswith("htmlcov/") or 
            f.endswith(".lock") or f.startswith(".coverage")
        )
    ]

    if not non_trivial:
        print("Only generated artifacts staged; skipping CHANGELOG requirement.")
        return 0

    # Enforce CHANGELOG presence and update
    if not CHANGELOG_PATH.exists():
        print("❌ Rule 19: docs/CHANGELOG.md not found. Create it and document your change.")
        return 1

    if not file_staged(CHANGELOG_PATH):
        print("❌ Rule 19: You must update docs/CHANGELOG.md in every commit.")
        print("   Add a new entry with the format: [Time] - [Date] - [Version] - [Component] - [Change Type] - [Description]")
        return 1

    if not changelog_has_new_entry():
        print("❌ Rule 19: docs/CHANGELOG.md is staged but no new properly formatted entry was detected.")
        print("   Ensure you add a line like: '## [2025-08-08 12:34 UTC] - [vX.Y] - [Component] - [Type] - [Description]' ")
        return 1

    print("✅ Rule 19: CHANGELOG update detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


