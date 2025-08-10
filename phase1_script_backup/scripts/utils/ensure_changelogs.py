#!/usr/bin/env python3
"""
Purpose: Ensure every directory in the repository contains a CHANGELOG.md.

- Reads template from docs/templates/CHANGELOG_TEMPLATE.md
- Skips common vendor/build/cache directories (configurable via CLI)
- Supports dry-run and verbose modes

Usage:
  python scripts/utils/ensure_changelogs.py --write

Args:
  --write        Actually create missing CHANGELOG.md files (default: dry-run)
  --root PATH    Repository root (default: current working directory)
  --skip DIR     Additional dirs to skip (can repeat)
  --verbose      Print each action

Notes:
- This script writes minimal, standards-compliant placeholders and preserves existing files.
- Author: Coding Agent
- Date: 2025-08-08
"""

import argparse
import os
from pathlib import Path


DEFAULT_SKIPS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".hypothesis",
    ".claude",
    "venv",
    ".venv",
    "repos",
    "backups",
    "secrets_secure",
}


def load_template(root: Path) -> str:
    template_path = root / "docs" / "templates" / "CHANGELOG_TEMPLATE.md"
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")
    return template_path.read_text(encoding="utf-8")


def ensure_changelogs(root: Path, do_write: bool, extra_skips: list[str], verbose: bool) -> int:
    template = load_template(root)
    skips = set(DEFAULT_SKIPS) | set(extra_skips)
    created = 0

    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        # Normalize and skip hidden/vendor dirs early
        parts = Path(rel).parts
        if any(p in skips for p in parts):
            # Prevent descent into skipped directories
            dirnames[:] = [d for d in dirnames if d not in skips]
            continue

        # Do not modify repo root CHANGELOG.md behavior; create if missing too
        cl_path = Path(dirpath) / "CHANGELOG.md"
        if not cl_path.exists():
            created += 1
            if verbose:
                print(f"[missing] {cl_path}")
            if do_write:
                header = f"\n> Path: /{rel if rel != '.' else ''}\n\n" if rel != "." else "\n> Path: /\n\n"
                content = template + header
                cl_path.write_text(content, encoding="utf-8")
                if verbose:
                    print(f"[created] {cl_path}")

    if verbose and not do_write:
        print(f"Dry-run complete. {created} files would be created.")
    return created


def main():
    parser = argparse.ArgumentParser(description="Ensure CHANGELOG.md exists in every directory")
    parser.add_argument("--write", action="store_true", help="Create files instead of dry-run")
    parser.add_argument("--root", type=str, default=".", help="Repository root path")
    parser.add_argument("--skip", action="append", default=[], help="Additional directory names to skip")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    count = ensure_changelogs(root, args.write, args.skip, args.verbose)
    if args.verbose:
        print(f"Total missing CHANGELOG.md files: {count}")


if __name__ == "__main__":
    main()

