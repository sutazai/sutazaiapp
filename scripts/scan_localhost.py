#!/usr/bin/env python3
"""
scan_localhost.py

Purpose: Find hardcoded localhost/127.0.0.1 references and suggest service-name replacements.

Exit codes:
  0 -> No issues
  1 -> Found issues
  2 -> Script error
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
PATTERN = re.compile(r"(localhost|127\.0\.0\.1)(:\d+)?", re.IGNORECASE)
IGNORE_DIRS = {"node_modules", ".git", "venv", "__pycache__", "htmlcov"}
TEXT_EXT = {".py", ".ts", ".tsx", ".js", ".json", ".yml", ".yaml", ".md", ".sh", ".toml"}


SVC_HINTS = {
    ":10010": "backend:8000",
    ":10011": "frontend:8501",
    ":10000": "postgres:5432",
    ":10001": "redis:6379",
    ":10101": "qdrant:6333",
    ":10102": "qdrant:6334",
    ":10100": "chromadb:8000",
    ":10103": "faiss:8000",
    ":10002": "neo4j:7474",
    ":10003": "neo4j:7687",
    ":10104": "ollama:10104",
    ":10200": "prometheus:9090",
    ":10201": "grafana:3000",
}


def should_scan(path: Path) -> bool:
    parts = set(path.parts)
    if parts & IGNORE_DIRS:
        return False
    if path.suffix.lower() in TEXT_EXT:
        return True
    return False


def scan_file(path: Path) -> List[Tuple[int, str]]:
    findings: List[Tuple[int, str]] = []
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return findings
    for idx, line in enumerate(content.splitlines(), 1):
        if PATTERN.search(line):
            findings.append((idx, line.strip()))
    return findings


def suggest_replacement(line: str) -> str | None:
    for host_port, svc in SVC_HINTS.items():
        if host_port in line:
            return line.replace("localhost" + host_port, svc).replace("127.0.0.1" + host_port, svc)
    # generic replacement
    if "localhost" in line:
        return line.replace("localhost", "<service-name>")
    if "127.0.0.1" in line:
        return line.replace("127.0.0.1", "<service-name>")
    return None


def main() -> int:
    try:
        issues = 0
        for path in REPO_ROOT.rglob("*"):
            if path.is_file() and should_scan(path):
                findings = scan_file(path)
                if findings:
                    print(f"\n{path}")
                    for lineno, line in findings:
                        suggestion = suggest_replacement(line) or "(replace with docker service name)"
                        print(f"  L{lineno}: {line}\n    -> {suggestion}")
                        issues += 1
        if issues:
            print(f"\nFound {issues} localhost references. Align to IMPORTANT (service names, not host ports).")
            return 1
        print("OK: No localhost/127.0.0.1 references found in scanned files.")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


