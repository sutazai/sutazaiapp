#!/usr/bin/env python3
"""
Analyze duplicate directories/modules differing only by '-' vs '_' or case.

This is a read-only analyzer to surface consolidation opportunities without
deleting or modifying any files. Outputs a JSON report to stdout.
"""
import json
import os
from pathlib import Path
from collections import defaultdict


ROOT = Path("/opt/sutazaiapp")


def canonical(name: str) -> str:
    return name.lower().replace('-', '_')


def find_duplicates(base: Path):
    groups = defaultdict(list)
    for p in base.iterdir():
        if p.is_dir():
            groups[canonical(p.name)].append(p.name)
    dups = {k: v for k, v in groups.items() if len(v) > 1}
    return dups


def main():
    hotspots = [ROOT / "agents", ROOT / "backend" / "ai_agents", ROOT / "backend" / "agent_orchestration"]
    report = {}
    for h in hotspots:
        if h.exists():
            report[str(h)] = find_duplicates(h)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

