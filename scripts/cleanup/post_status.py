#!/usr/bin/env python3
"""Append a status message to coordination bus and ledger.
Usage: post_status.py AGENT_ID PHASE TASK_ID STATUS SUMMARY [ARTIFACT]
"""
from __future__ import annotations
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path


def main(argv):
    if len(argv) < 6:
        print("Usage: post_status.py AGENT_ID PHASE TASK_ID STATUS SUMMARY [ARTIFACT]", file=sys.stderr)
        return 2
    agent_id, phase, task_id, status, summary = argv[1:6]
    artifact = argv[6] if len(argv) > 6 else ''
    ts = datetime.now(timezone.utc).isoformat()
    msg = {
        'ts': ts,
        'agent_id': agent_id,
        'phase': phase,
        'task_id': task_id,
        'status': status,
        'summary': summary,
        'artifact': artifact,
    }
    Path('coordination_bus/messages').mkdir(parents=True, exist_ok=True)
    with open('coordination_bus/messages/status.jsonl','a') as f:
        f.write(json.dumps(msg) + "\n")
    Path('reports/ledger').mkdir(parents=True, exist_ok=True)
    with open('reports/ledger/ledger.jsonl','a') as f:
        f.write(json.dumps({'ts': ts, 'event':'status', 'detail': msg}) + "\n")
    print(json.dumps(msg))
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))

