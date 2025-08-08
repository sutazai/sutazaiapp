#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
import uuid
from datetime import datetime, timezone

ROOT = os.getcwd()
BUS_DIR = os.path.join(ROOT, 'coordination_bus')
MSG_DIR = os.path.join(BUS_DIR, 'messages')
AGENTS_CSV = os.path.join(BUS_DIR, 'agents.csv')
STATUS_FILE = os.path.join(MSG_DIR, 'status.jsonl')
DIRECTIVES_FILE = os.path.join(MSG_DIR, 'directives.jsonl')
HEARTBEATS_FILE = os.path.join(MSG_DIR, 'heartbeats.jsonl')
LEDGER_FILE = os.path.join(ROOT, 'reports', 'cleanup', 'ledger.jsonl')


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def ensure_paths():
    os.makedirs(MSG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LEDGER_FILE), exist_ok=True)
    for f in (STATUS_FILE, DIRECTIVES_FILE, HEARTBEATS_FILE, LEDGER_FILE):
        if not os.path.exists(f):
            open(f, 'a', encoding='utf-8').close()


def append_jsonl(path, obj):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_agents():
    agents = []
    if not os.path.exists(AGENTS_CSV):
        return agents
    with open(AGENTS_CSV, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            agents.append(row)
    return agents


def cmd_list_agents(args):
    agents = load_agents()
    print(json.dumps({"count": len(agents), "sample": agents[:5]}, indent=2))


def cmd_post_status(args):
    ensure_paths()
    obj = {
        "ts": now_iso(),
        "agent_id": args.agent_id,
        "phase": args.phase,
        "task_id": args.task_id or f"CLN-{datetime.now():%Y%m%d}-{uuid.uuid4().hex[:6]}",
        "status": args.status,
        "summary": args.summary,
        "artifact": args.artifact or ""
    }
    append_jsonl(STATUS_FILE, obj)
    append_jsonl(LEDGER_FILE, {"event": "status", **obj})
    print(json.dumps(obj, indent=2))


def cmd_post_directive(args):
    ensure_paths()
    obj = {
        "ts": now_iso(),
        "directive_id": f"DIR-{datetime.now():%Y%m%d}-{uuid.uuid4().hex[:8]}",
        "from": args.from_id,
        "to": args.to,
        "phase": args.phase or "",
        "task_id": args.task_id or "",
        "priority": args.priority,
        "command": args.command,
        "notes": args.notes or ""
    }
    append_jsonl(DIRECTIVES_FILE, obj)
    append_jsonl(LEDGER_FILE, {"event": "directive", **obj})
    print(json.dumps(obj, indent=2))


def cmd_post_heartbeat(args):
    ensure_paths()
    obj = {
        "ts": now_iso(),
        "agent_id": args.agent_id,
        "status": args.status,
        "load": args.load,
        "queue": args.queue,
        "notes": args.notes or ""
    }
    append_jsonl(HEARTBEATS_FILE, obj)
    append_jsonl(LEDGER_FILE, {"event": "heartbeat", **obj})
    print(json.dumps(obj, indent=2))


def build_parser():
    p = argparse.ArgumentParser(description="Local coordination bus CLI")
    sub = p.add_subparsers(dest='cmd', required=True)

    s_list = sub.add_parser('list-agents', help='List agents roster (sample)')
    s_list.set_defaults(func=cmd_list_agents)

    s_stat = sub.add_parser('post-status', help='Append a status update')
    s_stat.add_argument('--agent-id', required=True)
    s_stat.add_argument('--phase', required=True, choices=['discovery','resolution','integration','validation'])
    s_stat.add_argument('--status', required=True, choices=['queued','in_progress','blocked','done','approve'])
    s_stat.add_argument('--summary', required=True)
    s_stat.add_argument('--artifact')
    s_stat.add_argument('--task-id')
    s_stat.set_defaults(func=cmd_post_status)

    s_dir = sub.add_parser('post-directive', help='Append an architect directive')
    s_dir.add_argument('--from-id', required=True)
    s_dir.add_argument('--to', required=True, help='agent_id|role|scope')
    s_dir.add_argument('--phase', choices=['discovery','resolution','integration','validation'])
    s_dir.add_argument('--task-id')
    s_dir.add_argument('--priority', default='normal', choices=['low','normal','high','urgent'])
    s_dir.add_argument('--command', required=True)
    s_dir.add_argument('--notes')
    s_dir.set_defaults(func=cmd_post_directive)

    s_hb = sub.add_parser('heartbeat', help='Append an agent heartbeat')
    s_hb.add_argument('--agent-id', required=True)
    s_hb.add_argument('--status', required=True, choices=['READY','BUSY','OFFLINE','DEGRADED'])
    s_hb.add_argument('--load', type=float, default=0.0, help='0.0-1.0')
    s_hb.add_argument('--queue', type=int, default=0)
    s_hb.add_argument('--notes')
    s_hb.set_defaults(func=cmd_post_heartbeat)

    return p


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()

