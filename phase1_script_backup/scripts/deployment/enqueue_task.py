#!/usr/bin/env python3
"""
Enqueue a task into the lightweight mesh (Redis Streams).

Usage:
  python3 scripts/mesh/enqueue_task.py --topic nlp --task '{"prompt":"hello"}'
Env:
  REDIS_URL=redis://redis:6379/0
"""
import argparse
import json
import sys
from backend.app.mesh.redis_bus import enqueue_task


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True)
    ap.add_argument("--task", required=True, help="JSON payload")
    args = ap.parse_args()

    try:
        payload = json.loads(args.task)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON for --task: {e}")
        return 2

    msg_id = enqueue_task(args.topic, payload)
    print(msg_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())

