#!/usr/bin/env python3
"""
Tail results from the lightweight mesh (Redis Streams).

Usage:
  python3 scripts/mesh/tail_results.py --topic nlp --count 10
Env:
  REDIS_URL=redis://redis:6379/0
"""
import argparse
import json
import sys
from backend.app.mesh.redis_bus import tail_results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True)
    ap.add_argument("--count", type=int, default=10)
    args = ap.parse_args()

    items = tail_results(args.topic, args.count)
    print(json.dumps([{"id": mid, "data": data} for (mid, data) in items], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

