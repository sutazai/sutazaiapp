#!/usr/bin/env python3
"""
Agent Stream Consumer (optional)

Runs a minimal Redis Streams consumer loop and, for each task, invokes the
GenericAgentWithHealth's `_execute_task` implementation to process the task.

Usage:
  python3 scripts/mesh/agent_stream_consumer.py \
    --topic nlp --group nlp --consumer local-01 --results nlp

Env:
  REDIS_URL=redis://redis:6379/0
  LOG_LEVEL=INFO

Notes:
- Does not modify agent core; this is an opt-in helper.
- Safely acks messages only after processing.
"""
import argparse
import asyncio
import json
import os
import signal
import sys
import time
from typing import Any, Dict

from backend.app.mesh.redis_bus import (
    create_consumer_group,
    read_group,
    ack,
    result_stream,
    get_redis,
)


async def process_loop(topic: str, group: str, consumer: str, results_topic: str, agent_name: str, agent_type: str) -> None:
    # Lazy import to avoid heavy deps at module import time
    from agents.agent_with_health import GenericAgentWithHealth

    # Instantiate agent (no health server needed here)
    os.environ.setdefault("AGENT_NAME", agent_name)
    os.environ.setdefault("AGENT_TYPE", agent_type)
    agent = GenericAgentWithHealth()

    r = get_redis()

    create_consumer_group(topic, group)

    while True:
        msgs = read_group(topic, group, consumer, count=1, block_ms=1000)
        if not msgs:
            await asyncio.sleep(0.1)
            continue

        for msg_id, payload in msgs:
            try:
                # Execute using agent logic
                result = await agent._execute_task(payload)
                # Emit result
                r.xadd(result_stream(results_topic), {"json": json.dumps({
                    "task": payload,
                    "result": result,
                    "agent": agent_name,
                    "ts": int(time.time())
                })}, maxlen=10000, approximate=True)
                ack(topic, group, msg_id)
            except Exception as e:
                # Do not crash loop; leave message pending for retry policies
                sys.stderr.write(f"Error processing {msg_id}: {e}\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True)
    ap.add_argument("--group", required=True)
    ap.add_argument("--consumer", required=True)
    ap.add_argument("--results", required=True, help="results topic")
    ap.add_argument("--agent-name", default="mesh-agent")
    ap.add_argument("--agent-type", default="mesh")
    args = ap.parse_args()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    task = loop.create_task(process_loop(args.topic, args.group, args.consumer, args.results, args.agent_name, args.agent_type))

    def shutdown(*_):
        task.cancel()
        loop.stop()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

