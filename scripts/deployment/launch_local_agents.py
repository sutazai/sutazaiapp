#!/usr/bin/env python3
"""
Local Multi-Agent Launcher

Purpose: Launch N local SutazAI agent instances concurrently for development/testing.
Author: Codex CLI (AI)
Date: 2025-08-07

Usage:
  python3 scripts/launch_local_agents.py --count 20 --base-port 8600 \
      --agent-type local-dev --backend-url http://localhost:8000 \
      --ollama-url http://localhost:11434

Notes:
- Spawns independent processes, each running `GenericAgentWithHealth` with a unique port/name.
- Uses only local services (backend and Ollama endpoints you provide).
- Gracefully terminates all child processes on Ctrl+C.
"""

import argparse
import asyncio
import os
import signal
import sys
from dataclasses import dataclass
from multiprocessing import Process
from typing import List


def _run_agent(name: str, agent_type: str, port: int, backend_url: str, ollama_url: str) -> None:
    """Child process target: run a single agent instance."""
    # Defer imports so parent process doesn't need agent deps at import time
    from agents.agent_with_health import GenericAgentWithHealth

    # Set environment for the agent instance
    os.environ["AGENT_NAME"] = name
    os.environ["AGENT_TYPE"] = agent_type
    os.environ["HEALTH_PORT"] = str(port)
    os.environ["BACKEND_URL"] = backend_url
    os.environ["OLLAMA_URL"] = ollama_url
    os.environ.setdefault("LOG_LEVEL", "INFO")

    agent = GenericAgentWithHealth()

    # Run until stopped; rely on agent's own shutdown handling
    try:
        asyncio.run(agent.start())
    except KeyboardInterrupt:
        pass


@dataclass
class AgentSpec:
    name: str
    agent_type: str
    port: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch multiple local SutazAI agent instances")
    parser.add_argument("--count", type=int, default=20, help="Number of agents to launch (default: 20)")
    parser.add_argument("--base-port", type=int, default=8600, help="Starting port for agent health endpoints")
    parser.add_argument("--agent-type", type=str, default="local-dev", help="Agent type label for all instances")
    parser.add_argument(
        "--backend-url",
        type=str,
        default=os.environ.get("BACKEND_URL", "http://localhost:8000"),
        help="Backend API base URL",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        help="Ollama base URL (local model endpoint)",
    )
    return parser.parse_args()


def build_specs(count: int, base_port: int, agent_type: str) -> List[AgentSpec]:
    specs: List[AgentSpec] = []
    for i in range(count):
        name = f"local-agent-{i+1:02d}"
        port = base_port + i
        specs.append(AgentSpec(name=name, agent_type=agent_type, port=port))
    return specs


def main() -> int:
    args = parse_args()
    specs = build_specs(args.count, args.base_port, args.agent_type)

    procs: List[Process] = []

    def terminate_all(signum=None, frame=None):
        for p in procs:
            if p.is_alive():
                try:
                    p.terminate()
                except Exception:
                    pass
        for p in procs:
            try:
                p.join(timeout=5)
            except Exception:
                pass
        return 0

    # Handle Ctrl+C and TERM for clean shutdown
    signal.signal(signal.SIGINT, terminate_all)
    signal.signal(signal.SIGTERM, terminate_all)

    print(
        f"Launching {len(specs)} agents starting at port {args.base_port} "
        f"(agent_type={args.agent_type})"
    )
    print(f"Backend: {args.backend_url} | Ollama: {args.ollama_url}")

    # Start all agents
    for spec in specs:
        p = Process(
            target=_run_agent,
            args=(spec.name, spec.agent_type, spec.port, args.backend_url, args.ollama_url),
            daemon=True,
        )
        p.start()
        procs.append(p)
        print(f"- Started {spec.name} on /health port {spec.port} (pid={p.pid})")

    # Keep parent alive while children run
    try:
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        terminate_all()

    return 0


if __name__ == "__main__":
    sys.exit(main())

