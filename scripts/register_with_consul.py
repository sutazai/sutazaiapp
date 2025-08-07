#!/usr/bin/env python3
"""
Idempotent Consul service registration utility.

Usage:
  python scripts/register_with_consul.py \
      --service_name jarvis-task-controller \
      --service_address 127.0.0.1 \
      --service_port 8000 \
      --tags api,jarvis

Behavior:
  - Validates CLI args and fails fast on invalid input.
  - Uses python-consul to register a service with Consul Agent.
  - Idempotent: if the same service (name+address+port) is already registered, exits 0.
  - On differences, updates registration (deregister by ID then register) and logs actions.
  - Emits timestamped logs to stdout/stderr; exits non-zero on failure.

Notes:
  - Consul Agent must be reachable on default host/port or via CONSUL_HOST/CONSUL_PORT env vars.
  - Service ID is derived as f"{name}-{address}-{port}" to ensure uniqueness across instances.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List

try:
    import consul  # python-consul
except Exception as e:  # pragma: no cover
    print(f"[ERROR] {time.strftime('%Y-%m-%d %H:%M:%S')} Failed to import python-consul: {e}", file=sys.stderr)
    sys.exit(2)


def ts() -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register a service with Consul (idempotent)")
    parser.add_argument("--service_name", required=True, type=str, help="Service name (e.g., jarvis-task-controller)")
    parser.add_argument("--service_address", required=True, type=str, help="Service address (IP or hostname)")
    parser.add_argument("--service_port", required=True, type=int, help="Service port (e.g., 8000)")
    parser.add_argument("--tags", default="", type=str, help="Comma-separated tags (optional)")
    return parser.parse_args()


def validate_args(name: str, address: str, port: int) -> None:
    if not name.strip():
        raise ValueError("service_name must be non-empty")
    if not address.strip():
        raise ValueError("service_address must be non-empty")
    if port <= 0 or port > 65535:
        raise ValueError("service_port must be between 1 and 65535")


def get_consul_client() -> consul.Consul:
    host = os.getenv("CONSUL_HOST", "127.0.0.1")
    port_env = os.getenv("CONSUL_PORT", "8500")
    try:
        port = int(port_env)
    except ValueError:
        raise ValueError("CONSUL_PORT must be an integer")
    return consul.Consul(host=host, port=port, scheme=os.getenv("CONSUL_SCHEME", "http"))


def main() -> int:
    args = parse_args()
    try:
        validate_args(args.service_name, args.service_address, args.service_port)
    except Exception as e:
        print(f"[ERROR] {ts()} Invalid arguments: {e}", file=sys.stderr)
        return 2

    service_id = f"{args.service_name}-{args.service_address}-{args.service_port}"
    tags: List[str] = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []

    try:
        c = get_consul_client()
        # Fetch current services via local agent
        services = c.agent.services()
    except Exception as e:
        print(f"[ERROR] {ts()} Failed to connect to Consul Agent: {e}", file=sys.stderr)
        return 1

    # Check for idempotency
    for svc in services.values():
        if svc.get("ID") == service_id:
            same = (
                svc.get("Service") == args.service_name and
                str(svc.get("Address") or "") == args.service_address and
                int(svc.get("Port") or 0) == int(args.service_port)
            )
            if same:
                print(f"[INFO] {ts()} Service already registered: {service_id}")
                return 0
            else:
                print(f"[INFO] {ts()} Updating existing service: {service_id}")
                try:
                    c.agent.service.deregister(service_id)
                except Exception as e:
                    print(f"[WARN] {ts()} Deregister failed (continuing): {e}")
                break

    # Register (no health check to avoid speculative endpoints)
    try:
        c.agent.service.register(
            name=args.service_name,
            service_id=service_id,
            address=args.service_address,
            port=args.service_port,
            tags=tags or None,
        )
        print(f"[INFO] {ts()} Registered service: {service_id} ({args.service_name}@{args.service_address}:{args.service_port})")
        return 0
    except Exception as e:
        print(f"[ERROR] {ts()} Registration failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

