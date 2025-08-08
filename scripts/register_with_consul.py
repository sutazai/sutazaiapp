#!/usr/bin/env python3
"""
Idempotent Consul service registration utility with retry logic and exponential backoff.

Usage:
  python scripts/register_with_consul.py \
      --service_name jarvis-task-controller \
      --service_address 127.0.0.1 \
      --service_port 8000 \
      --tags api,jarvis

Behavior:
  - Validates CLI args and fails fast on invalid input.
  - Uses python-consul library to register a service with Consul Agent.
  - Implements retry logic with exponential backoff for both connection and registration.
  - Idempotent: if the same service (name+address+port) is already registered, exits 0.
  - On differences, updates registration (deregister by ID then register) and logs actions.
  - Emits detailed timestamped logs to stdout/stderr; exits non-zero on failure.
  - Safe to run multiple times without side effects.

Features:
  - Exponential backoff retry mechanism (3 attempts with 1-2s base delay).
  - Comprehensive error handling and logging.
  - Environment variable configuration support.
  - Validation of input parameters.

Environment Variables:
  - CONSUL_HOST: Consul agent hostname (default: 127.0.0.1)
  - CONSUL_PORT: Consul agent port (default: 10006 for SutazAI system)
  - CONSUL_SCHEME: HTTP scheme (default: http)

Exit Codes:
  - 0: Success (service registered or already exists)
  - 1: Registration/connection failure
  - 2: Invalid arguments or missing dependencies

Notes:
  - Consul Agent must be reachable on configured host/port.
  - Service ID is derived as f"{name}-{address}-{port}" to ensure uniqueness across instances.
  - Default port changed to 10006 to match SutazAI system port registry.
  - Retry logic handles transient network issues and Consul startup delays.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import random
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


def retry_with_exponential_backoff(func, max_retries=3, base_delay=1.0):
    """
    Retry a function with exponential backoff.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for backoff
        
    Returns:
        Function result on success
        
    Raises:
        Last exception encountered after all retries failed
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"[WARN] {ts()} Attempt {attempt + 1}/{max_retries + 1} failed: {e}. Retrying in {delay:.2f}s...")
                time.sleep(delay)
            else:
                print(f"[ERROR] {ts()} All {max_retries + 1} attempts failed. Last error: {e}")
    
    raise last_exception


def get_consul_client() -> consul.Consul:
    """
    Create and return a Consul client using environment variables or defaults.
    Uses localhost:10006 as per SutazAI system configuration.
    """
    host = os.getenv("CONSUL_HOST", "127.0.0.1")
    port_env = os.getenv("CONSUL_PORT", "10006")  # Updated to match SutazAI port registry
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

    def connect_to_consul():
        c = get_consul_client()
        # Test connection and fetch current services via local agent
        services = c.agent.services()
        return c, services
    
    try:
        c, services = retry_with_exponential_backoff(connect_to_consul, max_retries=3, base_delay=2.0)
        print(f"[INFO] {ts()} Successfully connected to Consul")
    except Exception as e:
        print(f"[ERROR] {ts()} Failed to connect to Consul Agent after retries: {e}", file=sys.stderr)
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
    def register_service():
        c.agent.service.register(
            name=args.service_name,
            service_id=service_id,
            address=args.service_address,
            port=args.service_port,
            tags=tags or None,
        )
        return service_id
    
    try:
        result_service_id = retry_with_exponential_backoff(register_service, max_retries=3, base_delay=1.0)
        print(f"[INFO] {ts()} Registered service: {result_service_id} ({args.service_name}@{args.service_address}:{args.service_port})")
        return 0
    except Exception as e:
        print(f"[ERROR] {ts()} Registration failed after retries: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

