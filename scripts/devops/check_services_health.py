#!/usr/bin/env python3
"""
Infrastructure health verification and smoke tests (idempotent).

Usage example:
  python scripts/devops/check_services_health.py \
    --ollama localhost:10104 \
    --kong localhost:10005 \
    --consul localhost:10006 \
    --vector-range 10100-10103 \
    --redis localhost:6379 \
    --postgres localhost:5432 \
    --rabbitmq localhost:5672 \
    --prometheus localhost:9090 \
    --grafana localhost:3000

Notes:
  - Only checks provided endpoints; nothing is hardcoded.
  - Exits 0 if all provided services are reachable; non-zero otherwise.
  - Prints timestamped logs and basic latency measurements.
"""
from __future__ import annotations

import argparse
import socket
import sys
import time
from typing import Tuple, List
import urllib.request


def ts() -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S')


def parse_host_port(spec: str) -> Tuple[str, int]:
    host, _, port = spec.partition(":")
    if not host or not port:
        raise argparse.ArgumentTypeError("Must be in host:port format")
    try:
        p = int(port)
    except ValueError:
        raise argparse.ArgumentTypeError("Port must be an integer")
    if p <= 0 or p > 65535:
        raise argparse.ArgumentTypeError("Port must be 1-65535")
    return host, p


def parse_range(spec: str) -> Tuple[int, int]:
    a, _, b = spec.partition("-")
    if not a or not b:
        raise argparse.ArgumentTypeError("Range must be start-end")
    try:
        s, e = int(a), int(b)
    except ValueError:
        raise argparse.ArgumentTypeError("Range bounds must be integers")
    if s > e:
        raise argparse.ArgumentTypeError("Range start must be <= end")
    return s, e


def tcp_check(host: str, port: int, name: str, timeout: float = 2.5) -> bool:
    start = time.time()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            elapsed = int((time.time() - start) * 1000)
            print(f"[INFO] {ts()} {name} reachable at {host}:{port} (~{elapsed}ms)")
            return True
    except Exception as e:
        print(f"[ERROR] {ts()} {name} not reachable at {host}:{port} ({e})")
        return False


def http_check(url: str, name: str, timeout: float = 2.5) -> None:
    try:
        start = time.time()
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            elapsed = int((time.time() - start) * 1000)
            print(f"[INFO] {ts()} {name} {url} -> code={resp.status} time={elapsed}ms")
    except Exception as e:
        print(f"[WARN] {ts()} {name} {url} check failed: {e}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Check reachability of core infrastructure services")
    p.add_argument("--ollama", type=parse_host_port, help="host:port for Ollama/TinyLlama")
    p.add_argument("--kong", type=parse_host_port, help="host:port for Kong")
    p.add_argument("--consul", type=parse_host_port, help="host:port for Consul HTTP API")
    p.add_argument("--vector-range", type=parse_range, help="start-end port range for vector DBs on localhost")
    p.add_argument("--redis", type=parse_host_port, help="host:port for Redis")
    p.add_argument("--postgres", type=parse_host_port, help="host:port for PostgreSQL")
    p.add_argument("--rabbitmq", type=parse_host_port, help="host:port for RabbitMQ")
    p.add_argument("--prometheus", type=parse_host_port, help="host:port for Prometheus")
    p.add_argument("--grafana", type=parse_host_port, help="host:port for Grafana")
    return p


def main(argv: List[str]) -> int:
    args = build_parser().parse_args(argv)
    failures = 0

    if args.ollama:
        host, port = args.ollama
        if tcp_check(host, port, "Ollama"):
            http_check(f"http://{host}:{port}/", "Ollama-root")
        else:
            failures += 1

    if args.kong:
        host, port = args.kong
        if tcp_check(host, port, "Kong"):
            http_check(f"http://{host}:{port}/", "Kong-root")
        else:
            failures += 1

    if args.consul:
        host, port = args.consul
        if tcp_check(host, port, "Consul"):
            http_check(f"http://{host}:{port}/v1/status/leader", "Consul-leader")
        else:
            failures += 1

    if args.vector_range:
        start, end = args.vector_range
        for p in range(start, end + 1):
            if not tcp_check("localhost", p, f"VectorDB:{p}"):
                failures += 1

    if args.redis:
        host, port = args.redis
        if not tcp_check(host, port, "Redis"):
            failures += 1

    if args.postgres:
        host, port = args.postgres
        if not tcp_check(host, port, "PostgreSQL"):
            failures += 1

    if args.rabbitmq:
        host, port = args.rabbitmq
        if not tcp_check(host, port, "RabbitMQ"):
            failures += 1

    if args.prometheus:
        host, port = args.prometheus
        if tcp_check(host, port, "Prometheus"):
            http_check(f"http://{host}:{port}/-/ready", "Prometheus-ready")
        else:
            failures += 1

    if args.grafana:
        host, port = args.grafana
        if tcp_check(host, port, "Grafana"):
            http_check(f"http://{host}:{port}/login", "Grafana-login")
        else:
            failures += 1

    if failures == 0:
        print(f"[INFO] {ts()} All checks passed")
        return 0
    else:
        print(f"[ERROR] {ts()} {failures} checks failed")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

