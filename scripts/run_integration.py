#!/usr/bin/env python3
import subprocess
import shutil
import json
import os
from datetime import datetime, timezone

REPORT_DIR = os.path.join('reports', 'cleanup')
INTEG_FILE = os.path.join(REPORT_DIR, 'integration_results.jsonl')


def now():
    return datetime.now(timezone.utc).isoformat()


def write_event(event):
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(INTEG_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(event) + "\n")


def run_cmd(cmd, timeout=600):
    try:
        cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, text=True)
        return cp.returncode, cp.stdout, cp.stderr
    except FileNotFoundError as e:
        return 127, '', str(e)
    except subprocess.TimeoutExpired as e:
        return 124, e.stdout or '', e.stderr or 'timeout'


def record_suite(name, ok, stdout='', stderr='', extra=None):
    ev = {
        "ts": now(),
        "phase": "integration",
        "task_id": f"CLN-{datetime.now():%Y%m%d}-{name}",
        "suite": name,
        "passed": 1 if ok else 0,
        "failed": 0 if ok else 1,
        "duration_s": 0,
        "artifacts": [INTEG_FILE],
        "notes": (stderr or '')[:4000],
    }
    if extra:
        ev.update(extra)
    write_event(ev)


def try_compose_up():
    for exe in (("docker", "compose"), ("docker-compose",)):
        if shutil.which(exe[0]):
            cmd = list(exe) + ["up", "-d"]
            code, out, err = run_cmd(cmd, timeout=900)
            record_suite("compose_up", code == 0, out, err)
            return
    record_suite("compose_up", False, stderr="docker not available")


def try_health_checks():
    curl = shutil.which('curl')
    if not curl:
        record_suite("health_checks", False, stderr="curl not available")
        return
    endpoints = [
        ("backend_health", [curl, "-sf", "http://localhost:10010/health"]),
        ("ollama_tags", [curl, "-sf", "http://localhost:10104/api/tags"]),
        ("consul_leader", [curl, "-sf", "http://localhost:10006/v1/status/leader"]),
    ]
    all_ok = True
    results = {}
    for name, cmd in endpoints:
        code, out, err = run_cmd(cmd, timeout=30)
        ok = code == 0
        all_ok = all_ok and ok
        results[name] = {"ok": ok, "out": (out or '')[:500], "err": (err or '')[:500]}
    record_suite("health_checks", all_ok, extra={"results": results})


def try_pytest():
    if not shutil.which('pytest'):
        record_suite("pytest", False, stderr="pytest not available")
        return
    cmd = ["pytest", "-v", "backend/tests", "--cov=backend", "--cov-fail-under=80"]
    code, out, err = run_cmd(cmd, timeout=1800)
    record_suite("pytest", code == 0, out, err)


def try_linters():
    # black --check
    if shutil.which('black'):
        code, out, err = run_cmd(["black", "--check", "backend", "agents", "services"], timeout=600)
        record_suite("black_check", code == 0, out, err)
    else:
        record_suite("black_check", False, stderr="black not available")

    # flake8
    if shutil.which('flake8'):
        code, out, err = run_cmd(["flake8", "backend"], timeout=600)
        record_suite("flake8", code == 0, out, err)
    else:
        record_suite("flake8", False, stderr="flake8 not available")

    # mypy
    if shutil.which('mypy'):
        code, out, err = run_cmd(["mypy", "backend"], timeout=1200)
        record_suite("mypy", code == 0, out, err)
    else:
        record_suite("mypy", False, stderr="mypy not available")


def try_bandit():
    if not shutil.which('bandit'):
        record_suite("bandit", False, stderr="bandit not available")
        return
    code, out, err = run_cmd(["bandit", "-r", "backend", "agents", "services", "-f", "json"], timeout=1800)
    ok = code == 0
    # Store raw output for audit
    out_path = os.path.join(REPORT_DIR, 'bandit.json')
    os.makedirs(REPORT_DIR, exist_ok=True)
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(out)
    except Exception:
        pass
    record_suite("bandit", ok, out[:500], err[:500], extra={"artifact_bandit": out_path})


def main():
    try_compose_up()
    try_health_checks()
    try_pytest()
    try_linters()
    try_bandit()
    print(f"Integration results written to {INTEG_FILE}")


if __name__ == '__main__':
    main()

