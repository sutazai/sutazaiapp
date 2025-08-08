#!/usr/bin/env python3
import os
import json
from contextlib import contextmanager

os.environ.setdefault('JARVIS_FAKE', '1')
os.environ.setdefault('CONSUL_ENABLE', '0')

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
SVC = ROOT / 'services' / 'jarvis'
for p in (ROOT, SVC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fastapi.testclient import TestClient
from services.jarvis.main import app


@contextmanager
def client_ctx():
    with TestClient(app) as client:
        yield client


def assert_ok(resp, msg):
    if not (200 <= resp.status_code < 300):
        raise SystemExit(f"{msg}: HTTP {resp.status_code} -> {resp.text}")


def main():
    with client_ctx() as client:
        # Health
        r = client.get("/health")
        assert_ok(r, "GET /health failed")
        data = r.json()
        assert data.get('status') == 'healthy', f"Unexpected health: {data}"

        r = client.get("/jarvis/health")
        assert_ok(r, "GET /jarvis/health failed")

        # Task plan
        r = client.post("/jarvis/task/plan", json={
            "command": "ping",
            "context": {"x": 1},
            "voice_enabled": False,
            "plugins": []
        })
        assert_ok(r, "POST /jarvis/task/plan failed")
        data = r.json()
        assert data.get('status') in ("completed", "processing"), data
        assert 'result' in data, data

        # Voice process (send small dummy payload)
        files = {"audio": ("test.webm", b"dummy", "audio/webm")}
        r = client.post("/jarvis/voice/process", files=files)
        assert_ok(r, "POST /jarvis/voice/process failed")

        # Agents listing for static UI
        r = client.get("/api/agents")
        assert_ok(r, "GET /api/agents failed")

        # WebSocket
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"command": "hello ws", "context": {}, "voice_enabled": False})
            msg = ws.receive_json()
            assert isinstance(msg, dict) and 'result' in msg

    print("Jarvis smoke tests: OK")


if __name__ == '__main__':
    main()
