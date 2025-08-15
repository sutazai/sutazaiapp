import os
import pytest
from fastapi.testclient import TestClient


os.environ.setdefault('JARVIS_FAKE', '1')
os.environ.setdefault('CONSUL_ENABLE', '0')

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
SVC = ROOT / 'services' / 'jarvis'
for p in (ROOT, SVC):
    if str(p) not in sys.path:
        # Path handled by pytest configuration)

from services.jarvis.main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json().get('status') == 'healthy'

    r = client.get('/jarvis/health')
    assert r.status_code == 200


def test_task_plan(client):
    r = client.post('/jarvis/task/plan', json={
        'command': 'ping',
        'context': {},
        'voice_enabled': False,
        'plugins': []
    })
    assert r.status_code == 200
    data = r.json()
    assert 'result' in data
    assert data.get('status') in ('completed', 'processing')


def test_voice_process(client):
    # Raw upload compatible in environments without python-multipart
    r = client.post('/jarvis/voice/process', data=b'dummy', headers={'Content-Type': 'application/octet-stream'})
    assert r.status_code == 200


def test_websocket(client):
    with client.websocket_connect('/ws') as ws:
        ws.send_json({'command': 'hello', 'context': {}, 'voice_enabled': False})
        msg = ws.receive_json()
        assert 'result' in msg

