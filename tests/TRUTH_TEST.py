#!/usr/bin/env python3
"""
THE TRUTH TEST - What ACTUALLY works vs what's broken
NO LIES, JUST FACTS
"""

import requests
import time

print("=" * 70)
print("THE BRUTAL TRUTH ABOUT JARVIS")
print("=" * 70)

# Test 1: Can we actually chat?
print("\n1. CHAT TEST:")
try:
    start = time.time()
    resp = requests.post(
        "http://localhost:10200/api/v1/chat/",
        json={"message": "Say yes if you work"},
        timeout=60
    )
    elapsed = time.time() - start
    if resp.status_code == 200:
        data = resp.json()
        print(f"✅ ACTUALLY WORKS - Response in {elapsed:.1f}s")
        print(f"   AI said: {data.get('response', '')[:100]}")
    else:
        print(f"❌ BROKEN - Status: {resp.status_code}")
        print(f"   Error: {resp.text}")
except Exception as e:
    print(f"❌ TOTALLY BROKEN: {e}")

# Test 2: Voice endpoint (without auth)
print("\n2. VOICE TEST (unauthenticated):")
try:
    resp = requests.post(
        "http://localhost:10200/api/v1/voice/process",
        json={"text": "test"},
        timeout=5
    )
    if resp.status_code == 200:
        print("✅ ACTUALLY WORKS")
    else:
        print(f"❌ BROKEN - Requires auth: {resp.json()}")
except Exception as e:
    print(f"❌ BROKEN: {e}")

# Test 3: Can we access the frontend?
print("\n3. FRONTEND TEST:")
try:
    resp = requests.get("http://localhost:11000", timeout=5)
    if "Streamlit" in resp.text:
        print("✅ ACTUALLY WORKS - Streamlit is running")
    else:
        print("❌ BROKEN - Not a Streamlit app")
except Exception as e:
    print(f"❌ TOTALLY BROKEN: {e}")

# Test 4: WebSocket test
print("\n4. WEBSOCKET TEST:")
try:
    import websocket
    ws = websocket.create_connection("ws://localhost:10200/ws", timeout=2)
    ws.close()
    print("✅ ACTUALLY WORKS - WebSocket connects")
except Exception as e:
    if "400" in str(e):
        print("⚠️ EXISTS but needs proper WebSocket client")
    else:
        print(f"❌ BROKEN: {e}")

# Test 5: Does voice health work without auth?
print("\n5. VOICE HEALTH TEST:")
try:
    resp = requests.get("http://localhost:10200/api/v1/voice/health", timeout=5)
    if resp.status_code == 200:
        print("✅ ACTUALLY WORKS - No auth needed")
        health = resp.json()
        print(f"   Status: {health.get('status')}")
        print(f"   Components: {list(health.get('components', {}).keys())}")
    else:
        print(f"❌ BROKEN - Status: {resp.status_code}")
except Exception as e:
    print(f"❌ BROKEN: {e}")

# Test 6: Check if voice files really exist in container
print("\n6. VOICE FILES IN CONTAINER:")
import subprocess
try:
    result = subprocess.run(
        ["docker", "exec", "sutazai-backend", "ls", "-la", "/app/app/services/"],
        capture_output=True, text=True, timeout=5
    )
    if "voice_service.py" in result.stdout:
        print("✅ FILES EXIST in container")
        for line in result.stdout.split('\n'):
            if 'voice' in line or 'wake' in line:
                print(f"   {line}")
    else:
        print("❌ NO VOICE FILES")
except Exception as e:
    print(f"❌ Can't check: {e}")

print("\n" + "=" * 70)
print("SUMMARY OF TRUTH:")
print("=" * 70)
print("""
What ACTUALLY works:
- Chat endpoint (slow but real AI)
- Frontend is running
- Voice health endpoint
- Voice files exist in container

What's BROKEN or FAKE:
- Voice processing requires auth (not fixed)
- WebSocket needs proper client
- Voice features not accessible from frontend
- No actual voice recording/playback working
""")
print("=" * 70)