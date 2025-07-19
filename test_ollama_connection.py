#!/usr/bin/env python3
import requests
import time
import json

print("Testing Ollama connection...")

# Test 1: Basic connectivity
try:
    response = requests.get("http://localhost:11434/api/tags", timeout=5)
    print(f"✓ Ollama API accessible: {response.status_code}")
    models = response.json()
    print(f"✓ Available models: {[m['name'] for m in models.get('models', [])]}")
except Exception as e:
    print(f"✗ Failed to connect to Ollama: {e}")

# Test 2: Generate with different timeouts
test_prompt = "What is 2+2?"
model = "llama3.2:1b"

for timeout in [5, 10, 15]:
    print(f"\nTesting with {timeout}s timeout...")
    try:
        start = time.time()
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": test_prompt,
                "stream": False,
                "options": {
                    "num_predict": 30,
                    "temperature": 0.7
                }
            },
            timeout=timeout
        )
        elapsed = time.time() - start
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Success in {elapsed:.2f}s: {result['response'][:50]}...")
        else:
            print(f"✗ HTTP {response.status_code}: {response.text}")
    except requests.exceptions.Timeout:
        print(f"✗ Timeout after {timeout}s")
    except Exception as e:
        print(f"✗ Error: {e}")

# Test 3: Test with even shorter response
print("\nTesting with very short response (10 tokens)...")
try:
    start = time.time()
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": "Hi",
            "stream": False,
            "options": {
                "num_predict": 10,
                "temperature": 0.7
            }
        },
        timeout=10
    )
    elapsed = time.time() - start
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Success in {elapsed:.2f}s: {result['response']}")
        print(f"  Total duration: {result.get('total_duration', 0)/1e9:.2f}s")
        print(f"  Load duration: {result.get('load_duration', 0)/1e9:.2f}s")
        print(f"  Eval duration: {result.get('eval_duration', 0)/1e9:.2f}s")
except Exception as e:
    print(f"✗ Error: {e}")