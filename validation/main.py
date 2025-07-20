#!/usr/bin/env python3
"""
SutazAI System Validation Script

This script runs a suite of tests to validate the functionality
of the entire SutazAI system.
"""

import requests
import time

BACKEND_URL = "http://localhost:8000"

def run_test(test_name, test_function):
    print(f"Running test: {test_name}...")
    try:
        test_function()
        print(f"[SUCCESS] {test_name}")
    except Exception as e:
        print(f"[FAILURE] {test_name}: {e}")

def test_health_check():
    response = requests.get(f"{BACKEND_URL}/health")
    response.raise_for_status()
    assert response.json()["status"] == "healthy"

def test_get_models():
    response = requests.get(f"{BACKEND_URL}/api/models")
    response.raise_for_status()
    assert "models" in response.json()
    assert len(response.json()["models"]) > 0

def test_chat():
    response = requests.post(f"{BACKEND_URL}/api/chat", json={"message": "Hello", "model": "llama3"})
    response.raise_for_status()
    assert "response" in response.json()

def test_agent_status():
    response = requests.get(f"{BACKEND_URL}/api/agents")
    response.raise_for_status()
    assert len(response.json()) > 0

def test_system_status():
    response = requests.get(f"{BACKEND_URL}/api/system/status")
    response.raise_for_status()
    assert "agents" in response.json()
    assert "tasks" in response.json()

def main():
    print("--- SutazAI System Validation ---")
    run_test("Health Check", test_health_check)
    run_test("Get Models", test_get_models)
    run_test("Chat", test_chat)
    run_test("Agent Status", test_agent_status)
    run_test("System Status", test_system_status)
    print("--- Validation Complete ---")

if __name__ == "__main__":
    main()
