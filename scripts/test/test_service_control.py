#!/usr/bin/env python3
"""Test script to verify service control through the automation Coordinator API"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer sutazai-default-token"
}

test_commands = [
    "list all services",
    "check system status",
    "show resource usage",
    "show me the status of ollama",
    "restart n8n service",
    "stop flowise",
    "start flowise"
]

def test_service_command(command):
    """Test a service control command"""
    print(f"\n{'='*60}")
    print(f"Testing command: {command}")
    print('='*60)
    
    payload = {
        "input_data": {"text": command},
        "reasoning_type": "strategic"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/coordinator/think",
            headers=HEADERS,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            output = result.get("result", {}).get("output", "No output")
            reasoning = result.get("reasoning_type", "unknown")
            agents = result.get("agents_used", [])
            
            print(f"Status: SUCCESS")
            print(f"Reasoning Type: {reasoning}")
            print(f"Agents Used: {', '.join(agents)}")
            print(f"\nOutput:\n{output[:500]}...")  # Show first 500 chars
            
            # Check if it was processed by service controller
            if "unified_service_controller" in agents:
                print("\n✅ Command processed by Service Controller!")
            else:
                print("\n⚠️ Command NOT processed by Service Controller")
                
        else:
            print(f"Status: FAILED (HTTP {response.status_code})")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Status: ERROR")
        print(f"Exception: {str(e)}")
    
    time.sleep(2)  # Small delay between requests

def main():
    print("SutazAI Service Control Test")
    print("Testing service control commands through automation Coordinator API...")
    
    for command in test_commands:
        test_service_command(command)
    
    print("\n" + "="*60)
    print("Test completed!")
    print("="*60)

if __name__ == "__main__":
    main()