#!/usr/bin/env python3
"""
Example: Using Universal Agents

This example shows how to use the deployed universal agents
that are independent of Claude.
"""

import os
import sys
import requests
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.ai_agents.universal_agent_factory import universal_agent_factory


def example_direct_usage():
    """Example of using agents directly through the factory."""
    print("=== Direct Agent Usage ===")
    
    # List available agents
    agents = universal_agent_factory.list_available_agents()
    print(f"Available agents: {agents}")
    
    # Get a specific agent
    agent_name = "semgrep-security-analyzer"
    agent = universal_agent_factory.create_agent(agent_name)
    
    if agent:
        print(f"\nAgent: {agent['name']}")
        print(f"Capabilities: {', '.join(agent['capabilities'])}")
        print(f"Provider: {agent['provider']}")
        print("\nSystem Prompt Preview:")
        print(agent['system_prompt'][:200] + "...")


def example_api_usage():
    """Example of using agents through their API."""
    print("\n\n=== API Agent Usage ===")
    
    # Assuming the agent is running on port 8001
    agent_url = "http://localhost:8001"
    
    try:
        # Get agent info
        response = requests.get(f"{agent_url}/info")
        if response.status_code == 200:
            info = response.json()
            print(f"Agent: {info['name']}")
            print(f"Status: {info['status']}")
            print(f"Model: {info['model_provider']}/{info['model_name']}")
        
        # Send a chat request
        chat_request = {
            "messages": [
                {"role": "user", "content": "Analyze this Python code for security issues: exec(user_input)"}
            ]
        }
        
        response = requests.post(f"{agent_url}/chat", json=chat_request)
        if response.status_code == 200:
            result = response.json()
            print(f"\nAgent Response:")
            print(result['response'])
    
    except requests.exceptions.ConnectionError:
        print("Agent API not running. Start with: docker-compose -f docker-compose-universal-agents.yml up")


def example_ollama_usage():
    """Example of using agents with Ollama directly."""
    print("\n\n=== Ollama Agent Usage ===")
    
    try:
        # Use the agent model with Ollama
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "sutazai_semgrep-security-analyzer",
                "prompt": "What are the top 5 security vulnerabilities to check for in Python code?",
                "stream": False
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            print("Ollama Response:")
            print(result['response'])
    
    except requests.exceptions.ConnectionError:
        print("Ollama not running. Start Ollama and build models with: cd ollama/models && ./build_all_models.sh")


    
    try:
        response = requests.post(
            "http://localhost:4000/chat/completions",
            json={
                "model": "sutazai/semgrep-security-analyzer",
                "messages": [
                    {"role": "user", "content": "How do I prevent SQL injection in Python?"}
                ]
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            print(result['choices'][0]['message']['content'])
    
    except requests.exceptions.ConnectionError:


def main():
    """Run all examples."""
    print("SutazAI Universal Agents - Usage Examples")
    print("=" * 50)
    
    # Show how to use agents in different ways
    example_direct_usage()
    example_api_usage()
    example_ollama_usage()
    
    print("\n\nThese agents are now completely independent of Claude!")
    print("You can run them with local models using Ollama or any OpenAI-compatible API.")


if __name__ == "__main__":
    main()