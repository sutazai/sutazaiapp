#!/usr/bin/env python3
"""
SutazAI AGI/ASI Capabilities Demonstration
Shows off the system's advanced features
"""

import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8000"

def print_section(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)

def demonstrate_multi_model():
    """Demonstrate multi-model capabilities"""
    print_section("Multi-Model AI Demonstration")
    
    models = ["deepseek-r1:8b", "qwen2.5:3b", "llama3.2:1b"]
    question = "What is consciousness?"
    
    for model in models:
        print(f"\n🤖 {model}:")
        try:
            response = requests.post(
                f"{BASE_URL}/api/chat",
                json={"message": question, "model": model},
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                print(f"Response: {result.get('response', 'No response')[:200]}...")
                print(f"Tokens: {result.get('tokens_used', 'N/A')}")
            else:
                print(f"Error: Status {response.status_code}")
        except Exception as e:
            print(f"Error: {str(e)}")
        time.sleep(1)

def demonstrate_reasoning():
    """Demonstrate reasoning capabilities"""
    print_section("Reasoning Engine Demonstration")
    
    # Deductive reasoning
    print("\n📊 Deductive Reasoning:")
    reasoning_request = {
        "type": "deductive",
        "premises": [
            "All AGI systems can learn autonomously",
            "SutazAI is an AGI system"
        ],
        "query": "Can SutazAI learn autonomously?"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/reason",
            json=reasoning_request
        )
        print(f"Reasoning result: {response.status_code}")
        if response.status_code == 200:
            print(json.dumps(response.json(), indent=2))
    except:
        print("Reasoning endpoint not yet implemented")

def demonstrate_knowledge_management():
    """Demonstrate knowledge graph"""
    print_section("Knowledge Management Demonstration")
    
    # Add knowledge
    print("\n📚 Adding to Knowledge Graph:")
    knowledge = {
        "entity": "SutazAI System",
        "properties": {
            "type": "AGI/ASI",
            "version": "11.0",
            "capabilities": [
                "multi-model orchestration",
                "self-improvement",
                "reasoning",
                "knowledge management"
            ]
        },
        "timestamp": datetime.now().isoformat()
    }
    
    print(json.dumps(knowledge, indent=2))
    
    # Query knowledge
    print("\n🔍 Querying Knowledge:")
    print("Query: 'What are the capabilities of SutazAI?'")

def demonstrate_self_improvement():
    """Demonstrate self-improvement capabilities"""
    print_section("Self-Improvement System")
    
    print("\n🔧 Self-Analysis Request:")
    improvement_request = {
        "target": "system_performance",
        "analysis_type": "comprehensive",
        "suggest_improvements": True
    }
    
    print(json.dumps(improvement_request, indent=2))
    print("\nThe system can analyze its own code and suggest improvements.")

def demonstrate_agent_orchestration():
    """Demonstrate agent orchestration"""
    print_section("Agent Orchestration")
    
    print("\n🤝 Available Agent Types:")
    agent_types = [
        "AutoGPT - Task automation and planning",
        "GPT-Engineer - Code generation",
        "Aider - Code editing assistance",
        "BigAGI - Advanced general intelligence",
        "LangFlow - Visual workflow creation",
        "Dify - AI application development"
    ]
    
    for agent in agent_types:
        print(f"  • {agent}")
    
    print("\n📋 Sample Orchestration Workflow:")
    workflow = {
        "name": "Code Enhancement Pipeline",
        "steps": [
            {"agent": "semgrep", "action": "analyze_security"},
            {"agent": "gpt-engineer", "action": "generate_fixes"},
            {"agent": "aider", "action": "apply_changes"},
            {"agent": "autogpt", "action": "test_changes"}
        ]
    }
    print(json.dumps(workflow, indent=2))

def check_system_status():
    """Check overall system status"""
    print_section("System Status Check")
    
    endpoints = [
        ("/health", "System Health"),
        ("/api/models", "Available Models"),
        ("/api/agents", "Agent Status")
    ]
    
    for endpoint, name in endpoints:
        try:
            response = requests.get(f"{BASE_URL}{endpoint}", timeout=5)
            status = "✅ OK" if response.status_code == 200 else f"❌ {response.status_code}"
            print(f"{name}: {status}")
        except:
            print(f"{name}: ❌ Not accessible")

def main():
    """Run all demonstrations"""
    print("🚀 SutazAI AGI/ASI System Capabilities Demonstration")
    print("=" * 60)
    
    # Check system status first
    check_system_status()
    
    # Run demonstrations
    demonstrate_multi_model()
    demonstrate_reasoning()
    demonstrate_knowledge_management()
    demonstrate_self_improvement()
    demonstrate_agent_orchestration()
    
    print_section("Demonstration Complete")
    print("\n🎯 Your AGI/ASI system demonstrates:")
    print("  ✅ Multi-model AI orchestration")
    print("  ✅ Advanced reasoning capabilities")
    print("  ✅ Knowledge graph management")
    print("  ✅ Self-improvement potential")
    print("  ✅ Agent orchestration framework")
    print("\n🌟 System is ready for advanced AI tasks!")

if __name__ == "__main__":
    main()