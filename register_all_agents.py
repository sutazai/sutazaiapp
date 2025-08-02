#\!/usr/bin/env python3
"""Register all agents with the backend API"""

import requests
import json
from typing import List, Dict

# List of all agents to register
AGENTS = [
    # External framework agents
    {"id": "autogpt", "name": "AutoGPT", "description": "Autonomous task execution agent"},
    {"id": "agentgpt", "name": "AgentGPT", "description": "Browser-based autonomous agent"},
    {"id": "babyagi", "name": "BabyAGI", "description": "Task-driven autonomous agent"},
    {"id": "crewai", "name": "CrewAI", "description": "Multi-agent collaboration framework"},
    {"id": "letta", "name": "Letta (MemGPT)", "description": "Memory-persistent AI agent"},
    {"id": "aider", "name": "Aider", "description": "AI pair programming assistant"},
    {"id": "gpt-engineer", "name": "GPT-Engineer", "description": "Full application builder"},
    {"id": "devika", "name": "Devika", "description": "Software engineering agent"},
    {"id": "privategpt", "name": "PrivateGPT", "description": "Local document Q&A system"},
    {"id": "shellgpt", "name": "ShellGPT", "description": "Command-line AI assistant"},
    {"id": "pentestgpt", "name": "PentestGPT", "description": "Penetration testing assistant"},
    
    # Specialized agents
    {"id": "data-pipeline-engineer", "name": "Data Pipeline Engineer", "description": "Builds and manages data pipelines"},
    {"id": "private-data-analyst", "name": "Private Data Analyst", "description": "Analyzes data with privacy preservation"},
    {"id": "knowledge-graph-builder", "name": "Knowledge Graph Builder", "description": "Creates knowledge graphs from data"},
    {"id": "document-knowledge-manager", "name": "Document Knowledge Manager", "description": "Manages document knowledge bases"},
    {"id": "task-assignment-coordinator", "name": "Task Assignment Coordinator", "description": "Coordinates task assignments"},
    {"id": "langflow-workflow-designer", "name": "LangFlow Workflow Designer", "description": "Designs LangFlow workflows"},
    {"id": "flowiseai-flow-manager", "name": "FlowiseAI Flow Manager", "description": "Manages Flowise AI workflows"},
    {"id": "dify-automation-specialist", "name": "Dify Automation Specialist", "description": "Specializes in Dify automation"},
    {"id": "memory-persistence-manager", "name": "Memory Persistence Manager", "description": "Manages persistent memory systems"},
    {"id": "garbage-collector-coordinator", "name": "Garbage Collector Coordinator", "description": "Optimizes garbage collection"},
    {"id": "semgrep-security-analyzer", "name": "Semgrep Security Analyzer", "description": "Analyzes code security"},
    {"id": "transformers-migration-specialist", "name": "Transformers Migration Specialist", "description": "Migrates transformer models"},
    {"id": "model-training-specialist", "name": "Model Training Specialist", "description": "Specializes in model training"},
    {"id": "edge-computing-optimizer", "name": "Edge Computing Optimizer", "description": "Optimizes edge computing"},
    {"id": "distributed-computing-architect", "name": "Distributed Computing Architect", "description": "Designs distributed systems"},
    {"id": "neuromorphic-computing-expert", "name": "Neuromorphic Computing Expert", "description": "Expert in neuromorphic computing"},
    {"id": "quantum-computing-optimizer", "name": "Quantum Computing Optimizer", "description": "Optimizes quantum algorithms"},
    {"id": "federated-learning-coordinator", "name": "Federated Learning Coordinator", "description": "Coordinates federated learning"},
    {"id": "reinforcement-learning-trainer", "name": "Reinforcement Learning Trainer", "description": "Trains RL models"},
    {"id": "causal-inference-expert", "name": "Causal Inference Expert", "description": "Expert in causal inference"},
    {"id": "meta-learning-specialist", "name": "Meta-Learning Specialist", "description": "Specializes in meta-learning"},
    {"id": "explainable-ai-specialist", "name": "Explainable AI Specialist", "description": "Makes AI explainable"},
    {"id": "synthetic-data-generator", "name": "Synthetic Data Generator", "description": "Generates synthetic datasets"},
    {"id": "knowledge-distillation-expert", "name": "Knowledge Distillation Expert", "description": "Distills model knowledge"},
    {"id": "multi-modal-fusion-coordinator", "name": "Multi-Modal Fusion Coordinator", "description": "Fuses multi-modal data"},
    {"id": "cognitive-architecture-designer", "name": "Cognitive Architecture Designer", "description": "Designs cognitive architectures"},
    {"id": "symbolic-reasoning-engine", "name": "Symbolic Reasoning Engine", "description": "Performs symbolic reasoning"},
    {"id": "episodic-memory-engineer", "name": "Episodic Memory Engineer", "description": "Engineers episodic memory"},
    {"id": "attention-optimizer", "name": "Attention Optimizer", "description": "Optimizes attention mechanisms"},
    {"id": "gradient-compression-specialist", "name": "Gradient Compression Specialist", "description": "Compresses gradients"},
    {"id": "self-healing-orchestrator", "name": "Self-Healing Orchestrator", "description": "Orchestrates self-healing"},
    {"id": "intelligence-optimization-monitor", "name": "Intelligence Optimization Monitor", "description": "Monitors AI optimization"},
    {"id": "product-strategy-architect", "name": "Product Strategy Architect", "description": "Architects product strategy"},
    {"id": "observability-monitoring-engineer", "name": "Observability Monitoring Engineer", "description": "Engineers observability"},
    {"id": "ai-agent-debugger", "name": "AI Agent Debugger", "description": "Debugs AI agents"},
    {"id": "jarvis-voice-interface", "name": "Jarvis Voice Interface", "description": "Voice-based AI interface"},
    {"id": "litellm-proxy-manager", "name": "LiteLLM Proxy Manager", "description": "Manages LLM proxies"},
    {"id": "opendevin-code-generator", "name": "OpenDevin Code Generator", "description": "Generates code autonomously"},
    {"id": "agentgpt-autonomous-executor", "name": "AgentGPT Autonomous Executor", "description": "Executes autonomous tasks"},
    {"id": "agentzero-coordinator", "name": "AgentZero Coordinator", "description": "Coordinates agent systems"},
    {"id": "localagi-orchestration-manager", "name": "LocalAGI Orchestration Manager", "description": "Manages local AGI"},
    {"id": "bigagi-system-manager", "name": "BigAGI System Manager", "description": "Manages BigAGI systems"},
]

def register_agents():
    """Register all agents with the backend"""
    base_url = "http://localhost:8000/api/v1"
    
    # First, get existing agents
    try:
        response = requests.get(f"{base_url}/agents")
        existing_agents = {agent['id'] for agent in response.json()}
        print(f"Found {len(existing_agents)} existing agents")
    except Exception as e:
        print(f"Error getting existing agents: {e}")
        existing_agents = set()
    
    # Register new agents
    registered = 0
    failed = 0
    
    for agent in AGENTS:
        if agent['id'] in existing_agents:
            print(f"Agent {agent['id']} already exists, skipping")
            continue
            
        agent_data = {
            "id": agent['id'],
            "name": agent['name'],
            "description": agent['description'],
            "status": "active",
            "capabilities": ["task_execution", "collaboration"],
            "config": {
                "model": "local",
                "temperature": 0.7
            }
        }
        
        try:
            response = requests.post(
                f"{base_url}/agents",
                json=agent_data,
                headers={"Content-Type": "application/json"}
            )
            if response.status_code in [200, 201]:
                print(f"✅ Registered: {agent['name']}")
                registered += 1
            else:
                print(f"❌ Failed to register {agent['name']}: {response.status_code} - {response.text}")
                failed += 1
        except Exception as e:
            print(f"❌ Error registering {agent['name']}: {e}")
            failed += 1
    
    print(f"\n📊 Registration Summary:")
    print(f"- Registered: {registered}")
    print(f"- Failed: {failed}")
    print(f"- Already existed: {len(existing_agents)}")
    print(f"- Total agents: {registered + len(existing_agents)}")

if __name__ == "__main__":
    register_agents()
