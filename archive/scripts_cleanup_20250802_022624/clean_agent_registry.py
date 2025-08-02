#!/usr/bin/env python3
"""
Clean up agent registry to remove fantasy elements and focus on practical capabilities
"""

import json
import re
from pathlib import Path

# Agents to remove (too fantasy-focused)
REMOVE_AGENTS = [
    'agi-system-architect',
    'deep-learning-coordinator-manager',
    'autonomous-system-controller'
]

# Fantasy term replacements for descriptions
REPLACEMENTS = {
    r'automation/advanced automation': 'automation',
    r'automation': 'automation',
    r'advanced automation': 'advanced automation',
    r'system_state': 'system state',
    r'coordinator': 'system',
    r'cognitive': 'processing',
    r'processing intelligence': 'machine learning',
    r'emergent': 'optimized',
    r'transfer_learning': 'transfer learning',
    r'self-improving': 'continuously improving',
    r'recursive continuous_improvement': 'continuous optimization',
    r'evolution': 'improvement',
    r'cognitive architecture': 'system architecture',
    r'processing plasticity': 'model adaptation',
    r'synaptic': 'connection',
    r'system_state modeling': 'state monitoring',
}

def clean_description(description):
    """Remove fantasy elements from agent description"""
    cleaned = description
    for pattern, replacement in REPLACEMENTS.items():
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    return cleaned

def clean_agent_registry():
    """Clean the agent registry file"""
    registry_path = Path('/opt/sutazaiapp/agents/agent_registry.json')
    
    # Load registry
    with open(registry_path, 'r') as f:
        registry = json.load(f)
    
    # Remove fantasy agents
    for agent in REMOVE_AGENTS:
        if agent in registry['agents']:
            print(f"Removing fantasy agent: {agent}")
            del registry['agents'][agent]
    
    # Clean descriptions
    for agent_name, agent_data in registry['agents'].items():
        if 'description' in agent_data:
            original = agent_data['description']
            cleaned = clean_description(original)
            if cleaned != original:
                print(f"Cleaning description for: {agent_name}")
                agent_data['description'] = cleaned
    
    # Update specific agent descriptions to be more practical
    practical_updates = {
        'ai-agent-orchestrator': {
            'description': """Use this agent when you need to:

- Coordinate multi-agent workflows and task distribution
- Manage agent discovery and registration
- Handle distributed task execution across agents
- Implement agent communication protocols
- Design workflow execution plans
- Monitor agent performance and health
- Manage agent lifecycle (start, stop, restart)
- Implement load balancing across agents
- Handle agent failover and recovery
- Create agent collaboration patterns
- Design task routing mechanisms
- Implement agent state synchronization
- Build event-driven architectures
- Create agent monitoring dashboards
- Design agent deployment strategies
- Implement agent version management
- Build agent performance benchmarks
- Design agent cost optimization
- Implement resource allocation
- Create debugging tools

Do NOT use this agent for:
- Simple single-agent tasks
- Direct code implementation
- Infrastructure management
- Testing individual components

This agent specializes in orchestrating multi-agent systems for efficient task distribution."""
        },
        'senior-ai-engineer': {
            'description': """Use this agent when you need to:

- Design and implement ML/AI pipelines
- Build RAG (Retrieval Augmented Generation) systems
- Integrate various LLMs and AI models
- Implement machine learning workflows
- Build model training and evaluation systems
- Create embeddings and vector databases
- Implement semantic search systems
- Design model serving infrastructure
- Build model monitoring systems
- Create A/B testing for models
- Implement model versioning
- Build ML debugging tools
- Design data preprocessing pipelines
- Create model deployment strategies
- Implement MLOps practices
- Build model registries
- Design experimentation platforms
- Create performance benchmarks
- Implement cost optimization

Do NOT use this agent for:
- Frontend development
- Backend API development
- Infrastructure management
- Basic data analysis

This agent specializes in ML/AI engineering and model deployment."""
        }
    }
    
    # Apply practical updates
    for agent_name, updates in practical_updates.items():
        if agent_name in registry['agents']:
            registry['agents'][agent_name].update(updates)
    
    # Save cleaned registry
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    
    print(f"\nCleaned agent registry saved to {registry_path}")
    print(f"Removed {len(REMOVE_AGENTS)} fantasy agents")
    print(f"Updated {len(registry['agents'])} agent descriptions")

if __name__ == "__main__":
    clean_agent_registry()