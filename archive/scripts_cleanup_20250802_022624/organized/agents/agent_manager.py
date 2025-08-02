#!/usr/bin/env python3
"""
SutazAI Agent Manager - Complete Agent Control System
=====================================================

Manage, monitor, and utilize all 39 AI agents in your SutazAI ecosystem.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import asyncio
from datetime import datetime

class SutazAIAgentManager:
    """Complete agent management system for SutazAI"""
    
    def __init__(self):
        self.project_agents_dir = Path("/opt/sutazaiapp/.claude/agents")
        self.personal_agents_dir = Path("/root/.claude/agents")
        self.agents = self._load_all_agents()
    
    def _load_all_agents(self) -> Dict[str, Dict]:
        """Load all agents from both directories"""
        agents = {}
        
        # Load project agents (these take precedence)
        for agent_file in self.project_agents_dir.glob("*.md"):
            agent_data = self._parse_agent_file(agent_file)
            if agent_data:
                agents[agent_data['name']] = {
                    **agent_data,
                    'location': 'project',
                    'file_path': str(agent_file)
                }
        
        # Load personal agents (only if not overridden)
        for agent_file in self.personal_agents_dir.glob("*.md"):
            agent_data = self._parse_agent_file(agent_file)
            if agent_data and agent_data['name'] not in agents:
                agents[agent_data['name']] = {
                    **agent_data,
                    'location': 'personal',
                    'file_path': str(agent_file)
                }
        
        return agents
    
    def _parse_agent_file(self, file_path: Path) -> Optional[Dict]:
        """Parse an agent markdown file"""
        try:
            content = file_path.read_text()
            lines = content.split('\n')
            
            agent_data = {
                'name': None,
                'model': None,
                'description': None,
                'capabilities': []
            }
            
            for line in lines[:20]:  # Check first 20 lines
                if line.startswith('name:'):
                    agent_data['name'] = line.split(':', 1)[1].strip()
                elif line.startswith('model:'):
                    agent_data['model'] = line.split(':', 1)[1].strip()
                elif line.startswith('description:'):
                    agent_data['description'] = line.split(':', 1)[1].strip()
            
            # Extract capabilities from the content
            if 'capabilities:' in content:
                capabilities_section = content.split('capabilities:')[1].split('\n\n')[0]
                agent_data['capabilities'] = [
                    cap.strip('- ').strip() 
                    for cap in capabilities_section.split('\n') 
                    if cap.strip().startswith('-')
                ]
            
            return agent_data if agent_data['name'] else None
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None
    
    def list_all_agents(self, group_by: str = 'category'):
        """List all agents grouped by category"""
        print("\n🤖 SutazAI Complete Agent Registry")
        print("=" * 80)
        print(f"Total Agents: {len(self.agents)}")
        print("=" * 80)
        
        if group_by == 'category':
            categories = self._categorize_agents()
            for category, agent_names in categories.items():
                print(f"\n📁 {category} ({len(agent_names)} agents)")
                print("-" * 60)
                for name in sorted(agent_names):
                    agent = self.agents[name]
                    model = agent.get('model', 'unknown')
                    location = '📍' if agent['location'] == 'project' else '👤'
                    print(f"  {location} {name:<35} [{model}]")
        
        elif group_by == 'model':
            by_model = {}
            for name, agent in self.agents.items():
                model = agent.get('model', 'unknown')
                if model not in by_model:
                    by_model[model] = []
                by_model[model].append(name)
            
            for model, names in sorted(by_model.items()):
                print(f"\n🧠 Model: {model} ({len(names)} agents)")
                print("-" * 60)
                for name in sorted(names):
                    print(f"  • {name}")
    
    def _categorize_agents(self) -> Dict[str, List[str]]:
        """Categorize agents by their primary function"""
        categories = {
            "Security & Testing": [],
            "Development & Engineering": [],
            "System & Infrastructure": [],
            "AI & Machine Learning": [],
            "Automation & Workflows": [],
            "Management & Coordination": [],
            "Specialized Domains": [],
            "User Experience": []
        }
        
        # Keywords for categorization
        category_keywords = {
            "Security & Testing": ["security", "kali", "semgrep", "pentesting", "testing", "qa"],
            "Development & Engineering": ["developer", "engineer", "code", "frontend", "backend"],
            "System & Infrastructure": ["system", "infrastructure", "devops", "deployment", "hardware"],
            "AI & Machine Learning": ["ai", "learning", "coordinator", "processing", "agent-creator"],
            "Automation & Workflows": ["automation", "workflow", "flow", "dify", "langflow"],
            "Management & Coordination": ["manager", "coordinator", "scrum", "orchestrator"],
            "Specialized Domains": ["financial", "document", "knowledge", "data"],
            "User Experience": ["jarvis", "voice", "context", "interface"]
        }
        
        for agent_name in self.agents:
            categorized = False
            for category, keywords in category_keywords.items():
                if any(keyword in agent_name.lower() for keyword in keywords):
                    categories[category].append(agent_name)
                    categorized = True
                    break
            
            if not categorized:
                categories["Specialized Domains"].append(agent_name)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
    
    def show_agent_details(self, agent_name: str):
        """Show detailed information about a specific agent"""
        if agent_name not in self.agents:
            print(f"❌ Agent '{agent_name}' not found")
            return
        
        agent = self.agents[agent_name]
        print(f"\n🤖 Agent Details: {agent_name}")
        print("=" * 60)
        print(f"Model: {agent.get('model', 'unknown')}")
        print(f"Location: {agent['location']} ({agent['file_path']})")
        print(f"Description: {agent.get('description', 'No description available')}")
        
        if agent.get('capabilities'):
            print("\nCapabilities:")
            for cap in agent['capabilities']:
                print(f"  • {cap}")
    
    def generate_agent_usage_examples(self):
        """Generate usage examples for common scenarios"""
        print("\n📚 Agent Usage Examples")
        print("=" * 80)
        
        scenarios = [
            {
                "title": "🔒 Secure Development Pipeline",
                "agents": ["kali-security-specialist", "semgrep-security-analyzer", "security-pentesting-specialist"],
                "example": """
# 1. Initial security audit
Task(subagent_type='kali-security-specialist', 
     prompt='Perform comprehensive security audit of the application')

# 2. Code vulnerability scanning  
Task(subagent_type='semgrep-security-analyzer',
     prompt='Scan codebase for OWASP Top 10 vulnerabilities')

# 3. Penetration testing
Task(subagent_type='security-pentesting-specialist',
     prompt='Execute penetration testing on deployed application')
"""
            },
            {
                "title": "🚀 Full-Stack Development",
                "agents": ["senior-backend-developer", "senior-frontend-developer", "opendevin-code-generator"],
                "example": """
# 1. Backend API development
Task(subagent_type='senior-backend-developer',
     prompt='Design and implement RESTful API for user management')

# 2. Frontend UI creation
Task(subagent_type='senior-frontend-developer', 
     prompt='Create responsive React components for user dashboard')

# 3. Code generation assistance
Task(subagent_type='opendevin-code-generator',
     prompt='Generate boilerplate code for authentication system')
"""
            },
            {
                "title": "🤖 AI System Development",
                "agents": ["ai-agent-creator", "deep-learning-coordinator-manager", "ollama-integration-specialist"],
                "example": """
# 1. Create new AI agent
Task(subagent_type='ai-agent-creator',
     prompt='Design a new agent for automated data analysis')

# 2. Implement processing network
Task(subagent_type='deep-learning-coordinator-manager',
     prompt='Create processing network for pattern recognition')

# 3. Deploy with Ollama
Task(subagent_type='ollama-integration-specialist',
     prompt='Configure and optimize Ollama for local AI deployment')
"""
            }
        ]
        
        for scenario in scenarios:
            print(f"\n{scenario['title']}")
            print("-" * 60)
            print(f"Agents: {', '.join(scenario['agents'])}")
            print("\nExample Usage:")
            print(scenario['example'])
    
    def create_agent_workflow(self, workflow_name: str, tasks: List[Tuple[str, str]]):
        """Create a multi-agent workflow"""
        print(f"\n🔄 Creating Workflow: {workflow_name}")
        print("=" * 60)
        
        workflow = {
            "name": workflow_name,
            "created": datetime.now().isoformat(),
            "tasks": []
        }
        
        for i, (agent_name, task_description) in enumerate(tasks, 1):
            if agent_name not in self.agents:
                print(f"⚠️  Warning: Agent '{agent_name}' not found")
                continue
            
            agent = self.agents[agent_name]
            task = {
                "step": i,
                "agent": agent_name,
                "model": agent.get('model', 'unknown'),
                "description": task_description
            }
            workflow["tasks"].append(task)
            
            print(f"\nStep {i}: {agent_name}")
            print(f"  Model: {agent.get('model', 'unknown')}")
            print(f"  Task: {task_description}")
        
        # Save workflow
        workflow_file = Path(f"/opt/sutazaiapp/workflows/{workflow_name.replace(' ', '_')}.json")
        workflow_file.parent.mkdir(exist_ok=True)
        workflow_file.write_text(json.dumps(workflow, indent=2))
        
        print(f"\n✅ Workflow saved to: {workflow_file}")
        return workflow
    
    def analyze_agent_coverage(self):
        """Analyze capability coverage across all agents"""
        print("\n📊 Agent Capability Analysis")
        print("=" * 80)
        
        # Model distribution
        model_count = {}
        for agent in self.agents.values():
            model = agent.get('model', 'unknown')
            model_count[model] = model_count.get(model, 0) + 1
        
        print("\n🧠 Model Distribution:")
        for model, count in sorted(model_count.items(), key=lambda x: -x[1]):
            percentage = (count / len(self.agents)) * 100
            print(f"  {model}: {count} agents ({percentage:.1f}%)")
        
        # Category coverage
        categories = self._categorize_agents()
        print("\n📁 Category Coverage:")
        for category, agents in sorted(categories.items(), key=lambda x: -len(x[1])):
            print(f"  {category}: {len(agents)} agents")
        
        # Capability keywords
        all_capabilities = []
        for agent in self.agents.values():
            all_capabilities.extend(agent.get('capabilities', []))
        
        print(f"\n🎯 Total Unique Capabilities: {len(set(all_capabilities))}")


def main():
    """Main CLI interface"""
    manager = SutazAIAgentManager()
    
    while True:
        print("\n🤖 SutazAI Agent Manager")
        print("=" * 40)
        print("1. List all agents")
        print("2. Show agent details")
        print("3. Generate usage examples")
        print("4. Create workflow")
        print("5. Analyze agent coverage")
        print("6. Exit")
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == '1':
            group = input("Group by (category/model) [category]: ").strip() or 'category'
            manager.list_all_agents(group_by=group)
        
        elif choice == '2':
            agent_name = input("Enter agent name: ").strip()
            manager.show_agent_details(agent_name)
        
        elif choice == '3':
            manager.generate_agent_usage_examples()
        
        elif choice == '4':
            workflow_name = input("Workflow name: ").strip()
            tasks = []
            print("Enter tasks (agent_name: task_description), empty line to finish:")
            while True:
                task_input = input("Task: ").strip()
                if not task_input:
                    break
                if ':' in task_input:
                    agent, desc = task_input.split(':', 1)
                    tasks.append((agent.strip(), desc.strip()))
            
            if tasks:
                manager.create_agent_workflow(workflow_name, tasks)
        
        elif choice == '5':
            manager.analyze_agent_coverage()
        
        elif choice == '6':
            print("\n👋 Goodbye!")
            break
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    # Quick demo if run without interaction
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        manager = SutazAIAgentManager()
        manager.list_all_agents()
        manager.analyze_agent_coverage()
        manager.generate_agent_usage_examples()
    else:
        main()