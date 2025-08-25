#!/usr/bin/env python3
"""
SuperClaude Framework Feature Demonstration
Shows how the framework components work together
"""

import os
import json
from pathlib import Path

def demonstrate_commands():
    """Show available SuperClaude commands"""
    print("\n" + "="*60)
    print("SUPERCLAUDE SLASH COMMANDS (/sc:)")
    print("="*60)
    
    commands_dir = Path.home() / ".claude" / "commands" / "sc"
    
    # Read command metadata
    command_info = {
        "brainstorm": "Interactive requirements discovery through Socratic dialogue",
        "implement": "Systematic implementation with best practices",
        "analyze": "Deep code analysis and architecture review",
        "troubleshoot": "Systematic debugging and problem-solving",
        "test": "Comprehensive testing strategy and execution",
        "document": "Intelligent documentation generation",
        "improve": "Code optimization and refactoring",
        "design": "System architecture and design patterns",
        "workflow": "Multi-step workflow orchestration",
        "task": "Task decomposition and management"
    }
    
    print("\nKey Commands:")
    for cmd, desc in command_info.items():
        cmd_file = commands_dir / f"{cmd}.md"
        if cmd_file.exists():
            print(f"  /sc:{cmd:<15} - {desc}")
    
    print("\nUsage Example:")
    print("  Type '/sc:brainstorm create a REST API' to start interactive design")
    print("  Type '/sc:implement user authentication' for guided implementation")

def demonstrate_agents():
    """Show available specialized agents"""
    print("\n" + "="*60)
    print("SPECIALIZED AI AGENTS")
    print("="*60)
    
    agents_dir = Path.home() / ".claude" / "Agents"
    
    agent_info = {
        "backend-architect": "Backend systems, APIs, databases",
        "frontend-architect": "UI/UX, React, responsive design",
        "security-engineer": "Security audits, threat modeling",
        "devops-architect": "CI/CD, Docker, Kubernetes",
        "performance-engineer": "Optimization, profiling, scaling",
        "python-expert": "Python best practices, frameworks",
        "quality-engineer": "Testing strategies, QA processes",
        "system-architect": "High-level design, patterns"
    }
    
    print("\nAvailable Agents:")
    for agent, focus in agent_info.items():
        agent_file = agents_dir / f"{agent}.md"
        if agent_file.exists():
            print(f"  {agent:<20} - {focus}")
    
    print("\nAgent Activation:")
    print("  Agents activate automatically based on context")
    print("  Example: Mention 'API design' -> backend-architect activates")

def demonstrate_modes():
    """Show behavioral modes"""
    print("\n" + "="*60)
    print("BEHAVIORAL MODES")
    print("="*60)
    
    modes_dir = Path.home() / ".claude" / "Modes"
    
    mode_info = {
        "Brainstorming": "Socratic questioning, idea exploration",
        "Token_Efficiency": "Compressed communication, symbols",
        "Task_Management": "Systematic task tracking, TodoWrite",
        "Orchestration": "Multi-tool coordination, parallel execution",
        "Introspection": "Meta-cognitive analysis, self-reflection"
    }
    
    print("\nActive Modes:")
    for mode, behavior in mode_info.items():
        mode_file = modes_dir / f"MODE_{mode}.md"
        if mode_file.exists():
            print(f"  {mode:<20} - {behavior}")
    
    print("\nMode Triggers:")
    print("  - High context usage -> Token Efficiency Mode")
    print("  - Complex tasks -> Task Management Mode")
    print("  - Multiple operations -> Orchestration Mode")

def show_integration():
    """Show how components work together"""
    print("\n" + "="*60)
    print("FRAMEWORK INTEGRATION")
    print("="*60)
    
    print("\nExample Workflow:")
    print("1. User: '/sc:brainstorm e-commerce platform'")
    print("   -> Brainstorming Mode activates")
    print("   -> Multiple agents coordinate (backend, frontend, security)")
    print("   -> Socratic dialogue explores requirements")
    print()
    print("2. User: '/sc:implement user authentication'")
    print("   -> Task Management Mode activates")
    print("   -> Backend-architect leads implementation")
    print("   -> Security-engineer reviews for vulnerabilities")
    print("   -> Automatic TodoWrite tracking")
    print()
    print("3. User: '/sc:test authentication flow'")
    print("   -> Quality-engineer designs test strategy")
    print("   -> Orchestration Mode for parallel test execution")
    print("   -> Comprehensive test coverage report")

def check_claude_md():
    """Verify CLAUDE.md configuration"""
    print("\n" + "="*60)
    print("CLAUDE.MD CONFIGURATION")
    print("="*60)
    
    claude_md = Path.home() / ".claude" / "CLAUDE.md"
    
    if claude_md.exists():
        with open(claude_md, encoding='utf-8') as f:
            content = f.read()
            
        components = {
            "Core Framework": ["FLAGS.md", "PRINCIPLES.md", "RULES.md"],
            "Modes": ["MODE_Brainstorming", "MODE_Token_Efficiency"],
            "Agents": ["backend-architect", "security-engineer"]
        }
        
        print("\nActive Components in CLAUDE.md:")
        for category, items in components.items():
            active = [item for item in items if item in content]
            if active:
                print(f"\n{category}:")
                for item in active:
                    print(f"  [OK] {item}")

def main():
    """Run feature demonstration"""
    print("="*60)
    print("  SUPERCLAUDE FRAMEWORK v4.0.8 - FEATURE DEMONSTRATION")
    print("="*60)
    
    # Show all components
    demonstrate_commands()
    demonstrate_agents()
    demonstrate_modes()
    show_integration()
    check_claude_md()
    
    print("\n" + "="*60)
    print("READY TO USE!")
    print("="*60)
    print("\nQuick Start:")
    print("  1. Type '/sc:brainstorm [your idea]' to explore concepts")
    print("  2. Type '/sc:implement [feature]' for guided development")
    print("  3. Type '/sc:troubleshoot [issue]' for debugging help")
    print("\nThe framework will automatically:")
    print("  - Activate relevant agents based on context")
    print("  - Switch behavioral modes as needed")
    print("  - Coordinate multiple tools efficiently")
    print("="*60)

if __name__ == "__main__":
    main()