#!/usr/bin/env python3
"""
Test Script for AI Agent Setup Verification
Tests basic functionality of deployed agents
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(title):
    """Print formatted header"""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{title}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")

def test_agent_directory_structure():
    """Test if agent directories are properly created"""
    print_header("Testing Agent Directory Structure")
    
    base_dir = Path("/opt/sutazaiapp/agents")
    required_dirs = [
        "task-automation",
        "code-generation", 
        "orchestration",
        "document-processing",
        "financial",
        "code-security",
        "development-tools",
        "frameworks",
        "chat-interfaces"
    ]
    
    all_exist = True
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            print(f"  {GREEN}✓{RESET} {dir_name}")
        else:
            print(f"  {RED}✗{RESET} {dir_name}")
            all_exist = False
    
    return all_exist

def test_cloned_agents():
    """Test which agents have been successfully cloned"""
    print_header("Testing Cloned Agents")
    
    agents = {
        "task-automation/letta": "Letta (MemGPT)",
        "task-automation/autogpt": "AutoGPT",
        "orchestration/crewai": "CrewAI",
        "code-generation/aider": "Aider",
        "document-processing/private-gpt": "Private-GPT"
    }
    
    base_dir = Path("/opt/sutazaiapp/agents")
    cloned_count = 0
    
    for path, name in agents.items():
        full_path = base_dir / path
        if full_path.exists() and (full_path / ".git").exists():
            print(f"  {GREEN}✓{RESET} {name}: {path}")
            cloned_count += 1
            
            # Check for key files
            if (full_path / "README.md").exists():
                print(f"    └─ README.md found")
            if (full_path / "requirements.txt").exists():
                print(f"    └─ requirements.txt found")
            elif (full_path / "pyproject.toml").exists():
                print(f"    └─ pyproject.toml found")
        else:
            print(f"  {RED}✗{RESET} {name}: Not found")
    
    print(f"\n  Summary: {cloned_count}/{len(agents)} agents cloned")
    return cloned_count

def test_docker_compose():
    """Test if Docker Compose file is valid"""
    print_header("Testing Docker Compose Configuration")
    
    compose_file = Path("/opt/sutazaiapp/agents/docker-compose-agents.yml")
    
    if not compose_file.exists():
        print(f"  {RED}✗{RESET} docker-compose-agents.yml not found")
        return False
    
    print(f"  {GREEN}✓{RESET} docker-compose-agents.yml exists")
    
    # Validate compose file
    try:
        result = subprocess.run(
            ["docker", "compose", "-f", str(compose_file), "config"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print(f"  {GREEN}✓{RESET} Docker Compose configuration is valid")
            
            # Parse services
            import yaml
            with open(compose_file, 'r') as f:
                config = yaml.safe_load(f)
                if 'services' in config:
                    print(f"  Services defined: {list(config['services'].keys())}")
            return True
        else:
            print(f"  {RED}✗{RESET} Docker Compose validation failed")
            print(f"    Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"  {YELLOW}⚠{RESET} Could not validate compose file: {e}")
        return False

def test_agent_registry():
    """Test agent registry file"""
    print_header("Testing Agent Registry")
    
    registry_file = Path("/opt/sutazaiapp/agents/agent_registry.json")
    
    if not registry_file.exists():
        print(f"  {RED}✗{RESET} agent_registry.json not found")
        return False
    
    print(f"  {GREEN}✓{RESET} agent_registry.json exists")
    
    try:
        with open(registry_file, 'r') as f:
            registry = json.load(f)
        
        print(f"  Total agents registered: {registry.get('total', 0)}")
        print(f"  Agents deployed: {registry.get('deployed', 0)}")
        
        if 'agents' in registry:
            for agent in registry['agents'][:5]:  # Show first 5
                status_color = GREEN if agent['status'] == 'active' else YELLOW
                print(f"    {status_color}●{RESET} {agent['name']} (Port: {agent['port']})")
        
        return True
    except Exception as e:
        print(f"  {RED}✗{RESET} Failed to parse registry: {e}")
        return False

def test_python_requirements():
    """Test if requirements file exists"""
    print_header("Testing Python Requirements")
    
    req_file = Path("/opt/sutazaiapp/agents/requirements_all.txt")
    
    if not req_file.exists():
        print(f"  {RED}✗{RESET} requirements_all.txt not found")
        return False
    
    print(f"  {GREEN}✓{RESET} requirements_all.txt exists")
    
    # Count requirements
    with open(req_file, 'r') as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
        print(f"  Total requirements: {len(lines)}")
        
        # Show some key requirements
        key_deps = ['openai', 'langchain', 'crewai', 'letta', 'aider-chat']
        for dep in key_deps:
            if any(dep in line for line in lines):
                print(f"    {GREEN}✓{RESET} {dep}")
    
    return True

def test_agent_connectivity():
    """Test if agents can be accessed (when running)"""
    print_header("Testing Agent Connectivity (if running)")
    
    agents_ports = [
        ("Letta", 11100),
        ("AutoGPT", 11101),
        ("CrewAI", 11102)
    ]
    
    import socket
    
    for name, port in agents_ports:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        
        if result == 0:
            print(f"  {GREEN}✓{RESET} {name} is accessible on port {port}")
        else:
            print(f"  {YELLOW}○{RESET} {name} not running on port {port} (expected)")
    
    return True

def generate_summary_report():
    """Generate summary report"""
    print_header("AI Agents Deployment Summary")
    
    print(f"\n  Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Platform Phase: 6 - AI Agents Setup")
    print(f"  Status: IN PROGRESS")
    print()
    
    # Run all tests
    results = {
        "Directory Structure": test_agent_directory_structure(),
        "Cloned Agents": test_cloned_agents() > 0,
        "Docker Compose": test_docker_compose() if Path("/opt/sutazaiapp/agents/docker-compose-agents.yml").exists() else False,
        "Agent Registry": test_agent_registry(),
        "Requirements File": test_python_requirements()
    }
    
    print_header("Test Results Summary")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = f"{GREEN}PASSED{RESET}" if result else f"{RED}FAILED{RESET}"
        print(f"  {test_name}: {status}")
    
    print(f"\n  Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print(f"\n  {GREEN}✅ All tests passed! Agents are ready for configuration.{RESET}")
    else:
        print(f"\n  {YELLOW}⚠ Some tests failed. Please review and fix issues.{RESET}")
    
    print("\n  Next Steps:")
    print("  1. Configure agent API keys and settings")
    print("  2. Deploy agents using: docker compose -f docker-compose-agents.yml up -d")
    print("  3. Access agent interfaces on ports 11100-11199")
    print("  4. Integrate agents with JARVIS frontend")

if __name__ == "__main__":
    try:
        # Try to import yaml for docker-compose parsing
        import yaml
    except ImportError:
        print(f"{YELLOW}Installing PyYAML for docker-compose parsing...{RESET}")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyyaml", "--quiet"])
        import yaml
    
    print(f"{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}   SutazAI Platform - AI Agents Test Suite{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")
    
    generate_summary_report()
    
    print(f"\n{BLUE}{'='*60}{RESET}")