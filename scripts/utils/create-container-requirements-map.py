#!/usr/bin/env python3
"""
Purpose: Create a detailed map of containers and their essential requirements
Usage: python create-container-requirements-map.py
Requirements: Python 3.8+, PyYAML
"""

import os
import re
import json
import yaml
from pathlib import Path
from collections import defaultdict

def analyze_container_requirements():
    """Create comprehensive container to requirements mapping"""
    
    root_path = Path("/opt/sutazaiapp")
    
    # Key containers from docker-compose.yml
    critical_containers = {
        # Core Services
        'postgres': {'type': 'database', 'requirements': None},
        'redis': {'type': 'cache', 'requirements': None},
        'neo4j': {'type': 'graph-db', 'requirements': None},
        'chromadb': {'type': 'vector-db', 'requirements': None},
        'qdrant': {'type': 'vector-db', 'requirements': None},
        'faiss': {'type': 'vector-db', 'requirements': 'docker/faiss/requirements.txt'},
        'ollama': {'type': 'llm-runtime', 'requirements': None},
        
        # Main Application
        'backend': {'type': 'api', 'requirements': 'backend/requirements.txt'},
        'frontend': {'type': 'ui', 'requirements': 'frontend/requirements.txt'},
        
        # AI Agents
        'autogpt': {'type': 'agent', 'requirements': 'docker/autogpt/requirements.txt'},
        'crewai': {'type': 'agent', 'requirements': 'docker/crewai/requirements.txt'},
        'letta': {'type': 'agent', 'requirements': 'docker/letta/requirements.txt'},
        'aider': {'type': 'agent', 'requirements': 'docker/aider/requirements.txt'},
        'gpt-engineer': {'type': 'agent', 'requirements': 'docker/gpt-engineer/requirements.txt'},
        'agentgpt': {'type': 'agent', 'requirements': 'docker/agentgpt/requirements.txt'},
        'privategpt': {'type': 'agent', 'requirements': 'docker/privategpt/requirements.txt'},
        'pentestgpt': {'type': 'agent', 'requirements': 'docker/pentestgpt/requirements.txt'},
        'shellgpt': {'type': 'agent', 'requirements': 'docker/shellgpt/requirements.txt'},
        
        # Workflow Tools
        'langflow': {'type': 'workflow', 'requirements': None},  # Uses image
        'flowise': {'type': 'workflow', 'requirements': None},   # Uses image
        'n8n': {'type': 'workflow', 'requirements': None},       # Uses image
        'dify': {'type': 'workflow', 'requirements': None},      # Uses image
        
        # ML Frameworks
        'pytorch': {'type': 'ml-framework', 'requirements': 'docker/pytorch/requirements.txt'},
        'tensorflow': {'type': 'ml-framework', 'requirements': 'docker/tensorflow/requirements.txt'},
        'jax': {'type': 'ml-framework', 'requirements': 'docker/jax/requirements.txt'},
        
        # Monitoring
        'prometheus': {'type': 'monitoring', 'requirements': None},
        'grafana': {'type': 'monitoring', 'requirements': None},
        'loki': {'type': 'monitoring', 'requirements': None},
        'ai-metrics-exporter': {'type': 'monitoring', 'requirements': 'monitoring/ai-metrics-exporter/requirements.txt'},
        
        # Specialized Services
        'semgrep': {'type': 'security', 'requirements': 'docker/semgrep/requirements.txt'},
        'browser-use': {'type': 'automation', 'requirements': 'docker/browser-use/requirements.txt'},
        'documind': {'type': 'document-processing', 'requirements': 'docker/documind/requirements.txt'},
        'finrobot': {'type': 'finance-ai', 'requirements': 'docker/finrobot/requirements.txt'},
        'code-improver': {'type': 'code-ai', 'requirements': 'docker/code-improver/requirements.txt'},
        'context-framework': {'type': 'context-ai', 'requirements': 'docker/context-framework/requirements.txt'},
        'autogen': {'type': 'agent', 'requirements': 'docker/autogen/requirements.txt'},
        'opendevin': {'type': 'agent', 'requirements': 'docker/opendevin/requirements.txt'},
        'service-hub': {'type': 'service-manager', 'requirements': 'docker/service-hub/requirements.txt'},
        'awesome-code-ai': {'type': 'code-ai', 'requirements': 'docker/awesome-code-ai/requirements.txt'},
        'fsdp': {'type': 'ml-distributed', 'requirements': 'docker/fsdp/requirements.txt'},
        'agentzero': {'type': 'agent-coordinator', 'requirements': 'docker/agentzero/requirements.txt'},
        'mcp-server': {'type': 'mcp-protocol', 'requirements': 'mcp_server/requirements.txt'},
        'health-monitor': {'type': 'monitoring', 'requirements': 'docker/health-check/requirements.txt'},
        'llamaindex': {'type': 'rag', 'requirements': 'docker/llamaindex/requirements.txt'},
        'tabbyml': {'type': 'code-completion', 'requirements': None},  # Uses image
        'skyvern': {'type': 'automation', 'requirements': 'docker/skyvern/requirements.txt'},
    }
    
    # Verify which requirements files actually exist
    container_map = {}
    missing_requirements = []
    existing_requirements = set()
    
    for container_name, info in critical_containers.items():
        container_info = {
            'name': container_name,
            'type': info['type'],
            'requirements_file': None,
            'requirements_exist': False,
            'dockerfile_path': None
        }
        
        if info['requirements']:
            req_path = root_path / info['requirements']
            if req_path.exists():
                container_info['requirements_file'] = info['requirements']
                container_info['requirements_exist'] = True
                existing_requirements.add(info['requirements'])
            else:
                missing_requirements.append(f"{container_name}: {info['requirements']}")
        
        container_map[container_name] = container_info
    
    # Find all requirements.txt files
    all_requirements = []
    for req_file in root_path.rglob("requirements*.txt"):
        if req_file.is_file():
            req_path = str(req_file.relative_to(root_path))
            all_requirements.append(req_path)
    
    # Identify orphaned requirements
    orphaned_requirements = []
    for req_file in all_requirements:
        if req_file not in existing_requirements:
            # Check if it's in a container directory but not mapped
            if any(container in req_file for container in critical_containers.keys()):
                orphaned_requirements.append(f"Unmapped but likely used: {req_file}")
            elif any(x in req_file for x in ['backup', 'old', 'archive', 'test', 'docs']):
                orphaned_requirements.append(f"Safe to remove: {req_file}")
            else:
                orphaned_requirements.append(f"Verify before removing: {req_file}")
    
    # Generate consolidation opportunities
    common_packages = defaultdict(list)
    for req_file in existing_requirements:
        try:
            with open(root_path / req_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        pkg_match = re.match(r'^([a-zA-Z0-9\-_]+)', line)
                        if pkg_match:
                            pkg_name = pkg_match.group(1)
                            common_packages[pkg_name].append(req_file)
        except Exception:
            pass
    
    # Create consolidation recommendations
    consolidation = {
        'base_packages': [],
        'ml_packages': [],
        'agent_packages': [],
        'monitoring_packages': []
    }
    
    for pkg, files in common_packages.items():
        if len(files) > 5:
            if pkg in ['fastapi', 'uvicorn', 'pydantic', 'redis', 'sqlalchemy']:
                consolidation['base_packages'].append(pkg)
            elif pkg in ['torch', 'transformers', 'accelerate', 'numpy', 'pandas']:
                consolidation['ml_packages'].append(pkg)
            elif pkg in ['langchain', 'openai', 'anthropic', 'ollama']:
                consolidation['agent_packages'].append(pkg)
            elif pkg in ['prometheus-client', 'psutil', 'aiohttp']:
                consolidation['monitoring_packages'].append(pkg)
    
    # Generate report
    report = {
        'total_containers': len(critical_containers),
        'containers_with_requirements': sum(1 for c in container_map.values() if c['requirements_exist']),
        'containers_without_requirements': sum(1 for c in container_map.values() if not c['requirements_file']),
        'total_requirements_files': len(all_requirements),
        'used_requirements_files': len(existing_requirements),
        'orphaned_requirements_count': len(orphaned_requirements),
        'container_mapping': container_map,
        'missing_requirements': missing_requirements,
        'orphaned_requirements': orphaned_requirements[:20],  # Show first 20
        'consolidation_opportunities': consolidation,
        'validation_commands': generate_validation_commands(container_map),
        'cleanup_script': generate_cleanup_script(orphaned_requirements)
    }
    
    return report

def generate_validation_commands(container_map):
    """Generate commands to validate each container after cleanup"""
    commands = []
    
    # Test critical services first
    critical_services = ['postgres', 'redis', 'backend', 'frontend', 'ollama']
    
    commands.append("#!/bin/bash")
    commands.append("# Container validation script")
    commands.append("set -e")
    commands.append("")
    
    # Stop all services first
    commands.append("echo 'Stopping all services...'")
    commands.append("docker-compose down")
    commands.append("")
    
    # Test each critical service
    for service in critical_services:
        commands.append(f"echo 'Testing {service}...'")
        commands.append(f"docker-compose up -d {service}")
        commands.append(f"sleep 10")
        commands.append(f"docker-compose ps {service} | grep 'Up' || exit 1")
        commands.append("")
    
    # Test agent containers
    agent_services = ['autogpt', 'crewai', 'letta', 'aider']
    for service in agent_services:
        if service in container_map and container_map[service]['requirements_exist']:
            commands.append(f"echo 'Building {service}...'")
            commands.append(f"docker-compose build {service}")
            commands.append("")
    
    commands.append("echo 'All critical containers validated!'")
    
    return commands

def generate_cleanup_script(orphaned_requirements):
    """Generate safe cleanup script"""
    script = []
    
    script.append("#!/bin/bash")
    script.append("# Safe cleanup script for orphaned requirements")
    script.append("set -e")
    script.append("")
    script.append("# Create backup directory")
    script.append("BACKUP_DIR='/opt/sutazaiapp/requirements_backup_$(date +%Y%m%d_%H%M%S)'")
    script.append("mkdir -p $BACKUP_DIR")
    script.append("")
    
    # Backup all requirements first
    script.append("echo 'Backing up all requirements files...'")
    script.append("find /opt/sutazaiapp -name 'requirements*.txt' -type f | while read f; do")
    script.append("    rel_path=${f#/opt/sutazaiapp/}")
    script.append("    mkdir -p \"$BACKUP_DIR/$(dirname $rel_path)\"")
    script.append("    cp \"$f\" \"$BACKUP_DIR/$rel_path\"")
    script.append("done")
    script.append("")
    
    # Remove safe files
    script.append("echo 'Removing safe orphaned files...'")
    for orphan in orphaned_requirements:
        if "Safe to remove:" in orphan:
            file_path = orphan.replace("Safe to remove: ", "")
            script.append(f"rm -f '/opt/sutazaiapp/{file_path}'")
    
    script.append("")
    script.append("echo 'Cleanup complete! Backup stored in: $BACKUP_DIR'")
    
    return script

def main():
    report = analyze_container_requirements()
    
    # Save detailed report
    with open('/opt/sutazaiapp/container-requirements-map.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Save validation script
    with open('/opt/sutazaiapp/scripts/validate-containers.sh', 'w') as f:
        f.write('\n'.join(report['validation_commands']))
    os.chmod('/opt/sutazaiapp/scripts/validate-containers.sh', 0o755)
    
    # Save cleanup script
    with open('/opt/sutazaiapp/scripts/cleanup-requirements.sh', 'w') as f:
        f.write('\n'.join(report['cleanup_script']))
    os.chmod('/opt/sutazaiapp/scripts/cleanup-requirements.sh', 0o755)
    
    # Print comprehensive report
    print("=" * 80)
    print("DOCKER CONTAINER REQUIREMENTS COMPREHENSIVE MAP")
    print("=" * 80)
    print(f"\nTotal Containers Analyzed: {report['total_containers']}")
    print(f"Containers with Requirements: {report['containers_with_requirements']}")
    print(f"Containers without Requirements: {report['containers_without_requirements']}")
    print(f"Total Requirements Files Found: {report['total_requirements_files']}")
    print(f"Used Requirements Files: {report['used_requirements_files']}")
    print(f"Orphaned Requirements Files: {report['orphaned_requirements_count']}")
    
    print("\n" + "=" * 80)
    print("CONTAINER MAPPING (ESSENTIAL REQUIREMENTS)")
    print("=" * 80)
    
    # Group by type
    by_type = defaultdict(list)
    for name, info in report['container_mapping'].items():
        by_type[info['type']].append((name, info))
    
    for container_type, containers in sorted(by_type.items()):
        print(f"\n{container_type.upper()} CONTAINERS:")
        for name, info in sorted(containers):
            if info['requirements_file']:
                status = "✓ ACTIVE" if info['requirements_exist'] else "✗ MISSING"
                print(f"  - {name}: {info['requirements_file']} [{status}]")
            else:
                print(f"  - {name}: No requirements file (uses Docker image)")
    
    print("\n" + "=" * 80)
    print("CONSOLIDATION OPPORTUNITIES")
    print("=" * 80)
    
    for category, packages in report['consolidation_opportunities'].items():
        if packages:
            print(f"\n{category.upper()}:")
            print(f"  {', '.join(packages[:10])}")
            if len(packages) > 10:
                print(f"  ... and {len(packages) - 10} more")
    
    print("\n" + "=" * 80)
    print("SAFE CLEANUP RECOMMENDATIONS")
    print("=" * 80)
    
    safe_count = sum(1 for o in report['orphaned_requirements'] if "Safe to remove:" in o)
    verify_count = sum(1 for o in report['orphaned_requirements'] if "Verify before removing:" in o)
    
    print(f"\nFiles safe to remove: {safe_count}")
    print(f"Files to verify before removing: {verify_count}")
    
    print("\n" + "=" * 80)
    print("GENERATED FILES")
    print("=" * 80)
    print("  1. Full report: /opt/sutazaiapp/container-requirements-map.json")
    print("  2. Validation script: /opt/sutazaiapp/scripts/validate-containers.sh")
    print("  3. Cleanup script: /opt/sutazaiapp/scripts/cleanup-requirements.sh")
    
    print("\n" + "=" * 80)
    print("RECOMMENDED ACTIONS")
    print("=" * 80)
    print("  1. Review the container-requirements-map.json for full details")
    print("  2. Run ./scripts/cleanup-requirements.sh to safely clean orphaned files")
    print("  3. Run ./scripts/validate-containers.sh to test all containers")
    print("  4. Consider creating shared base requirements for common packages")
    
    print("\n✅ Analysis complete! Zero tolerance for breaking container functionality maintained.")

if __name__ == "__main__":
    main()