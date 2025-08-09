#!/usr/bin/env python3
"""
Update all service ports to use only approved port list:
10010, 10104, 11015, 8589, 8587, 8551, 8002

This script reassigns ports for all services to comply with the port policy.
"""

import yaml
import json
import os
import shutil
from datetime import datetime

# Approved port list
APPROVED_PORTS = [10010, 10104, 11015, 8589, 8587, 8551, 8002]

# Port mapping for critical services (must keep these assignments)
CRITICAL_PORT_MAPPING = {
    'backend': 10010,          # Backend API (already correct)
    'ollama': 10104,           # Ollama LLM server (already correct)
    'ollama-integration-specialist': 11015,  # Already correct
    'ai-agent-orchestrator': 8589,  # Already correct
    'multi-agent-coordinator': 8587,  # Already correct
    'task-assignment-coordinator': 8551,  # Already correct
    'hardware-resource-optimizer': 8002,  # Reassign from 11110 to 8002
}

# Services that should NOT have ports (internal only or not needed)
NO_PORT_SERVICES = [
    'postgres', 'redis', 'neo4j', 'chromadb', 'qdrant', 'faiss',
    'prometheus', 'grafana', 'loki', 'alertmanager', 'node-exporter',
    'cadvisor', 'blackbox-exporter', 'postgres-exporter', 'redis-exporter',
    'promtail', 'semgrep', 'fsdp', 'autogpt', 'letta', 'pentestgpt', 'skyvern'
]

def backup_file(filepath):
    """Create a backup of the file before modifying"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{filepath}.backup_{timestamp}"
    shutil.copy2(filepath, backup_path)
    print(f"Created backup: {backup_path}")
    return backup_path

def update_docker_compose():
    """Update docker-compose.yml to use only approved ports"""
    compose_file = '/opt/sutazaiapp/docker-compose.yml'
    
    # Backup the file first
    backup_file(compose_file)
    
    # Read the docker-compose file
    with open(compose_file, 'r') as f:
        compose_data = yaml.safe_load(f)
    
    # Track changes
    changes = []
    
    # Update services
    for service_name, service_config in compose_data.get('services', {}).items():
        if 'ports' in service_config:
            old_ports = service_config['ports'].copy()
            
            # Check if this is a critical service
            if service_name in CRITICAL_PORT_MAPPING:
                new_port = CRITICAL_PORT_MAPPING[service_name]
                # Update port mapping
                if isinstance(service_config['ports'], list) and len(service_config['ports']) > 0:
                    port_mapping = service_config['ports'][0]
                    if ':' in str(port_mapping):
                        host_port, container_port = str(port_mapping).split(':')
                        service_config['ports'] = [f"{new_port}:{container_port}"]
                        changes.append(f"  {service_name}: {host_port} -> {new_port}")
                    
            elif service_name in NO_PORT_SERVICES:
                # These services should not have external ports
                del service_config['ports']
                changes.append(f"  {service_name}: Removed external ports (internal only)")
                
            else:
                # Non-critical services - remove external ports
                del service_config['ports']
                changes.append(f"  {service_name}: Removed external ports (not in approved list)")
    
    # Write updated docker-compose file
    with open(compose_file, 'w') as f:
        yaml.dump(compose_data, f, default_flow_style=False, sort_keys=False)
    
    print(f"Updated {compose_file}")
    print("\nChanges made:")
    for change in changes:
        print(change)
    
    return changes

def update_agent_configs():
    """Update agent configuration files to use approved ports"""
    configs_dir = '/opt/sutazaiapp/agents/configs'
    changes = []
    
    if os.path.exists(configs_dir):
        for config_file in os.listdir(configs_dir):
            if config_file.endswith('.json'):
                filepath = os.path.join(configs_dir, config_file)
                
                with open(filepath, 'r') as f:
                    config = json.load(f)
                
                if 'port' in config:
                    old_port = config['port']
                    agent_name = config.get('name', config_file)
                    
                    # Check if this agent has an assigned port
                    if agent_name in CRITICAL_PORT_MAPPING:
                        new_port = CRITICAL_PORT_MAPPING[agent_name]
                        if old_port != new_port:
                            config['port'] = new_port
                            
                            with open(filepath, 'w') as f:
                                json.dump(config, f, indent=2)
                            
                            changes.append(f"  {agent_name}: {old_port} -> {new_port}")
    
    if changes:
        print("\nAgent config changes:")
        for change in changes:
            print(change)
    
    return changes

def update_port_registry():
    """Update the port registry configuration"""
    registry_file = '/opt/sutazaiapp/config/port-registry-actual.yaml'
    
    if os.path.exists(registry_file):
        backup_file(registry_file)
        
        port_registry = {
            'approved_ports': APPROVED_PORTS,
            'port_assignments': CRITICAL_PORT_MAPPING,
            'internal_only_services': NO_PORT_SERVICES,
            'updated_at': datetime.now().isoformat()
        }
        
        with open(registry_file, 'w') as f:
            yaml.dump(port_registry, f, default_flow_style=False)
        
        print(f"\nUpdated port registry: {registry_file}")

def update_resource_arbitration_agent():
    """Update the resource-arbitration-agent to use an approved port"""
    agent_file = '/opt/sutazaiapp/docker/resource-arbitration-agent/app.py'
    
    if os.path.exists(agent_file):
        backup_file(agent_file)
        
        with open(agent_file, 'r') as f:
            content = f.read()
        
        # Replace port 8588 with 8002 (reassigned to hardware-resource-optimizer's port)
        # Since hardware-resource-optimizer will use port 8002, resource-arbitration won't have external port
        content = content.replace('port = int(os.getenv("PORT", "8588"))', 
                                 'port = int(os.getenv("PORT", "8080"))')  # Internal port only
        
        with open(agent_file, 'w') as f:
            f.write(content)
        
        print(f"\nUpdated resource-arbitration-agent to use internal port only (was 8588)")

def verify_changes():
    """Verify that all ports are now in the approved list"""
    compose_file = '/opt/sutazaiapp/docker-compose.yml'
    
    with open(compose_file, 'r') as f:
        compose_data = yaml.safe_load(f)
    
    print("\n" + "="*60)
    print("VERIFICATION: Services with external ports:")
    print("="*60)
    
    for service_name, service_config in compose_data.get('services', {}).items():
        if 'ports' in service_config:
            ports = service_config['ports']
            if isinstance(ports, list) and len(ports) > 0:
                port_mapping = str(ports[0])
                if ':' in port_mapping:
                    host_port = int(port_mapping.split(':')[0])
                    status = "✅ APPROVED" if host_port in APPROVED_PORTS else "❌ UNAPPROVED"
                    print(f"  {service_name}: {host_port} {status}")

def main():
    print("="*60)
    print("PORT POLICY ENFORCEMENT")
    print("Updating all services to use only approved ports:")
    print(", ".join(map(str, APPROVED_PORTS)))
    print("="*60)
    
    # Update docker-compose.yml
    update_docker_compose()
    
    # Update agent configurations
    update_agent_configs()
    
    # Update port registry
    update_port_registry()
    
    # Update resource-arbitration-agent
    update_resource_arbitration_agent()
    
    # Verify changes
    verify_changes()
    
    print("\n" + "="*60)
    print("PORT POLICY UPDATE COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Review the changes above")
    print("2. Restart services: docker-compose down && docker-compose up -d")
    print("3. Verify services are accessible on approved ports only")

if __name__ == "__main__":
    main()