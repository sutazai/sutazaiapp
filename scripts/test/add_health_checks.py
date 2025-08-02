#!/usr/bin/env python3
"""
Script to add health checks to all agent containers in docker-compose files
"""

import re
import yaml
import sys
from pathlib import Path

def add_health_check_to_agent_service(service_config):
    """Add health check configuration to an agent service"""
    
    # Add agent_with_health.py to volumes if not present
    volumes = service_config.get('volumes', [])
    health_volume = "./agents/agent_with_health.py:/app/shared/agent_with_health.py:ro"
    if health_volume not in volumes:
        volumes.append(health_volume)
        service_config['volumes'] = volumes
    
    # Add HEALTH_PORT environment variable
    environment = service_config.get('environment', [])
    if isinstance(environment, list):
        # Check if HEALTH_PORT is already set
        has_health_port = any('HEALTH_PORT=' in env for env in environment if isinstance(env, str))
        if not has_health_port:
            environment.append('HEALTH_PORT=8080')
            service_config['environment'] = environment
    elif isinstance(environment, dict):
        if 'HEALTH_PORT' not in environment:
            environment['HEALTH_PORT'] = '8080'
    
    # Add health check configuration
    healthcheck = {
        'test': ['CMD', 'curl', '-f', 'http://localhost:8080/health'],
        'interval': '30s',
        'timeout': '10s',
        'retries': 3,
        'start_period': '60s'
    }
    service_config['healthcheck'] = healthcheck
    
    return service_config

def process_docker_compose_file(file_path):
    """Process a docker-compose file and add health checks to agent services"""
    
    print(f"Processing {file_path}...")
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Load YAML content
        compose_data = yaml.safe_load(content)
        
        if not compose_data or 'services' not in compose_data:
            print(f"No services found in {file_path}")
            return False
        
        services_updated = 0
        
        # Process each service
        for service_name, service_config in compose_data['services'].items():
            # Check if this is an agent service
            if is_agent_service(service_name, service_config):
                print(f"  Adding health check to {service_name}")
                compose_data['services'][service_name] = add_health_check_to_agent_service(service_config)
                services_updated += 1
        
        if services_updated > 0:
            # Write back the updated content
            with open(file_path, 'w') as f:
                yaml.dump(compose_data, f, default_flow_style=False, sort_keys=False)
            
            print(f"Updated {services_updated} services in {file_path}")
            return True
        else:
            print(f"No agent services found in {file_path}")
            return False
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def is_agent_service(service_name, service_config):
    """Check if a service is an agent service that needs health checks"""
    
    # Agent service indicators
    agent_indicators = [
        'sutazai-',
        'agent',
        'specialist',
        'coordinator',
        'manager',
        'orchestrator',
        'master',
        'engineer',
        'analyst',
        'optimizer',
        'creator',
        'solver'
    ]
    
    # Check service name
    for indicator in agent_indicators:
        if indicator in service_name.lower():
            # Make sure it's using python:3.11-slim image (our agent containers)
            image = service_config.get('image', '')
            if 'python:3.11-slim' in image:
                return True
            
            # Check if it has agent-related volumes
            volumes = service_config.get('volumes', [])
            for volume in volumes:
                if isinstance(volume, str) and ('agent_base.py' in volume or 'generic_agent.py' in volume):
                    return True
    
    return False

def main():
    """Main function"""
    
    # List of docker-compose files to process
    compose_files = [
        '/opt/sutazaiapp/docker-compose.agents-simple.yml',
        '/opt/sutazaiapp/docker-compose.complete-agents.yml',
        '/opt/sutazaiapp/docker-compose.agents-extended.yml'
    ]
    
    total_updated = 0
    
    for compose_file in compose_files:
        if Path(compose_file).exists():
            if process_docker_compose_file(compose_file):
                total_updated += 1
        else:
            print(f"File not found: {compose_file}")
    
    print(f"\nProcessed {total_updated} docker-compose files")
    print("Health checks have been added to agent services")

if __name__ == "__main__":
    main()