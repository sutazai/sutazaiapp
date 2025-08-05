#!/usr/bin/env python3
"""
Fix memory limits for all containers
Implements Rules 20-21 from IMPROVED_CODEBASE_RULES_v2.0.md
Adds memory limits to the 34 containers currently missing them
"""

import yaml
import os
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Memory allocation based on agent phases (Rule 20)
MEMORY_LIMITS = {
    # Critical agents (ports 10300-10319)
    "critical": {
        "limit": "4Gi",
        "reservation": "2Gi"
    },
    # Performance agents (ports 10320-10419)
    "performance": {
        "limit": "2Gi",
        "reservation": "1Gi"
    },
    # Specialized agents (ports 10420-10599)
    "specialized": {
        "limit": "1Gi",
        "reservation": "512Mi"
    },
    # Infrastructure services
    "infrastructure": {
        "postgres": {"limit": "2Gi", "reservation": "1Gi"},
        "redis": {"limit": "1Gi", "reservation": "512Mi"},
        "neo4j": {"limit": "4Gi", "reservation": "2Gi"},
        "chromadb": {"limit": "2Gi", "reservation": "1Gi"},
        "qdrant": {"limit": "2Gi", "reservation": "1Gi"},
        "faiss": {"limit": "1Gi", "reservation": "512Mi"},
        "ollama": {"limit": "8Gi", "reservation": "4Gi"}
    },
    # Monitoring services
    "monitoring": {
        "prometheus": {"limit": "512Mi", "reservation": "256Mi"},
        "grafana": {"limit": "512Mi", "reservation": "256Mi"},
        "loki": {"limit": "512Mi", "reservation": "256Mi"},
        "alertmanager": {"limit": "256Mi", "reservation": "128Mi"}
    }
}

def get_agent_phase(port: int) -> str:
    """Determine agent phase based on port number"""
    if 10300 <= port <= 10319:
        return "critical"
    elif 10320 <= port <= 10419:
        return "performance"
    elif 10420 <= port <= 10599:
        return "specialized"
    else:
        return "infrastructure"

def get_current_containers() -> List[Dict]:
    """Get list of current running containers"""
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{json .}}"],
            capture_output=True,
            text=True,
            check=True
        )
        containers = []
        for line in result.stdout.strip().split('\n'):
            if line:
                containers.append(json.loads(line))
        return containers
    except subprocess.CalledProcessError:
        print("Error: Failed to get container list")
        return []

def check_memory_limits() -> Tuple[List[str], List[str]]:
    """Check which containers have memory limits"""
    containers_with_limits = []
    containers_without_limits = []
    
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            check=True
        )
        
        for container_name in result.stdout.strip().split('\n'):
            if container_name:
                # Check if container has memory limit
                inspect_result = subprocess.run(
                    ["docker", "inspect", container_name, "--format", "{{.HostConfig.Memory}}"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                memory_limit = inspect_result.stdout.strip()
                
                if memory_limit == "0":
                    containers_without_limits.append(container_name)
                else:
                    containers_with_limits.append(container_name)
                    
    except subprocess.CalledProcessError as e:
        print(f"Error inspecting containers: {e}")
        
    return containers_with_limits, containers_without_limits

def update_compose_file(file_path: Path, updates: Dict) -> bool:
    """Update docker-compose file with memory limits"""
    try:
        with open(file_path, 'r') as f:
            compose_data = yaml.safe_load(f)
        
        if not compose_data or 'services' not in compose_data:
            return False
            
        modified = False
        
        for service_name, service_config in compose_data['services'].items():
            # Check if service needs memory limits
            if 'deploy' not in service_config:
                service_config['deploy'] = {}
            
            if 'resources' not in service_config['deploy']:
                service_config['deploy']['resources'] = {}
                
            if 'limits' not in service_config['deploy']['resources']:
                service_config['deploy']['resources']['limits'] = {}
                
            if 'memory' not in service_config['deploy']['resources']['limits']:
                # Determine appropriate memory limit
                port = None
                if 'ports' in service_config and service_config['ports']:
                    # Extract port number
                    port_mapping = str(service_config['ports'][0])
                    if ':' in port_mapping:
                        port = int(port_mapping.split(':')[0])
                
                # Apply memory limit based on service type or port
                if service_name in MEMORY_LIMITS['infrastructure']:
                    limits = MEMORY_LIMITS['infrastructure'][service_name]
                elif service_name in MEMORY_LIMITS['monitoring']:
                    limits = MEMORY_LIMITS['monitoring'][service_name]
                elif port:
                    phase = get_agent_phase(port)
                    limits = MEMORY_LIMITS[phase]
                else:
                    # Default to specialized limits
                    limits = MEMORY_LIMITS['specialized']
                
                service_config['deploy']['resources']['limits']['memory'] = limits['limit']
                
                if 'reservations' not in service_config['deploy']['resources']:
                    service_config['deploy']['resources']['reservations'] = {}
                service_config['deploy']['resources']['reservations']['memory'] = limits['reservation']
                
                modified = True
                print(f"  ✓ Added memory limits to {service_name}: {limits['limit']} (reserved: {limits['reservation']})")
        
        if modified:
            # Backup original file
            backup_path = file_path.with_suffix('.yml.backup')
            os.rename(file_path, backup_path)
            
            # Write updated file
            with open(file_path, 'w') as f:
                yaml.dump(compose_data, f, default_flow_style=False, sort_keys=False)
            
            print(f"  ✓ Updated {file_path} (backup: {backup_path})")
            
        return modified
        
    except Exception as e:
        print(f"  ✗ Error updating {file_path}: {e}")
        return False

def create_memory_enforcement_script():
    """Create script to enforce memory limits on running containers"""
    script_content = '''#!/bin/bash
# Enforce memory limits on running containers
# This is a temporary fix until services are redeployed

set -euo pipefail

echo "Enforcing memory limits on running containers..."

# Function to convert memory string to bytes
convert_to_bytes() {
    local memory=$1
    local number=${memory//[^0-9]/}
    local unit=${memory//[0-9]/}
    
    case $unit in
        G|Gi) echo $((number * 1024 * 1024 * 1024)) ;;
        M|Mi) echo $((number * 1024 * 1024)) ;;
        K|Ki) echo $((number * 1024)) ;;
        *) echo $number ;;
    esac
}

# Apply limits to running containers
apply_memory_limit() {
    local container=$1
    local limit=$2
    
    # Convert limit to bytes
    local limit_bytes=$(convert_to_bytes $limit)
    
    echo "  Setting memory limit for $container to $limit"
    docker update --memory=$limit --memory-swap=$limit $container || {
        echo "  ⚠️  Failed to update $container"
    }
}

# Critical agents (ports 10300-10319)
for port in {10300..10319}; do
    container=$(docker ps --filter "publish=$port" --format "{{.Names}}" 2>/dev/null)
    if [ -n "$container" ]; then
        apply_memory_limit "$container" "4g"
    fi
done

# Performance agents (ports 10320-10419)
for port in {10320..10419}; do
    container=$(docker ps --filter "publish=$port" --format "{{.Names}}" 2>/dev/null)
    if [ -n "$container" ]; then
        apply_memory_limit "$container" "2g"
    fi
done

# Specialized agents (ports 10420-10599)
for port in {10420..10599}; do
    container=$(docker ps --filter "publish=$port" --format "{{.Names}}" 2>/dev/null)
    if [ -n "$container" ]; then
        apply_memory_limit "$container" "1g"
    fi
done

# Infrastructure services
docker update --memory=2g --memory-swap=2g sutazai-postgres 2>/dev/null || true
docker update --memory=1g --memory-swap=1g sutazai-redis 2>/dev/null || true
docker update --memory=4g --memory-swap=4g sutazai-neo4j 2>/dev/null || true
docker update --memory=2g --memory-swap=2g sutazai-chromadb 2>/dev/null || true
docker update --memory=2g --memory-swap=2g sutazai-qdrant 2>/dev/null || true
docker update --memory=1g --memory-swap=1g sutazai-faiss 2>/dev/null || true
docker update --memory=8g --memory-swap=8g sutazai-ollama 2>/dev/null || true

# Monitoring services
docker update --memory=512m --memory-swap=512m sutazai-prometheus 2>/dev/null || true
docker update --memory=512m --memory-swap=512m sutazai-grafana 2>/dev/null || true
docker update --memory=512m --memory-swap=512m sutazai-loki 2>/dev/null || true
docker update --memory=256m --memory-swap=256m sutazai-alertmanager 2>/dev/null || true

echo "✓ Memory limits enforcement completed"
echo ""
echo "Note: These are runtime updates. For permanent changes,"
echo "redeploy services with updated docker-compose files."
'''
    
    script_path = Path("/opt/sutazaiapp/scripts/enforce-memory-limits.sh")
    with open(script_path, 'w') as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)
    print(f"\n✓ Created memory enforcement script: {script_path}")

def main():
    print("🚀 Starting memory limits fix...")
    print("Target: Add memory limits to 34 containers missing them")
    print("=" * 50)
    
    # Step 1: Check current status
    print("\n📊 Step 1: Checking current container memory limits...")
    containers_with, containers_without = check_memory_limits()
    
    print(f"Containers with memory limits: {len(containers_with)}")
    print(f"Containers without memory limits: {len(containers_without)}")
    
    if containers_without:
        print("\nContainers missing memory limits:")
        for container in containers_without[:10]:  # Show first 10
            print(f"  - {container}")
        if len(containers_without) > 10:
            print(f"  ... and {len(containers_without) - 10} more")
    
    # Step 2: Update docker-compose files
    print("\n📝 Step 2: Updating docker-compose files...")
    compose_files = list(Path("/opt/sutazaiapp").glob("docker-compose*.yml"))
    
    updated_files = 0
    for compose_file in compose_files:
        print(f"\nProcessing: {compose_file}")
        if update_compose_file(compose_file, MEMORY_LIMITS):
            updated_files += 1
    
    print(f"\n✓ Updated {updated_files} docker-compose files")
    
    # Step 3: Create enforcement script
    print("\n🔧 Step 3: Creating runtime enforcement script...")
    create_memory_enforcement_script()
    
    # Step 4: Summary
    print("\n" + "=" * 50)
    print("✅ Memory limits fix completed!")
    print("\n📋 Summary of memory allocations:")
    print("  Critical agents: 4GB limit, 2GB reserved")
    print("  Performance agents: 2GB limit, 1GB reserved")
    print("  Specialized agents: 1GB limit, 512MB reserved")
    print("  Ollama: 8GB limit, 4GB reserved")
    print("  PostgreSQL: 2GB limit, 1GB reserved")
    print("  Neo4j: 4GB limit, 2GB reserved")
    
    print("\n🔧 Next steps:")
    print("  1. Apply runtime limits: ./scripts/enforce-memory-limits.sh")
    print("  2. Redeploy services: docker-compose up -d")
    print("  3. Monitor memory usage: docker stats")
    
    print("\n🎯 Expected outcome: All containers will have appropriate memory limits")

if __name__ == "__main__":
    main()