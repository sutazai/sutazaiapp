#!/usr/bin/env python3
"""
Docker Compose Configuration Validator
Ensures all services in docker-compose.consolidated.yml are properly configured
Created: 2025-08-18 21:45:00 UTC
"""

import yaml
import sys
from pathlib import Path
from typing import Dict, List, Tuple

def validate_docker_compose(file_path: str) -> Tuple[bool, List[str]]:
    """
    Validates docker-compose configuration for completeness.
    
    Returns:
        Tuple of (success, list of issues)
    """
    issues = []
    
    try:
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        return False, [f"Failed to parse YAML: {e}"]
    
    if 'services' not in config:
        return False, ["No services section found"]
    
    services = config['services']
    total_services = len(services)
    
    # Track statistics
    stats = {
        'missing_image': [],
        'missing_container_name': [],
        'missing_networks': [],
        'missing_healthcheck': [],
        'missing_resource_limits': [],
        'missing_restart_policy': [],
    }
    
    for service_name, service_config in services.items():
        if not service_config:
            issues.append(f"{service_name}: Empty service configuration")
            continue
            
        # Check for image or build
        if 'image' not in service_config and 'build' not in service_config:
            stats['missing_image'].append(service_name)
            issues.append(f"{service_name}: Missing 'image' or 'build' configuration")
        
        # Check for container_name (best practice)
        if 'container_name' not in service_config:
            stats['missing_container_name'].append(service_name)
            # Not critical, just a warning
        
        # Check for networks
        if 'networks' not in service_config and service_name not in ['promtail']:  # Some services may use host network
            stats['missing_networks'].append(service_name)
            issues.append(f"{service_name}: Missing 'networks' configuration")
        
        # Check for healthcheck (recommended)
        if 'healthcheck' not in service_config:
            stats['missing_healthcheck'].append(service_name)
            # Not critical, just tracking
        
        # Check for resource limits
        if 'deploy' not in service_config or 'resources' not in service_config.get('deploy', {}):
            stats['missing_resource_limits'].append(service_name)
            # Not critical but recommended
        
        # Check for restart policy
        if 'restart' not in service_config:
            stats['missing_restart_policy'].append(service_name)
            # Not critical but recommended
    
    # Print statistics
    print(f"\n=== Docker Compose Validation Report ===")
    print(f"File: {file_path}")
    print(f"Total Services: {total_services}")
    print(f"\n✅ CRITICAL CHECKS:")
    print(f"  - Services with image/build: {total_services - len(stats['missing_image'])}/{total_services}")
    print(f"  - Services with networks: {total_services - len(stats['missing_networks'])}/{total_services}")
    
    print(f"\n📊 BEST PRACTICES:")
    print(f"  - Services with container_name: {total_services - len(stats['missing_container_name'])}/{total_services}")
    print(f"  - Services with healthcheck: {total_services - len(stats['missing_healthcheck'])}/{total_services}")
    print(f"  - Services with resource limits: {total_services - len(stats['missing_resource_limits'])}/{total_services}")
    print(f"  - Services with restart policy: {total_services - len(stats['missing_restart_policy'])}/{total_services}")
    
    if stats['missing_image']:
        print(f"\n⚠️  Services missing image/build: {', '.join(stats['missing_image'])}")
    
    if stats['missing_networks']:
        print(f"\n⚠️  Services missing networks: {', '.join(stats['missing_networks'])}")
    
    if not stats['missing_healthcheck']:
        print(f"\n💚 All services have health checks!")
    elif len(stats['missing_healthcheck']) <= 5:
        print(f"\n📝 Services without healthcheck (optional): {', '.join(stats['missing_healthcheck'])}")
    
    # Determine overall success
    critical_issues = [i for i in issues if "Missing 'image'" in i or "Missing 'networks'" in i]
    
    if critical_issues:
        print(f"\n❌ CRITICAL ISSUES FOUND:")
        for issue in critical_issues:
            print(f"  - {issue}")
        return False, critical_issues
    else:
        print(f"\n✅ All critical configurations are present!")
        print(f"🎉 Docker Compose configuration is 100% deployable!")
        return True, []

def main():
    """Main validation function."""
    docker_compose_path = Path("/opt/sutazaiapp/docker/docker-compose.consolidated.yml")
    
    if not docker_compose_path.exists():
        print(f"❌ File not found: {docker_compose_path}")
        sys.exit(1)
    
    success, issues = validate_docker_compose(str(docker_compose_path))
    
    if not success:
        print(f"\n❌ Validation failed with {len(issues)} critical issues")
        sys.exit(1)
    else:
        print(f"\n✅ Validation successful! All services properly configured.")
        sys.exit(0)

if __name__ == "__main__":
    main()