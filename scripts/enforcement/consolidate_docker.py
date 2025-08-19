"""
Docker Configuration Consolidation Script
Created: 2025-08-18 16:55:00 UTC
Purpose: Consolidate all docker-compose files into single authoritative file
"""

import os
import yaml
import sys
from pathlib import Path
from datetime import datetime

def load_docker_compose(file_path):
    """Load a docker-compose file safely."""
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Could not load {file_path}: {e}")
        return None

def merge_services(base, override):
    """Merge service definitions intelligently."""
    for service_name, service_config in override.get('services', {}).items():
        if service_name not in base['services']:
            base['services'][service_name] = service_config
        else:
            for key, value in service_config.items():
                if key == 'environment':
                    if key not in base['services'][service_name]:
                        base['services'][service_name][key] = value
                    else:
                        if isinstance(value, dict) and isinstance(base['services'][service_name][key], dict):
                            base['services'][service_name][key].update(value)
                        elif isinstance(value, list):
                            if isinstance(base['services'][service_name][key], list):
                                base['services'][service_name][key].extend(
                                    v for v in value if v not in base['services'][service_name][key]
                                )
                            else:
                                base['services'][service_name][key] = value
                        else:
                            base['services'][service_name][key] = value
                elif key == 'volumes':
                    if key not in base['services'][service_name]:
                        base['services'][service_name][key] = []
                    base['services'][service_name][key].extend(
                        v for v in value if v not in base['services'][service_name][key]
                    )
                elif key == 'ports':
                    if key not in base['services'][service_name]:
                        base['services'][service_name][key] = []
                    base['services'][service_name][key].extend(
                        p for p in value if p not in base['services'][service_name][key]
                    )
                else:
                    base['services'][service_name][key] = value
    return base

def main():
    docker_dir = Path('/opt/sutazaiapp/docker')
    output_file = docker_dir / 'docker-compose.consolidated.yml'
    
    compose_files = list(docker_dir.glob('docker-compose*.yml'))
    compose_files.extend(docker_dir.glob('docker-compose*.yaml'))
    
    root_compose = Path('/opt/sutazaiapp/docker-compose.yml')
    if root_compose.exists():
        compose_files.append(root_compose)
    
    print(f"Found {len(compose_files)} docker-compose files to consolidate")
    
    consolidated = {
        'version': '3.8',
        'services': {},
        'networks': {
            'sutazai-network': {
                'driver': 'bridge',
                'name': 'sutazai-network'
            }
        },
        'volumes': {}
    }
    
    priority_order = [
        'docker-compose.yml',
        'docker-compose.base.yml',
        'docker-compose.secure.yml',
        'docker-compose.performance.yml',
        'docker-compose.optimized.yml'
    ]
    
    sorted_files = []
    for priority_name in priority_order:
        for f in compose_files:
            if f.name == priority_name:
                sorted_files.append(f)
                compose_files.remove(f)
                break
    sorted_files.extend(compose_files)  # Add remaining files
    
    for compose_file in sorted_files:
        print(f"Processing: {compose_file}")
        config = load_docker_compose(compose_file)
        if config:
            consolidated = merge_services(consolidated, config)
            
            if 'networks' in config:
                for net_name, net_config in config['networks'].items():
                    if net_name not in consolidated['networks']:
                        consolidated['networks'][net_name] = net_config
            
            if 'volumes' in config:
                for vol_name, vol_config in config['volumes'].items():
                    if vol_name not in consolidated['volumes']:
                        consolidated['volumes'][vol_name] = vol_config
    
    header = f"""# CONSOLIDATED DOCKER CONFIGURATION
"""
    
    with open(output_file, 'w') as f:
        f.write(header)
        f.write('\n')
        yaml.dump(consolidated, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✅ Consolidated configuration written to: {output_file}")
    print(f"   Total services: {len(consolidated['services'])}")
    print(f"   Total networks: {len(consolidated['networks'])}")
    print(f"   Total volumes: {len(consolidated['volumes'])}")
    
    deprecation_notice = """# DEPRECATED - DO NOT USE
""".format(datetime.utcnow().isoformat())
    
    for compose_file in sorted_files:
        if compose_file != output_file:
            deprecated_file = compose_file.with_suffix('.yml.deprecated')
            print(f"   Marking as deprecated: {compose_file} -> {deprecated_file}")
    
    print("\n⚠️  Next steps:")
    print("1. Review docker-compose.consolidated.yml for correctness")
    print("2. Test with: docker-compose -f docker-compose.consolidated.yml config")
    print("3. Once validated, rename old files to .deprecated")
    print("4. Update all scripts to use consolidated file")

if __name__ == '__main__':
    main()