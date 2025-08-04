#!/usr/bin/env python3
"""
SutazAI Docker Build Analysis
Analyzes which Docker images need to be built
"""

import yaml
import os
import sys

def main():
    compose_file = 'docker-compose.yml'
    
    if not os.path.exists(compose_file):
        print(f"Error: {compose_file} not found")
        sys.exit(1)
    
    with open(compose_file, 'r') as f:
        compose = yaml.safe_load(f)
    
    services_to_build = []
    for service_name, service_config in compose.get('services', {}).items():
        if 'build' in service_config:
            build_config = service_config['build']
            if isinstance(build_config, dict):
                context = build_config.get('context', '.')
                dockerfile = build_config.get('dockerfile', 'Dockerfile')
            else:
                context = build_config
                dockerfile = 'Dockerfile'
            
            services_to_build.append({
                'service': service_name,
                'context': context,
                'dockerfile': dockerfile,
            })
    
    print(f"Found {len(services_to_build)} services requiring Docker builds:")
    print("=" * 80)
    
    ready_count = 0
    missing_count = 0
    
    for service in services_to_build:
        context_path = service['context']
        dockerfile_path = f"{context_path}/{service['dockerfile']}"
        
        dockerfile_exists = os.path.isfile(dockerfile_path)
        context_exists = os.path.isdir(context_path)
        
        if dockerfile_exists and context_exists:
            status = "✓ READY"
            ready_count += 1
        else:
            status = "✗ MISSING"
            missing_count += 1
        
        print(f"{status:10} {service['service']:25} -> {dockerfile_path}")
    
    print("=" * 80)
    print(f"Summary: {ready_count} ready, {missing_count} missing")
    
    if missing_count > 0:
        print(f"\nMissing Dockerfiles need to be created before building.")

if __name__ == "__main__":
    main()