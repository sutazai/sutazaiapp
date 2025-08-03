#!/usr/bin/env python3
"""
Purpose: Quick container analysis without builds - assess scope and dependencies
Usage: python quick-container-analysis.py
Requirements: Python 3.8+, yaml

Fast analysis to understand container landscape before full validation.
"""

import os
import json
import yaml
import logging
from pathlib import Path
from collections import defaultdict
import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def discover_containers():
    """Quick discovery of all containers and requirements"""
    project_root = Path("/opt/sutazaiapp")
    
    # Find all Dockerfiles
    dockerfiles = {}
    for dockerfile in project_root.rglob("Dockerfile*"):
        if any(skip in str(dockerfile) for skip in ['.git', '__pycache__', 'venv']):
            continue
        
        service_name = extract_service_name(dockerfile)
        dockerfiles[service_name] = str(dockerfile)
    
    # Find all requirements
    requirements = defaultdict(list)
    patterns = ['requirements*.txt', 'package.json', 'pyproject.toml']
    
    for pattern in patterns:
        for req_file in project_root.rglob(pattern):
            if any(skip in str(req_file) for skip in ['.git', '__pycache__', 'venv']):
                continue
            
            service_name = extract_service_name(req_file)
            requirements[service_name].append(str(req_file))
    
    return dockerfiles, dict(requirements)

def extract_service_name(file_path):
    """Extract service name from path"""
    parts = file_path.parts
    
    for i, part in enumerate(parts):
        if part in ['docker', 'agents', 'services']:
            if i + 1 < len(parts):
                return parts[i + 1]
        elif part.endswith(('-service', '-agent', '-manager')):
            return part
    
    return file_path.parent.name

def analyze_compose_files():
    """Analyze docker-compose files to identify active services"""
    project_root = Path("/opt/sutazaiapp")
    active_services = set()
    
    for compose_file in project_root.glob("docker-compose*.yml"):
        try:
            with open(compose_file) as f:
                data = yaml.safe_load(f)
                if 'services' in data:
                    active_services.update(data['services'].keys())
                    logger.info(f"Found {len(data['services'])} services in {compose_file.name}")
        except Exception as e:
            logger.warning(f"Could not parse {compose_file}: {e}")
    
    return active_services

def identify_critical_services(active_services, dockerfiles):
    """Identify critical services based on composition and dependencies"""
    critical_base = {
        'backend', 'frontend', 'ollama', 'postgres', 'redis', 'nginx',
        'monitoring', 'loki', 'grafana', 'prometheus', 'chromadb', 'qdrant'
    }
    
    # Add services from main compose
    critical = critical_base.union(active_services)
    
    # Add any service that has a Dockerfile in main areas
    main_services = set()
    for service, dockerfile_path in dockerfiles.items():
        if any(main_area in dockerfile_path for main_area in ['/backend/', '/frontend/', '/docker/']):
            main_services.add(service)
    
    critical.update(main_services)
    
    return critical

def main():
    logger.info("🔍 Quick container analysis starting...")
    
    # Discovery
    dockerfiles, requirements = discover_containers()
    active_services = analyze_compose_files()
    critical_services = identify_critical_services(active_services, dockerfiles)
    
    # Summary
    results = {
        'timestamp': datetime.datetime.now().isoformat(),
        'total_dockerfiles': len(dockerfiles),
        'total_services_with_requirements': len(requirements),
        'active_services_in_compose': len(active_services),
        'identified_critical_services': len(critical_services),
        'critical_services': sorted(list(critical_services)),
        'active_services': sorted(list(active_services))
    }
    
    # Service breakdown
    services_with_both = set(dockerfiles.keys()).intersection(set(requirements.keys()))
    services_dockerfile_only = set(dockerfiles.keys()) - set(requirements.keys())
    services_requirements_only = set(requirements.keys()) - set(dockerfiles.keys())
    
    results['services_analysis'] = {
        'with_both_dockerfile_and_requirements': len(services_with_both),
        'dockerfile_only': len(services_dockerfile_only),
        'requirements_only': len(services_requirements_only)
    }
    
    # Save results
    os.makedirs("/opt/sutazaiapp/reports", exist_ok=True)
    report_path = "/opt/sutazaiapp/reports/quick_container_analysis.json"
    
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n📊 Container Analysis Summary")
    print(f"{'='*50}")
    print(f"Total Dockerfiles: {results['total_dockerfiles']}")
    print(f"Services with Requirements: {results['total_services_with_requirements']}")
    print(f"Active in Compose: {results['active_services_in_compose']}")
    print(f"Critical Services: {results['identified_critical_services']}")
    print(f"\n🔧 Service Breakdown:")
    print(f"  - Complete (Dockerfile + Requirements): {results['services_analysis']['with_both_dockerfile_and_requirements']}")
    print(f"  - Dockerfile Only: {results['services_analysis']['dockerfile_only']}")
    print(f"  - Requirements Only: {results['services_analysis']['requirements_only']}")
    
    print(f"\n🎯 Critical Services to Validate:")
    for service in sorted(critical_services):
        has_dockerfile = service in dockerfiles
        has_requirements = service in requirements
        status = "✅" if (has_dockerfile and has_requirements) else "⚠️" if has_dockerfile else "❌"
        print(f"  {status} {service}")
    
    print(f"\n📄 Report saved: {report_path}")
    
    return results

if __name__ == "__main__":
    main()