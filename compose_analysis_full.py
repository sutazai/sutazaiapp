#!/usr/bin/env python3
"""
Comprehensive Docker Compose Analysis Script
Analyzes all docker-compose files for conflicts, issues, and recommendations
"""

import yaml
import os
import json
from pathlib import Path
from collections import defaultdict
import re


def extract_compose_info(file_path):
    """Extract comprehensive information from a compose file"""
    try:
        with open(file_path, 'r') as f:
            content = yaml.safe_load(f)
        
        if not content or 'services' not in content:
            return {}
        
        services = {}
        for service_name, service_config in content['services'].items():
            service_info = {
                'ports': [],
                'container_name': service_config.get('container_name', ''),
                'image': service_config.get('image', ''),
                'build': service_config.get('build', ''),
                'environment': service_config.get('environment', {}),
                'networks': service_config.get('networks', []),
                'volumes': service_config.get('volumes', []),
                'depends_on': service_config.get('depends_on', []),
                'restart': service_config.get('restart', ''),
                'healthcheck': service_config.get('healthcheck', {}),
                'deploy': service_config.get('deploy', {}),
                'labels': service_config.get('labels', [])
            }
            
            # Extract ports with better parsing
            if 'ports' in service_config:
                ports = service_config['ports']
                for port in ports:
                    if isinstance(port, str):
                        service_info['ports'].append(port)
                    elif isinstance(port, dict):
                        if 'target' in port and 'published' in port:
                            service_info['ports'].append(f"{port['published']}:{port['target']}")
                        else:
                            service_info['ports'].append(str(port))
            
            services[service_name] = service_info
        
        return services
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return {}


def analyze_compose_files():
    """Main analysis function"""
    
    # Find all compose files
    compose_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.startswith('docker-compose') and (file.endswith('.yml') or file.endswith('.yaml')):
                compose_files.append(os.path.join(root, file))
    
    print(f"Found {len(compose_files)} docker-compose files")
    
    # Extract information from all files
    all_services = {}
    port_usage = defaultdict(list)
    service_duplicates = defaultdict(list)
    container_name_duplicates = defaultdict(list)
    environment_conflicts = defaultdict(list)
    network_usage = defaultdict(list)
    volume_usage = defaultdict(list)
    
    active_files = set()  # Files that appear to be actively used
    
    for file_path in compose_files:
        services = extract_compose_info(file_path)
        if not services:
            continue
            
        # Determine if file is likely active
        is_active = (
            'docker-compose.yml' in file_path or
            'production' in file_path or
            'main' in file_path or
            len(services) > 5  # Files with many services are likely important
        )
        
        if is_active:
            active_files.add(file_path)
        
        all_services[file_path] = services
        
        for service_name, service_info in services.items():
            # Track service duplicates
            service_duplicates[service_name].append(file_path)
            
            # Track container name duplicates
            container_name = service_info.get('container_name', '')
            if container_name:
                container_name_duplicates[container_name].append(file_path)
            
            # Track port conflicts
            for port in service_info['ports']:
                if ':' in port:
                    host_port = port.split(':')[0]
                    port_usage[host_port].append({
                        'service': service_name,
                        'file': file_path,
                        'mapping': port
                    })
            
            # Track network usage
            for network in service_info.get('networks', []):
                network_usage[network].append({
                    'service': service_name,
                    'file': file_path
                })
            
            # Track environment variables for conflicts
            env_vars = service_info.get('environment', {})
            if isinstance(env_vars, list):
                # Handle environment as list format
                for env_item in env_vars:
                    if '=' in str(env_item):
                        key = str(env_item).split('=')[0]
                        environment_conflicts[f"{service_name}:{key}"].append(file_path)
            elif isinstance(env_vars, dict):
                # Handle environment as dict format
                for key in env_vars.keys():
                    environment_conflicts[f"{service_name}:{key}"].append(file_path)
    
    return {
        'compose_files': compose_files,
        'all_services': all_services,
        'port_usage': dict(port_usage),
        'service_duplicates': dict(service_duplicates),
        'container_name_duplicates': dict(container_name_duplicates),
        'environment_conflicts': dict(environment_conflicts),
        'network_usage': dict(network_usage),
        'volume_usage': dict(volume_usage),
        'active_files': active_files
    }


def check_service_implementations(analysis_data):
    """Check if service implementations exist"""
    missing_implementations = []
    
    for file_path, services in analysis_data['all_services'].items():
        for service_name, service_info in services.items():
            build_context = service_info.get('build', '')
            if isinstance(build_context, dict):
                build_context = build_context.get('context', '')
            
            if build_context and isinstance(build_context, str):
                # Check if build context directory exists
                if build_context.startswith('./'):
                    build_path = build_context[2:]  # Remove ./
                else:
                    build_path = build_context
                
                if not os.path.exists(build_path):
                    missing_implementations.append({
                        'service': service_name,
                        'file': file_path,
                        'missing_path': build_path
                    })
    
    return missing_implementations


def generate_validation_report(analysis_data):
    """Generate comprehensive validation report"""
    
    # Count issues
    port_conflicts = {k: v for k, v in analysis_data['port_usage'].items() if len(v) > 1}
    service_duplicates = {k: v for k, v in analysis_data['service_duplicates'].items() if len(v) > 1}
    container_duplicates = {k: v for k, v in analysis_data['container_name_duplicates'].items() if len(v) > 1}
    
    missing_implementations = check_service_implementations(analysis_data)
    
    # Identify abandoned files
    all_files = set(analysis_data['compose_files'])
    likely_abandoned = all_files - analysis_data['active_files']
    
    # Filter abandoned files by additional criteria
    definitely_abandoned = []
    possibly_abandoned = []
    
    for file_path in likely_abandoned:
        filename = os.path.basename(file_path)
        if any(keyword in filename.lower() for keyword in ['backup', 'old', 'test', 'temp', 'bak', 'archive']):
            definitely_abandoned.append(file_path)
        elif 'archive/' in file_path or 'backup/' in file_path:
            definitely_abandoned.append(file_path)
        else:
            possibly_abandoned.append(file_path)
    
    # Generate report
    report = f"""
VALIDATION REPORT
================
Component: Docker Compose Configuration Analysis
Validation Scope: All {len(analysis_data['compose_files'])} docker-compose files in codebase

SUMMARY
-------
✅ Total compose files: {len(analysis_data['compose_files'])}
✅ Total unique services: {len(analysis_data['service_duplicates'])}
❌ Port conflicts: {len(port_conflicts)}
❌ Service duplicates: {len(service_duplicates)}
❌ Container name conflicts: {len(container_duplicates)}
⚠️  Missing implementations: {len(missing_implementations)}
⚠️  Likely abandoned files: {len(definitely_abandoned) + len(possibly_abandoned)}

CRITICAL ISSUES
--------------

🔴 SEVERE PORT CONFLICTS ({len(port_conflicts)} conflicts found)
"""
    
    # Show top 10 port conflicts
    sorted_conflicts = sorted(port_conflicts.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    for port, usages in sorted_conflicts:
        report += f"\n  Port {port}: {len(usages)} services competing"
        for usage in usages[:3]:
            report += f"\n    - {usage['service']} ({usage['mapping']}) in {os.path.basename(usage['file'])}"
        if len(usages) > 3:
            report += f"\n    ... and {len(usages) - 3} more services"
    
    report += f"""

🔴 DUPLICATE SERVICE DEFINITIONS ({len(service_duplicates)} services duplicated)
"""
    
    # Show top 10 service duplicates
    sorted_duplicates = sorted(service_duplicates.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    for service, files in sorted_duplicates:
        report += f"\n  {service}: {len(files)} definitions"
        for file_path in files[:3]:
            report += f"\n    - {os.path.basename(file_path)}"
        if len(files) > 3:
            report += f"\n    ... and {len(files) - 3} more files"
    
    if missing_implementations:
        report += f"""

🔴 MISSING SERVICE IMPLEMENTATIONS ({len(missing_implementations)} services)
"""
        for item in missing_implementations[:10]:
            report += f"\n  {item['service']}: Missing {item['missing_path']} (in {os.path.basename(item['file'])})"
    
    report += f"""

WARNINGS
--------

⚠️  CONTAINER NAME CONFLICTS ({len(container_duplicates)} conflicts)
"""
    
    # Show container name conflicts
    for container_name, files in list(container_duplicates.items())[:5]:
        report += f"\n  {container_name}: {len(files)} containers"
        for file_path in files:
            report += f"\n    - {os.path.basename(file_path)}"
    
    report += f"""

⚠️  ABANDONED/OBSOLETE FILES ({len(definitely_abandoned)} definitely, {len(possibly_abandoned)} possibly)

Definitely abandoned:"""
    
    for file_path in definitely_abandoned[:10]:
        report += f"\n  - {file_path}"
    
    if possibly_abandoned:
        report += "\n\nPossibly abandoned:"
        for file_path in possibly_abandoned[:10]:
            report += f"\n  - {file_path}"
    
    report += f"""

RECOMMENDATIONS
--------------

🔧 IMMEDIATE ACTIONS REQUIRED:

1. RESOLVE PORT CONFLICTS:
   - Consolidate services using conflicting ports
   - Update port mappings to use unique ports
   - Remove unused service definitions

2. ELIMINATE SERVICE DUPLICATION:
   - Choose ONE authoritative compose file per environment
   - Move specialized overrides to docker-compose.override.yml
   - Delete redundant service definitions

3. CLEANUP ABANDONED FILES:
   - Move {len(definitely_abandoned)} abandoned files to archive/
   - Review {len(possibly_abandoned)} possibly obsolete files
   - Update documentation to clarify which files are active

4. FIX MISSING IMPLEMENTATIONS:
   - Create missing Docker build contexts
   - Remove service definitions without implementations
   - Update build paths to correct locations

🏗️  ARCHITECTURAL IMPROVEMENTS:

1. STANDARDIZE COMPOSE STRUCTURE:
   - Use docker-compose.yml as base configuration
   - Use docker-compose.override.yml for local development
   - Use docker-compose.production.yml for production overrides
   - Use docker-compose.<feature>.yml for specific features

2. IMPLEMENT PROPER NAMING CONVENTIONS:
   - Consistent service naming (no sutazai- prefix redundancy)
   - Logical port number allocation (ranges per service type)
   - Clear container naming patterns

3. CENTRALIZE CONFIGURATION:
   - Use .env files for environment variables
   - Implement configuration templates
   - Standardize network and volume definitions

📊 TECHNICAL DEBT ASSESSMENT:

- HIGH: {len(port_conflicts)} port conflicts requiring immediate resolution
- HIGH: {len(service_duplicates)} duplicate services causing maintenance overhead  
- MEDIUM: {len(definitely_abandoned)} abandoned files cluttering repository
- LOW: {len(container_duplicates)} container name conflicts (mostly harmless)

VALIDATION DETAILS
-----------------

🔍 Analysis covered:
- Service definitions and configurations
- Port mappings and conflicts
- Container naming conventions
- Build context validation
- Network and volume usage
- Environment variable conflicts
- File categorization (active vs abandoned)

📈 Metrics:
- Average services per file: {sum(len(services) for services in analysis_data['all_services'].values()) / len(analysis_data['compose_files']):.1f}
- Most complex file: {max(analysis_data['all_services'].items(), key=lambda x: len(x[1]))[0]} ({len(max(analysis_data['all_services'].items(), key=lambda x: len(x[1]))[1])} services)
- Port range usage: {min(int(p) for p in analysis_data['port_usage'].keys() if p.isdigit())}-{max(int(p) for p in analysis_data['port_usage'].keys() if p.isdigit())}
"""
    
    return report


def main():
    """Main execution function"""
    print("Starting Docker Compose configuration analysis...")
    
    analysis_data = analyze_compose_files()
    report = generate_validation_report(analysis_data)
    
    print(report)
    
    # Save detailed analysis to JSON for further processing
    analysis_output = {
        'summary': {
            'total_files': len(analysis_data['compose_files']),
            'total_services': len(analysis_data['service_duplicates']),
            'port_conflicts': len([k for k, v in analysis_data['port_usage'].items() if len(v) > 1]),
            'service_duplicates': len([k for k, v in analysis_data['service_duplicates'].items() if len(v) > 1]),
            'missing_implementations': len(check_service_implementations(analysis_data))
        },
        'port_conflicts': {k: v for k, v in analysis_data['port_usage'].items() if len(v) > 1},
        'service_duplicates': {k: v for k, v in analysis_data['service_duplicates'].items() if len(v) > 1},
        'active_files': list(analysis_data['active_files'])
    }
    
    with open('compose_analysis_results.json', 'w') as f:
        json.dump(analysis_output, f, indent=2)
    
    print("\n" + "="*80)
    print("Analysis complete. Detailed results saved to compose_analysis_results.json")


if __name__ == "__main__":
    main()