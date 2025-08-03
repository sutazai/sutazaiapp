#!/usr/bin/env python3
"""
Purpose: Analyze Docker containers and map them to their requirements.txt files
Usage: python analyze-docker-requirements.py
Requirements: Python 3.8+, PyYAML
"""

import os
import re
import json
import yaml
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple

class DockerRequirementsAnalyzer:
    def __init__(self, root_path: str = "/opt/sutazaiapp"):
        self.root_path = Path(root_path)
        self.dockerfiles = []
        self.requirements_files = []
        self.container_map = {}
        self.orphaned_requirements = []
        self.duplicate_requirements = defaultdict(list)
        
    def find_all_files(self):
        """Find all Dockerfiles and requirements.txt files"""
        # Find Dockerfiles
        for dockerfile in self.root_path.rglob("Dockerfile*"):
            if dockerfile.is_file():
                self.dockerfiles.append(dockerfile)
                
        # Find requirements files
        for req_file in self.root_path.rglob("requirements*.txt"):
            if req_file.is_file():
                self.requirements_files.append(req_file)
    
    def analyze_dockerfile(self, dockerfile_path: Path) -> Dict:
        """Analyze a Dockerfile to find its requirements references"""
        container_info = {
            'dockerfile': str(dockerfile_path.relative_to(self.root_path)),
            'container_name': dockerfile_path.parent.name,
            'requirements': [],
            'base_image': None,
            'build_context': str(dockerfile_path.parent.relative_to(self.root_path))
        }
        
        try:
            with open(dockerfile_path, 'r') as f:
                content = f.read()
                
            # Extract base image
            base_match = re.search(r'^FROM\s+(\S+)', content, re.MULTILINE)
            if base_match:
                container_info['base_image'] = base_match.group(1)
            
            # Find requirements references
            req_patterns = [
                r'COPY\s+.*?(requirements.*?\.txt)',
                r'pip install.*?-r\s+(requirements.*?\.txt)',
                r'ADD\s+.*?(requirements.*?\.txt)'
            ]
            
            for pattern in req_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    req_path = dockerfile_path.parent / match
                    if req_path.exists():
                        container_info['requirements'].append({
                            'file': str(req_path.relative_to(self.root_path)),
                            'exists': True
                        })
                    else:
                        container_info['requirements'].append({
                            'file': match,
                            'exists': False
                        })
                        
        except Exception as e:
            container_info['error'] = str(e)
            
        return container_info
    
    def analyze_docker_compose(self) -> Dict[str, List[str]]:
        """Analyze docker-compose files to get service definitions"""
        services = {}
        compose_files = list(self.root_path.glob("docker-compose*.yml")) + \
                       list(self.root_path.glob("docker-compose*.yaml"))
        
        for compose_file in compose_files:
            try:
                with open(compose_file, 'r') as f:
                    compose_data = yaml.safe_load(f)
                    
                if 'services' in compose_data:
                    for service_name, service_config in compose_data['services'].items():
                        if 'build' in service_config:
                            build_context = service_config['build']
                            if isinstance(build_context, dict):
                                context = build_context.get('context', '.')
                                dockerfile = build_context.get('dockerfile', 'Dockerfile')
                            else:
                                context = build_context
                                dockerfile = 'Dockerfile'
                            
                            dockerfile_path = self.root_path / context / dockerfile
                            if dockerfile_path.exists():
                                services[service_name] = str(dockerfile_path.relative_to(self.root_path))
                                
            except Exception as e:
                print(f"Error parsing {compose_file}: {e}")
                
        return services
    
    def find_orphaned_requirements(self):
        """Find requirements.txt files not referenced by any Dockerfile"""
        referenced_requirements = set()
        
        for container in self.container_map.values():
            for req in container.get('requirements', []):
                if req['exists']:
                    referenced_requirements.add(req['file'])
        
        for req_file in self.requirements_files:
            req_path = str(req_file.relative_to(self.root_path))
            if req_path not in referenced_requirements:
                self.orphaned_requirements.append(req_path)
    
    def analyze_dependencies(self) -> Dict[str, Set[str]]:
        """Analyze common dependencies across requirements files"""
        dependencies = defaultdict(set)
        
        for req_file in self.requirements_files:
            try:
                with open(req_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            # Extract package name without version
                            pkg_match = re.match(r'^([a-zA-Z0-9\-_]+)', line)
                            if pkg_match:
                                pkg_name = pkg_match.group(1)
                                dependencies[pkg_name].add(str(req_file.relative_to(self.root_path)))
            except Exception:
                pass
                
        return dependencies
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        # Find all files
        self.find_all_files()
        
        # Analyze each Dockerfile
        for dockerfile in self.dockerfiles:
            container_info = self.analyze_dockerfile(dockerfile)
            self.container_map[str(dockerfile)] = container_info
        
        # Analyze docker-compose services
        compose_services = self.analyze_docker_compose()
        
        # Find orphaned requirements
        self.find_orphaned_requirements()
        
        # Analyze common dependencies
        common_deps = self.analyze_dependencies()
        shared_deps = {pkg: list(files) for pkg, files in common_deps.items() if len(files) > 3}
        
        # Generate report
        report = {
            'summary': {
                'total_containers': len(self.dockerfiles),
                'total_requirements_files': len(self.requirements_files),
                'orphaned_requirements': len(self.orphaned_requirements),
                'containers_without_requirements': sum(1 for c in self.container_map.values() if not c['requirements'])
            },
            'containers': self.container_map,
            'compose_services': compose_services,
            'orphaned_requirements': self.orphaned_requirements,
            'shared_dependencies': shared_deps,
            'validation_commands': self.generate_validation_commands(),
            'cleanup_recommendations': self.generate_cleanup_recommendations()
        }
        
        return report
    
    def generate_validation_commands(self) -> List[str]:
        """Generate commands to validate each container"""
        commands = []
        
        for service in ['backend', 'frontend', 'autogpt', 'crewai', 'letta', 'aider']:
            commands.append(f"# Validate {service} container")
            commands.append(f"docker-compose build {service} --no-cache")
            commands.append(f"docker-compose run --rm {service} python -c 'import sys; print(sys.version)'")
            commands.append("")
            
        return commands
    
    def generate_cleanup_recommendations(self) -> Dict[str, List[str]]:
        """Generate safe cleanup recommendations"""
        recommendations = {
            'safe_to_remove': [],
            'consolidate': [],
            'backup_first': [],
            'keep': []
        }
        
        # Orphaned requirements can be removed after backup
        for orphan in self.orphaned_requirements:
            if any(x in orphan for x in ['backup', 'old', 'copy', 'test', 'archive']):
                recommendations['safe_to_remove'].append(orphan)
            else:
                recommendations['backup_first'].append(orphan)
        
        # Identify consolidation opportunities
        for pkg, files in self.analyze_dependencies().items():
            if len(files) > 5:
                recommendations['consolidate'].append(f"{pkg} appears in {len(files)} files")
        
        return recommendations

def main():
    analyzer = DockerRequirementsAnalyzer()
    report = analyzer.generate_report()
    
    # Save detailed report
    with open('/opt/sutazaiapp/docker-requirements-analysis.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("=== Docker Container Requirements Analysis ===\n")
    print(f"Total Containers Found: {report['summary']['total_containers']}")
    print(f"Total Requirements Files: {report['summary']['total_requirements_files']}")
    print(f"Orphaned Requirements: {report['summary']['orphaned_requirements']}")
    print(f"Containers Without Requirements: {report['summary']['containers_without_requirements']}")
    
    print("\n=== Orphaned Requirements Files ===")
    for orphan in report['orphaned_requirements'][:10]:
        print(f"  - {orphan}")
    
    if len(report['orphaned_requirements']) > 10:
        print(f"  ... and {len(report['orphaned_requirements']) - 10} more")
    
    print("\n=== Shared Dependencies (appearing in >3 files) ===")
    for dep, files in list(report['shared_dependencies'].items())[:10]:
        print(f"  - {dep}: {len(files)} files")
    
    print("\n=== Cleanup Recommendations ===")
    print(f"Safe to Remove: {len(report['cleanup_recommendations']['safe_to_remove'])} files")
    print(f"Backup First: {len(report['cleanup_recommendations']['backup_first'])} files")
    
    print("\nFull report saved to: /opt/sutazaiapp/docker-requirements-analysis.json")

if __name__ == "__main__":
    main()