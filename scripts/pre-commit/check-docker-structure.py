#!/usr/bin/env python3
"""
Purpose: Validate Docker structure is clean and modular (Rule 11 enforcement)
Usage: python check-docker-structure.py <file1> <file2> ...
Requirements: Python 3.8+, docker (optional for advanced checks)
"""

import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple
import yaml
import json

class DockerStructureValidator:
    """Validates Docker structure compliance with Rule 11."""
    
    def __init__(self):
        self.violations = []
        self.warnings = []
        
    def check_dockerfile(self, filepath: Path) -> bool:
        """Check Dockerfile for best practices and structure."""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
                lines = content.splitlines()
            
            # Check for unpinned base images
            from_pattern = re.compile(r'^FROM\s+([^:]+)(:(.+))?', re.IGNORECASE)
            for line_num, line in enumerate(lines, 1):
                match = from_pattern.match(line.strip())
                if match:
                    image = match.group(1)
                    tag = match.group(3)
                    
                    if not tag or tag == 'latest':
                        self.violations.append((
                            filepath,
                            line_num,
                            f"Unpinned base image: {image}",
                            "Use specific version tags (e.g., node:18.17)"
                        ))
            
            # Check for multi-stage builds where appropriate
            from_count = len(re.findall(r'^FROM\s+', content, re.MULTILINE | re.IGNORECASE))
            if len(content) > 1000 and from_count == 1:
                self.warnings.append((
                    filepath,
                    "Consider using multi-stage build for smaller images"
                ))
            
            # Check for package installation best practices
            if 'apt-get install' in content or 'apk add' in content:
                if 'apt-get update' in content and 'apt-get install' not in content.replace('\n', ' '):
                    self.violations.append((
                        filepath,
                        0,
                        "apt-get update and install should be in same RUN command",
                        "Combine to avoid stale package cache"
                    ))
            
            # Check for .dockerignore reference
            dockerignore = filepath.parent / '.dockerignore'
            if not dockerignore.exists():
                self.warnings.append((
                    filepath,
                    "Missing .dockerignore file to exclude build context"
                ))
            
            # Check for COPY vs ADD usage
            add_count = len(re.findall(r'^ADD\s+', content, re.MULTILINE))
            if add_count > 0:
                self.warnings.append((
                    filepath,
                    f"Found {add_count} ADD commands - prefer COPY unless extracting archives"
                ))
                
            return len(self.violations) == 0
            
        except Exception as e:
            self.violations.append((
                filepath,
                0,
                f"Error reading Dockerfile: {e}",
                "Ensure file is valid"
            ))
            return False
    
    def check_docker_compose(self, filepath: Path) -> bool:
        """Check docker-compose.yml for structure and best practices."""
        try:
            with open(filepath, 'r') as f:
                compose_data = yaml.safe_load(f)
            
            if not isinstance(compose_data, dict):
                self.violations.append((
                    filepath,
                    0,
                    "Invalid docker-compose structure",
                    "File should contain a YAML dictionary"
                ))
                return False
            
            services = compose_data.get('services', {})
            
            # Check each service
            for service_name, service_config in services.items():
                # Check for build context
                if 'build' in service_config:
                    build_config = service_config['build']
                    if isinstance(build_config, dict):
                        dockerfile = build_config.get('dockerfile', 'Dockerfile')
                        context = build_config.get('context', '.')
                    else:
                        dockerfile = 'Dockerfile'
                        context = build_config
                    
                    # Verify Dockerfile exists
                    dockerfile_path = filepath.parent / context / dockerfile
                    if not dockerfile_path.exists():
                        self.violations.append((
                            filepath,
                            0,
                            f"Service '{service_name}' references missing Dockerfile: {dockerfile}",
                            "Ensure Dockerfile exists at specified path"
                        ))
                
                # Check for image versioning
                if 'image' in service_config:
                    image = service_config['image']
                    if ':' not in image or image.endswith(':latest'):
                        self.warnings.append((
                            filepath,
                            f"Service '{service_name}' uses unpinned image: {image}"
                        ))
                
                # Check for proper volume usage
                if 'volumes' in service_config:
                    for volume in service_config['volumes']:
                        if isinstance(volume, str) and volume.startswith('/'):
                            self.warnings.append((
                                filepath,
                                f"Service '{service_name}' uses absolute path volume: {volume}"
                            ))
            
            # Check for network configuration
            if 'networks' not in compose_data and len(services) > 1:
                self.warnings.append((
                    filepath,
                    "Consider defining custom networks for service isolation"
                ))
                
            return len(self.violations) == 0
            
        except yaml.YAMLError as e:
            self.violations.append((
                filepath,
                0,
                f"Invalid YAML syntax: {e}",
                "Fix YAML formatting"
            ))
            return False
        except Exception as e:
            self.violations.append((
                filepath,
                0,
                f"Error reading docker-compose file: {e}",
                "Ensure file is valid"
            ))
            return False
    
    def check_docker_structure(self, project_root: Path) -> bool:
        """Check overall Docker structure in the project."""
        # Expected structure
        docker_files = {
            'root_level': [],
            'docker_dir': [],
            'service_level': []
        }
        
        # Find all Dockerfiles and docker-compose files
        for pattern in ['Dockerfile*', 'docker-compose*.yml', 'docker-compose*.yaml']:
            for filepath in project_root.rglob(pattern):
                if 'archive' in str(filepath) or '.git' in str(filepath):
                    continue
                    
                relative_path = filepath.relative_to(project_root)
                path_parts = relative_path.parts
                
                if len(path_parts) == 1:
                    docker_files['root_level'].append(filepath)
                elif path_parts[0] == 'docker':
                    docker_files['docker_dir'].append(filepath)
                else:
                    docker_files['service_level'].append(filepath)
        
        # Check for scattered Docker files
        total_files = sum(len(files) for files in docker_files.values())
        if total_files > 10 and not docker_files['docker_dir']:
            self.warnings.append((
                project_root,
                f"Found {total_files} Docker files scattered across project - consider consolidating in /docker directory"
            ))
        
        return True

def main():
    """Main function to validate Docker structure."""
    validator = DockerStructureValidator()
    
    # If no files specified, check project structure
    if len(sys.argv) < 2:
        project_root = Path("/opt/sutazaiapp")
        validator.check_docker_structure(project_root)
    else:
        # Check specified files
        for filepath_str in sys.argv[1:]:
            filepath = Path(filepath_str)
            
            if 'Dockerfile' in filepath.name:
                validator.check_dockerfile(filepath)
            elif filepath.name.startswith('docker-compose'):
                validator.check_docker_compose(filepath)
    
    # Report results
    if validator.violations:
        print("❌ Rule 11 Violations: Docker structure issues detected")
        print("\n📋 Violations that must be fixed:")
        
        for item in validator.violations:
            if len(item) == 4:
                filepath, line_num, message, fix = item
                print(f"\n  File: {filepath}")
                if line_num > 0:
                    print(f"  Line: {line_num}")
                print(f"  Issue: {message}")
                print(f"  Fix: {fix}")
            else:
                filepath, line_num, message, fix = item[0], 0, str(item), ""
                print(f"\n  Issue: {message}")
        
        return 1
    
    if validator.warnings:
        print("⚠️  Docker structure warnings:")
        for warning in validator.warnings:
            if isinstance(warning, tuple) and len(warning) == 2:
                filepath, message = warning
                print(f"\n  File: {filepath}")
                print(f"  Warning: {message}")
            else:
                print(f"\n  Warning: {warning}")
    
    if not validator.violations:
        print("✅ Rule 11: Docker structure validation passed")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())