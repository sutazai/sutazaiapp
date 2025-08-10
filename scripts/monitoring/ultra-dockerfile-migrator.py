#!/usr/bin/env python3
"""
ULTRA Dockerfile Migration Tool
Migrates Dockerfiles to use master base images with zero downtime
Author: ULTRA SYSTEM ARCHITECT
Date: August 10, 2025
"""

import os
import re
import shutil
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

class UltraDockerfileMigrator:
    """Orchestrates the migration of Dockerfiles to master base images"""
    
    def __init__(self):
        self.python_master = "sutazai-python-agent-master:latest"
        self.nodejs_master = "sutazai-nodejs-agent-master:latest"
        self.project_root = Path("/opt/sutazaiapp")
        self.migration_report = []
        self.skip_patterns = [
            # Infrastructure services that should NOT be migrated
            'postgres', 'redis', 'neo4j', 'rabbitmq', 'mongodb',
            'qdrant', 'chromadb', 'elasticsearch', 'influxdb',
            'prometheus', 'grafana', 'loki', 'alertmanager',
            'kong', 'consul', 'ollama/ollama', 'jaeger',
            'nvidia/cuda', 'tensorflow/tensorflow', 'pytorch/pytorch',
            'mcr.microsoft.com', 'hashicorp/', 'prom/', 'grafana/',
        ]
        self.stats = {
            'total': 0,
            'migrated': 0,
            'skipped': 0,
            'failed': 0,
            'already_migrated': 0
        }
        
    def scan_dockerfiles(self) -> List[Path]:
        """Find all Dockerfiles in the project"""
        dockerfiles = []
        for dockerfile in self.project_root.rglob("Dockerfile*"):
            if dockerfile.is_file():
                dockerfiles.append(dockerfile)
        self.stats['total'] = len(dockerfiles)
        return dockerfiles
    
    def detect_technology(self, dockerfile_path: Path) -> str:
        """Detect the primary technology stack of a Dockerfile"""
        try:
            content = dockerfile_path.read_text()
            content_lower = content.lower()
            
            # Check for Python indicators
            python_indicators = ['python', 'pip', 'requirements.txt', 'poetry', 'conda', 'uvicorn', 'fastapi', 'flask', 'django']
            python_score = sum(1 for indicator in python_indicators if indicator in content_lower)
            
            # Check for Node.js indicators
            node_indicators = ['node', 'npm', 'yarn', 'package.json', 'express', 'react', 'vue', 'angular', 'next']
            node_score = sum(1 for indicator in node_indicators if indicator in content_lower)
            
            # Check for Go indicators
            go_indicators = ['golang', 'go build', 'go mod', 'go.mod', 'go.sum']
            go_score = sum(1 for indicator in go_indicators if indicator in content_lower)
            
            # Determine primary technology
            if python_score > node_score and python_score > go_score:
                return 'python'
            elif node_score > python_score and node_score > go_score:
                return 'nodejs'
            elif go_score > 0:
                return 'golang'
            else:
                return 'unknown'
                
        except Exception as e:
            print(f"Error detecting technology for {dockerfile_path}: {e}")
            return 'unknown'
    
    def should_migrate(self, dockerfile_path: Path) -> Tuple[bool, str]:
        """Determine if a Dockerfile should be migrated"""
        try:
            content = dockerfile_path.read_text()
            first_line = content.split('\n')[0] if content else ''
            
            # Check if already migrated
            if 'sutazai-python-agent-master' in content or 'sutazai-nodejs-agent-master' in content:
                return False, "already_migrated"
            
            # Check against skip patterns
            for pattern in self.skip_patterns:
                if pattern in first_line.lower() or pattern in content.lower()[:500]:
                    return False, f"infrastructure_service ({pattern})"
            
            # Check if it's a base image itself
            if 'base' in dockerfile_path.name.lower() or '/base/' in str(dockerfile_path):
                return False, "base_image"
                
            return True, "eligible"
            
        except Exception as e:
            print(f"Error checking {dockerfile_path}: {e}")
            return False, f"error: {e}"
    
    def extract_dockerfile_components(self, content: str) -> Dict:
        """Extract key components from a Dockerfile"""
        components = {
            'base_image': None,
            'workdir': '/app',
            'expose_ports': [],
            'env_vars': [],
            'copy_commands': [],
            'run_commands': [],
            'cmd': None,
            'entrypoint': None,
            'user': None,
            'healthcheck': None,
            'requirements_files': [],
            'package_files': []
        }
        
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            
            if line.startswith('FROM '):
                components['base_image'] = line[5:].strip()
            elif line.startswith('WORKDIR '):
                components['workdir'] = line[8:].strip()
            elif line.startswith('EXPOSE '):
                ports = re.findall(r'\d+', line)
                components['expose_ports'].extend(ports)
            elif line.startswith('ENV '):
                components['env_vars'].append(line[4:].strip())
            elif line.startswith('COPY '):
                components['copy_commands'].append(line[5:].strip())
                if 'requirements' in line.lower():
                    components['requirements_files'].append(line)
                if 'package.json' in line.lower():
                    components['package_files'].append(line)
            elif line.startswith('RUN '):
                components['run_commands'].append(line[4:].strip())
            elif line.startswith('CMD '):
                components['cmd'] = line[4:].strip()
            elif line.startswith('ENTRYPOINT '):
                components['entrypoint'] = line[11:].strip()
            elif line.startswith('USER '):
                components['user'] = line[5:].strip()
            elif line.startswith('HEALTHCHECK '):
                components['healthcheck'] = line
                
        return components
    
    def generate_migrated_dockerfile(self, dockerfile_path: Path, technology: str) -> Optional[str]:
        """Generate a migrated version of the Dockerfile"""
        try:
            content = dockerfile_path.read_text()
            components = self.extract_dockerfile_components(content)
            
            # Select appropriate base image
            if technology == 'python':
                base_image = self.python_master
            elif technology == 'nodejs':
                base_image = self.nodejs_master
            else:
                return None  # Can't migrate unknown technology
            
            # Build new Dockerfile
            lines = [
                f"# Migrated to Master Base Image",
                f"# Original: {components['base_image']}",
                f"# Migration Date: {datetime.now().isoformat()}",
                f"FROM {base_image}",
                ""
            ]
            
            # Add workdir if different from default
            if components['workdir'] != '/app':
                lines.extend([f"WORKDIR {components['workdir']}", ""])
            
            # Add requirements/package installation
            if technology == 'python' and components['requirements_files']:
                for req_file in components['requirements_files']:
                    lines.append(f"COPY {req_file}")
                lines.extend([
                    "RUN pip install --no-cache-dir -r requirements.txt && rm requirements.txt",
                    ""
                ])
            elif technology == 'nodejs' and components['package_files']:
                for pkg_file in components['package_files']:
                    lines.append(f"COPY {pkg_file}")
                lines.extend([
                    "RUN npm install --only=production && npm cache clean --force",
                    ""
                ])
            
            # Add service-specific RUN commands (filtered)
            important_run_commands = []
            for run_cmd in components['run_commands']:
                # Skip apt-get, system package installations (handled by base)
                if not any(skip in run_cmd.lower() for skip in ['apt-get', 'apk add', 'yum install', 'pip install numpy', 'pip install pandas']):
                    important_run_commands.append(run_cmd)
            
            if important_run_commands:
                lines.append("# Service-specific setup")
                for run_cmd in important_run_commands[:3]:  # Limit to avoid bloat
                    lines.append(f"RUN {run_cmd}")
                lines.append("")
            
            # Add copy commands
            lines.append("# Copy application code")
            app_copied = False
            for copy_cmd in components['copy_commands']:
                if '. .' in copy_cmd or '. /' in copy_cmd:
                    if not app_copied:
                        lines.append("COPY . .")
                        app_copied = True
                elif 'requirements' not in copy_cmd.lower() and 'package.json' not in copy_cmd.lower():
                    lines.append(f"COPY {copy_cmd}")
            if not app_copied:
                lines.append("COPY . .")
            lines.append("")
            
            # Add environment variables
            if components['env_vars']:
                lines.append("# Environment configuration")
                for env_var in components['env_vars'][:5]:  # Limit to avoid bloat
                    lines.append(f"ENV {env_var}")
            
            # Add service port
            if components['expose_ports']:
                port = components['expose_ports'][0]
                lines.extend([
                    f"ENV SERVICE_PORT={port}",
                    f"EXPOSE {port}",
                    ""
                ])
            
            # Add healthcheck if exists
            if components['healthcheck']:
                lines.extend([components['healthcheck'], ""])
            
            # Switch to non-root user
            lines.extend(["# Security: Run as non-root user", "USER appuser", ""])
            
            # Add CMD or ENTRYPOINT
            if components['entrypoint']:
                lines.append(f"ENTRYPOINT {components['entrypoint']}")
            if components['cmd']:
                lines.append(f"CMD {components['cmd']}")
            elif not components['entrypoint']:
                # Default CMD based on technology
                if technology == 'python':
                    lines.append('CMD ["python", "app.py"]')
                elif technology == 'nodejs':
                    lines.append('CMD ["node", "index.js"]')
            
            return '\n'.join(lines)
            
        except Exception as e:
            print(f"Error generating migrated Dockerfile for {dockerfile_path}: {e}")
            return None
    
    def validate_migration(self, dockerfile_path: Path, new_content: str) -> bool:
        """Validate that the migrated Dockerfile is correct"""
        # Basic validation checks
        required_elements = ['FROM sutazai-', 'USER appuser', 'COPY']
        for element in required_elements:
            if element not in new_content:
                print(f"Validation failed for {dockerfile_path}: Missing {element}")
                return False
        
        # Check that it's not too short (indicates missing content)
        if len(new_content.split('\n')) < 10:
            print(f"Validation failed for {dockerfile_path}: Content too short")
            return False
            
        return True
    
    def migrate_dockerfile(self, dockerfile_path: Path, dry_run: bool = False) -> bool:
        """Migrate a single Dockerfile"""
        # Check if should migrate
        should_migrate, reason = self.should_migrate(dockerfile_path)
        
        if not should_migrate:
            if reason == "already_migrated":
                self.stats['already_migrated'] += 1
            else:
                self.stats['skipped'] += 1
            self.migration_report.append({
                'path': str(dockerfile_path),
                'status': 'skipped',
                'reason': reason
            })
            return False
        
        # Detect technology
        technology = self.detect_technology(dockerfile_path)
        if technology == 'unknown':
            self.stats['skipped'] += 1
            self.migration_report.append({
                'path': str(dockerfile_path),
                'status': 'skipped',
                'reason': 'unknown_technology'
            })
            return False
        
        # Generate migrated version
        new_content = self.generate_migrated_dockerfile(dockerfile_path, technology)
        if not new_content:
            self.stats['failed'] += 1
            self.migration_report.append({
                'path': str(dockerfile_path),
                'status': 'failed',
                'reason': 'generation_failed'
            })
            return False
        
        # Validate migration
        if not self.validate_migration(dockerfile_path, new_content):
            self.stats['failed'] += 1
            self.migration_report.append({
                'path': str(dockerfile_path),
                'status': 'failed',
                'reason': 'validation_failed'
            })
            return False
        
        if not dry_run:
            # Backup original
            backup_path = dockerfile_path.with_suffix('.backup')
            shutil.copy2(dockerfile_path, backup_path)
            
            # Write new content
            dockerfile_path.write_text(new_content)
        
        self.stats['migrated'] += 1
        self.migration_report.append({
            'path': str(dockerfile_path),
            'status': 'migrated',
            'technology': technology,
            'base_image': self.python_master if technology == 'python' else self.nodejs_master
        })
        
        print(f"✅ Migrated: {dockerfile_path.relative_to(self.project_root)} ({technology})")
        return True
    
    def run_migration(self, dry_run: bool = False, target_priority: str = None):
        """Run the full migration process"""
        print("=" * 80)
        print("ULTRA DOCKERFILE MIGRATION TOOL")
        print(f"Mode: {'DRY RUN' if dry_run else 'LIVE MIGRATION'}")
        print("=" * 80)
        
        # Scan for Dockerfiles
        dockerfiles = self.scan_dockerfiles()
        print(f"Found {len(dockerfiles)} Dockerfiles")
        
        # Categorize by priority
        priorities = {
            'P0': [],  # Critical - Running in production (but most are infrastructure)
            'P1': [],  # Important - AI/ML services
            'P2': [],  # Standard - Utility services
            'P3': []   # Optional - Experimental/deprecated
        }
        
        for dockerfile in dockerfiles:
            path_str = str(dockerfile)
            
            # Categorize based on path and content
            if any(x in path_str for x in ['/agents/', '/backend/', '/frontend/']):
                priorities['P0'].append(dockerfile)
            elif any(x in path_str for x in ['/docker/ai', '/docker/ml', '/docker/model']):
                priorities['P1'].append(dockerfile)
            elif any(x in path_str for x in ['/docker/', '/services/']):
                priorities['P2'].append(dockerfile)
            else:
                priorities['P3'].append(dockerfile)
        
        # Filter by target priority if specified
        if target_priority:
            dockerfiles_to_migrate = priorities.get(target_priority, [])
            print(f"Migrating {target_priority} priority services only: {len(dockerfiles_to_migrate)} files")
        else:
            dockerfiles_to_migrate = dockerfiles
            print(f"Migrating all priorities: {len(dockerfiles_to_migrate)} files")
        
        # Migrate each Dockerfile
        for dockerfile in dockerfiles_to_migrate:
            try:
                self.migrate_dockerfile(dockerfile, dry_run)
            except Exception as e:
                print(f"❌ Error migrating {dockerfile}: {e}")
                self.stats['failed'] += 1
        
        # Generate report
        self.generate_report()
    
    def generate_report(self):
        """Generate migration report"""
        print("\n" + "=" * 80)
        print("MIGRATION REPORT")
        print("=" * 80)
        
        print(f"Total Dockerfiles: {self.stats['total']}")
        print(f"Successfully Migrated: {self.stats['migrated']}")
        print(f"Already Migrated: {self.stats['already_migrated']}")
        print(f"Skipped (Infrastructure): {self.stats['skipped']}")
        print(f"Failed: {self.stats['failed']}")
        
        # Save detailed report
        report_path = self.project_root / f"migration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'stats': self.stats,
                'details': self.migration_report
            }, f, indent=2)
        
        print(f"\nDetailed report saved to: {report_path}")
        
        # Show migration success rate
        if self.stats['migrated'] > 0:
            success_rate = (self.stats['migrated'] / (self.stats['migrated'] + self.stats['failed'])) * 100
            print(f"\nMigration Success Rate: {success_rate:.1f}%")
    
    def validate_service(self, service_name: str) -> bool:
        """Validate a migrated service is working correctly"""
        try:
            # Build the new image
            build_result = subprocess.run(
                ["docker", "build", "-t", f"{service_name}:test", "."],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if build_result.returncode != 0:
                print(f"Build failed for {service_name}: {build_result.stderr}")
                return False
            
            # Run container and test health
            run_result = subprocess.run(
                ["docker", "run", "-d", "--name", f"{service_name}-test", f"{service_name}:test"],
                capture_output=True,
                text=True
            )
            
            if run_result.returncode != 0:
                print(f"Run failed for {service_name}: {run_result.stderr}")
                return False
            
            # Wait for container to start
            import time
            time.sleep(5)
            
            # Check if container is still running
            ps_result = subprocess.run(
                ["docker", "ps", "-f", f"name={service_name}-test"],
                capture_output=True,
                text=True
            )
            
            if service_name in ps_result.stdout:
                print(f"✅ {service_name} validated successfully")
                # Cleanup
                subprocess.run(["docker", "stop", f"{service_name}-test"])
                subprocess.run(["docker", "rm", f"{service_name}-test"])
                return True
            else:
                print(f"❌ {service_name} validation failed - container not running")
                return False
                
        except Exception as e:
            print(f"Validation error for {service_name}: {e}")
            return False


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ULTRA Dockerfile Migration Tool')
    parser.add_argument('--dry-run', action='store_true', help='Perform dry run without making changes')
    parser.add_argument('--priority', choices=['P0', 'P1', 'P2', 'P3'], help='Migrate specific priority only')
    parser.add_argument('--validate', type=str, help='Validate a specific service after migration')
    
    args = parser.parse_args()
    
    migrator = UltraDockerfileMigrator()
    
    if args.validate:
        # Validate specific service
        success = migrator.validate_service(args.validate)
        exit(0 if success else 1)
    else:
        # Run migration
        migrator.run_migration(dry_run=args.dry_run, target_priority=args.priority)


if __name__ == "__main__":
    main()