#!/usr/bin/env python3
"""
Purpose: Comprehensive container infrastructure validation for SutazAI
Usage: python validate-container-infrastructure.py [--critical-only] [--report-format json|markdown]
Requirements: Docker, Python 3.8+, PyYAML, docker-py

This script validates all container infrastructure, identifies dependencies, 
and creates safe cleanup recommendations with full rollback capability.
"""

import os
import sys
import json
import subprocess
import yaml
import datetime
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import hashlib
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/logs/container-validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ContainerInfrastructureValidator:
    def __init__(self, project_root: str = "/opt/sutazaiapp"):
        self.project_root = Path(project_root)
        self.validation_results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "dockerfiles": {},
            "requirements": {},
            "critical_services": {},
            "build_status": {},
            "dependency_graph": {},
            "cleanup_recommendations": {},
            "rollback_plan": {}
        }
        self.critical_services = self._identify_critical_services()
        
    def _identify_critical_services(self) -> Set[str]:
        """Identify critical services based on dependency analysis"""
        critical = {
            'ollama', 'backend', 'frontend', 'postgres', 'redis', 'nginx',
            'monitoring', 'loki', 'grafana', 'prometheus', 'chromadb',
            'qdrant', 'neo4j', 'health-monitor', 'api-gateway'
        }
        
        # Add services from main docker-compose.yml
        compose_file = self.project_root / "docker-compose.yml"
        if compose_file.exists():
            try:
                with open(compose_file) as f:
                    compose_data = yaml.safe_load(f)
                    if 'services' in compose_data:
                        critical.update(compose_data['services'].keys())
            except Exception as e:
                logger.warning(f"Could not parse docker-compose.yml: {e}")
                
        return critical

    def find_all_dockerfiles(self) -> Dict[str, Path]:
        """Find all Dockerfiles in the project"""
        dockerfiles = {}
        
        for dockerfile_path in self.project_root.rglob("Dockerfile*"):
            # Skip files in certain directories
            if any(skip in str(dockerfile_path) for skip in ['.git', '__pycache__', 'venv', 'node_modules']):
                continue
                
            relative_path = dockerfile_path.relative_to(self.project_root)
            service_name = self._extract_service_name(dockerfile_path)
            dockerfiles[service_name] = dockerfile_path
            
        logger.info(f"Found {len(dockerfiles)} Dockerfiles")
        return dockerfiles

    def find_all_requirements(self) -> Dict[str, List[Path]]:
        """Find all requirements files"""
        requirements = defaultdict(list)
        
        patterns = ['requirements*.txt', 'requirements*.yml', 'requirements*.yaml', 
                   'package.json', 'pyproject.toml', 'Pipfile', 'environment.yml']
        
        for pattern in patterns:
            for req_file in self.project_root.rglob(pattern):
                if any(skip in str(req_file) for skip in ['.git', '__pycache__', 'venv', 'node_modules']):
                    continue
                    
                service_name = self._extract_service_name(req_file)
                requirements[service_name].append(req_file)
                
        logger.info(f"Found requirements for {len(requirements)} services")
        return dict(requirements)

    def _extract_service_name(self, file_path: Path) -> str:
        """Extract service name from file path"""
        path_parts = file_path.parts
        
        # Look for service-specific directories
        for i, part in enumerate(path_parts):
            if part in ['docker', 'agents', 'services']:
                if i + 1 < len(path_parts):
                    return path_parts[i + 1]
            elif part.endswith(('-service', '-agent', '-manager')):
                return part
                
        # Fallback to parent directory name
        return file_path.parent.name

    def validate_dockerfile_build(self, service_name: str, dockerfile_path: Path) -> Dict:
        """Validate if Dockerfile can build successfully"""
        result = {
            "service": service_name,
            "dockerfile": str(dockerfile_path),
            "build_success": False,
            "build_time": 0,
            "image_size": 0,
            "errors": [],
            "warnings": []
        }
        
        try:
            start_time = datetime.datetime.now()
            
            # Build the image
            build_cmd = [
                "docker", "build", 
                "-t", f"sutazai-test-{service_name}:validation",
                "-f", str(dockerfile_path),
                str(dockerfile_path.parent)
            ]
            
            logger.info(f"Building {service_name} from {dockerfile_path}")
            process = subprocess.run(
                build_cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            build_time = (datetime.datetime.now() - start_time).total_seconds()
            result["build_time"] = build_time
            
            if process.returncode == 0:
                result["build_success"] = True
                
                # Get image size
                size_cmd = ["docker", "images", f"sutazai-test-{service_name}:validation", "--format", "{{.Size}}"]
                size_result = subprocess.run(size_cmd, capture_output=True, text=True)
                if size_result.returncode == 0:
                    result["image_size"] = size_result.stdout.strip()
                    
                logger.info(f"✅ {service_name} built successfully in {build_time:.1f}s")
            else:
                result["errors"].append(process.stderr)
                logger.error(f"❌ {service_name} build failed: {process.stderr[:200]}...")
                
        except subprocess.TimeoutExpired:
            result["errors"].append("Build timeout (10 minutes)")
            logger.error(f"❌ {service_name} build timed out")
        except Exception as e:
            result["errors"].append(str(e))
            logger.error(f"❌ {service_name} build error: {e}")
            
        return result

    def validate_container_startup(self, service_name: str) -> Dict:
        """Test if container can start successfully"""
        result = {
            "service": service_name,
            "startup_success": False,
            "startup_time": 0,
            "health_check": False,
            "exposed_ports": [],
            "errors": []
        }
        
        try:
            start_time = datetime.datetime.now()
            
            # Start container
            run_cmd = [
                "docker", "run", "-d", 
                "--name", f"sutazai-test-{service_name}-startup",
                f"sutazai-test-{service_name}:validation"
            ]
            
            process = subprocess.run(run_cmd, capture_output=True, text=True)
            
            if process.returncode == 0:
                container_id = process.stdout.strip()
                
                # Wait a moment for startup
                import time
                time.sleep(5)
                
                # Check if container is still running
                inspect_cmd = ["docker", "inspect", container_id, "--format", "{{.State.Running}}"]
                inspect_result = subprocess.run(inspect_cmd, capture_output=True, text=True)
                
                if inspect_result.returncode == 0 and inspect_result.stdout.strip() == "true":
                    result["startup_success"] = True
                    result["startup_time"] = (datetime.datetime.now() - start_time).total_seconds()
                    
                    # Get port information
                    port_cmd = ["docker", "port", container_id]
                    port_result = subprocess.run(port_cmd, capture_output=True, text=True)
                    if port_result.returncode == 0:
                        result["exposed_ports"] = port_result.stdout.strip().split('\n')
                        
                    logger.info(f"✅ {service_name} started successfully")
                else:
                    result["errors"].append("Container stopped after startup")
                    
                # Clean up
                subprocess.run(["docker", "stop", container_id], capture_output=True)
                subprocess.run(["docker", "rm", container_id], capture_output=True)
                
            else:
                result["errors"].append(process.stderr)
                
        except Exception as e:
            result["errors"].append(str(e))
            logger.error(f"❌ {service_name} startup test failed: {e}")
            
        return result

    def analyze_dependencies(self, dockerfiles: Dict[str, Path], requirements: Dict[str, List[Path]]) -> Dict:
        """Analyze inter-service dependencies"""
        dependency_graph = defaultdict(set)
        shared_dependencies = defaultdict(set)
        
        # Analyze Dockerfile dependencies
        for service, dockerfile_path in dockerfiles.items():
            try:
                with open(dockerfile_path) as f:
                    content = f.read()
                    
                # Look for FROM statements
                for line in content.split('\n'):
                    line = line.strip()
                    if line.startswith('FROM '):
                        base_image = line.split()[1].split(':')[0]
                        dependency_graph[service].add(base_image)
                        
            except Exception as e:
                logger.warning(f"Could not analyze {dockerfile_path}: {e}")
                
        # Analyze requirements dependencies
        for service, req_files in requirements.items():
            for req_file in req_files:
                deps = self._extract_dependencies(req_file)
                shared_dependencies[service].update(deps)
                
        return {
            "service_dependencies": dict(dependency_graph),
            "shared_dependencies": dict(shared_dependencies)
        }

    def _extract_dependencies(self, req_file: Path) -> Set[str]:
        """Extract dependencies from requirements file"""
        dependencies = set()
        
        try:
            if req_file.name == 'package.json':
                with open(req_file) as f:
                    data = json.load(f)
                    if 'dependencies' in data:
                        dependencies.update(data['dependencies'].keys())
                    if 'devDependencies' in data:
                        dependencies.update(data['devDependencies'].keys())
                        
            elif req_file.suffix in ['.txt']:
                with open(req_file) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            # Extract package name
                            dep_name = line.split('==')[0].split('>=')[0].split('<=')[0].split('~=')[0]
                            dependencies.add(dep_name.strip())
                            
        except Exception as e:
            logger.warning(f"Could not parse {req_file}: {e}")
            
        return dependencies

    def create_cleanup_plan(self, dockerfiles: Dict[str, Path], requirements: Dict[str, List[Path]]) -> Dict:
        """Create safe cleanup plan with verification"""
        cleanup_plan = {
            "safe_to_remove": [],
            "consolidation_opportunities": [],
            "backup_required": [],
            "high_risk": []
        }
        
        # Identify duplicate/similar requirements files
        req_hashes = {}
        for service, req_files in requirements.items():
            for req_file in req_files:
                try:
                    with open(req_file, 'rb') as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()
                        
                    if file_hash in req_hashes:
                        cleanup_plan["consolidation_opportunities"].append({
                            "duplicate_of": req_hashes[file_hash],
                            "duplicate": str(req_file),
                            "service": service
                        })
                    else:
                        req_hashes[file_hash] = str(req_file)
                        
                except Exception as e:
                    logger.warning(f"Could not hash {req_file}: {e}")
                    
        # Identify unused Dockerfiles
        for service, dockerfile_path in dockerfiles.items():
            if service not in self.critical_services:
                # Check if referenced in docker-compose files
                is_referenced = self._is_dockerfile_referenced(dockerfile_path)
                if not is_referenced:
                    cleanup_plan["safe_to_remove"].append({
                        "type": "dockerfile",
                        "path": str(dockerfile_path),
                        "service": service,
                        "reason": "Not referenced in compose files"
                    })
                    
        return cleanup_plan

    def _is_dockerfile_referenced(self, dockerfile_path: Path) -> bool:
        """Check if Dockerfile is referenced in any compose file"""
        compose_files = list(self.project_root.rglob("docker-compose*.yml")) + \
                      list(self.project_root.rglob("docker-compose*.yaml"))
        
        dockerfile_rel = str(dockerfile_path.relative_to(self.project_root))
        
        for compose_file in compose_files:
            try:
                with open(compose_file) as f:
                    content = f.read()
                    if dockerfile_rel in content or dockerfile_path.name in content:
                        return True
            except Exception:
                continue
                
        return False

    def create_rollback_plan(self) -> Dict:
        """Create comprehensive rollback plan"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.project_root / "archive" / f"container_validation_{timestamp}"
        
        rollback_plan = {
            "backup_directory": str(backup_dir),
            "backup_commands": [],
            "restore_commands": [],
            "validation_commands": []
        }
        
        # Create backup directory
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup critical files
        critical_patterns = [
            "docker-compose*.yml", "docker-compose*.yaml",
            "Dockerfile*", "requirements*.txt", "package.json"
        ]
        
        for pattern in critical_patterns:
            for file_path in self.project_root.rglob(pattern):
                if any(skip in str(file_path) for skip in ['.git', '__pycache__', 'venv']):
                    continue
                    
                rel_path = file_path.relative_to(self.project_root)
                backup_path = backup_dir / rel_path
                
                rollback_plan["backup_commands"].append(f"cp {file_path} {backup_path}")
                rollback_plan["restore_commands"].append(f"cp {backup_path} {file_path}")
                
        return rollback_plan

    def run_comprehensive_validation(self, critical_only: bool = False) -> Dict:
        """Run complete infrastructure validation"""
        logger.info("🚀 Starting comprehensive container infrastructure validation")
        
        # Step 1: Discovery
        logger.info("📋 Discovering containers and requirements...")
        dockerfiles = self.find_all_dockerfiles()
        requirements = self.find_all_requirements()
        
        self.validation_results["dockerfiles"] = {k: str(v) for k, v in dockerfiles.items()}
        self.validation_results["requirements"] = {k: [str(f) for f in v] for k, v in requirements.items()}
        
        # Step 2: Dependency Analysis
        logger.info("🔍 Analyzing dependencies...")
        dependency_analysis = self.analyze_dependencies(dockerfiles, requirements)
        self.validation_results["dependency_graph"] = dependency_analysis
        
        # Step 3: Build Validation
        logger.info("🔨 Validating container builds...")
        services_to_test = dockerfiles.keys()
        if critical_only:
            services_to_test = [s for s in services_to_test if s in self.critical_services]
            
        build_results = {}
        for service in services_to_test:
            if service in dockerfiles:
                result = self.validate_dockerfile_build(service, dockerfiles[service])
                build_results[service] = result
                
                # Test startup only if build succeeded
                if result["build_success"]:
                    startup_result = self.validate_container_startup(service)
                    result["startup_test"] = startup_result
                    
        self.validation_results["build_status"] = build_results
        
        # Step 4: Cleanup Planning
        logger.info("🧹 Creating cleanup plan...")
        cleanup_plan = self.create_cleanup_plan(dockerfiles, requirements)
        self.validation_results["cleanup_recommendations"] = cleanup_plan
        
        # Step 5: Rollback Planning
        logger.info("🔄 Creating rollback plan...")
        rollback_plan = self.create_rollback_plan()
        self.validation_results["rollback_plan"] = rollback_plan
        
        # Step 6: Summary
        total_containers = len(dockerfiles)
        successful_builds = sum(1 for r in build_results.values() if r["build_success"])
        successful_startups = sum(1 for r in build_results.values() 
                                if r.get("startup_test", {}).get("startup_success", False))
        
        logger.info(f"✅ Validation complete!")
        logger.info(f"📊 Total containers: {total_containers}")
        logger.info(f"🔨 Successful builds: {successful_builds}/{len(build_results)}")
        logger.info(f"🚀 Successful startups: {successful_startups}/{len(build_results)}")
        
        return self.validation_results

    def cleanup_test_images(self):
        """Clean up test images created during validation"""
        logger.info("🧹 Cleaning up test images...")
        try:
            # Remove test images
            subprocess.run(["docker", "images", "-q", "--filter", "reference=sutazai-test-*"], 
                         capture_output=True, text=True)
            cleanup_cmd = ["docker", "rmi", "-f"] + \
                         subprocess.run(["docker", "images", "-q", "--filter", "reference=sutazai-test-*"],
                                      capture_output=True, text=True).stdout.strip().split('\n')
            if len(cleanup_cmd) > 3:  # Only run if there are images to remove
                subprocess.run(cleanup_cmd, capture_output=True)
                
        except Exception as e:
            logger.warning(f"Could not clean up test images: {e}")

    def generate_report(self, format_type: str = "json") -> str:
        """Generate validation report"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format_type == "json":
            report_path = f"/opt/sutazaiapp/reports/container_validation_{timestamp}.json"
            with open(report_path, 'w') as f:
                json.dump(self.validation_results, f, indent=2)
                
        elif format_type == "markdown":
            report_path = f"/opt/sutazaiapp/reports/container_validation_{timestamp}.md"
            with open(report_path, 'w') as f:
                f.write(self._generate_markdown_report())
                
        logger.info(f"📄 Report generated: {report_path}")
        return report_path

    def _generate_markdown_report(self) -> str:
        """Generate markdown format report"""
        results = self.validation_results
        
        md = f"""# Container Infrastructure Validation Report
        
Generated: {results['timestamp']}

## Summary

- **Total Dockerfiles**: {len(results['dockerfiles'])}
- **Total Services with Requirements**: {len(results['requirements'])}
- **Build Tests**: {len(results['build_status'])}
- **Successful Builds**: {sum(1 for r in results['build_status'].values() if r['build_success'])}

## Build Results

| Service | Build Status | Build Time | Image Size | Startup Test |
|---------|-------------|------------|-----------|--------------|
"""
        
        for service, result in results['build_status'].items():
            status = "✅" if result['build_success'] else "❌"
            startup = "✅" if result.get('startup_test', {}).get('startup_success', False) else "❌"
            md += f"| {service} | {status} | {result['build_time']:.1f}s | {result.get('image_size', 'N/A')} | {startup} |\n"
            
        md += f"""
## Cleanup Recommendations

### Safe to Remove ({len(results['cleanup_recommendations']['safe_to_remove'])})
"""
        for item in results['cleanup_recommendations']['safe_to_remove']:
            md += f"- `{item['path']}` - {item['reason']}\n"
            
        md += f"""
### Consolidation Opportunities ({len(results['cleanup_recommendations']['consolidation_opportunities'])})
"""
        for item in results['cleanup_recommendations']['consolidation_opportunities']:
            md += f"- `{item['duplicate']}` duplicates `{item['duplicate_of']}`\n"
            
        return md


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate SutazAI container infrastructure")
    parser.add_argument("--critical-only", action="store_true", 
                       help="Only test critical services")
    parser.add_argument("--report-format", choices=["json", "markdown"], 
                       default="json", help="Report format")
    parser.add_argument("--no-cleanup", action="store_true",
                       help="Skip cleanup of test images")
    
    args = parser.parse_args()
    
    # Ensure reports directory exists
    os.makedirs("/opt/sutazaiapp/reports", exist_ok=True)
    
    try:
        validator = ContainerInfrastructureValidator()
        results = validator.run_comprehensive_validation(critical_only=args.critical_only)
        
        # Generate report
        report_path = validator.generate_report(args.report_format)
        
        # Cleanup test images unless disabled
        if not args.no_cleanup:
            validator.cleanup_test_images()
            
        print(f"\n✅ Validation complete! Report: {report_path}")
        
        # Return appropriate exit code
        failed_builds = sum(1 for r in results['build_status'].values() if not r['build_success'])
        if failed_builds > 0:
            print(f"⚠️  {failed_builds} containers failed to build")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()