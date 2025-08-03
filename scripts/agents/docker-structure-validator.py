#!/usr/bin/env python3
"""
Purpose: Docker Structure Validator - Enforces Rule 11 for clean, modular Docker infrastructure
Usage: python docker-structure-validator.py [--auto-fix] [--strict] [--report-format json|markdown]
Requirements: Python 3.8+, PyYAML, docker-py

This agent validates Docker structure compliance and provides auto-fix capabilities.
"""

import os
import sys
import json
import yaml
import re
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from datetime import datetime
from collections import defaultdict
import difflib
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/logs/docker-validator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DockerStructureValidator:
    """Validates Docker infrastructure compliance with Rule 11"""
    
    def __init__(self, project_root: str = "/opt/sutazaiapp"):
        self.project_root = Path(project_root)
        self.validation_results = {
            "timestamp": datetime.now().isoformat(),
            "structure_compliance": {},
            "dockerfile_analysis": {},
            "compose_validation": {},
            "security_issues": {},
            "modularity_check": {},
            "auto_fixes": [],
            "recommendations": []
        }
        
        # Required Docker structure
        self.required_structure = {
            "docker": {
                "backend": ["Dockerfile"],
                "frontend": ["Dockerfile"],
                "base": [],  # Base images
                ".dockerignore": None
            }
        }
        
        # Dockerfile best practices patterns
        self.dockerfile_patterns = {
            "version_pinning": re.compile(r'FROM\s+([^:]+):?([^\s]*)'),
            "multi_stage": re.compile(r'FROM\s+.*\s+AS\s+', re.IGNORECASE),
            "apt_cleanup": re.compile(r'rm\s+-rf\s+/var/lib/apt/lists/?\*'),
            "cache_mount": re.compile(r'--mount=type=cache'),
            "secrets": re.compile(r'(PASSWORD|SECRET|KEY|TOKEN)\s*=\s*["\']?[^"\'\s]+'),
            "user_context": re.compile(r'USER\s+(?!root)'),
            "workdir": re.compile(r'WORKDIR\s+'),
            "healthcheck": re.compile(r'HEALTHCHECK\s+'),
            "layer_optimization": re.compile(r'&&\s*\\')
        }
        
        # Banned patterns
        self.security_issues = {
            "sudo_nopasswd": re.compile(r'NOPASSWD:\s*ALL'),
            "curl_pipe_sh": re.compile(r'curl.*\|\s*(ba)?sh'),
            "hardcoded_secrets": re.compile(r'(api_key|password|secret|token)\s*=\s*["\'][\w\d]+["\']', re.IGNORECASE),
            "root_user_end": re.compile(r'USER\s+root\s*$'),
            "latest_tag": re.compile(r'FROM\s+[^:]+:latest'),
            "add_instead_copy": re.compile(r'^ADD\s+(?!http)', re.MULTILINE)
        }
        
        # Official base images
        self.official_images = {
            'alpine', 'ubuntu', 'debian', 'centos', 'fedora', 'amazonlinux',
            'python', 'node', 'golang', 'ruby', 'php', 'java', 'openjdk',
            'nginx', 'httpd', 'redis', 'postgres', 'mysql', 'mongo',
            'busybox', 'scratch', 'distroless'
        }

    def validate_structure(self) -> Dict[str, Any]:
        """Validate Docker directory structure"""
        logger.info("📁 Validating Docker directory structure...")
        
        structure_issues = []
        structure_status = "compliant"
        
        # Check if docker directory exists
        docker_dir = self.project_root / "docker"
        if not docker_dir.exists():
            structure_issues.append({
                "type": "missing_directory",
                "path": "docker/",
                "severity": "critical",
                "fix": "Create docker directory structure"
            })
            structure_status = "non_compliant"
        
        # Check required subdirectories
        for subdir, required_files in self.required_structure["docker"].items():
            if subdir == ".dockerignore":
                continue
                
            subdir_path = docker_dir / subdir
            if not subdir_path.exists():
                structure_issues.append({
                    "type": "missing_subdirectory",
                    "path": f"docker/{subdir}/",
                    "severity": "high",
                    "fix": f"Create docker/{subdir}/ directory"
                })
                structure_status = "non_compliant"
            elif required_files:
                for req_file in required_files:
                    file_path = subdir_path / req_file
                    if not file_path.exists():
                        structure_issues.append({
                            "type": "missing_file",
                            "path": f"docker/{subdir}/{req_file}",
                            "severity": "high",
                            "fix": f"Create {req_file} in docker/{subdir}/"
                        })
                        structure_status = "non_compliant"
        
        # Check .dockerignore
        dockerignore_path = docker_dir / ".dockerignore"
        if not dockerignore_path.exists():
            structure_issues.append({
                "type": "missing_dockerignore",
                "path": "docker/.dockerignore",
                "severity": "medium",
                "fix": "Create .dockerignore file"
            })
        
        # Find misplaced Dockerfiles
        misplaced_dockerfiles = self._find_misplaced_dockerfiles()
        for dockerfile in misplaced_dockerfiles:
            structure_issues.append({
                "type": "misplaced_dockerfile",
                "path": str(dockerfile),
                "severity": "medium",
                "fix": f"Move to appropriate docker/ subdirectory"
            })
            structure_status = "non_compliant"
        
        self.validation_results["structure_compliance"] = {
            "status": structure_status,
            "issues": structure_issues,
            "misplaced_files": [str(f) for f in misplaced_dockerfiles]
        }
        
        return self.validation_results["structure_compliance"]

    def _find_misplaced_dockerfiles(self) -> List[Path]:
        """Find Dockerfiles outside the docker/ directory"""
        misplaced = []
        
        for dockerfile in self.project_root.rglob("Dockerfile*"):
            # Skip those in proper locations
            if "docker" in dockerfile.parts:
                continue
            # Skip test/example directories
            if any(skip in str(dockerfile) for skip in ['.git', 'node_modules', 'venv', 'test', 'example']):
                continue
                
            misplaced.append(dockerfile)
            
        return misplaced

    def validate_dockerfile(self, dockerfile_path: Path) -> Dict[str, Any]:
        """Validate individual Dockerfile compliance"""
        logger.info(f"🔍 Analyzing {dockerfile_path}...")
        
        issues = []
        recommendations = []
        security_problems = []
        
        try:
            with open(dockerfile_path, 'r') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Check version pinning
            from_matches = self.dockerfile_patterns["version_pinning"].findall(content)
            for image, tag in from_matches:
                if not tag or tag == "latest":
                    issues.append({
                        "line": self._find_line_number(lines, f"FROM {image}"),
                        "type": "no_version_pinning",
                        "severity": "high",
                        "message": f"Image '{image}' not version-pinned",
                        "fix": f"Use specific version like '{image}:1.2.3'"
                    })
                
                # Check if official image
                base_image = image.split('/')[-1]
                if base_image not in self.official_images and '/' not in image:
                    recommendations.append({
                        "type": "unofficial_image",
                        "message": f"Consider using official image instead of '{image}'",
                        "severity": "low"
                    })
            
            # Check for multi-stage build opportunities
            if len(from_matches) == 1 and len(content) > 1000:
                recommendations.append({
                    "type": "multi_stage_opportunity",
                    "message": "Large Dockerfile could benefit from multi-stage build",
                    "severity": "medium"
                })
            
            # Check for apt cleanup
            if "apt-get install" in content and not self.dockerfile_patterns["apt_cleanup"].search(content):
                issues.append({
                    "type": "no_apt_cleanup",
                    "severity": "medium",
                    "message": "Missing apt cleanup after install",
                    "fix": "Add '&& rm -rf /var/lib/apt/lists/*' after apt-get install"
                })
            
            # Security checks
            for issue_name, pattern in self.security_issues.items():
                matches = pattern.findall(content)
                if matches:
                    for match in matches:
                        security_problems.append({
                            "type": issue_name,
                            "severity": "critical",
                            "match": match if isinstance(match, str) else match[0],
                            "line": self._find_line_number(lines, match if isinstance(match, str) else match[0])
                        })
            
            # Check for USER directive
            if not self.dockerfile_patterns["user_context"].search(content):
                issues.append({
                    "type": "no_user_context",
                    "severity": "high",
                    "message": "Dockerfile runs as root user",
                    "fix": "Add 'USER <non-root-user>' directive"
                })
            
            # Check for WORKDIR
            if not self.dockerfile_patterns["workdir"].search(content):
                issues.append({
                    "type": "no_workdir",
                    "severity": "low",
                    "message": "No WORKDIR specified",
                    "fix": "Add 'WORKDIR /app' or appropriate directory"
                })
            
            # Check for HEALTHCHECK in service containers
            service_name = dockerfile_path.parent.name
            if service_name in ['backend', 'frontend'] and not self.dockerfile_patterns["healthcheck"].search(content):
                recommendations.append({
                    "type": "no_healthcheck",
                    "message": f"Consider adding HEALTHCHECK for {service_name} service",
                    "severity": "medium"
                })
            
            # Check for layer optimization
            run_commands = [line for line in lines if line.strip().startswith("RUN")]
            if len(run_commands) > 5:
                recommendations.append({
                    "type": "too_many_layers",
                    "message": f"Dockerfile has {len(run_commands)} RUN commands, consider combining",
                    "severity": "medium"
                })
            
            # Check .dockerignore usage
            dockerignore_path = dockerfile_path.parent / ".dockerignore"
            if not dockerignore_path.exists():
                issues.append({
                    "type": "missing_dockerignore",
                    "severity": "medium",
                    "message": "No .dockerignore file found",
                    "fix": "Create .dockerignore to exclude unnecessary files"
                })
            
            analysis = {
                "path": str(dockerfile_path),
                "issues": issues,
                "security_problems": security_problems,
                "recommendations": recommendations,
                "metrics": {
                    "total_lines": len(lines),
                    "from_statements": len(from_matches),
                    "run_commands": len(run_commands),
                    "multi_stage": bool(self.dockerfile_patterns["multi_stage"].search(content))
                }
            }
            
            # Determine overall status
            if security_problems:
                analysis["status"] = "critical"
            elif any(i["severity"] == "high" for i in issues):
                analysis["status"] = "needs_fixes"
            else:
                analysis["status"] = "compliant"
                
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing {dockerfile_path}: {e}")
            return {
                "path": str(dockerfile_path),
                "status": "error",
                "error": str(e)
            }

    def _find_line_number(self, lines: List[str], search_text: str) -> int:
        """Find line number containing search text"""
        for i, line in enumerate(lines, 1):
            if search_text in line:
                return i
        return 0

    def validate_compose_files(self) -> Dict[str, Any]:
        """Validate docker-compose.yml files"""
        logger.info("📋 Validating docker-compose files...")
        
        compose_files = list(self.project_root.rglob("docker-compose*.yml")) + \
                       list(self.project_root.rglob("docker-compose*.yaml"))
        
        validation = {
            "files_found": len(compose_files),
            "issues": [],
            "service_consistency": {}
        }
        
        all_services = {}
        
        for compose_file in compose_files:
            try:
                with open(compose_file, 'r') as f:
                    compose_data = yaml.safe_load(f)
                
                if not compose_data or 'services' not in compose_data:
                    validation["issues"].append({
                        "file": str(compose_file),
                        "type": "invalid_structure",
                        "message": "No services defined"
                    })
                    continue
                
                # Check each service
                for service_name, service_config in compose_data['services'].items():
                    all_services[service_name] = compose_file
                    
                    # Check build context
                    if 'build' in service_config:
                        build_config = service_config['build']
                        if isinstance(build_config, dict):
                            dockerfile = build_config.get('dockerfile', 'Dockerfile')
                            context = build_config.get('context', '.')
                        else:
                            dockerfile = 'Dockerfile'
                            context = build_config
                        
                        # Verify Dockerfile exists
                        dockerfile_path = self.project_root / context / dockerfile
                        if not dockerfile_path.exists():
                            validation["issues"].append({
                                "file": str(compose_file),
                                "service": service_name,
                                "type": "missing_dockerfile",
                                "message": f"Dockerfile not found: {dockerfile_path}"
                            })
                    
                    # Check for image version pinning
                    if 'image' in service_config:
                        image = service_config['image']
                        if ':' not in image or image.endswith(':latest'):
                            validation["issues"].append({
                                "file": str(compose_file),
                                "service": service_name,
                                "type": "unpinned_image",
                                "message": f"Image not version-pinned: {image}"
                            })
                    
                    # Check for security issues
                    if 'privileged' in service_config and service_config['privileged']:
                        validation["issues"].append({
                            "file": str(compose_file),
                            "service": service_name,
                            "type": "privileged_container",
                            "severity": "high",
                            "message": "Container runs in privileged mode"
                        })
                    
                    # Check for explicit container names
                    if 'container_name' not in service_config:
                        validation["issues"].append({
                            "file": str(compose_file),
                            "service": service_name,
                            "type": "no_container_name",
                            "severity": "low",
                            "message": "No explicit container_name specified"
                        })
                        
            except Exception as e:
                validation["issues"].append({
                    "file": str(compose_file),
                    "type": "parse_error",
                    "message": str(e)
                })
        
        validation["service_consistency"] = all_services
        self.validation_results["compose_validation"] = validation
        
        return validation

    def check_modularity(self) -> Dict[str, Any]:
        """Check for service modularity"""
        logger.info("🧩 Checking service modularity...")
        
        modularity = {
            "services": {},
            "shared_dependencies": [],
            "monolithic_indicators": []
        }
        
        # Find all Dockerfiles
        dockerfiles = list(self.project_root.rglob("Dockerfile*"))
        
        # Analyze each service
        for dockerfile in dockerfiles:
            service_name = dockerfile.parent.name
            
            # Check if service has its own directory
            if dockerfile.parent == self.project_root:
                modularity["monolithic_indicators"].append({
                    "type": "root_dockerfile",
                    "path": str(dockerfile),
                    "message": "Dockerfile in root suggests monolithic structure"
                })
                continue
            
            # Analyze service isolation
            service_info = {
                "dockerfile": str(dockerfile),
                "has_own_directory": True,
                "config_files": [],
                "dependencies": []
            }
            
            # Look for service-specific configs
            for config_pattern in ['*.yml', '*.yaml', '*.json', '*.env']:
                configs = list(dockerfile.parent.glob(config_pattern))
                service_info["config_files"].extend([str(c) for c in configs])
            
            modularity["services"][service_name] = service_info
        
        # Check for shared base images
        base_dockerfiles = list((self.project_root / "docker" / "base").glob("Dockerfile*")) if (self.project_root / "docker" / "base").exists() else []
        if base_dockerfiles:
            modularity["shared_dependencies"].append({
                "type": "base_images",
                "count": len(base_dockerfiles),
                "files": [str(f) for f in base_dockerfiles]
            })
        
        self.validation_results["modularity_check"] = modularity
        return modularity

    def auto_fix_issues(self) -> List[Dict[str, Any]]:
        """Automatically fix common issues"""
        logger.info("🔧 Attempting auto-fixes...")
        
        fixes_applied = []
        
        # Fix 1: Create missing Docker structure
        docker_dir = self.project_root / "docker"
        if not docker_dir.exists():
            docker_dir.mkdir(parents=True)
            for subdir in ['backend', 'frontend', 'base']:
                (docker_dir / subdir).mkdir(exist_ok=True)
            fixes_applied.append({
                "type": "created_structure",
                "description": "Created docker/ directory structure"
            })
        
        # Fix 2: Create .dockerignore if missing
        dockerignore_path = docker_dir / ".dockerignore"
        if not dockerignore_path.exists():
            dockerignore_content = """# Dependencies
node_modules/
venv/
__pycache__/
*.pyc
.env
.env.*

# Build artifacts
dist/
build/
*.egg-info/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Git
.git/
.gitignore

# Documentation
*.md
docs/

# Tests
tests/
test/
*.test.js
*.spec.js
"""
            with open(dockerignore_path, 'w') as f:
                f.write(dockerignore_content)
            fixes_applied.append({
                "type": "created_dockerignore",
                "path": str(dockerignore_path)
            })
        
        # Fix 3: Move misplaced Dockerfiles
        misplaced = self._find_misplaced_dockerfiles()
        for dockerfile in misplaced:
            # Determine appropriate location
            if 'backend' in str(dockerfile).lower():
                target_dir = docker_dir / "backend"
            elif 'frontend' in str(dockerfile).lower():
                target_dir = docker_dir / "frontend"
            else:
                target_dir = docker_dir / dockerfile.parent.name
                
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / dockerfile.name
            
            try:
                shutil.move(str(dockerfile), str(target_path))
                fixes_applied.append({
                    "type": "moved_dockerfile",
                    "from": str(dockerfile),
                    "to": str(target_path)
                })
            except Exception as e:
                logger.error(f"Could not move {dockerfile}: {e}")
        
        # Fix 4: Update Dockerfiles with basic fixes
        for dockerfile in docker_dir.rglob("Dockerfile*"):
            self._auto_fix_dockerfile(dockerfile, fixes_applied)
        
        self.validation_results["auto_fixes"] = fixes_applied
        return fixes_applied

    def _auto_fix_dockerfile(self, dockerfile_path: Path, fixes_applied: List[Dict]) -> None:
        """Apply automatic fixes to a Dockerfile"""
        try:
            with open(dockerfile_path, 'r') as f:
                original_content = f.read()
                lines = original_content.split('\n')
            
            modified = False
            new_lines = []
            
            for i, line in enumerate(lines):
                new_line = line
                
                # Fix apt-get without cleanup
                if "apt-get install" in line and "rm -rf /var/lib/apt/lists" not in line:
                    if line.rstrip().endswith('\\'):
                        # Multi-line command, add cleanup at the end
                        j = i + 1
                        while j < len(lines) and lines[j].rstrip().endswith('\\'):
                            j += 1
                        if j < len(lines):
                            lines[j] = lines[j] + " && rm -rf /var/lib/apt/lists/*"
                            modified = True
                    else:
                        # Single line command
                        new_line = line.rstrip() + " && rm -rf /var/lib/apt/lists/*"
                        modified = True
                
                # Fix FROM without version
                if line.strip().startswith("FROM "):
                    parts = line.strip().split()
                    if len(parts) >= 2 and ':' not in parts[1]:
                        # Add :latest as a temporary fix (should be replaced with specific version)
                        new_line = f"{parts[0]} {parts[1]}:latest  # TODO: Pin to specific version"
                        modified = True
                
                new_lines.append(new_line)
            
            # Add USER directive if missing
            if modified or not any("USER " in line for line in lines):
                # Add before the last CMD/ENTRYPOINT if present
                user_added = False
                for i in range(len(new_lines) - 1, -1, -1):
                    if new_lines[i].strip().startswith(("CMD", "ENTRYPOINT")):
                        new_lines.insert(i, "\n# Run as non-root user")
                        new_lines.insert(i + 1, "USER nobody")
                        user_added = True
                        modified = True
                        break
                
                if not user_added and len(new_lines) > 0:
                    new_lines.append("\n# Run as non-root user")
                    new_lines.append("USER nobody")
                    modified = True
            
            if modified:
                new_content = '\n'.join(new_lines)
                
                # Backup original
                backup_path = dockerfile_path.with_suffix('.bak')
                shutil.copy2(dockerfile_path, backup_path)
                
                # Write fixed version
                with open(dockerfile_path, 'w') as f:
                    f.write(new_content)
                
                fixes_applied.append({
                    "type": "fixed_dockerfile",
                    "path": str(dockerfile_path),
                    "backup": str(backup_path),
                    "changes": self._get_diff(original_content, new_content)
                })
                
        except Exception as e:
            logger.error(f"Error fixing {dockerfile_path}: {e}")

    def _get_diff(self, original: str, modified: str) -> List[str]:
        """Get diff between original and modified content"""
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile='original',
            tofile='modified',
            n=3
        )
        return list(diff)[:20]  # Limit diff size

    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        logger.info("📊 Generating compliance report...")
        
        # Structure compliance
        structure = self.validation_results.get("structure_compliance", {})
        structure_score = 100 if structure.get("status") == "compliant" else 50
        
        # Dockerfile compliance
        dockerfile_issues = sum(
            len(d.get("issues", [])) + len(d.get("security_problems", []))
            for d in self.validation_results.get("dockerfile_analysis", {}).values()
        )
        dockerfile_score = max(0, 100 - (dockerfile_issues * 5))
        
        # Compose compliance
        compose_issues = len(self.validation_results.get("compose_validation", {}).get("issues", []))
        compose_score = max(0, 100 - (compose_issues * 10))
        
        # Overall score
        overall_score = (structure_score + dockerfile_score + compose_score) / 3
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_score": round(overall_score, 2),
            "compliance_status": "compliant" if overall_score >= 80 else "non_compliant",
            "scores": {
                "structure": structure_score,
                "dockerfiles": dockerfile_score,
                "compose": compose_score
            },
            "summary": {
                "total_dockerfiles": len(self.validation_results.get("dockerfile_analysis", {})),
                "structure_issues": len(structure.get("issues", [])),
                "dockerfile_issues": dockerfile_issues,
                "compose_issues": compose_issues,
                "auto_fixes_applied": len(self.validation_results.get("auto_fixes", []))
            },
            "critical_issues": self._get_critical_issues(),
            "recommendations": self._generate_recommendations()
        }
        
        return report

    def _get_critical_issues(self) -> List[Dict[str, Any]]:
        """Extract critical issues from validation results"""
        critical = []
        
        # Security issues from Dockerfiles
        for path, analysis in self.validation_results.get("dockerfile_analysis", {}).items():
            for issue in analysis.get("security_problems", []):
                critical.append({
                    "type": "security",
                    "location": path,
                    "issue": issue["type"],
                    "severity": "critical"
                })
        
        # Privileged containers
        for issue in self.validation_results.get("compose_validation", {}).get("issues", []):
            if issue.get("type") == "privileged_container":
                critical.append({
                    "type": "security",
                    "location": issue["file"],
                    "issue": "privileged_container",
                    "service": issue.get("service"),
                    "severity": "critical"
                })
        
        return critical

    def _generate_recommendations(self) -> List[Dict[str, str]]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Structure recommendations
        if self.validation_results.get("structure_compliance", {}).get("status") != "compliant":
            recommendations.append({
                "category": "structure",
                "priority": "high",
                "action": "Reorganize Docker files into proper docker/ directory structure"
            })
        
        # Multi-stage build recommendations
        large_dockerfiles = [
            path for path, analysis in self.validation_results.get("dockerfile_analysis", {}).items()
            if analysis.get("metrics", {}).get("total_lines", 0) > 50 and not analysis.get("metrics", {}).get("multi_stage")
        ]
        if large_dockerfiles:
            recommendations.append({
                "category": "optimization",
                "priority": "medium",
                "action": f"Consider multi-stage builds for {len(large_dockerfiles)} large Dockerfiles"
            })
        
        # Security recommendations
        if self._get_critical_issues():
            recommendations.append({
                "category": "security",
                "priority": "critical",
                "action": "Address critical security issues immediately"
            })
        
        return recommendations

    def run_validation(self, auto_fix: bool = False, strict: bool = False) -> Dict[str, Any]:
        """Run complete Docker structure validation"""
        logger.info("🐳 Starting Docker Structure Validation...")
        
        # Step 1: Validate structure
        self.validate_structure()
        
        # Step 2: Find and analyze all Dockerfiles
        docker_dir = self.project_root / "docker"
        dockerfiles = list(docker_dir.rglob("Dockerfile*")) if docker_dir.exists() else []
        dockerfiles.extend(self._find_misplaced_dockerfiles())
        
        dockerfile_analysis = {}
        for dockerfile in dockerfiles:
            analysis = self.validate_dockerfile(dockerfile)
            dockerfile_analysis[str(dockerfile)] = analysis
        
        self.validation_results["dockerfile_analysis"] = dockerfile_analysis
        
        # Step 3: Validate docker-compose files
        self.validate_compose_files()
        
        # Step 4: Check modularity
        self.check_modularity()
        
        # Step 5: Apply auto-fixes if requested
        if auto_fix:
            self.auto_fix_issues()
        
        # Step 6: Generate compliance report
        compliance_report = self.generate_compliance_report()
        self.validation_results["compliance_report"] = compliance_report
        
        # Log summary
        logger.info(f"✅ Validation complete!")
        logger.info(f"📊 Overall compliance score: {compliance_report['overall_score']}%")
        logger.info(f"🔍 Found {len(dockerfiles)} Dockerfiles")
        total_issues = (compliance_report['summary'].get('structure_issues', 0) +
                       compliance_report['summary'].get('dockerfile_issues', 0) +
                       compliance_report['summary'].get('compose_issues', 0))
        logger.info(f"⚠️  Total issues: {total_issues}")
        
        if strict and compliance_report['overall_score'] < 80:
            logger.error("❌ Validation failed in strict mode")
            sys.exit(1)
        
        return self.validation_results

    def save_report(self, format_type: str = "json") -> str:
        """Save validation report to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        reports_dir = self.project_root / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        if format_type == "json":
            report_path = reports_dir / f"docker_validation_{timestamp}.json"
            with open(report_path, 'w') as f:
                json.dump(self.validation_results, f, indent=2, default=str)
        else:
            report_path = reports_dir / f"docker_validation_{timestamp}.md"
            with open(report_path, 'w') as f:
                f.write(self._generate_markdown_report())
        
        logger.info(f"📄 Report saved: {report_path}")
        return str(report_path)

    def _generate_markdown_report(self) -> str:
        """Generate markdown format report"""
        report = self.validation_results.get("compliance_report", {})
        
        md = f"""# Docker Structure Validation Report

Generated: {report.get('timestamp', 'N/A')}

## Compliance Summary

**Overall Score: {report.get('overall_score', 0)}%**
**Status: {report.get('compliance_status', 'unknown').upper()}**

### Component Scores
- Structure Compliance: {report['scores']['structure']}%
- Dockerfile Standards: {report['scores']['dockerfiles']}%
- Compose Configuration: {report['scores']['compose']}%

## Issues Summary

- Structure Issues: {report['summary']['structure_issues']}
- Dockerfile Issues: {report['summary']['dockerfile_issues']}
- Compose Issues: {report['summary']['compose_issues']}
- Auto-fixes Applied: {report['summary']['auto_fixes_applied']}

## Critical Issues
"""
        
        critical_issues = report.get('critical_issues', [])
        if critical_issues:
            for issue in critical_issues:
                md += f"\n### ⚠️ {issue['issue']}\n"
                md += f"- **Location**: `{issue['location']}`\n"
                md += f"- **Type**: {issue['type']}\n"
                md += f"- **Severity**: {issue['severity']}\n"
        else:
            md += "\n✅ No critical issues found\n"
        
        md += "\n## Recommendations\n"
        for rec in report.get('recommendations', []):
            md += f"\n### {rec['priority'].upper()}: {rec['category'].title()}\n"
            md += f"- {rec['action']}\n"
        
        # Add detailed findings
        md += "\n## Detailed Findings\n"
        
        # Structure compliance
        structure = self.validation_results.get("structure_compliance", {})
        if structure.get("issues"):
            md += "\n### Structure Issues\n"
            for issue in structure["issues"]:
                md += f"- **{issue['type']}**: `{issue['path']}` - {issue['fix']}\n"
        
        # Dockerfile analysis
        md += "\n### Dockerfile Analysis\n"
        for path, analysis in self.validation_results.get("dockerfile_analysis", {}).items():
            if analysis.get("status") != "compliant":
                md += f"\n#### `{path}` - Status: {analysis.get('status', 'unknown')}\n"
                for issue in analysis.get("issues", []):
                    md += f"- {issue['type']}: {issue['message']}\n"
        
        return md


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Docker Structure Validator - Enforces Rule 11 compliance"
    )
    parser.add_argument("--auto-fix", action="store_true",
                       help="Automatically fix common issues")
    parser.add_argument("--strict", action="store_true",
                       help="Exit with error code if non-compliant")
    parser.add_argument("--report-format", choices=["json", "markdown"],
                       default="json", help="Report output format")
    parser.add_argument("--project-root", default="/opt/sutazaiapp",
                       help="Project root directory")
    
    args = parser.parse_args()
    
    # Ensure logs directory exists
    os.makedirs("/opt/sutazaiapp/logs", exist_ok=True)
    
    try:
        validator = DockerStructureValidator(args.project_root)
        results = validator.run_validation(
            auto_fix=args.auto_fix,
            strict=args.strict
        )
        
        # Save report
        report_path = validator.save_report(args.report_format)
        
        # Print summary
        compliance = results.get("compliance_report", {})
        print(f"\n{'='*60}")
        print(f"Docker Structure Validation Complete")
        print(f"{'='*60}")
        print(f"Overall Score: {compliance.get('overall_score', 0)}%")
        print(f"Status: {compliance.get('compliance_status', 'unknown').upper()}")
        print(f"\nReport saved: {report_path}")
        
        if compliance.get('critical_issues'):
            print(f"\n⚠️  Found {len(compliance['critical_issues'])} CRITICAL issues!")
            print("Run with --auto-fix to attempt automatic fixes")
        
        # Exit code based on compliance
        if args.strict and compliance.get('overall_score', 0) < 80:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()