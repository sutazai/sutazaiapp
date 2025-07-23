#!/usr/bin/env python3
"""
SutazAI Dependency Analyzer - Comprehensive dependency conflict and compatibility analysis
"""

import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import logging
from packaging import version
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DependencyAnalyzer:
    """Comprehensive dependency analyzer for SutazAI system"""
    
    def __init__(self, project_root: str = "/opt/sutazaiapp"):
        self.project_root = Path(project_root)
        self.findings = {
            "python_conflicts": [],
            "docker_issues": [],
            "security_vulnerabilities": [],
            "version_mismatches": [],
            "compatibility_issues": [],
            "recommendations": []
        }
        
    def analyze_all_dependencies(self):
        """Run comprehensive dependency analysis"""
        logger.info("Starting comprehensive dependency analysis...")
        
        # 1. Python Dependencies Analysis
        self.analyze_python_requirements()
        
        # 2. Docker Dependencies Analysis
        self.analyze_docker_dependencies()
        
        # 3. JavaScript/Node Dependencies
        self.analyze_node_dependencies()
        
        # 4. Security Vulnerability Analysis
        self.analyze_security_vulnerabilities()
        
        # 5. Framework Compatibility Analysis
        self.analyze_framework_compatibility()
        
        # 6. Generate comprehensive report
        self.generate_report()
        
    def analyze_python_requirements(self):
        """Analyze Python requirement files for conflicts"""
        logger.info("Analyzing Python requirements...")
        
        # Find all requirements files
        req_files = list(self.project_root.rglob("requirements*.txt"))
        
        all_deps = defaultdict(list)
        
        for req_file in req_files:
            try:
                with open(req_file, 'r') as f:
                    lines = f.readlines()
                    
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Parse package name and version
                        match = re.match(r'^([a-zA-Z0-9_-]+)([><=!~]+.*)?', line)
                        if match:
                            pkg_name = match.group(1).lower()
                            version_spec = match.group(2) or ""
                            all_deps[pkg_name].append({
                                'file': str(req_file),
                                'line': line,
                                'version_spec': version_spec
                            })
                            
            except Exception as e:
                logger.error(f"Error reading {req_file}: {e}")
                
        # Find conflicts
        for pkg_name, occurrences in all_deps.items():
            if len(occurrences) > 1:
                # Check for version conflicts
                versions = [occ['version_spec'] for occ in occurrences if occ['version_spec']]
                if len(set(versions)) > 1:
                    self.findings["python_conflicts"].append({
                        "package": pkg_name,
                        "conflicts": occurrences,
                        "severity": "high"
                    })
                    
        # Check for known problematic combinations
        self.check_known_conflicts(all_deps)
        
    def check_known_conflicts(self, all_deps: Dict):
        """Check for known problematic package combinations"""
        
        # FastAPI + Pydantic version conflicts
        if 'fastapi' in all_deps and 'pydantic' in all_deps:
            fastapi_versions = [dep['version_spec'] for dep in all_deps['fastapi']]
            pydantic_versions = [dep['version_spec'] for dep in all_deps['pydantic']]
            
            # Check compatibility
            if any('>=0.104' in v for v in fastapi_versions) and any('>=2.5' not in v for v in pydantic_versions):
                self.findings["compatibility_issues"].append({
                    "issue": "FastAPI 0.104+ requires Pydantic v2.5+",
                    "packages": ["fastapi", "pydantic"],
                    "severity": "critical"
                })
                
        # PyTorch + CUDA compatibility
        if 'torch' in all_deps:
            torch_versions = [dep['version_spec'] for dep in all_deps['torch']]
            if 'onnxruntime-gpu' in all_deps:
                self.findings["compatibility_issues"].append({
                    "issue": "PyTorch and ONNX Runtime GPU may have CUDA version conflicts",
                    "packages": ["torch", "onnxruntime-gpu"],
                    "severity": "medium",
                    "recommendation": "Verify CUDA version compatibility"
                })
                
        # Streamlit + FastAPI potential conflicts
        if 'streamlit' in all_deps and 'fastapi' in all_deps:
            self.findings["compatibility_issues"].append({
                "issue": "Streamlit and FastAPI may conflict on async event loops",
                "packages": ["streamlit", "fastapi"],
                "severity": "low",
                "recommendation": "Use separate containers or proper async handling"
            })
            
    def analyze_docker_dependencies(self):
        """Analyze Docker configurations for issues"""
        logger.info("Analyzing Docker dependencies...")
        
        docker_files = list(self.project_root.rglob("Dockerfile*"))
        compose_files = list(self.project_root.rglob("docker-compose*.yml"))
        
        base_images = {}
        port_usage = defaultdict(list)
        
        # Analyze Dockerfiles
        for docker_file in docker_files:
            try:
                with open(docker_file, 'r') as f:
                    content = f.read()
                    
                # Extract base images
                from_matches = re.findall(r'^FROM\s+([^\s]+)', content, re.MULTILINE)
                for base_image in from_matches:
                    if base_image not in base_images:
                        base_images[base_image] = []
                    base_images[base_image].append(str(docker_file))
                    
            except Exception as e:
                logger.error(f"Error reading {docker_file}: {e}")
                
        # Check for base image security issues
        for base_image, files in base_images.items():
            if 'python:3.11-slim' in base_image:
                # Check if it's the latest secure version
                self.findings["docker_issues"].append({
                    "issue": f"Base image {base_image} may have security updates available",
                    "affected_files": files,
                    "severity": "medium",
                    "recommendation": "Consider updating to python:3.12-slim or latest LTS"
                })
                
        # Analyze docker-compose files for port conflicts
        for compose_file in compose_files:
            try:
                import yaml
                with open(compose_file, 'r') as f:
                    compose_data = yaml.safe_load(f)
                    
                services = compose_data.get('services', {})
                for service_name, service_config in services.items():
                    ports = service_config.get('ports', [])
                    for port_mapping in ports:
                        if isinstance(port_mapping, str):
                            host_port = port_mapping.split(':')[0]
                            port_usage[host_port].append({
                                'service': service_name,
                                'file': str(compose_file)
                            })
                            
            except Exception as e:
                logger.error(f"Error reading {compose_file}: {e}")
                
        # Check for port conflicts
        for port, services in port_usage.items():
            if len(services) > 1:
                self.findings["docker_issues"].append({
                    "issue": f"Port {port} is used by multiple services",
                    "services": services,
                    "severity": "critical"
                })
                
    def analyze_node_dependencies(self):
        """Analyze Node.js dependencies"""
        logger.info("Analyzing Node.js dependencies...")
        
        package_files = list(self.project_root.rglob("package.json"))
        
        for pkg_file in package_files:
            try:
                with open(pkg_file, 'r') as f:
                    package_data = json.load(f)
                    
                dependencies = package_data.get('dependencies', {})
                dev_dependencies = package_data.get('devDependencies', {})
                
                # Check for security vulnerabilities in Node packages
                all_node_deps = {**dependencies, **dev_dependencies}
                
                # Check for known vulnerable packages
                vulnerable_packages = {
                    'next': {'min_safe': '14.2.0', 'issues': 'XSS vulnerabilities in older versions'},
                    'react': {'min_safe': '18.2.0', 'issues': 'Security fixes in recent versions'},
                    'axios': {'min_safe': '1.6.0', 'issues': 'SSRF vulnerabilities in older versions'}
                }
                
                for pkg, ver in all_node_deps.items():
                    if pkg in vulnerable_packages:
                        current_ver = ver.replace('^', '').replace('~', '')
                        min_safe = vulnerable_packages[pkg]['min_safe']
                        
                        try:
                            if version.parse(current_ver) < version.parse(min_safe):
                                self.findings["security_vulnerabilities"].append({
                                    "package": pkg,
                                    "current_version": ver,
                                    "min_safe_version": min_safe,
                                    "issue": vulnerable_packages[pkg]['issues'],
                                    "file": str(pkg_file),
                                    "severity": "high"
                                })
                        except:
                            pass  # Version parsing failed
                            
            except Exception as e:
                logger.error(f"Error reading {pkg_file}: {e}")
                
    def analyze_security_vulnerabilities(self):
        """Analyze for security vulnerabilities"""
        logger.info("Analyzing security vulnerabilities...")
        
        # Check for hardcoded secrets
        sensitive_patterns = [
            (r'password\s*=\s*["\'][^"\']{8,}["\']', 'Potential hardcoded password'),
            (r'secret\s*=\s*["\'][^"\']{20,}["\']', 'Potential hardcoded secret'),
            (r'key\s*=\s*["\'][^"\']{20,}["\']', 'Potential hardcoded API key'),
            (r'token\s*=\s*["\'][^"\']{20,}["\']', 'Potential hardcoded token'),
        ]
        
        python_files = list(self.project_root.rglob("*.py"))
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                for pattern, description in sensitive_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        self.findings["security_vulnerabilities"].append({
                            "file": str(py_file),
                            "issue": description,
                            "matches": len(matches),
                            "severity": "critical"
                        })
                        
            except Exception as e:
                logger.error(f"Error reading {py_file}: {e}")
                
        # Check file permissions for sensitive files
        sensitive_files = ['.env', '.env.local', 'secrets.json', 'private.key']
        
        for sens_file in sensitive_files:
            file_path = self.project_root / sens_file
            if file_path.exists():
                stat_info = file_path.stat()
                if stat_info.st_mode & 0o077:  # Check if others have permissions
                    self.findings["security_vulnerabilities"].append({
                        "file": str(file_path),
                        "issue": "Sensitive file has insecure permissions",
                        "severity": "high",
                        "recommendation": f"chmod 600 {file_path}"
                    })
                    
    def analyze_framework_compatibility(self):
        """Analyze AI/ML framework compatibility"""
        logger.info("Analyzing framework compatibility...")
        
        # Check for CUDA compatibility issues
        req_files = list(self.project_root.rglob("requirements*.txt"))
        all_packages = set()
        
        for req_file in req_files:
            try:
                with open(req_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            pkg_name = re.match(r'^([a-zA-Z0-9_-]+)', line)
                            if pkg_name:
                                all_packages.add(pkg_name.group(1).lower())
            except:
                pass
                
        # Check for potential CUDA conflicts
        gpu_packages = ['torch', 'tensorflow', 'onnxruntime-gpu', 'cupy', 'nvidia-ml-py']
        found_gpu_packages = [pkg for pkg in gpu_packages if pkg in all_packages]
        
        if len(found_gpu_packages) > 1:
            self.findings["compatibility_issues"].append({
                "issue": "Multiple GPU frameworks detected - potential CUDA version conflicts",
                "packages": found_gpu_packages,
                "severity": "medium",
                "recommendation": "Verify CUDA version compatibility across all GPU packages"
            })
            
        # Check for vector database conflicts
        vector_dbs = ['chromadb', 'qdrant-client', 'faiss-cpu', 'faiss-gpu', 'pinecone-client']
        found_vector_dbs = [pkg for pkg in vector_dbs if pkg in all_packages]
        
        if 'faiss-cpu' in found_vector_dbs and 'faiss-gpu' in found_vector_dbs:
            self.findings["compatibility_issues"].append({
                "issue": "Both faiss-cpu and faiss-gpu detected",
                "packages": ['faiss-cpu', 'faiss-gpu'],
                "severity": "high",
                "recommendation": "Use only faiss-gpu for GPU systems or faiss-cpu for CPU-only"
            })
            
    def generate_report(self):
        """Generate comprehensive dependency analysis report"""
        logger.info("Generating dependency analysis report...")
        
        # Calculate statistics
        total_issues = (
            len(self.findings["python_conflicts"]) +
            len(self.findings["docker_issues"]) +
            len(self.findings["security_vulnerabilities"]) +
            len(self.findings["version_mismatches"]) +
            len(self.findings["compatibility_issues"])
        )
        
        critical_issues = sum(1 for category in self.findings.values() 
                             for issue in category 
                             if isinstance(issue, dict) and issue.get('severity') == 'critical')
        
        # Generate recommendations
        self.generate_recommendations()
        
        # Create report
        report = {
            "analysis_summary": {
                "total_issues": total_issues,
                "critical_issues": critical_issues,
                "categories": {
                    "python_conflicts": len(self.findings["python_conflicts"]),
                    "docker_issues": len(self.findings["docker_issues"]),
                    "security_vulnerabilities": len(self.findings["security_vulnerabilities"]),
                    "compatibility_issues": len(self.findings["compatibility_issues"])
                }
            },
            "detailed_findings": self.findings,
            "timestamp": subprocess.check_output(['date']).decode().strip()
        }
        
        # Save report
        report_path = self.project_root / "logs" / "dependency_analysis_report.json"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        # Print summary
        self.print_summary(report)
        
        return report
        
    def generate_recommendations(self):
        """Generate specific recommendations for fixing issues"""
        
        recommendations = []
        
        # Python conflict recommendations
        if self.findings["python_conflicts"]:
            recommendations.append({
                "category": "Python Dependencies",
                "action": "Consolidate conflicting package versions",
                "commands": [
                    "pip-tools compile --upgrade requirements.txt",
                    "pip-tools sync requirements.txt"
                ]
            })
            
        # Docker security recommendations
        if any('python:3.11' in str(issue) for issue in self.findings["docker_issues"]):
            recommendations.append({
                "category": "Docker Security",
                "action": "Update base images to latest secure versions",
                "commands": [
                    "sed -i 's/python:3.11-slim/python:3.12-slim/g' docker/*/Dockerfile",
                    "docker system prune -af"
                ]
            })
            
        # Security vulnerability recommendations
        if self.findings["security_vulnerabilities"]:
            recommendations.append({
                "category": "Security",
                "action": "Fix security vulnerabilities",
                "commands": [
                    "chmod 600 .env*",
                    "pip install --upgrade pip",
                    "npm audit fix"
                ]
            })
            
        # Framework compatibility recommendations
        if any('CUDA' in str(issue) for issue in self.findings["compatibility_issues"]):
            recommendations.append({
                "category": "Framework Compatibility",
                "action": "Verify CUDA compatibility",
                "commands": [
                    "nvidia-smi",
                    "python -c 'import torch; print(torch.cuda.is_available())'",
                    "python -c 'import tensorflow as tf; print(tf.config.list_physical_devices())'"
                ]
            })
            
        self.findings["recommendations"] = recommendations
        
    def print_summary(self, report: Dict):
        """Print analysis summary"""
        print("\n" + "="*80)
        print("SUTAZAI DEPENDENCY ANALYSIS REPORT")
        print("="*80)
        
        summary = report["analysis_summary"]
        print(f"Total Issues Found: {summary['total_issues']}")
        print(f"Critical Issues: {summary['critical_issues']}")
        print()
        
        print("Issues by Category:")
        for category, count in summary["categories"].items():
            if count > 0:
                severity_indicator = "🔴" if "security" in category or "critical" in category else "🟡"
                print(f"  {severity_indicator} {category.replace('_', ' ').title()}: {count}")
        print()
        
        # Print critical issues
        if summary['critical_issues'] > 0:
            print("CRITICAL ISSUES REQUIRING IMMEDIATE ATTENTION:")
            print("-" * 50)
            
            for category, issues in self.findings.items():
                for issue in issues:
                    if isinstance(issue, dict) and issue.get('severity') == 'critical':
                        print(f"❌ {issue.get('issue', issue.get('package', 'Unknown issue'))}")
                        if 'recommendation' in issue:
                            print(f"   Fix: {issue['recommendation']}")
                        print()
                        
        # Print recommendations
        if self.findings["recommendations"]:
            print("RECOMMENDED ACTIONS:")
            print("-" * 50)
            for i, rec in enumerate(self.findings["recommendations"], 1):
                print(f"{i}. {rec['category']}: {rec['action']}")
                for cmd in rec['commands']:
                    print(f"   $ {cmd}")
                print()
                
        print("="*80)
        print(f"Detailed report saved to: {self.project_root}/logs/dependency_analysis_report.json")
        print("="*80)

def main():
    """Main execution function"""
    analyzer = DependencyAnalyzer()
    analyzer.analyze_all_dependencies()

if __name__ == "__main__":
    main()