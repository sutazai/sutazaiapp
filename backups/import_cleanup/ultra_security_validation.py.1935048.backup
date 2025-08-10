#!/usr/bin/env python3
"""
ULTRA SECURITY VALIDATION SCRIPT
Comprehensive security audit and validation after final hardening
"""

import os
import json
import yaml
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

class UltraSecurityValidator:
    def __init__(self):
        self.base_path = Path("/opt/sutazaiapp")
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "total_score": 0,
            "max_score": 100,
            "critical_issues": [],
            "resolved_issues": [],
            "validation_details": {}
        }
        
    def check_docker_socket_mounts(self) -> Tuple[int, List[str]]:
        """Check for Docker socket mounts in compose files"""
        issues = []
        compose_files = list(self.base_path.glob("docker-compose*.yml"))
        compose_files.extend(list((self.base_path / "docker").glob("docker-compose*.yml")))
        
        for compose_file in compose_files:
            if "backup" in str(compose_file).lower():
                continue
                
            try:
                with open(compose_file, 'r') as f:
                    content = f.read()
                    
                # Check for active Docker socket mounts (not commented)
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    if '/var/run/docker.sock' in line and not line.strip().startswith('#'):
                        # Special exception for cAdvisor which requires it
                        context = '\n'.join(lines[max(0, i-10):min(len(lines), i+10)])
                        if 'cadvisor' not in context.lower():
                            issues.append(f"{compose_file.relative_to(self.base_path)}:{i} - Docker socket mount found")
                            
            except Exception as e:
                issues.append(f"Error reading {compose_file}: {e}")
                
        score = 100 if len(issues) == 0 else max(0, 100 - len(issues) * 20)
        return score, issues
        
    def check_tls_verification(self) -> Tuple[int, List[str]]:
        """Check for disabled TLS verification"""
        issues = []
        python_files = list(self.base_path.rglob("*.py"))
        
        dangerous_patterns = [
            ('verify=False', 'TLS verification disabled'),
            ('verify=0', 'TLS verification disabled'),
            ('SSL_VERIFY=false', 'SSL verification disabled'),
            ('InsecureRequestWarning', 'Insecure request warning suppressed'),
            ('disable_warnings()', 'Security warnings disabled')
        ]
        
        for py_file in python_files:
            if any(skip in str(py_file) for skip in ['backup', 'test', '__pycache__', 'venv']):
                continue
                
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                for i, line in enumerate(lines, 1):
                    if line.strip().startswith('#'):
                        continue
                        
                    for pattern, desc in dangerous_patterns:
                        if pattern in line:
                            issues.append(f"{py_file.relative_to(self.base_path)}:{i} - {desc}: {line.strip()}")
                            
            except Exception as e:
                continue
                
        score = 100 if len(issues) == 0 else max(0, 100 - len(issues) * 10)
        return score, issues
        
    def check_container_users(self) -> Tuple[int, List[str]]:
        """Check for containers running as root"""
        issues = []
        resolved = []
        
        dockerfiles = list(self.base_path.rglob("Dockerfile*"))
        
        for dockerfile in dockerfiles:
            if any(skip in str(dockerfile) for skip in ['backup', 'test', 'archive']):
                continue
                
            try:
                with open(dockerfile, 'r') as f:
                    content = f.read()
                    
                has_user = False
                user_is_root = False
                
                for line in content.split('\n'):
                    if line.strip().startswith('USER'):
                        has_user = True
                        user_val = line.strip().split()[1] if len(line.strip().split()) > 1 else ''
                        if user_val in ['0', 'root', '0:0', 'root:root']:
                            user_is_root = True
                        else:
                            resolved.append(f"{dockerfile.relative_to(self.base_path)} - Running as non-root: {user_val}")
                            
                if not has_user:
                    issues.append(f"{dockerfile.relative_to(self.base_path)} - No USER directive (defaults to root)")
                elif user_is_root:
                    issues.append(f"{dockerfile.relative_to(self.base_path)} - Explicitly running as root")
                    
            except Exception as e:
                continue
                
        # Calculate percentage of non-root containers
        total_containers = len(dockerfiles)
        root_containers = len(issues)
        non_root_percentage = ((total_containers - root_containers) / total_containers * 100) if total_containers > 0 else 0
        
        score = min(100, int(non_root_percentage))
        return score, issues
        
    def check_resource_limits(self) -> Tuple[int, List[str]]:
        """Check for proper resource limits in compose files"""
        issues = []
        services_checked = 0
        services_with_limits = 0
        
        compose_files = [
            self.base_path / "docker-compose.yml",
            self.base_path / "docker-compose.optimized.yml"
        ]
        
        for compose_file in compose_files:
            if not compose_file.exists():
                continue
                
            try:
                with open(compose_file, 'r') as f:
                    data = yaml.safe_load(f)
                    
                if 'services' in data:
                    for service_name, service_config in data['services'].items():
                        services_checked += 1
                        
                        # Check for resource limits
                        if 'deploy' in service_config and 'resources' in service_config['deploy']:
                            if 'limits' in service_config['deploy']['resources']:
                                services_with_limits += 1
                            else:
                                issues.append(f"{service_name} - No resource limits defined")
                        else:
                            issues.append(f"{service_name} - No deploy.resources configuration")
                            
            except Exception as e:
                issues.append(f"Error parsing {compose_file}: {e}")
                
        score = (services_with_limits / services_checked * 100) if services_checked > 0 else 0
        return int(score), issues
        
    def check_secrets_management(self) -> Tuple[int, List[str]]:
        """Check for hardcoded secrets"""
        issues = []
        files_to_check = list(self.base_path.rglob("*.py"))
        files_to_check.extend(list(self.base_path.rglob("*.yml")))
        files_to_check.extend(list(self.base_path.rglob("*.yaml")))
        
        # Patterns that indicate hardcoded secrets
        secret_patterns = [
            'password=',
            'secret=',
            'api_key=',
            'token=',
            'JWT_SECRET=',
            'SECRET_KEY='
        ]
        
        for file_path in files_to_check:
            if any(skip in str(file_path) for skip in ['backup', 'test', '.env', 'example']):
                continue
                
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    
                for i, line in enumerate(lines, 1):
                    if line.strip().startswith('#'):
                        continue
                        
                    for pattern in secret_patterns:
                        if pattern in line.lower():
                            # Check if it's using environment variable
                            if 'os.getenv' not in line and '${' not in line and 'environ' not in line:
                                # Check if value looks like a hardcoded secret
                                parts = line.split(pattern)
                                if len(parts) > 1:
                                    value = parts[1].split()[0].strip('"\'')
                                    if value and value not in ['', 'None', 'null', '""', "''", '{', '[']:
                                        issues.append(f"{file_path.relative_to(self.base_path)}:{i} - Possible hardcoded secret")
                                        
            except Exception as e:
                continue
                
        score = 100 if len(issues) == 0 else max(0, 100 - len(issues) * 5)
        return score, issues
        
    def generate_report(self) -> str:
        """Generate comprehensive security report"""
        print("\n" + "="*80)
        print("ULTRA SECURITY VALIDATION REPORT")
        print("="*80)
        print(f"Timestamp: {self.results['timestamp']}")
        print("="*80 + "\n")
        
        # Run all checks
        checks = [
            ("Docker Socket Mounts", self.check_docker_socket_mounts()),
            ("TLS Verification", self.check_tls_verification()),
            ("Container Users", self.check_container_users()),
            ("Resource Limits", self.check_resource_limits()),
            ("Secrets Management", self.check_secrets_management())
        ]
        
        total_score = 0
        total_issues = 0
        
        for check_name, (score, issues) in checks:
            self.results["validation_details"][check_name] = {
                "score": score,
                "issues": issues,
                "status": "PASS" if score >= 80 else "WARN" if score >= 60 else "FAIL"
            }
            
            total_score += score / len(checks)
            total_issues += len(issues)
            
            print(f"\n## {check_name}")
            print(f"Score: {score}/100")
            print(f"Status: {'✅ PASS' if score >= 80 else '⚠️ WARNING' if score >= 60 else '❌ FAIL'}")
            
            if issues:
                print(f"Issues Found ({len(issues)}):")
                for issue in issues[:10]:  # Show first 10 issues
                    print(f"  - {issue}")
                if len(issues) > 10:
                    print(f"  ... and {len(issues) - 10} more issues")
            else:
                print("✅ No issues found")
                
        self.results["total_score"] = int(total_score)
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Overall Security Score: {self.results['total_score']}/100")
        print(f"Total Issues Found: {total_issues}")
        
        # Critical recommendations
        print("\n## CRITICAL SECURITY STATUS:")
        
        if self.results["total_score"] >= 90:
            print("✅ EXCELLENT - System is highly secure")
        elif self.results["total_score"] >= 80:
            print("✅ GOOD - System is secure with minor improvements needed")
        elif self.results["total_score"] >= 70:
            print("⚠️ WARNING - System has security concerns that should be addressed")
        else:
            print("❌ CRITICAL - System has serious security vulnerabilities")
            
        # Save report
        report_path = self.base_path / f"ULTRA_SECURITY_VALIDATION_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
            
        print(f"\nDetailed report saved to: {report_path}")
        
        return str(report_path)
        
if __name__ == "__main__":
    validator = UltraSecurityValidator()
    validator.generate_report()