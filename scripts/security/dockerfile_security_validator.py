#!/usr/bin/env python3
"""
SutazAI Dockerfile Security Validation Module
Ultra QA Validator - Security & Compliance Testing for Consolidation

This module validates that Dockerfile consolidation maintains or improves
security posture and compliance standards.

Author: ULTRA QA VALIDATOR  
Date: August 10, 2025
Version: 1.0.0
"""

import os
import re
import json
import logging
import subprocess
import tempfile
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
from datetime import datetime
import docker
import yaml

logger = logging.getLogger(__name__)

class DockerfileSecurityValidator:
    """Security validation for Dockerfile consolidation."""
    
    def __init__(self):
        """Initialize security validator."""
        self.docker_client = docker.from_env()
        self.security_patterns = self._load_security_patterns()
        self.compliance_rules = self._load_compliance_rules()
        
    def _load_security_patterns(self) -> Dict:
        """Load security vulnerability patterns."""
        return {
            'hardcoded_secrets': [
                r'(?i)(password|passwd|pwd)\s*[=:]\s*["\']?[a-zA-Z0-9]+["\']?',
                r'(?i)(secret|api_?key|token)\s*[=:]\s*["\']?[a-zA-Z0-9]+["\']?',
                r'(?i)(access_?key|private_?key)\s*[=:]\s*["\']?[a-zA-Z0-9]+["\']?',
            ],
            'insecure_protocols': [
                r'http://(?!localhost|127\.0\.0\.1)',  # HTTP instead of HTTPS
                r'ftp://',  # Insecure FTP
                r'telnet://',  # Insecure telnet
            ],
            'privileged_operations': [
                r'(?i)--privileged',
                r'(?i)--cap-add\s+SYS_ADMIN',
                r'(?i)--security-opt\s+seccomp=unconfined',
                r'(?i)chmod\s+777',
                r'(?i)chown.*root:root',
            ],
            'insecure_package_installs': [
                r'pip install.*--trusted-host',
                r'pip install.*--allow-external',
                r'pip install.*--allow-unverified',
                r'apt-get install.*--allow-unauthenticated',
                r'yum install.*--nogpgcheck',
            ],
            'root_user_usage': [
                r'^USER\s+root\s*$',
                r'^USER\s+0\s*$',
                r'sudo\s+',  # Running commands as sudo
            ],
            'missing_security_updates': [
                r'apt-get\s+install(?!.*update)',
                r'yum\s+install(?!.*update)',
                r'apk\s+add(?!.*update)',
            ]
        }
    
    def _load_compliance_rules(self) -> Dict:
        """Load compliance rules (CIS, NIST, etc.)."""
        return {
            'cis_docker_benchmark': {
                'user_namespace': 'Container should not run as root user',
                'health_check': 'Container should have health check configured',
                'pid_cgroup': 'Container should not share host PID namespace',
                'network_namespace': 'Container should not share host network namespace',
                'memory_limit': 'Container should have memory limit set',
                'cpu_limit': 'Container should have CPU limit set',
                'read_only_fs': 'Container root filesystem should be read-only when possible',
                'no_privileged': 'Container should not run in privileged mode',
            },
            'owasp_docker_security': {
                'secrets_management': 'Secrets should not be embedded in images',
                'minimal_attack_surface': 'Image should have minimal attack surface',
                'trusted_base_images': 'Should use trusted base images only',
                'image_scanning': 'Images should be scanned for vulnerabilities',
                'runtime_security': 'Runtime security controls should be in place',
            }
        }
    
    def analyze_dockerfile_security(self, dockerfile_path: Path) -> Dict:
        """Comprehensive security analysis of a Dockerfile."""
        logger.info(f"Analyzing security for: {dockerfile_path}")
        
        try:
            content = dockerfile_path.read_text()
            lines = content.splitlines()
            
            analysis = {
                'dockerfile_path': str(dockerfile_path),
                'vulnerability_scan': self._scan_vulnerabilities(content, lines),
                'compliance_check': self._check_compliance(content, lines),
                'security_score': 0,
                'security_grade': 'F',
                'recommendations': []
            }
            
            # Calculate security score
            analysis['security_score'] = self._calculate_security_score(analysis)
            analysis['security_grade'] = self._grade_security_score(analysis['security_score'])
            
            # Generate recommendations
            analysis['recommendations'] = self._generate_security_recommendations(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Security analysis failed for {dockerfile_path}: {e}")
            return {
                'dockerfile_path': str(dockerfile_path),
                'error': str(e),
                'security_score': 0,
                'security_grade': 'F'
            }
    
    def _scan_vulnerabilities(self, content: str, lines: List[str]) -> Dict:
        """Scan for security vulnerabilities in Dockerfile."""
        vulnerabilities = {
            'critical': [],
            'high': [],
            'medium': [],
            'low': [],
            'total_count': 0
        }
        
        for category, patterns in self.security_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE)
                for match in matches:
                    # Find line number
                    line_num = content[:match.start()].count('\n') + 1
                    line_content = lines[line_num - 1] if line_num <= len(lines) else ""
                    
                    vuln = {
                        'category': category,
                        'line': line_num,
                        'content': line_content.strip(),
                        'pattern': pattern,
                        'severity': self._get_vulnerability_severity(category)
                    }
                    
                    severity = vuln['severity']
                    vulnerabilities[severity].append(vuln)
                    vulnerabilities['total_count'] += 1
        
        return vulnerabilities
    
    def _get_vulnerability_severity(self, category: str) -> str:
        """Get severity level for vulnerability category."""
        severity_map = {
            'hardcoded_secrets': 'critical',
            'privileged_operations': 'critical',
            'root_user_usage': 'high',
            'insecure_protocols': 'high',
            'insecure_package_installs': 'medium',
            'missing_security_updates': 'medium'
        }
        return severity_map.get(category, 'low')
    
    def _check_compliance(self, content: str, lines: List[str]) -> Dict:
        """Check compliance with security standards."""
        compliance = {
            'cis_docker_benchmark': {},
            'owasp_docker_security': {},
            'compliance_score': 0,
            'total_rules': 0,
            'passed_rules': 0
        }
        
        # CIS Docker Benchmark checks
        cis_results = {}
        
        # Check for non-root user
        has_non_root_user = any(
            line.strip().startswith('USER ') and 
            'root' not in line.lower() and 
            line.strip() != 'USER 0'
            for line in lines
        )
        cis_results['user_namespace'] = {
            'passed': has_non_root_user,
            'description': self.compliance_rules['cis_docker_benchmark']['user_namespace']
        }
        
        # Check for health check
        has_health_check = any(line.strip().startswith('HEALTHCHECK') for line in lines)
        cis_results['health_check'] = {
            'passed': has_health_check,
            'description': self.compliance_rules['cis_docker_benchmark']['health_check']
        }
        
        # Check for privileged mode
        no_privileged = not any('--privileged' in line.lower() for line in lines)
        cis_results['no_privileged'] = {
            'passed': no_privileged,
            'description': self.compliance_rules['cis_docker_benchmark']['no_privileged']
        }
        
        compliance['cis_docker_benchmark'] = cis_results
        
        # OWASP Docker Security checks
        owasp_results = {}
        
        # Check for secrets management
        no_embedded_secrets = len(self._scan_vulnerabilities(content, lines)['critical']) == 0
        owasp_results['secrets_management'] = {
            'passed': no_embedded_secrets,
            'description': self.compliance_rules['owasp_docker_security']['secrets_management']
        }
        
        # Check for trusted base images
        base_images = [line.strip() for line in lines if line.strip().startswith('FROM')]
        trusted_bases = all(
            any(trusted in img.lower() for trusted in ['python:', 'node:', 'alpine:', 'ubuntu:'])
            for img in base_images
        )
        owasp_results['trusted_base_images'] = {
            'passed': trusted_bases,
            'description': self.compliance_rules['owasp_docker_security']['trusted_base_images']
        }
        
        compliance['owasp_docker_security'] = owasp_results
        
        # Calculate compliance score
        all_checks = {**cis_results, **owasp_results}
        compliance['total_rules'] = len(all_checks)
        compliance['passed_rules'] = sum(1 for check in all_checks.values() if check['passed'])
        compliance['compliance_score'] = (
            compliance['passed_rules'] / compliance['total_rules'] * 100
            if compliance['total_rules'] > 0 else 0
        )
        
        return compliance
    
    def _calculate_security_score(self, analysis: Dict) -> float:
        """Calculate overall security score (0-100)."""
        # Start with perfect score
        score = 100.0
        
        # Deduct points for vulnerabilities
        vuln_scan = analysis.get('vulnerability_scan', {})
        score -= vuln_scan.get('critical', []) * 25  # Critical: -25 points each
        score -= len(vuln_scan.get('high', [])) * 15     # High: -15 points each
        score -= len(vuln_scan.get('medium', [])) * 10   # Medium: -10 points each
        score -= len(vuln_scan.get('low', [])) * 5       # Low: -5 points each
        
        # Factor in compliance score (weight: 40%)
        compliance_score = analysis.get('compliance_check', {}).get('compliance_score', 0)
        score = (score * 0.6) + (compliance_score * 0.4)
        
        return max(0.0, min(100.0, score))  # Clamp between 0-100
    
    def _grade_security_score(self, score: float) -> str:
        """Convert security score to letter grade."""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def _generate_security_recommendations(self, analysis: Dict) -> List[str]:
        """Generate actionable security recommendations."""
        recommendations = []
        
        # Vulnerability-based recommendations
        vuln_scan = analysis.get('vulnerability_scan', {})
        
        if vuln_scan.get('critical'):
            recommendations.append("CRITICAL: Remove hardcoded secrets and privileged operations")
        
        if vuln_scan.get('high'):
            recommendations.append("HIGH: Implement non-root user and secure protocols")
        
        if vuln_scan.get('medium'):
            recommendations.append("MEDIUM: Use secure package installation methods")
        
        # Compliance-based recommendations
        compliance = analysis.get('compliance_check', {})
        cis_checks = compliance.get('cis_docker_benchmark', {})
        owasp_checks = compliance.get('owasp_docker_security', {})
        
        failed_checks = []
        for check_name, check_result in {**cis_checks, **owasp_checks}.items():
            if not check_result.get('passed', True):
                failed_checks.append(check_result.get('description', check_name))
        
        if failed_checks:
            recommendations.append(f"COMPLIANCE: Address {len(failed_checks)} failed compliance checks")
        
        return recommendations
    
    def scan_container_vulnerabilities(self, image_name: str) -> Dict:
        """Scan container image for vulnerabilities using Trivy."""
        logger.info(f"Scanning container vulnerabilities for: {image_name}")
        
        try:
            # Use Trivy for vulnerability scanning
            result = subprocess.run([
                'trivy', 'image', '--format', 'json', '--quiet', image_name
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                scan_data = json.loads(result.stdout)
                
                # Process Trivy results
                vulnerabilities = {
                    'critical': 0,
                    'high': 0,
                    'medium': 0,
                    'low': 0,
                    'total': 0,
                    'details': []
                }
                
                for target in scan_data.get('Results', []):
                    for vuln in target.get('Vulnerabilities', []):
                        severity = vuln.get('Severity', 'UNKNOWN').lower()
                        if severity in vulnerabilities:
                            vulnerabilities[severity] += 1
                            vulnerabilities['total'] += 1
                        
                        vulnerabilities['details'].append({
                            'vulnerability_id': vuln.get('VulnerabilityID'),
                            'package_name': vuln.get('PkgName'),
                            'installed_version': vuln.get('InstalledVersion'),
                            'fixed_version': vuln.get('FixedVersion'),
                            'severity': severity,
                            'title': vuln.get('Title', '')
                        })
                
                return {
                    'scan_successful': True,
                    'vulnerabilities': vulnerabilities,
                    'scan_timestamp': datetime.now().isoformat()
                }
            else:
                logger.warning(f"Trivy scan failed: {result.stderr}")
                return {
                    'scan_successful': False,
                    'error': result.stderr,
                    'fallback_scan': True
                }
        
        except FileNotFoundError:
            logger.warning("Trivy not found, skipping container vulnerability scan")
            return {
                'scan_successful': False,
                'error': 'Trivy not installed',
                'fallback_scan': True
            }
        except Exception as e:
            logger.error(f"Container vulnerability scan failed: {e}")
            return {
                'scan_successful': False,
                'error': str(e),
                'fallback_scan': True
            }
    
    def validate_container_runtime_security(self, container_name: str) -> Dict:
        """Validate runtime security configuration of a container."""
        logger.info(f"Validating runtime security for: {container_name}")
        
        try:
            containers = self.docker_client.containers.list()
            container = None
            
            for c in containers:
                if container_name.lower() in c.name.lower():
                    container = c
                    break
            
            if not container:
                return {
                    'container_found': False,
                    'error': f'Container {container_name} not found'
                }
            
            container.reload()
            config = container.attrs
            
            # Runtime security checks
            security_checks = {
                'non_root_user': self._check_non_root_user(config),
                'no_privileged_mode': self._check_privileged_mode(config),
                'resource_limits': self._check_resource_limits(config),
                'network_isolation': self._check_network_isolation(config),
                'readonly_rootfs': self._check_readonly_rootfs(config),
                'security_options': self._check_security_options(config)
            }
            
            passed_checks = sum(1 for check in security_checks.values() if check['passed'])
            total_checks = len(security_checks)
            
            return {
                'container_found': True,
                'container_name': container.name,
                'security_checks': security_checks,
                'passed_checks': passed_checks,
                'total_checks': total_checks,
                'security_score': (passed_checks / total_checks) * 100,
                'runtime_security_grade': self._grade_security_score(
                    (passed_checks / total_checks) * 100
                )
            }
            
        except Exception as e:
            logger.error(f"Runtime security validation failed: {e}")
            return {
                'container_found': False,
                'error': str(e)
            }
    
    def _check_non_root_user(self, config: Dict) -> Dict:
        """Check if container runs as non-root user."""
        user = config.get('Config', {}).get('User', '')
        is_non_root = user and user not in ['root', '0', '']
        
        return {
            'passed': is_non_root,
            'details': f"Running as user: {user if user else 'root (default)'}",
            'recommendation': "Configure container to run as non-root user" if not is_non_root else None
        }
    
    def _check_privileged_mode(self, config: Dict) -> Dict:
        """Check if container runs in privileged mode."""
        privileged = config.get('HostConfig', {}).get('Privileged', False)
        
        return {
            'passed': not privileged,
            'details': f"Privileged mode: {'enabled' if privileged else 'disabled'}",
            'recommendation': "Disable privileged mode" if privileged else None
        }
    
    def _check_resource_limits(self, config: Dict) -> Dict:
        """Check if container has resource limits set."""
        host_config = config.get('HostConfig', {})
        memory_limit = host_config.get('Memory', 0)
        cpu_limit = host_config.get('CpuQuota', 0)
        
        has_limits = memory_limit > 0 or cpu_limit > 0
        
        return {
            'passed': has_limits,
            'details': f"Memory limit: {memory_limit}, CPU limit: {cpu_limit}",
            'recommendation': "Set resource limits for container" if not has_limits else None
        }
    
    def _check_network_isolation(self, config: Dict) -> Dict:
        """Check network isolation configuration."""
        network_mode = config.get('HostConfig', {}).get('NetworkMode', '')
        isolated = network_mode not in ['host', 'none']
        
        return {
            'passed': isolated,
            'details': f"Network mode: {network_mode}",
            'recommendation': "Use isolated network mode" if not isolated else None
        }
    
    def _check_readonly_rootfs(self, config: Dict) -> Dict:
        """Check if root filesystem is read-only."""
        readonly = config.get('HostConfig', {}).get('ReadonlyRootfs', False)
        
        return {
            'passed': readonly,
            'details': f"Read-only root filesystem: {'enabled' if readonly else 'disabled'}",
            'recommendation': "Enable read-only root filesystem where possible" if not readonly else None
        }
    
    def _check_security_options(self, config: Dict) -> Dict:
        """Check security options configuration."""
        security_opt = config.get('HostConfig', {}).get('SecurityOpt', [])
        
        # Look for good security options
        good_options = ['no-new-privileges:true', 'apparmor:', 'seccomp:']
        has_good_options = any(
            any(good in opt for good in good_options)
            for opt in security_opt
        )
        
        return {
            'passed': has_good_options or len(security_opt) == 0,  # Default is usually fine
            'details': f"Security options: {security_opt}",
            'recommendation': "Review security options configuration" if not has_good_options else None
        }
    
    def run_comprehensive_security_validation(self, dockerfiles: Dict[str, Path], 
                                            containers: List[str] = None) -> Dict:
        """Run comprehensive security validation."""
        logger.info("Starting comprehensive security validation")
        
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'dockerfile_analysis': {},
            'container_scans': {},
            'runtime_security': {},
            'summary': {
                'total_dockerfiles': len(dockerfiles),
                'secure_dockerfiles': 0,
                'vulnerable_dockerfiles': 0,
                'total_vulnerabilities': 0,
                'critical_vulnerabilities': 0,
                'overall_security_grade': 'F'
            }
        }
        
        # Analyze Dockerfiles
        for service_name, dockerfile_path in dockerfiles.items():
            analysis = self.analyze_dockerfile_security(dockerfile_path)
            validation_results['dockerfile_analysis'][service_name] = analysis
            
            # Update summary
            if analysis.get('security_grade', 'F') in ['A', 'B', 'C']:
                validation_results['summary']['secure_dockerfiles'] += 1
            else:
                validation_results['summary']['vulnerable_dockerfiles'] += 1
            
            vuln_scan = analysis.get('vulnerability_scan', {})
            validation_results['summary']['total_vulnerabilities'] += vuln_scan.get('total_count', 0)
            validation_results['summary']['critical_vulnerabilities'] += len(vuln_scan.get('critical', []))
        
        # Scan container images (if requested)
        if containers:
            for container_name in containers:
                image_name = f"sutazai-{container_name.lower()}:latest"
                scan_result = self.scan_container_vulnerabilities(image_name)
                validation_results['container_scans'][container_name] = scan_result
                
                # Validate runtime security
                runtime_result = self.validate_container_runtime_security(container_name)
                validation_results['runtime_security'][container_name] = runtime_result
        
        # Calculate overall security grade
        secure_pct = (
            validation_results['summary']['secure_dockerfiles'] / 
            validation_results['summary']['total_dockerfiles'] * 100
            if validation_results['summary']['total_dockerfiles'] > 0 else 0
        )
        
        if secure_pct >= 90 and validation_results['summary']['critical_vulnerabilities'] == 0:
            validation_results['summary']['overall_security_grade'] = 'A'
        elif secure_pct >= 80 and validation_results['summary']['critical_vulnerabilities'] <= 2:
            validation_results['summary']['overall_security_grade'] = 'B'
        elif secure_pct >= 70:
            validation_results['summary']['overall_security_grade'] = 'C'
        elif secure_pct >= 60:
            validation_results['summary']['overall_security_grade'] = 'D'
        else:
            validation_results['summary']['overall_security_grade'] = 'F'
        
        logger.info(f"Security validation completed. Overall grade: {validation_results['summary']['overall_security_grade']}")
        return validation_results
    
    def save_security_results(self, results: Dict, output_file: str):
        """Save security validation results to JSON file."""
        project_root = Path(__file__).parent.parent
        output_path = project_root / output_file
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Security validation results saved to: {output_path}")

def main():
    """Main execution function for security validation."""
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    validator = DockerfileSecurityValidator()
    
    # Discover Dockerfiles
    project_root = Path(__file__).parent.parent
    dockerfiles = {}
    
    for dockerfile in project_root.rglob("Dockerfile*"):
        if any(skip in str(dockerfile) for skip in ['.backup', 'test-', 'backup']):
            continue
        
        service_name = dockerfile.parent.name
        if dockerfile.name != "Dockerfile":
            service_name = f"{service_name}_{dockerfile.stem}"
        
        dockerfiles[service_name] = dockerfile
    
    # Key containers to validate
    key_containers = [
        'backend', 'frontend', 'ai-agent-orchestrator', 
        'hardware-resource-optimizer', 'ollama-integration'
    ]
    
    # Run comprehensive security validation
    results = validator.run_comprehensive_security_validation(dockerfiles, key_containers)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"dockerfile_security_validation_{timestamp}.json"
    validator.save_security_results(results, results_file)
    
    # Print summary
    print("\n" + "="*70)
    print("  DOCKERFILE CONSOLIDATION SECURITY VALIDATION RESULTS")
    print("="*70)
    print(f"Dockerfiles Analyzed:     {results['summary']['total_dockerfiles']}")
    print(f"Secure Dockerfiles:       {results['summary']['secure_dockerfiles']}")
    print(f"Vulnerable Dockerfiles:   {results['summary']['vulnerable_dockerfiles']}")
    print(f"Total Vulnerabilities:    {results['summary']['total_vulnerabilities']}")
    print(f"Critical Vulnerabilities: {results['summary']['critical_vulnerabilities']}")
    print(f"Overall Security Grade:   {results['summary']['overall_security_grade']}")
    print("="*70)
    
    # Show critical issues
    critical_issues = []
    for service, analysis in results['dockerfile_analysis'].items():
        if analysis.get('vulnerability_scan', {}).get('critical'):
            critical_issues.append(service)
    
    if critical_issues:
        print("\n🚨 CRITICAL SECURITY ISSUES FOUND:")
        for service in critical_issues:
            print(f"  • {service}")
        print("\n⚠️  Address critical issues before production deployment!")
    else:
        print("\n✅ No critical security issues found")
    
    print(f"\nDetailed Report: {results_file}")
    
    # Return appropriate exit code
    return 0 if results['summary']['overall_security_grade'] in ['A', 'B'] else 1

if __name__ == "__main__":
    import sys
    sys.exit(main())