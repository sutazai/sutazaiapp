#!/usr/bin/env python3
"""
Container Security Auditor for SutazAI System
Comprehensive audit of Docker container security configurations
"""

import subprocess
import json
import re
from datetime import datetime
import os
import yaml

class ContainerSecurityAuditor:
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'containers_audited': 0,
            'security_issues': [],
            'compliance_score': 0.0,
            'recommendations': []
        }
        
    def get_running_containers(self):
        """Get list of running containers"""
        try:
            result = subprocess.run(
                ['docker', 'ps', '--format', 'json'],
                capture_output=True, text=True, check=True
            )
            containers = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    containers.append(json.loads(line))
            return containers
        except subprocess.CalledProcessError as e:
            print(f"Error getting containers: {e}")
            return []
    
    def inspect_container(self, container_id):
        """Inspect container security configuration"""
        try:
            result = subprocess.run(
                ['docker', 'inspect', container_id],
                capture_output=True, text=True, check=True
            )
            return json.loads(result.stdout)[0]
        except subprocess.CalledProcessError as e:
            print(f"Error inspecting container {container_id}: {e}")
            return None
    
    def audit_container_security(self, container_info, inspect_data):
        """Audit security configuration of a single container"""
        issues = []
        container_id = container_info['ID']
        container_name = container_info['Names']
        
        # Check 1: Root user
        config = inspect_data['Config']
        if config.get('User') == '' or config.get('User') == 'root' or not config.get('User'):
            issues.append({
                'severity': 'HIGH',
                'type': 'ROOT_USER',
                'container': container_name,
                'description': 'Container running as root user',
                'recommendation': 'Create and use non-root user'
            })
        
        # Check 2: Privileged mode
        host_config = inspect_data['HostConfig']
        if host_config.get('Privileged', False):
            issues.append({
                'severity': 'CRITICAL',
                'type': 'PRIVILEGED_MODE',
                'container': container_name,
                'description': 'Container running in privileged mode',
                'recommendation': 'Remove privileged flag unless absolutely necessary'
            })
        
        # Check 3: Capabilities
        cap_add = host_config.get('CapAdd') or []
        dangerous_caps = ['SYS_ADMIN', 'SYS_PTRACE', 'SYS_MODULE', 'DAC_OVERRIDE']
        for cap in cap_add:
            if cap in dangerous_caps:
                issues.append({
                    'severity': 'HIGH',
                    'type': 'DANGEROUS_CAPABILITY',
                    'container': container_name,
                    'description': f'Container has dangerous capability: {cap}',
                    'recommendation': f'Remove capability {cap} if not required'
                })
        
        # Check 4: Resource limits
        memory_limit = host_config.get('Memory', 0)
        if memory_limit == 0:
            issues.append({
                'severity': 'MEDIUM',
                'type': 'NO_MEMORY_LIMIT',
                'container': container_name,
                'description': 'No memory limit set',
                'recommendation': 'Set appropriate memory limits'
            })
        
        cpu_quota = host_config.get('CpuQuota', 0)
        if cpu_quota == 0:
            issues.append({
                'severity': 'MEDIUM',
                'type': 'NO_CPU_LIMIT',
                'container': container_name,
                'description': 'No CPU limit set',
                'recommendation': 'Set appropriate CPU limits'
            })
        
        # Check 5: Network mode
        network_mode = host_config.get('NetworkMode', '')
        if network_mode == 'host':
            issues.append({
                'severity': 'HIGH',
                'type': 'HOST_NETWORK',
                'container': container_name,
                'description': 'Container using host network mode',
                'recommendation': 'Use bridge or custom networks instead of host network'
            })
        
        # Check 6: Volume mounts
        mounts = inspect_data.get('Mounts', [])
        for mount in mounts:
            if mount.get('Type') == 'bind':
                source = mount.get('Source', '')
                if source in ['/etc', '/proc', '/sys', '/']:
                    issues.append({
                        'severity': 'HIGH',
                        'type': 'SENSITIVE_MOUNT',
                        'container': container_name,
                        'description': f'Sensitive directory mounted: {source}',
                        'recommendation': 'Avoid mounting sensitive system directories'
                    })
                
                if not mount.get('ReadOnly', False) and source.startswith('/etc'):
                    issues.append({
                        'severity': 'MEDIUM',
                        'type': 'WRITABLE_SYSTEM_MOUNT',
                        'container': container_name,
                        'description': f'System directory mounted as writable: {source}',
                        'recommendation': 'Mount system directories as read-only'
                    })
        
        # Check 7: Port bindings
        port_bindings = host_config.get('PortBindings') or {}
        for port, bindings in port_bindings.items():
            for binding in bindings:
                host_ip = binding.get('HostIp', '')
                if host_ip == '0.0.0.0' or host_ip == '':
                    issues.append({
                        'severity': 'MEDIUM',
                        'type': 'EXPOSED_PORT',
                        'container': container_name,
                        'description': f'Port {port} exposed on all interfaces',
                        'recommendation': 'Bind ports to localhost (127.0.0.1) if not needed externally'
                    })
        
        # Check 8: Security options
        security_opt = host_config.get('SecurityOpt') or []
        has_apparmor = any('apparmor' in opt for opt in security_opt)
        has_seccomp = any('seccomp' in opt for opt in security_opt)
        
        if not has_apparmor and not has_seccomp:
            issues.append({
                'severity': 'MEDIUM',
                'type': 'NO_SECURITY_PROFILE',
                'container': container_name,
                'description': 'No security profile (AppArmor/SELinux/seccomp) enabled',
                'recommendation': 'Enable security profiles for additional protection'
            })
        
        # Check 9: Environment variables (secrets)
        env_vars = config.get('Env', [])
        for env_var in env_vars:
            if any(keyword in env_var.upper() for keyword in ['PASSWORD', 'SECRET', 'KEY', 'TOKEN']):
                if '=' in env_var and len(env_var.split('=', 1)[1]) > 0:
                    issues.append({
                        'severity': 'HIGH',
                        'type': 'SECRETS_IN_ENV',
                        'container': container_name,
                        'description': 'Potential secrets in environment variables',
                        'recommendation': 'Use Docker secrets or external secret management'
                    })
        
        return issues
    
    def audit_docker_compose_files(self):
        """Audit Docker Compose files for security misconfigurations"""
        compose_files = [
            'docker-compose.yml',
            'docker-compose.auth.yml',
            'docker-compose.security.yml'
        ]
        
        compose_issues = []
        
        for compose_file in compose_files:
            if os.path.exists(compose_file):
                issues = self.audit_compose_file(compose_file)
                compose_issues.extend(issues)
        
        return compose_issues
    
    def audit_compose_file(self, filename):
        """Audit a single Docker Compose file"""
        issues = []
        try:
            with open(filename, 'r') as f:
                compose_data = yaml.safe_load(f)
            
            services = compose_data.get('services', {})
            
            for service_name, service_config in services.items():
                # Check for privileged mode
                if service_config.get('privileged', False):
                    issues.append({
                        'severity': 'CRITICAL',
                        'type': 'COMPOSE_PRIVILEGED',
                        'file': filename,
                        'service': service_name,
                        'description': f'Service {service_name} configured with privileged mode',
                        'recommendation': 'Remove privileged flag from compose file'
                    })
                
                # Check for host network
                if service_config.get('network_mode') == 'host':
                    issues.append({
                        'severity': 'HIGH',
                        'type': 'COMPOSE_HOST_NETWORK',
                        'file': filename,
                        'service': service_name,
                        'description': f'Service {service_name} using host network mode',
                        'recommendation': 'Use bridge or custom networks instead'
                    })
                
                # Check for exposed secrets in environment
                environment = service_config.get('environment', {})
                if isinstance(environment, dict):
                    for key, value in environment.items():
                        if any(keyword in key.upper() for keyword in ['PASSWORD', 'SECRET', 'KEY', 'TOKEN']):
                            if value and not str(value).startswith('${'):
                                issues.append({
                                    'severity': 'HIGH',
                                    'type': 'COMPOSE_HARDCODED_SECRET',
                                    'file': filename,
                                    'service': service_name,
                                    'description': f'Hardcoded secret in {key}',
                                    'recommendation': 'Use environment variables or Docker secrets'
                                })
                
                # Check volume mounts
                volumes = service_config.get('volumes', [])
                for volume in volumes:
                    if isinstance(volume, str) and ':' in volume:
                        host_path = volume.split(':')[0]
                        if host_path in ['/etc', '/proc', '/sys', '/']:
                            issues.append({
                                'severity': 'HIGH',
                                'type': 'COMPOSE_SENSITIVE_MOUNT',
                                'file': filename,
                                'service': service_name,
                                'description': f'Sensitive directory mounted: {host_path}',
                                'recommendation': 'Avoid mounting sensitive system directories'
                            })
        
        except Exception as e:
            print(f"Error auditing {filename}: {e}")
        
        return issues
    
    def audit_dockerfile_security(self):
        """Audit Dockerfiles for security best practices"""
        dockerfile_issues = []
        
        # Find all Dockerfiles
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file == 'Dockerfile' or file.startswith('Dockerfile.'):
                    dockerfile_path = os.path.join(root, file)
                    issues = self.audit_single_dockerfile(dockerfile_path)
                    dockerfile_issues.extend(issues)
        
        return dockerfile_issues
    
    def audit_single_dockerfile(self, dockerfile_path):
        """Audit a single Dockerfile"""
        issues = []
        try:
            with open(dockerfile_path, 'r') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            has_user_instruction = False
            has_update_without_clean = False
            
            for i, line in enumerate(lines, 1):
                line = line.strip()
                
                # Check for USER instruction
                if line.upper().startswith('USER '):
                    has_user_instruction = True
                    user = line.split()[1]
                    if user == 'root' or user == '0':
                        issues.append({
                            'severity': 'HIGH',
                            'type': 'DOCKERFILE_ROOT_USER',
                            'file': dockerfile_path,
                            'line': i,
                            'description': 'Dockerfile explicitly sets USER to root',
                            'recommendation': 'Use non-root user'
                        })
                
                # Check for package update without cleanup
                if 'apt-get update' in line.lower() or 'yum update' in line.lower():
                    has_update_without_clean = True
                
                if ('apt-get clean' in line.lower() or 'yum clean' in line.lower() or 
                    'rm -rf /var/lib/apt/lists/*' in line.lower()):
                    has_update_without_clean = False
                
                # Check for ADD vs COPY
                if line.upper().startswith('ADD ') and not line.lower().endswith('.tar.gz'):
                    issues.append({
                        'severity': 'MEDIUM',
                        'type': 'DOCKERFILE_ADD_INSTEAD_COPY',
                        'file': dockerfile_path,
                        'line': i,
                        'description': 'Using ADD instead of COPY',
                        'recommendation': 'Use COPY instead of ADD for files'
                    })
                
                # Check for latest tag
                if 'FROM' in line.upper() and ':latest' in line:
                    issues.append({
                        'severity': 'MEDIUM',
                        'type': 'DOCKERFILE_LATEST_TAG',
                        'file': dockerfile_path,
                        'line': i,
                        'description': 'Using :latest tag',
                        'recommendation': 'Pin to specific version tags'
                    })
            
            # Final checks
            if not has_user_instruction:
                issues.append({
                    'severity': 'HIGH',
                    'type': 'DOCKERFILE_NO_USER',
                    'file': dockerfile_path,
                    'description': 'No USER instruction found',
                    'recommendation': 'Add USER instruction to run as non-root'
                })
            
            if has_update_without_clean:
                issues.append({
                    'severity': 'MEDIUM',
                    'type': 'DOCKERFILE_NO_CLEANUP',
                    'file': dockerfile_path,
                    'description': 'Package update without cleanup',
                    'recommendation': 'Clean package cache after updates'
                })
        
        except Exception as e:
            print(f"Error auditing {dockerfile_path}: {e}")
        
        return issues
    
    def calculate_compliance_score(self, all_issues):
        """Calculate security compliance score"""
        total_issues = len(all_issues)
        if total_issues == 0:
            return 10.0
        
        severity_weights = {
            'CRITICAL': 3.0,
            'HIGH': 2.0,
            'MEDIUM': 1.0,
            'LOW': 0.5
        }
        
        total_weight = sum(severity_weights.get(issue['severity'], 1.0) for issue in all_issues)
        max_possible_weight = total_issues * 3.0  # If all were critical
        
        score = max(0.0, 10.0 - (total_weight / max_possible_weight * 10.0))
        return round(score, 1)
    
    def generate_security_recommendations(self, all_issues):
        """Generate prioritized security recommendations"""
        recommendations = []
        
        # Group issues by type
        issue_counts = {}
        for issue in all_issues:
            issue_type = issue['type']
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
        
        # Generate recommendations based on most common issues
        if issue_counts.get('ROOT_USER', 0) > 0:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'User Security',
                'title': 'Implement Non-Root Users',
                'description': f"{issue_counts['ROOT_USER']} containers running as root",
                'action': 'Create non-root users in all Dockerfiles and compose files'
            })
        
        if issue_counts.get('NO_MEMORY_LIMIT', 0) > 0:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Resource Limits',
                'title': 'Set Memory Limits',
                'description': f"{issue_counts['NO_MEMORY_LIMIT']} containers without memory limits",
                'action': 'Configure memory limits for all containers'
            })
        
        if issue_counts.get('EXPOSED_PORT', 0) > 0:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Network Security',
                'title': 'Restrict Port Bindings',
                'description': f"{issue_counts['EXPOSED_PORT']} ports exposed on all interfaces",
                'action': 'Bind ports to localhost (127.0.0.1) when possible'
            })
        
        return recommendations
    
    def run_comprehensive_audit(self):
        """Run complete container security audit"""
        print("=" * 60)
        print("Container Security Audit")
        print("=" * 60)
        
        all_issues = []
        
        # Audit running containers
        print("[*] Auditing running containers...")
        containers = self.get_running_containers()
        self.results['containers_audited'] = len(containers)
        
        for container in containers:
            inspect_data = self.inspect_container(container['ID'])
            if inspect_data:
                issues = self.audit_container_security(container, inspect_data)
                all_issues.extend(issues)
        
        print(f"[+] Audited {len(containers)} running containers")
        
        # Audit Docker Compose files
        print("[*] Auditing Docker Compose files...")
        compose_issues = self.audit_docker_compose_files()
        all_issues.extend(compose_issues)
        print(f"[+] Found {len(compose_issues)} compose file issues")
        
        # Audit Dockerfiles
        print("[*] Auditing Dockerfiles...")
        dockerfile_issues = self.audit_dockerfile_security()
        all_issues.extend(dockerfile_issues)
        print(f"[+] Found {len(dockerfile_issues)} Dockerfile issues")
        
        # Calculate compliance score
        compliance_score = self.calculate_compliance_score(all_issues)
        self.results['compliance_score'] = compliance_score
        self.results['security_issues'] = all_issues
        
        # Generate recommendations
        recommendations = self.generate_security_recommendations(all_issues)
        self.results['recommendations'] = recommendations
        
        print(f"\n[*] Container security audit complete!")
        print(f"[*] Compliance Score: {compliance_score}/10.0")
        print(f"[*] Total issues found: {len(all_issues)}")
        
        return self.results
    
    def save_results(self, filename='container_security_audit.json'):
        """Save audit results to file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"[*] Results saved to {filename}")

def main():
    auditor = ContainerSecurityAuditor()
    results = auditor.run_comprehensive_audit()
    auditor.save_results('/opt/sutazaiapp/container_security_audit.json')
    
    # Print summary
    print("\n" + "=" * 60)
    print("CONTAINER SECURITY AUDIT SUMMARY")
    print("=" * 60)
    print(f"Containers Audited: {results['containers_audited']}")
    print(f"Compliance Score: {results['compliance_score']}/10.0")
    print(f"Total Issues: {len(results['security_issues'])}")
    
    # Group by severity
    severity_counts = {}
    for issue in results['security_issues']:
        sev = issue['severity']
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
    
    for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
        count = severity_counts.get(severity, 0)
        print(f"  - {severity}: {count}")
    
    if results['recommendations']:
        print(f"\nTop Recommendations:")
        for i, rec in enumerate(results['recommendations'][:3], 1):
            print(f"{i}. [{rec['priority']}] {rec['title']}")
            print(f"   {rec['description']}")

if __name__ == "__main__":
    main()