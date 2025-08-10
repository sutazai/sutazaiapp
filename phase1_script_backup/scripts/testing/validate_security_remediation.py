#!/usr/bin/env python3
"""
SutazAI Security Validation Script
Validates that security remediation has been properly implemented
"""

import re
import subprocess
from pathlib import Path


class SecurityValidator:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.violations = []
        self.warnings = []
        self.fixes_applied = []
    
    def scan_hardcoded_secrets(self) -> List[Dict]:
        """Scan for remaining hardcoded secrets"""
        print("🔍 Scanning for hardcoded secrets...")
        
        # Patterns for hardcoded secrets
        secret_patterns = [
            (r'password\s*=\s*["\']([^"\']{3,})["\']', 'hardcoded password'),
            (r'secret\s*=\s*["\']([^"\']{3,})["\']', 'hardcoded secret'),
            (r'api_key\s*=\s*["\']([^"\']{3,})["\']', 'hardcoded API key'),
            (r'token\s*=\s*["\']([^"\']{3,})["\']', 'hardcoded token'),
            (r'["\'][a-f0-9]{32,}["\']', 'potential hardcoded hex key'),
            (r'["\'][A-Za-z0-9+/]{20,}={0,2}["\']', 'potential base64 secret'),
        ]
        
        violations = []
        
        # Scan Python files
        for py_file in self.project_root.rglob("*.py"):
            if 'test' in py_file.name.lower() or 'example' in py_file.name.lower():
                continue  # Skip test files
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for line_num, line in enumerate(content.split('\n'), 1):
                    for pattern, description in secret_patterns:
                        matches = re.finditer(pattern, line, re.IGNORECASE)
                        for match in matches:
                            # Skip obvious false positives
                            if any(fp in match.group().lower() for fp in ['example', 'test', 'placeholder', 'change_me']):
                                continue
                            
                            violations.append({
                                'file': str(py_file),
                                'line': line_num,
                                'type': description,
                                'content': line.strip(),
                                'severity': 'HIGH'
                            })
            except Exception as e:
                print(f"⚠️  Error scanning {py_file}: {e}")
        
        return violations
    
    def check_container_security(self) -> List[Dict]:
        """Check Dockerfiles for security issues"""
        print("🐳 Checking container security...")
        
        violations = []
        
        for dockerfile in self.project_root.rglob("Dockerfile*"):
            try:
                with open(dockerfile, 'r') as f:
                    content = f.read()
                
                # Check for USER root
                if 'USER root' in content:
                    violations.append({
                        'file': str(dockerfile),
                        'type': 'Container running as root',
                        'severity': 'HIGH',
                        'recommendation': 'Use non-root user'
                    })
                
                # Check if any USER directive exists
                if 'USER ' not in content:
                    violations.append({
                        'file': str(dockerfile),
                        'type': 'No USER directive found',
                        'severity': 'MEDIUM',
                        'recommendation': 'Add USER directive to run as non-root'
                    })
                
            except Exception as e:
                print(f"⚠️  Error scanning {dockerfile}: {e}")
        
        return violations
    
    def check_environment_security(self) -> List[Dict]:
        """Check environment configuration security"""
        print("🔧 Checking environment security...")
        
        violations = []
        
        # Check if .env.secure.template exists
        secure_template = self.project_root / '.env.secure.template'
        if not secure_template.exists():
            violations.append({
                'file': 'Missing .env.secure.template',
                'type': 'No secure environment template',
                'severity': 'HIGH',
                'recommendation': 'Create secure environment template'
            })
        
        # Check docker-compose for default passwords
        compose_file = self.project_root / 'docker-compose.yml'
        if compose_file.exists():
            with open(compose_file, 'r') as f:
                content = f.read()
            
            # Check for hardcoded defaults
            if 'sutazai_grafana' in content:
                violations.append({
                    'file': str(compose_file),
                    'type': 'Hardcoded Grafana password',
                    'severity': 'HIGH',
                    'recommendation': 'Use environment variable'
                })
            
            if 'sutazai_rabbit' in content:
                violations.append({
                    'file': str(compose_file),
                    'type': 'Hardcoded RabbitMQ password',
                    'severity': 'HIGH',
                    'recommendation': 'Use environment variable'
                })
        
        return violations
    
    def check_authentication_security(self) -> List[Dict]:
        """Check authentication and authorization security"""
        print("🔐 Checking authentication security...")
        
        violations = []
        
        # Check JWT implementation
        auth_files = list(self.project_root.rglob("*auth*.py")) + list(self.project_root.rglob("*jwt*.py"))
        
        for auth_file in auth_files:
            try:
                with open(auth_file, 'r') as f:
                    content = f.read()
                
                # Check for hardcoded JWT secrets
                if 'jwt_secret' in content.lower() and ('=' in content):
                    lines = content.split('\n')
                    for line_num, line in enumerate(lines, 1):
                        if 'jwt_secret' in line.lower() and '=' in line:
                            if not 'os.getenv' in line or 'default' in line.lower():
                                violations.append({
                                    'file': str(auth_file),
                                    'line': line_num,
                                    'type': 'Potentially hardcoded JWT secret',
                                    'severity': 'CRITICAL',
                                    'recommendation': 'Use environment variable without default'
                                })
            except Exception as e:
                print(f"⚠️  Error scanning {auth_file}: {e}")
        
        return violations
    
    def generate_security_report(self) -> Dict:
        """Generate comprehensive security report"""
        
        report = {
            'timestamp': subprocess.check_output(['date'], text=True).strip(),
            'hardcoded_secrets': self.scan_hardcoded_secrets(),
            'container_security': self.check_container_security(),
            'environment_security': self.check_environment_security(),
            'authentication_security': self.check_authentication_security(),
        }
        
        # Calculate severity counts
        all_violations = (report['hardcoded_secrets'] + 
                         report['container_security'] + 
                         report['environment_security'] + 
                         report['authentication_security'])
        
        severity_counts = {
            'CRITICAL': len([v for v in all_violations if v.get('severity') == 'CRITICAL']),
            'HIGH': len([v for v in all_violations if v.get('severity') == 'HIGH']),
            'MEDIUM': len([v for v in all_violations if v.get('severity') == 'MEDIUM']),
            'LOW': len([v for v in all_violations if v.get('severity') == 'LOW'])
        }
        
        report['summary'] = {
            'total_violations': len(all_violations),
            'severity_counts': severity_counts,
            'status': 'FAIL' if severity_counts['CRITICAL'] + severity_counts['HIGH'] > 0 else 'PASS'
        }
        
        return report
    
    def print_report(self, report: Dict):
        """Print security report to console"""
        
        print("\n" + "="*60)
        print("🛡️  SUTAZAI SECURITY VALIDATION REPORT")
        print("="*60)
        
        summary = report['summary']
        print(f"📊 Total Violations: {summary['total_violations']}")
        print(f"🔴 Critical: {summary['severity_counts']['CRITICAL']}")
        print(f"🟠 High: {summary['severity_counts']['HIGH']}")
        print(f"🟡 Medium: {summary['severity_counts']['MEDIUM']}")
        print(f"🟢 Low: {summary['severity_counts']['LOW']}")
        print(f"✅ Overall Status: {summary['status']}")
        
        # Print violations by category
        categories = [
            ('hardcoded_secrets', '🔐 Hardcoded Secrets'),
            ('container_security', '🐳 Container Security'),
            ('environment_security', '🔧 Environment Security'),
            ('authentication_security', '🔑 Authentication Security')
        ]
        
        for category, title in categories:
            violations = report[category]
            if violations:
                print(f"\n{title} ({len(violations)} issues):")
                for violation in violations:
                    severity_icon = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'}.get(
                        violation.get('severity', 'LOW'), '⚪'
                    )
                    print(f"  {severity_icon} {violation.get('type', 'Unknown')}")
                    print(f"     📁 {violation.get('file', 'Unknown file')}")
                    if 'line' in violation:
                        print(f"     📍 Line {violation['line']}")
                    if 'recommendation' in violation:
                        print(f"     💡 {violation['recommendation']}")
                    print()
        
        print("="*60)
        if summary['status'] == 'PASS':
            print("🎉 Security validation PASSED!")
        else:
            print("❌ Security validation FAILED - Fix critical and high severity issues")
        print("="*60)


def main():
    """Main validation function"""
    
    project_root = Path(__file__).parent.parent
    validator = SecurityValidator(project_root)
    
    print("🛡️  Starting SutazAI Security Validation...")
    
    # Generate report
    report = validator.generate_security_report()
    
    # Print to console
    validator.print_report(report)
    
    # Save report to file
    report_file = project_root / 'security_validation_report.json'
    import json
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📋 Full report saved to: {report_file}")
    
    # Exit with appropriate code
    exit_code = 0 if report['summary']['status'] == 'PASS' else 1
    exit(exit_code)


if __name__ == '__main__':
    main()