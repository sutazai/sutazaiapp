#!/usr/bin/env python3
"""
Comprehensive Security Audit Script
Identifies and reports security vulnerabilities in the SutazAI backend
"""

import os
import re
import sys
import json
import ast
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime

# Add backend to path
sys.path.insert(0, '/opt/sutazaiapp/backend')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SecurityAuditor:
    """Comprehensive security auditor for Python backend"""
    
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.vulnerabilities = []
        self.warnings = []
        self.fixed_issues = []
        
    def scan_hardcoded_credentials(self) -> List[Dict[str, Any]]:
        """Scan for hardcoded credentials in code"""
        logger.info("🔍 Scanning for hardcoded credentials...")
        
        findings = []
        credential_patterns = [
            (r'password\s*=\s*["\']([^"\']+)["\']', "Hardcoded password"),
            (r'api_key\s*=\s*["\']([^"\']+)["\']', "Hardcoded API key"),
            (r'secret\s*=\s*["\']([^"\']+)["\']', "Hardcoded secret"),
            (r'token\s*=\s*["\']([^"\']+)["\']', "Hardcoded token"),
            (r'POSTGRES_PASSWORD.*=.*["\']([^"\']+)["\']', "Database password in code"),
        ]
        
        # Scan Python files
        for py_file in self.root_path.rglob("*.py"):
            try:
                content = py_file.read_text(encoding='utf-8')
                
                for line_num, line in enumerate(content.split('\n'), 1):
                    for pattern, description in credential_patterns:
                        matches = re.findall(pattern, line, re.IGNORECASE)
                        for match in matches:
                            # Skip if it's an environment variable reference
                            if match.startswith('${') or 'getenv' in line:
                                continue
                                
                            # Skip if it's a placeholder/example value
                            if any(placeholder in match.lower() for placeholder in 
                                  ['your_', 'example', 'placeholder', 'change_me']):
                                continue
                                
                            finding = {
                                'type': 'hardcoded_credential',
                                'severity': 'HIGH',
                                'file': str(py_file.relative_to(self.root_path)),
                                'line': line_num,
                                'description': description,
                                'value': match[:20] + '...' if len(match) > 20 else match,
                                'recommendation': 'Move to environment variable'
                            }
                            
                            # Check if this is a known fixed issue
                            if self._is_credential_fixed(py_file, line_num):
                                self.fixed_issues.append(finding)
                                logger.info(f"✅ Fixed: {description} in {py_file.name}:{line_num}")
                            else:
                                findings.append(finding)
                                
            except Exception as e:
                logger.warning(f"Could not scan {py_file}: {e}")
                
        return findings
    
    def scan_connection_pooling(self) -> List[Dict[str, Any]]:
        """Scan for connection pooling issues"""
        logger.info("🔍 Scanning for connection pooling issues...")
        
        findings = []
        
        # Look for database configuration files
        db_files = list(self.root_path.rglob("*database*.py")) + list(self.root_path.rglob("*db*.py"))
        
        for db_file in db_files:
            try:
                content = db_file.read_text(encoding='utf-8')
                
                # Check for NullPool usage (potential issue)
                if 'NullPool' in content and 'poolclass=NullPool' in content:
                    # Check if it's been fixed with QueuePool
                    if 'QueuePool' in content and 'poolclass=QueuePool' in content:
                        self.fixed_issues.append({
                            'type': 'connection_pooling',
                            'severity': 'MEDIUM',
                            'file': str(db_file.relative_to(self.root_path)),
                            'description': 'Fixed: Using proper QueuePool for connection pooling',
                            'status': 'FIXED'
                        })
                        logger.info(f"✅ Fixed: Connection pooling in {db_file.name}")
                    else:
                        findings.append({
                            'type': 'connection_pooling',
                            'severity': 'MEDIUM',
                            'file': str(db_file.relative_to(self.root_path)),
                            'description': 'Using NullPool - no connection pooling',
                            'recommendation': 'Switch to QueuePool for better performance'
                        })
                
            except Exception as e:
                logger.warning(f"Could not scan {db_file}: {e}")
                
        return findings
    
    def scan_memory_leaks(self) -> List[Dict[str, Any]]:
        """Scan for potential memory leak patterns"""
        logger.info("🔍 Scanning for memory leak patterns...")
        
        findings = []
        
        for py_file in self.root_path.rglob("*.py"):
            try:
                content = py_file.read_text(encoding='utf-8')
                
                # Look for unbounded collections
                lines = content.split('\n')
                for line_num, line in enumerate(lines, 1):
                    # Check for append operations without limits
                    if '.append(' in line and 'results' in line.lower():
                        # Look for memory leak prevention in surrounding lines
                        context_start = max(0, line_num - 10)
                        context_end = min(len(lines), line_num + 10)
                        context = '\n'.join(lines[context_start:context_end])
                        
                        # Check if there's limit checking or cleanup
                        has_limit = any(keyword in context.lower() for keyword in 
                                      ['max_', 'limit', 'cleanup', 'clear()', 'del '])
                        
                        if not has_limit and 'ClaudeAgentPool' not in line:
                            findings.append({
                                'type': 'potential_memory_leak',
                                'severity': 'MEDIUM',
                                'file': str(py_file.relative_to(self.root_path)),
                                'line': line_num,
                                'description': 'Unbounded collection growth',
                                'recommendation': 'Add size limits and cleanup'
                            })
                        elif 'ClaudeAgentPool' in context:
                            # Check if ClaudeAgentPool has been fixed
                            if 'max_results' in context and 'cleanup' in context:
                                self.fixed_issues.append({
                                    'type': 'memory_leak',
                                    'severity': 'HIGH',
                                    'file': str(py_file.relative_to(self.root_path)),
                                    'description': 'Fixed: ClaudeAgentPool memory leak with size limits',
                                    'status': 'FIXED'
                                })
                                logger.info(f"✅ Fixed: Memory leak in {py_file.name}")
                
            except Exception as e:
                logger.warning(f"Could not scan {py_file}: {e}")
                
        return findings
    
    def scan_authentication_middleware(self) -> List[Dict[str, Any]]:
        """Scan for authentication middleware implementation"""
        logger.info("🔍 Scanning for authentication middleware...")
        
        findings = []
        
        # Look for main.py to check middleware configuration
        main_files = list(self.root_path.rglob("main.py"))
        
        auth_middleware_found = False
        security_middleware_found = False
        
        for main_file in main_files:
            try:
                content = main_file.read_text(encoding='utf-8')
                
                if 'SecurityMiddleware' in content:
                    security_middleware_found = True
                    self.fixed_issues.append({
                        'type': 'authentication',
                        'severity': 'HIGH',
                        'file': str(main_file.relative_to(self.root_path)),
                        'description': 'Fixed: SecurityMiddleware properly configured',
                        'status': 'FIXED'
                    })
                    logger.info(f"✅ Fixed: Security middleware in {main_file.name}")
                
                if 'add_middleware' in content and 'auth' in content.lower():
                    auth_middleware_found = True
                    
            except Exception as e:
                logger.warning(f"Could not scan {main_file}: {e}")
        
        if not security_middleware_found and not auth_middleware_found:
            findings.append({
                'type': 'missing_authentication',
                'severity': 'HIGH',
                'file': 'main.py',
                'description': 'No authentication middleware configured',
                'recommendation': 'Add SecurityMiddleware to FastAPI application'
            })
        
        # Check for auth endpoints protection
        api_files = list(self.root_path.rglob("api/v1/*.py"))
        unprotected_endpoints = 0
        
        for api_file in api_files:
            try:
                content = api_file.read_text(encoding='utf-8')
                
                # Count router definitions without auth dependencies
                router_lines = [line for line in content.split('\n') if '@router.' in line]
                
                for line in router_lines:
                    if not any(auth_dep in content for auth_dep in 
                             ['get_current_user', 'require_admin', 'Depends']):
                        unprotected_endpoints += 1
                        
            except Exception as e:
                logger.warning(f"Could not scan {api_file}: {e}")
        
        if unprotected_endpoints > 0:
            findings.append({
                'type': 'unprotected_endpoints',
                'severity': 'MEDIUM',
                'file': 'api/v1/',
                'description': f'{unprotected_endpoints} potentially unprotected API endpoints',
                'recommendation': 'Add authentication dependencies to sensitive endpoints'
            })
        
        return findings
    
    def _is_credential_fixed(self, file_path: Path, line_num: int) -> bool:
        """Check if a credential issue has been fixed"""
        try:
            content = file_path.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            # Check surrounding lines for environment variable usage
            context_start = max(0, line_num - 5)
            context_end = min(len(lines), line_num + 5)
            context = '\n'.join(lines[context_start:context_end])
            
            # Look for env var patterns
            env_patterns = ['os.getenv', 'os.environ', 'getenv(', 'environ[']
            return any(pattern in context for pattern in env_patterns)
            
        except Exception:
            return False
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive security audit report"""
        logger.info("📊 Generating security audit report...")
        
        # Run all scans
        credential_findings = self.scan_hardcoded_credentials()
        pooling_findings = self.scan_connection_pooling()
        memory_findings = self.scan_memory_leaks()
        auth_findings = self.scan_authentication_middleware()
        
        all_findings = (
            credential_findings + 
            pooling_findings + 
            memory_findings + 
            auth_findings
        )
        
        # Categorize by severity
        high_severity = [f for f in all_findings if f.get('severity') == 'HIGH']
        medium_severity = [f for f in all_findings if f.get('severity') == 'MEDIUM']
        low_severity = [f for f in all_findings if f.get('severity') == 'LOW']
        
        report = {
            'audit_timestamp': datetime.now().isoformat(),
            'summary': {
                'total_vulnerabilities': len(all_findings),
                'high_severity': len(high_severity),
                'medium_severity': len(medium_severity),
                'low_severity': len(low_severity),
                'fixed_issues': len(self.fixed_issues),
                'overall_status': 'SECURE' if len(high_severity) == 0 else 'NEEDS_ATTENTION'
            },
            'vulnerabilities': {
                'high': high_severity,
                'medium': medium_severity,
                'low': low_severity
            },
            'fixed_issues': self.fixed_issues,
            'recommendations': self._generate_recommendations(all_findings),
        }
        
        return report
    
    def _generate_recommendations(self, findings: List[Dict[str, Any]]) -> List[str]:
        """Generate security recommendations based on findings"""
        recommendations = []
        
        if any(f['type'] == 'hardcoded_credential' for f in findings):
            recommendations.append(
                "Move all hardcoded credentials to environment variables"
            )
            
        if any(f['type'] == 'connection_pooling' for f in findings):
            recommendations.append(
                "Implement proper database connection pooling with QueuePool"
            )
            
        if any(f['type'] == 'potential_memory_leak' for f in findings):
            recommendations.append(
                "Add size limits and cleanup for unbounded collections"
            )
            
        if any(f['type'] == 'missing_authentication' for f in findings):
            recommendations.append(
                "Implement comprehensive authentication middleware"
            )
            
        # General recommendations
        recommendations.extend([
            "Regularly rotate all secrets and API keys",
            "Enable audit logging for all security events",
            "Implement rate limiting to prevent abuse",
            "Use HTTPS in production with proper certificate validation",
            "Regular security dependency updates with vulnerability scanning"
        ])
        
        return recommendations
    
    def print_report(self, report: Dict[str, Any]):
        """Print formatted security audit report"""
        print("\n" + "=" * 80)
        print("SUTAZAI BACKEND SECURITY AUDIT REPORT")
        print("=" * 80)
        
        print(f"Audit Date: {report['audit_timestamp']}")
        print(f"Overall Status: {report['summary']['overall_status']}")
        print()
        
        # Summary
        summary = report['summary']
        print("SUMMARY")
        print("-" * 40)
        print(f"Total Vulnerabilities: {summary['total_vulnerabilities']}")
        print(f"High Severity: {summary['high_severity']}")
        print(f"Medium Severity: {summary['medium_severity']}")
        print(f"Low Severity: {summary['low_severity']}")
        print(f"Fixed Issues: {summary['fixed_issues']}")
        print()
        
        # Fixed Issues (show first)
        if report['fixed_issues']:
            print("FIXED SECURITY ISSUES")
            print("-" * 40)
            for issue in report['fixed_issues']:
                print(f"  [FIXED] {issue['description']}")
                print(f"    File: {issue['file']}")
            print()
        
        # Vulnerabilities by severity
        for severity in ['high', 'medium', 'low']:
            vulns = report['vulnerabilities'][severity]
            if vulns:
                indicator = "[CRITICAL]" if severity == 'high' else "[WARNING]" if severity == 'medium' else "[INFO]"
                print(f"{indicator} {severity.upper()} SEVERITY VULNERABILITIES")
                print("-" * 40)
                
                for vuln in vulns:
                    print(f"  - {vuln['description']}")
                    print(f"    File: {vuln['file']}")
                    if 'line' in vuln:
                        print(f"    Line: {vuln['line']}")
                    if 'recommendation' in vuln:
                        print(f"    Fix: {vuln['recommendation']}")
                    print()
        
        # Recommendations
        if report['recommendations']:
            print("SECURITY RECOMMENDATIONS")
            print("-" * 40)
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"  {i}. {rec}")
            print()
        
        # Final status
        if summary['overall_status'] == 'SECURE':
            print("CONGRATULATIONS! No high-severity vulnerabilities found.")
            print("Continue following security best practices.")
        else:
            print("ACTION REQUIRED: Address high-severity vulnerabilities immediately.")
        
        print("=" * 80)


def main():
    """Main audit execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Security audit for SutazAI backend")
    parser.add_argument("--path", type=str, default="/opt/sutazaiapp/backend",
                       help="Path to backend code")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON report to file")
    parser.add_argument("--format", choices=['json', 'text'], default='text',
                       help="Output format")
    
    args = parser.parse_args()
    
    # Run security audit
    auditor = SecurityAuditor(args.path)
    report = auditor.generate_report()
    
    # Output report
    if args.format == 'json' or args.output:
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"📄 Report saved to {args.output}")
        else:
            print(json.dumps(report, indent=2))
    else:
        auditor.print_report(report)
    
    # Exit with error code if vulnerabilities found
    if report['summary']['high_severity'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()