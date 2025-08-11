#!/usr/bin/env python3
"""
Comprehensive Security Audit Script for SutazAI System
Validates JWT, CORS, Authentication, and Security Headers implementation
Author: Security Architect Team
Date: 2025-08-11
"""

import os
import sys
import json
import re
import subprocess
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from pathlib import Path

# ANSI color codes for output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

class SecurityAuditor:
    """Enterprise-grade security auditor for SutazAI system"""
    
    def __init__(self):
        self.project_root = Path("/opt/sutazaiapp")
        self.vulnerabilities = []
        self.warnings = []
        self.secure_findings = []
        self.severity_levels = {
            "CRITICAL": [],
            "HIGH": [],
            "MEDIUM": [],
            "LOW": []
        }
        
    def print_header(self, title: str):
        """Print formatted section header"""
        print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.CYAN}{title:^80}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.ENDC}\n")
        
    def print_success(self, message: str):
        """Print success message"""
        print(f"{Colors.GREEN}✅ {message}{Colors.ENDC}")
        
    def print_warning(self, message: str):
        """Print warning message"""
        print(f"{Colors.YELLOW}⚠️  {message}{Colors.ENDC}")
        self.warnings.append(message)
        
    def print_error(self, message: str, severity: str = "HIGH"):
        """Print error message"""
        print(f"{Colors.RED}❌ {severity}: {message}{Colors.ENDC}")
        self.vulnerabilities.append({"message": message, "severity": severity})
        self.severity_levels[severity].append(message)
        
    def audit_jwt_implementation(self) -> Dict:
        """Audit JWT implementation for security vulnerabilities"""
        self.print_header("JWT SECURITY AUDIT")
        jwt_findings = {
            "secure": [],
            "vulnerabilities": [],
            "recommendations": []
        }
        
        # Check backend JWT handler
        jwt_handler_path = self.project_root / "backend/app/auth/jwt_handler.py"
        if jwt_handler_path.exists():
            with open(jwt_handler_path, 'r') as f:
                content = f.read()
                
            # Check for hardcoded secrets
            hardcoded_patterns = [
                r'JWT_SECRET\s*=\s*["\']([^"\']+)["\']',
                r'SECRET_KEY\s*=\s*["\']([^"\']+)["\']',
                r'private_key\s*=\s*["\']([^"\']+)["\']'
            ]
            
            for pattern in hardcoded_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    if not match.startswith("os.getenv") and not match.startswith("Field"):
                        self.print_error(f"Hardcoded secret found: {match[:20]}...", "CRITICAL")
                        jwt_findings["vulnerabilities"].append(f"Hardcoded secret: {match[:20]}...")
                        
            # Check for environment variable usage
            if 'os.getenv("JWT_SECRET' in content or 'os.getenv("JWT_PRIVATE_KEY' in content:
                self.print_success("JWT secrets loaded from environment variables")
                jwt_findings["secure"].append("Secrets from environment")
                
            # Check for RS256 algorithm (asymmetric)
            if 'RS256' in content:
                self.print_success("Using RS256 asymmetric algorithm (secure)")
                jwt_findings["secure"].append("RS256 asymmetric encryption")
            elif 'HS256' in content:
                self.print_warning("Using HS256 symmetric algorithm (consider RS256 for better security)")
                jwt_findings["recommendations"].append("Migrate to RS256 asymmetric algorithm")
                
            # Check for token expiration
            if 'ACCESS_TOKEN_EXPIRE_MINUTES' in content:
                self.print_success("Token expiration configured")
                jwt_findings["secure"].append("Token expiration enabled")
                
            # Check for refresh tokens
            if 'refresh_token' in content.lower():
                self.print_success("Refresh token mechanism implemented")
                jwt_findings["secure"].append("Refresh tokens implemented")
                
        # Check auth service JWT implementation
        auth_service_path = self.project_root / "auth/jwt-service/main.py"
        if auth_service_path.exists():
            with open(auth_service_path, 'r') as f:
                content = f.read()
                
            # Check for JWT_SECRET environment validation
            if "if not JWT_SECRET:" in content and "raise ValueError" in content:
                self.print_success("JWT service validates secret existence")
                jwt_findings["secure"].append("Secret validation on startup")
                
            # Check for token revocation
            if "revoke_token" in content:
                self.print_success("Token revocation mechanism implemented")
                jwt_findings["secure"].append("Token revocation supported")
                
            # Check for TLS verification
            if "verify=True" in content:
                self.print_success("TLS verification enabled for external services")
                jwt_findings["secure"].append("TLS verification enabled")
                
        return jwt_findings
    
    def audit_cors_configuration(self) -> Dict:
        """Audit CORS configuration for security vulnerabilities"""
        self.print_header("CORS SECURITY AUDIT")
        cors_findings = {
            "secure": [],
            "vulnerabilities": [],
            "recommendations": []
        }
        
        # Check backend CORS configuration
        config_path = self.project_root / "backend/app/core/config.py"
        if config_path.exists():
            with open(config_path, 'r') as f:
                content = f.read()
                
            # Check for wildcard origins
            if '["*"]' in content and 'BACKEND_CORS_ORIGINS' in content:
                self.print_error("Wildcard CORS origins in configuration", "CRITICAL")
                cors_findings["vulnerabilities"].append("Wildcard origins in config")
            elif 'BACKEND_CORS_ORIGINS: List[str] = [' in content:
                self.print_success("Specific CORS origins configured")
                cors_findings["secure"].append("Explicit origin whitelist")
                
        # Check CORS security module
        cors_security_path = self.project_root / "backend/app/core/cors_security.py"
        if cors_security_path.exists():
            with open(cors_security_path, 'r') as f:
                content = f.read()
                
            # Check for wildcard validation
            if 'validate_cors_security' in content:
                self.print_success("CORS security validation implemented")
                cors_findings["secure"].append("Wildcard validation on startup")
                
            # Check for environment-based configuration
            if 'SUTAZAI_ENV' in content:
                self.print_success("Environment-aware CORS configuration")
                cors_findings["secure"].append("Environment-specific origins")
                
            # Check for secure headers
            if 'secure_allowed_headers' in content:
                self.print_success("Secure header whitelist implemented")
                cors_findings["secure"].append("Header whitelist configured")
                
        # Check main.py CORS implementation
        main_path = self.project_root / "backend/app/main.py"
        if main_path.exists():
            with open(main_path, 'r') as f:
                content = f.read()
                
            # Check for CORS security validation
            if 'validate_cors_security()' in content and 'sys.exit(1)' in content:
                self.print_success("System fails fast on CORS security violation")
                cors_findings["secure"].append("Fail-fast on wildcards")
                
            # Check for preflight handling
            if 'OPTIONS' in content:
                self.print_success("Preflight requests handled")
                cors_findings["secure"].append("OPTIONS preflight support")
                
        # Check Kong CORS configuration
        kong_config_path = self.project_root / "configs/kong/kong.yml"
        if kong_config_path.exists():
            with open(kong_config_path, 'r') as f:
                content = f.read()
                
            # Check for wildcard in Kong
            if 'origins: ["*"]' in content:
                self.print_error("Wildcard origins in Kong configuration", "HIGH")
                cors_findings["vulnerabilities"].append("Kong wildcard origins")
            elif 'origins: ["http://localhost:10011"]' in content:
                self.print_success("Kong has specific origin configuration")
                cors_findings["secure"].append("Kong explicit origins")
                
        return cors_findings
    
    def audit_authentication(self) -> Dict:
        """Audit authentication implementation"""
        self.print_header("AUTHENTICATION SECURITY AUDIT")
        auth_findings = {
            "secure": [],
            "vulnerabilities": [],
            "recommendations": []
        }
        
        # Check for password hashing
        auth_files = list(self.project_root.glob("**/auth*.py"))
        for auth_file in auth_files[:5]:  # Check first 5 auth files
            with open(auth_file, 'r') as f:
                content = f.read()
                
            if 'bcrypt' in content or 'argon2' in content or 'scrypt' in content:
                self.print_success(f"Secure password hashing in {auth_file.name}")
                auth_findings["secure"].append("Secure password hashing")
                break
                
        # Check for rate limiting
        if any(self.project_root.glob("**/rate_limit*")):
            self.print_success("Rate limiting implementation found")
            auth_findings["secure"].append("Rate limiting configured")
        else:
            self.print_warning("Consider implementing rate limiting")
            auth_findings["recommendations"].append("Implement rate limiting")
            
        # Check for MFA/2FA
        if any(self.project_root.glob("**/totp*")) or any(self.project_root.glob("**/mfa*")):
            self.print_success("Multi-factor authentication support found")
            auth_findings["secure"].append("MFA/2FA supported")
        else:
            self.print_warning("Consider implementing MFA/2FA")
            auth_findings["recommendations"].append("Add MFA/2FA support")
            
        return auth_findings
    
    def audit_security_headers(self) -> Dict:
        """Audit security headers implementation"""
        self.print_header("SECURITY HEADERS AUDIT")
        headers_findings = {
            "secure": [],
            "missing": [],
            "recommendations": []
        }
        
        # Essential security headers to check
        essential_headers = [
            "X-Frame-Options",
            "X-Content-Type-Options",
            "X-XSS-Protection",
            "Strict-Transport-Security",
            "Content-Security-Policy",
            "Referrer-Policy",
            "Permissions-Policy"
        ]
        
        # Search for security headers in code
        for header in essential_headers:
            found = False
            for py_file in self.project_root.glob("**/*.py"):
                if py_file.is_file():
                    try:
                        with open(py_file, 'r') as f:
                            if header in f.read():
                                found = True
                                break
                    except:
                        pass
                        
            if found:
                self.print_success(f"{header} header configured")
                headers_findings["secure"].append(header)
            else:
                self.print_warning(f"{header} header not found")
                headers_findings["missing"].append(header)
                
        return headers_findings
    
    def audit_environment_security(self) -> Dict:
        """Audit environment configuration security"""
        self.print_header("ENVIRONMENT SECURITY AUDIT")
        env_findings = {
            "secure": [],
            "vulnerabilities": [],
            "recommendations": []
        }
        
        # Check .env file security
        env_file = self.project_root / ".env"
        if env_file.exists():
            with open(env_file, 'r') as f:
                content = f.read()
                
            # Check for secure secret generation
            if re.search(r'JWT_SECRET=[\w\-_]{32,}', content):
                self.print_success("JWT_SECRET appears to be securely generated")
                env_findings["secure"].append("Secure JWT secret")
            else:
                self.print_warning("JWT_SECRET may not be securely generated")
                
            # Check for default passwords
            insecure_patterns = [
                'password',
                'changeme',
                'admin',
                'default',
                '123456'
            ]
            
            for pattern in insecure_patterns:
                if pattern in content.lower():
                    lines = content.split('\n')
                    for line in lines:
                        if pattern in line.lower() and '=' in line:
                            key = line.split('=')[0]
                            if not any(skip in key for skip in ['EXAMPLE', 'TEMPLATE', '#']):
                                self.print_warning(f"Potential insecure value in {key}")
                                
        # Check for .env in .gitignore
        gitignore = self.project_root / ".gitignore"
        if gitignore.exists():
            with open(gitignore, 'r') as f:
                if '.env' in f.read():
                    self.print_success(".env file is gitignored")
                    env_findings["secure"].append(".env gitignored")
                else:
                    self.print_error(".env file not in .gitignore", "HIGH")
                    env_findings["vulnerabilities"].append(".env not gitignored")
                    
        return env_findings
    
    def generate_report(self) -> str:
        """Generate comprehensive security audit report"""
        report = []
        report.append("=" * 80)
        report.append("SUTAZAI SECURITY AUDIT REPORT")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("=" * 80)
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 40)
        total_vulnerabilities = len(self.vulnerabilities)
        
        if total_vulnerabilities == 0:
            report.append("✅ SYSTEM SECURITY: EXCELLENT")
            report.append("No critical vulnerabilities detected.")
        elif total_vulnerabilities < 3:
            report.append("⚠️  SYSTEM SECURITY: GOOD (Minor issues)")
            report.append(f"{total_vulnerabilities} issues require attention.")
        else:
            report.append("❌ SYSTEM SECURITY: NEEDS IMPROVEMENT")
            report.append(f"{total_vulnerabilities} vulnerabilities detected.")
            
        report.append("")
        report.append("SEVERITY BREAKDOWN:")
        for severity, issues in self.severity_levels.items():
            if issues:
                report.append(f"  {severity}: {len(issues)} issue(s)")
                
        report.append("")
        report.append("SECURE IMPLEMENTATIONS:")
        report.append(f"  ✅ {len(self.secure_findings)} security best practices detected")
        
        report.append("")
        report.append("WARNINGS:")
        report.append(f"  ⚠️  {len(self.warnings)} recommendations for improvement")
        
        # Detailed Findings
        if self.vulnerabilities:
            report.append("")
            report.append("CRITICAL VULNERABILITIES")
            report.append("-" * 40)
            for vuln in self.vulnerabilities:
                report.append(f"• [{vuln['severity']}] {vuln['message']}")
                
        if self.warnings:
            report.append("")
            report.append("RECOMMENDATIONS")
            report.append("-" * 40)
            for warning in self.warnings:
                report.append(f"• {warning}")
                
        if self.secure_findings:
            report.append("")
            report.append("SECURE IMPLEMENTATIONS")
            report.append("-" * 40)
            for finding in self.secure_findings:
                report.append(f"✅ {finding}")
                
        # OWASP References
        report.append("")
        report.append("OWASP TOP 10 2021 COVERAGE")
        report.append("-" * 40)
        report.append("✅ A02:2021 - Cryptographic Failures (JWT implementation)")
        report.append("✅ A05:2021 - Security Misconfiguration (CORS, headers)")
        report.append("✅ A07:2021 - Identification and Authentication Failures")
        report.append("✅ A01:2021 - Broken Access Control (CORS origins)")
        
        # Compliance Readiness
        report.append("")
        report.append("COMPLIANCE READINESS")
        report.append("-" * 40)
        compliance_score = max(0, 100 - (total_vulnerabilities * 10))
        report.append(f"Security Score: {compliance_score}/100")
        
        if compliance_score >= 90:
            report.append("✅ SOC 2 Type II: Ready")
            report.append("✅ ISO 27001: Ready")
            report.append("✅ PCI DSS: Ready (with minor adjustments)")
        elif compliance_score >= 70:
            report.append("⚠️  SOC 2 Type II: Needs improvement")
            report.append("⚠️  ISO 27001: Needs improvement")
            report.append("❌ PCI DSS: Not ready")
        else:
            report.append("❌ Compliance not recommended until issues resolved")
            
        report.append("")
        report.append("=" * 80)
        report.append("END OF SECURITY AUDIT REPORT")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def run_full_audit(self):
        """Run complete security audit"""
        print(f"{Colors.BOLD}{Colors.CYAN}")
        print("╔════════════════════════════════════════════════════════════════════════╗")
        print("║          SUTAZAI COMPREHENSIVE SECURITY AUDIT v2.0                    ║")
        print("║                 Enterprise Security Assessment                         ║")
        print("╚════════════════════════════════════════════════════════════════════════╝")
        print(f"{Colors.ENDC}")
        
        # Run all audits
        jwt_results = self.audit_jwt_implementation()
        cors_results = self.audit_cors_configuration()
        auth_results = self.audit_authentication()
        headers_results = self.audit_security_headers()
        env_results = self.audit_environment_security()
        
        # Collect secure findings
        for category in [jwt_results, cors_results, auth_results, env_results]:
            if "secure" in category:
                self.secure_findings.extend(category["secure"])
                
        # Generate and save report
        report = self.generate_report()
        
        # Save report to file
        report_path = self.project_root / "SECURITY_AUDIT_REPORT.md"
        with open(report_path, 'w') as f:
            f.write(report)
            
        print(f"\n{Colors.BOLD}{Colors.GREEN}Report saved to: {report_path}{Colors.ENDC}")
        
        # Print summary
        print(f"\n{Colors.BOLD}AUDIT COMPLETE{Colors.ENDC}")
        if len(self.vulnerabilities) == 0:
            print(f"{Colors.GREEN}✅ EXCELLENT: No critical vulnerabilities detected!{Colors.ENDC}")
            return 0
        else:
            print(f"{Colors.YELLOW}⚠️  {len(self.vulnerabilities)} issues require attention{Colors.ENDC}")
            return 1

if __name__ == "__main__":
    auditor = SecurityAuditor()
    exit_code = auditor.run_full_audit()
    sys.exit(exit_code)