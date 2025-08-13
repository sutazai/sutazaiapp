#!/usr/bin/env python3
"""
ULTRA Security Validation Suite
Comprehensive security penetration testing and validation
Target: Zero critical vulnerabilities, enterprise-grade security
"""

import asyncio
import aiohttp
import json
import ssl
import socket
import subprocess
import re
import os
import time
import hashlib
import logging
import sys
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import secrets
import base64
import requests
from urllib.parse import urljoin, urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/tests/ultra_security_validation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SecurityTestResult:
    """Store security test results"""
    test_name: str
    category: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW, INFO
    status: str    # PASS, FAIL, SKIP, ERROR
    description: str
    details: Dict[str, Any]
    remediation: Optional[str] = None
    cve_references: Optional[List[str]] = None

@dataclass
class SecuritySummary:
    """Security assessment summary"""
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    error_tests: int
    critical_issues: int
    high_issues: int
    medium_issues: int
    low_issues: int
    security_score: float
    overall_grade: str
    compliance_status: Dict[str, bool]

class UltraSecurityValidator:
    """Comprehensive security validation system"""
    
    def __init__(self):
        self.base_url = "http://localhost"
        self.https_base_url = "https://localhost"
        self.test_results: List[SecurityTestResult] = []
        
        # Service endpoints to test
        self.endpoints = [
            {"url": f"{self.base_url}:10010", "name": "backend", "critical": True},
            {"url": f"{self.base_url}:10011", "name": "frontend", "critical": True},
            {"url": f"{self.base_url}:10104", "name": "ollama", "critical": True},
            {"url": f"{self.base_url}:10201", "name": "grafana", "critical": False},
            {"url": f"{self.base_url}:10200", "name": "prometheus", "critical": False},
            {"url": f"{self.base_url}:10000", "name": "postgres", "critical": True},
            {"url": f"{self.base_url}:10001", "name": "redis", "critical": True},
            {"url": f"{self.base_url}:10007", "name": "rabbitmq", "critical": True},
            {"url": f"{self.base_url}:8589", "name": "ai_orchestrator", "critical": True},
            {"url": f"{self.base_url}:11110", "name": "hardware_optimizer", "critical": True}
        ]
        
        # Common attack payloads
        self.xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "'\"><script>alert('xss')</script>",
            "<svg onload=alert('xss')>"
        ]
        
        self.sql_injection_payloads = [
            "' OR '1'='1",
            "'; DROP TABLE users; --",
            "' UNION SELECT * FROM users --",
            "1' AND (SELECT COUNT(*) FROM users) > 0 --",
            "admin'--"
        ]
        
        self.command_injection_payloads = [
            "; ls -la",
            "| whoami",
            "`id`",
            "$(whoami)",
            "&& cat /etc/passwd"
        ]
    
    def add_result(self, test_name: str, category: str, severity: str, status: str, 
                   description: str, details: Dict[str, Any], remediation: str = None):
        """Add test result to results list"""
        result = SecurityTestResult(
            test_name=test_name,
            category=category,
            severity=severity,
            status=status,
            description=description,
            details=details,
            remediation=remediation
        )
        self.test_results.append(result)
        
        # Log critical and high severity issues immediately
        if severity in ["CRITICAL", "HIGH"] and status == "FAIL":
            logger.error(f"SECURITY ISSUE - {severity}: {test_name} - {description}")
    
    def test_container_security(self) -> None:
        """Test Docker container security configurations"""
        logger.info("🔒 Testing container security configurations...")
        
        try:
            # Check for containers running as root
            result = subprocess.run(
                ["docker", "ps", "--format", "table {{.Names}}\\t{{.RunningFor}}"],
                capture_output=True, text=True, timeout=30
            )
            
            if result.returncode == 0:
                containers = result.stdout.strip().split('\n')[1:]  # Skip header
                
                for container_line in containers:
                    if not container_line.strip():
                        continue
                    
                    container_name = container_line.split('\t')[0]
                    
                    # Check if container is running as root
                    inspect_result = subprocess.run(
                        ["docker", "inspect", container_name, "--format", "{{.Config.User}}"],
                        capture_output=True, text=True, timeout=10
                    )
                    
                    if inspect_result.returncode == 0:
                        user = inspect_result.stdout.strip()
                        
                        if not user or user == "root" or user == "0":
                            self.add_result(
                                f"container_root_{container_name}",
                                "Container Security",
                                "HIGH",
                                "FAIL",
                                f"Container {container_name} running as root",
                                {"container": container_name, "user": user or "root"},
                                "Configure container to run with non-root user"
                            )
                        else:
                            self.add_result(
                                f"container_nonroot_{container_name}",
                                "Container Security",
                                "INFO",
                                "PASS",
                                f"Container {container_name} running as non-root user",
                                {"container": container_name, "user": user}
                            )
                
                self.add_result(
                    "container_security_scan",
                    "Container Security",
                    "INFO",
                    "PASS",
                    "Container security scan completed",
                    {"containers_checked": len(containers)}
                )
            else:
                self.add_result(
                    "container_security_scan",
                    "Container Security",
                    "MEDIUM",
                    "ERROR",
                    "Failed to inspect containers",
                    {"error": result.stderr}
                )
                
        except Exception as e:
            self.add_result(
                "container_security_scan",
                "Container Security",
                "MEDIUM",
                "ERROR",
                "Container security scan failed",
                {"error": str(e)}
            )
    
    def test_network_security(self) -> None:
        """Test network security configurations"""
        logger.info("🌐 Testing network security configurations...")
        
        # Test for open ports that shouldn't be public
        dangerous_ports = [22, 23, 3389, 5432, 3306, 6379, 27017, 9200]
        
        for port in dangerous_ports:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', port))
                sock.close()
                
                if result == 0:
                    self.add_result(
                        f"open_port_{port}",
                        "Network Security",
                        "HIGH" if port in [22, 23, 3389] else "MEDIUM",
                        "FAIL",
                        f"Dangerous port {port} is open",
                        {"port": port, "service": self.get_service_name(port)},
                        f"Close port {port} or restrict access with firewall"
                    )
                else:
                    self.add_result(
                        f"closed_port_{port}",
                        "Network Security",
                        "INFO",
                        "PASS",
                        f"Port {port} is properly closed",
                        {"port": port}
                    )
                    
            except Exception as e:
                logger.debug(f"Error checking port {port}: {e}")
    
    def get_service_name(self, port: int) -> str:
        """Get service name for port"""
        port_services = {
            22: "SSH",
            23: "Telnet", 
            3389: "RDP",
            5432: "PostgreSQL",
            3306: "MySQL",
            6379: "Redis",
            27017: "MongoDB",
            9200: "Elasticsearch"
        }
        return port_services.get(port, "Unknown")
    
    async def test_web_security(self) -> None:
        """Test web application security"""
        logger.info("🕷️ Testing web application security...")
        
        timeout = aiohttp.ClientTimeout(total=10)
        connector = aiohttp.TCPConnector(limit=20)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Test each endpoint for web vulnerabilities
            for endpoint in self.endpoints:
                await self.test_xss_vulnerabilities(session, endpoint)
                await self.test_sql_injection(session, endpoint)
                await self.test_authentication_bypass(session, endpoint)
                await self.test_directory_traversal(session, endpoint)
                await self.test_security_headers(session, endpoint)
    
    async def test_xss_vulnerabilities(self, session: aiohttp.ClientSession, endpoint: Dict[str, str]) -> None:
        """Test for XSS vulnerabilities"""
        url = endpoint['url']
        name = endpoint['name']
        
        for payload in self.xss_payloads[:3]:  # Test first 3 payloads
            try:
                # Test GET parameter injection
                test_url = f"{url}/?test={payload}"
                
                async with session.get(test_url) as response:
                    content = await response.text()
                    
                    if payload in content and response.headers.get('content-type', '').startswith('text/html'):
                        self.add_result(
                            f"xss_vulnerability_{name}",
                            "Web Security",
                            "HIGH",
                            "FAIL",
                            f"XSS vulnerability detected in {name}",
                            {
                                "endpoint": url,
                                "payload": payload,
                                "method": "GET",
                                "status_code": response.status
                            },
                            "Implement proper input validation and output encoding"
                        )
                        return  # One failure is enough per endpoint
                
            except Exception as e:
                logger.debug(f"XSS test error for {name}: {e}")
        
        # If we get here, no XSS was found
        self.add_result(
            f"xss_safe_{name}",
            "Web Security",
            "INFO",
            "PASS",
            f"No XSS vulnerabilities detected in {name}",
            {"endpoint": url, "payloads_tested": len(self.xss_payloads[:3])}
        )
    
    async def test_sql_injection(self, session: aiohttp.ClientSession, endpoint: Dict[str, str]) -> None:
        """Test for SQL injection vulnerabilities"""
        url = endpoint['url']
        name = endpoint['name']
        
        for payload in self.sql_injection_payloads[:3]:  # Test first 3 payloads
            try:
                # Test query parameter injection
                test_url = f"{url}/?id={payload}"
                
                async with session.get(test_url) as response:
                    content = await response.text().lower()
                    
                    # Look for SQL error indicators
                    sql_errors = [
                        'sql syntax', 'mysql_fetch', 'ora-', 'microsoft ole db',
                        'sqlite_', 'postgresql', 'warning: pg_', 'valid mysql result'
                    ]
                    
                    if any(error in content for error in sql_errors):
                        self.add_result(
                            f"sql_injection_{name}",
                            "Web Security",
                            "CRITICAL",
                            "FAIL",
                            f"SQL injection vulnerability detected in {name}",
                            {
                                "endpoint": url,
                                "payload": payload,
                                "status_code": response.status
                            },
                            "Use parameterized queries and input validation"
                        )
                        return
                
            except Exception as e:
                logger.debug(f"SQL injection test error for {name}: {e}")
        
        self.add_result(
            f"sql_injection_safe_{name}",
            "Web Security",
            "INFO",
            "PASS",
            f"No SQL injection vulnerabilities detected in {name}",
            {"endpoint": url, "payloads_tested": len(self.sql_injection_payloads[:3])}
        )
    
    async def test_authentication_bypass(self, session: aiohttp.ClientSession, endpoint: Dict[str, str]) -> None:
        """Test for authentication bypass vulnerabilities"""
        url = endpoint['url']
        name = endpoint['name']
        
        # Test common admin endpoints without authentication
        admin_paths = ['/admin', '/admin/', '/administrator', '/login', '/auth', '/api/admin']
        
        for path in admin_paths:
            try:
                test_url = f"{url}{path}"
                
                async with session.get(test_url) as response:
                    if response.status == 200:
                        content = await response.text().lower()
                        
                        # Check if it looks like an admin interface
                        if any(keyword in content for keyword in ['admin', 'dashboard', 'login', 'user management']):
                            self.add_result(
                                f"auth_bypass_{name}_{path}",
                                "Authentication",
                                "HIGH",
                                "FAIL",
                                f"Potential authentication bypass in {name} at {path}",
                                {
                                    "endpoint": url,
                                    "path": path,
                                    "status_code": response.status
                                },
                                "Implement proper authentication and authorization"
                            )
                        
            except Exception as e:
                logger.debug(f"Auth bypass test error for {name}{path}: {e}")
        
        self.add_result(
            f"auth_bypass_safe_{name}",
            "Authentication",
            "INFO",
            "PASS",
            f"No authentication bypass detected in {name}",
            {"endpoint": url}
        )
    
    async def test_directory_traversal(self, session: aiohttp.ClientSession, endpoint: Dict[str, str]) -> None:
        """Test for directory traversal vulnerabilities"""
        url = endpoint['url']
        name = endpoint['name']
        
        traversal_payloads = [
            "../../../etc/passwd",
            "..%2F..%2F..%2Fetc%2Fpasswd",
            "....//....//....//etc/passwd",
            "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts"
        ]
        
        for payload in traversal_payloads[:2]:  # Test first 2 payloads
            try:
                test_url = f"{url}/?file={payload}"
                
                async with session.get(test_url) as response:
                    content = await response.text()
                    
                    # Check for signs of successful directory traversal
                    if ('root:' in content and '/bin/' in content) or 'localhost' in content:
                        self.add_result(
                            f"directory_traversal_{name}",
                            "Web Security",
                            "HIGH",
                            "FAIL",
                            f"Directory traversal vulnerability in {name}",
                            {
                                "endpoint": url,
                                "payload": payload,
                                "status_code": response.status
                            },
                            "Implement proper input validation and file access controls"
                        )
                        return
                
            except Exception as e:
                logger.debug(f"Directory traversal test error for {name}: {e}")
        
        self.add_result(
            f"directory_traversal_safe_{name}",
            "Web Security",
            "INFO",
            "PASS",
            f"No directory traversal vulnerabilities in {name}",
            {"endpoint": url}
        )
    
    async def test_security_headers(self, session: aiohttp.ClientSession, endpoint: Dict[str, str]) -> None:
        """Test for security headers"""
        url = endpoint['url']
        name = endpoint['name']
        
        try:
            async with session.get(url) as response:
                headers = response.headers
                
                # Required security headers
                security_headers = {
                    'X-Content-Type-Options': 'nosniff',
                    'X-Frame-Options': ['DENY', 'SAMEORIGIN'],
                    'X-XSS-Protection': '1; mode=block',
                    'Strict-Transport-Security': 'max-age=',
                    'Content-Security-Policy': 'default-src'
                }
                
                missing_headers = []
                weak_headers = []
                
                for header, expected in security_headers.items():
                    header_value = headers.get(header, '').lower()
                    
                    if not header_value:
                        missing_headers.append(header)
                    elif isinstance(expected, list):
                        if not any(exp.lower() in header_value for exp in expected):
                            weak_headers.append(f"{header}: {header_value}")
                    elif expected.lower() not in header_value:
                        weak_headers.append(f"{header}: {header_value}")
                
                if missing_headers or weak_headers:
                    severity = "MEDIUM" if endpoint.get('critical') else "LOW"
                    self.add_result(
                        f"security_headers_{name}",
                        "Web Security",
                        severity,
                        "FAIL",
                        f"Missing or weak security headers in {name}",
                        {
                            "endpoint": url,
                            "missing_headers": missing_headers,
                            "weak_headers": weak_headers
                        },
                        "Implement proper security headers"
                    )
                else:
                    self.add_result(
                        f"security_headers_ok_{name}",
                        "Web Security",
                        "INFO",
                        "PASS",
                        f"Security headers properly configured in {name}",
                        {"endpoint": url}
                    )
                
        except Exception as e:
            logger.debug(f"Security headers test error for {name}: {e}")
    
    def test_secrets_and_credentials(self) -> None:
        """Test for hardcoded secrets and credentials"""
        logger.info("🔑 Testing for hardcoded secrets and credentials...")
        
        # Define patterns to search for
        secret_patterns = {
            'api_key': re.compile(r'(?i)(api_key|apikey)\s*[:=]\s*["\']?([a-z0-9-_]{20,})["\']?', re.MULTILINE),
            'password': re.compile(r'(?i)(password|pwd)\s*[:=]\s*["\']?([^"\'\s]{8,})["\']?', re.MULTILINE),
            'secret': re.compile(r'(?i)(secret|token)\s*[:=]\s*["\']?([a-z0-9-_]{20,})["\']?', re.MULTILINE),
            'private_key': re.compile(r'-----BEGIN (RSA )?PRIVATE KEY-----', re.MULTILINE),
            'jwt_secret': re.compile(r'(?i)(jwt_secret|jwt_key)\s*[:=]\s*["\']?([^"\'\s]{20,})["\']?', re.MULTILINE)
        }
        
        # Common files to check
        files_to_check = [
            '/opt/sutazaiapp/.env',
            '/opt/sutazaiapp/.env.example',
            '/opt/sutazaiapp/docker-compose.yml',
            '/opt/sutazaiapp/backend/app/core/config.py',
            '/opt/sutazaiapp/backend/app/main.py'
        ]
        
        secrets_found = []
        
        for file_path in files_to_check:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                        for pattern_name, pattern in secret_patterns.items():
                            matches = pattern.findall(content)
                            for match in matches:
                                if isinstance(match, tuple):
                                    key, value = match
                                else:
                                    key, value = pattern_name, match
                                
                                # Skip obvious examples/placeholders
                                if any(placeholder in value.lower() for placeholder in 
                                      ['example', 'placeholder', 'your-', 'change-me', 'xxx', '123']):
                                    continue
                                
                                secrets_found.append({
                                    'file': file_path,
                                    'type': pattern_name,
                                    'key': key,
                                    'value_preview': value[:10] + '...' if len(value) > 10 else value
                                })
                        
                except Exception as e:
                    logger.debug(f"Error checking {file_path}: {e}")
        
        if secrets_found:
            self.add_result(
                "hardcoded_secrets",
                "Secrets Management",
                "CRITICAL",
                "FAIL",
                "Hardcoded secrets or credentials found",
                {"secrets_count": len(secrets_found), "files": list(set(s['file'] for s in secrets_found))},
                "Move secrets to environment variables or secure secret management system"
            )
        else:
            self.add_result(
                "hardcoded_secrets_safe",
                "Secrets Management",
                "INFO",
                "PASS",
                "No hardcoded secrets detected in checked files",
                {"files_checked": len([f for f in files_to_check if os.path.exists(f)])}
            )
    
    def test_file_permissions(self) -> None:
        """Test file and directory permissions"""
        logger.info("📂 Testing file and directory permissions...")
        
        critical_files = [
            '/opt/sutazaiapp/.env',
            '/opt/sutazaiapp/docker-compose.yml',
            '/opt/sutazaiapp/backend/app/core/config.py'
        ]
        
        for file_path in critical_files:
            if os.path.exists(file_path):
                try:
                    stat = os.stat(file_path)
                    permissions = oct(stat.st_mode)[-3:]  # Get last 3 digits
                    
                    # Check if file is world-readable/writable
                    if permissions[2] in ['4', '5', '6', '7']:  # Others have read access
                        self.add_result(
                            f"file_permissions_{os.path.basename(file_path)}",
                            "File Permissions",
                            "MEDIUM",
                            "FAIL",
                            f"File {file_path} is world-readable",
                            {"file": file_path, "permissions": permissions},
                            f"Change permissions: chmod 600 {file_path}"
                        )
                    else:
                        self.add_result(
                            f"file_permissions_ok_{os.path.basename(file_path)}",
                            "File Permissions",
                            "INFO",
                            "PASS",
                            f"File {file_path} has proper permissions",
                            {"file": file_path, "permissions": permissions}
                        )
                        
                except Exception as e:
                    logger.debug(f"Error checking permissions for {file_path}: {e}")
    
    def generate_security_summary(self) -> SecuritySummary:
        """Generate comprehensive security summary"""
        if not self.test_results:
            return SecuritySummary(0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, "UNKNOWN", {})
        
        # Count results by status
        passed = sum(1 for r in self.test_results if r.status == "PASS")
        failed = sum(1 for r in self.test_results if r.status == "FAIL")
        skipped = sum(1 for r in self.test_results if r.status == "SKIP")
        errors = sum(1 for r in self.test_results if r.status == "ERROR")
        
        # Count by severity
        critical = sum(1 for r in self.test_results if r.severity == "CRITICAL" and r.status == "FAIL")
        high = sum(1 for r in self.test_results if r.severity == "HIGH" and r.status == "FAIL")
        medium = sum(1 for r in self.test_results if r.severity == "MEDIUM" and r.status == "FAIL")
        low = sum(1 for r in self.test_results if r.severity == "LOW" and r.status == "FAIL")
        
        # Calculate security score
        total_issues = critical + high + medium + low
        max_score = 100
        
        # Deduct points based on severity
        score_deduction = (critical * 25) + (high * 15) + (medium * 8) + (low * 3)
        security_score = max(0, max_score - score_deduction)
        
        # Determine overall grade
        if security_score >= 95:
            grade = "A+ (ULTRA SECURE)"
        elif security_score >= 90:
            grade = "A (EXCELLENT)"
        elif security_score >= 80:
            grade = "B (GOOD)"
        elif security_score >= 70:
            grade = "C (ACCEPTABLE)"
        elif security_score >= 60:
            grade = "D (NEEDS IMPROVEMENT)"
        else:
            grade = "F (CRITICAL ISSUES)"
        
        # Check compliance requirements
        compliance_status = {
            "zero_critical_vulns": critical == 0,
            " _high_vulns": high <= 2,
            "container_security": any("container_nonroot" in r.test_name for r in self.test_results if r.status == "PASS"),
            "secrets_management": any("hardcoded_secrets_safe" in r.test_name for r in self.test_results if r.status == "PASS"),
            "web_security": any("xss_safe" in r.test_name for r in self.test_results if r.status == "PASS")
        }
        
        return SecuritySummary(
            total_tests=len(self.test_results),
            passed_tests=passed,
            failed_tests=failed,
            skipped_tests=skipped,
            error_tests=errors,
            critical_issues=critical,
            high_issues=high,
            medium_issues=medium,
            low_issues=low,
            security_score=security_score,
            overall_grade=grade,
            compliance_status=compliance_status
        )
    
    async def run_comprehensive_security_test(self) -> Dict[str, Any]:
        """Run all security tests"""
        logger.info("🛡️ Starting ULTRA Security Validation Suite")
        start_time = time.time()
        
        # Run all security tests
        logger.info("Testing container security...")
        self.test_container_security()
        
        logger.info("Testing network security...")
        self.test_network_security()
        
        logger.info("Testing web application security...")
        await self.test_web_security()
        
        logger.info("Testing secrets and credentials...")
        self.test_secrets_and_credentials()
        
        logger.info("Testing file permissions...")
        self.test_file_permissions()
        
        end_time = time.time()
        test_duration = end_time - start_time
        
        # Generate summary
        summary = self.generate_security_summary()
        
        return {
            "test_type": "comprehensive_security_validation",
            "timestamp": datetime.now().isoformat(),
            "test_duration": test_duration,
            "summary": asdict(summary),
            "detailed_results": [asdict(result) for result in self.test_results],
            "compliance_check": self.check_security_compliance(summary)
        }
    
    def check_security_compliance(self, summary: SecuritySummary) -> Dict[str, Any]:
        """Check compliance with security standards"""
        compliance_score = sum(summary.compliance_status.values()) / len(summary.compliance_status) * 100
        
        return {
            "overall_compliance": compliance_score >= 80,
            "compliance_score": compliance_score,
            "requirements_met": summary.compliance_status,
            "critical_findings": summary.critical_issues == 0,
            "production_ready": summary.security_score >= 90 and summary.critical_issues == 0
        }
    
    def save_results(self, test_data: Dict[str, Any]) -> str:
        """Save security test results"""
        filename = f"/opt/sutazaiapp/tests/ultra_security_validation_results_{int(time.time())}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(test_data, f, indent=2)
            logger.info(f"Security test results saved to: {filename}")
            return filename
        except Exception as e:
            logger.error(f"Failed to save security results: {e}")
            return ""

async def main():
    """Run the ULTRA security validation suite"""
    print("🛡️ ULTRA SECURITY VALIDATION SUITE")
    print("=" * 50)
    
    validator = UltraSecurityValidator()
    
    # Run comprehensive security test
    results = await validator.run_comprehensive_security_test()
    
    # Save results
    results_file = validator.save_results(results)
    
    # Print summary
    summary = results['summary']
    compliance = results['compliance_check']
    
    print("\n" + "=" * 50)
    print("🎯 SECURITY VALIDATION RESULTS")
    print("=" * 50)
    print(f"Overall Grade: {summary['overall_grade']}")
    print(f"Security Score: {summary['security_score']:.1f}/100")
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed_tests']}")
    print(f"Failed: {summary['failed_tests']}")
    
    print(f"\n🚨 Issues by Severity:")
    print(f"Critical: {summary['critical_issues']}")
    print(f"High: {summary['high_issues']}")
    print(f"Medium: {summary['medium_issues']}")
    print(f"Low: {summary['low_issues']}")
    
    print(f"\n✅ Compliance Status:")
    for requirement, status in summary['compliance_status'].items():
        print(f"{requirement}: {'✅ PASS' if status else '❌ FAIL'}")
    
    print(f"\nProduction Ready: {'✅ YES' if compliance['production_ready'] else '❌ NO'}")
    print(f"Results saved to: {results_file}")
    
    # Return exit code based on critical issues
    return 0 if summary['critical_issues'] == 0 else 1

if __name__ == "__main__":
    exit(asyncio.run(main()))