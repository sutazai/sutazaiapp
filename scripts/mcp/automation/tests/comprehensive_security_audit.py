#!/usr/bin/env python3
"""
Comprehensive Security Audit and Error Scenario Testing for MCP Automation System
This script performs thorough security vulnerability assessment and error handling tests.
"""

import asyncio
import json
import logging
import os
import sys
import time
import random
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import aiohttp
import pytest
from dataclasses import dataclass, field
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('security_audit_report.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class VulnerabilityLevel(Enum):
    """Security vulnerability severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class TestCategory(Enum):
    """Security test categories"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    INJECTION = "injection"
    XSS = "xss"
    CSRF = "csrf"
    INPUT_VALIDATION = "input_validation"
    INFORMATION_DISCLOSURE = "information_disclosure"
    ACCESS_CONTROL = "access_control"
    ERROR_HANDLING = "error_handling"
    CONFIGURATION = "configuration"
    MCP_PROTECTION = "mcp_protection"


@dataclass
class SecurityFinding:
    """Security vulnerability finding"""
    category: TestCategory
    level: VulnerabilityLevel
    title: str
    description: str
    endpoint: Optional[str] = None
    remediation: Optional[str] = None
    evidence: Optional[Dict[str, Any]] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TestResult:
    """Test execution result"""
    test_name: str
    category: TestCategory
    passed: bool
    message: str
    duration: float
    details: Optional[Dict[str, Any]] = field(default_factory=dict)


class SecurityAuditor:
    """Comprehensive security auditor for MCP automation system"""
    
    def __init__(self, base_url: str = "http://localhost:10250"):
        self.base_url = base_url.rstrip('/')
        self.findings: List[SecurityFinding] = []
        self.test_results: List[TestResult] = []
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Setup async context"""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup async context"""
        if self.session:
            await self.session.close()
            
    def add_finding(self, finding: SecurityFinding):
        """Add security finding"""
        self.findings.append(finding)
        logger.warning(f"[{finding.level.value.upper()}] {finding.title}")
        
    def add_result(self, result: TestResult):
        """Add test result"""
        self.test_results.append(result)
        status = "✓" if result.passed else "✗"
        logger.info(f"[{status}] {result.test_name}: {result.message}")
        
    # ============== AUTHENTICATION & AUTHORIZATION TESTS ==============
    
    async def test_authentication(self) -> TestResult:
        """Test authentication mechanisms"""
        start_time = time.time()
        category = TestCategory.AUTHENTICATION
        
        try:
            # Test for missing authentication
            endpoints = ['/metrics', '/health/detailed', '/alerts', '/sla/status']
            
            for endpoint in endpoints:
                async with self.session.get(f"{self.base_url}{endpoint}") as resp:
                    if resp.status == 200:
                        # Check if authentication is required
                        if 'authorization' not in resp.headers and 'www-authenticate' not in resp.headers:
                            self.add_finding(SecurityFinding(
                                category=category,
                                level=VulnerabilityLevel.HIGH,
                                title="Missing Authentication",
                                description=f"Endpoint {endpoint} accessible without authentication",
                                endpoint=endpoint,
                                remediation="Implement authentication middleware for sensitive endpoints",
                                evidence={"status": resp.status, "headers": dict(resp.headers)}
                            ))
                            
            # Test for weak authentication schemes
            auth_headers = [
                ('Authorization', 'Basic YWRtaW46YWRtaW4='),  # admin:admin
                ('Authorization', 'Bearer test123'),
                ('X-API-Key', 'test'),
            ]
            
            for header_name, header_value in auth_headers:
                headers = {header_name: header_value}
                async with self.session.get(f"{self.base_url}/alerts", headers=headers) as resp:
                    if resp.status == 200:
                        self.add_finding(SecurityFinding(
                            category=category,
                            level=VulnerabilityLevel.CRITICAL,
                            title="Weak Authentication Accepted",
                            description=f"Weak credentials accepted: {header_name}",
                            endpoint="/alerts",
                            remediation="Implement strong authentication validation",
                            evidence={"header": header_name, "status": resp.status}
                        ))
                        
            duration = time.time() - start_time
            passed = len([f for f in self.findings if f.category == category and f.level in [VulnerabilityLevel.CRITICAL, VulnerabilityLevel.HIGH]]) == 0
            
            return TestResult(
                test_name="Authentication Security",
                category=category,
                passed=passed,
                message=f"Found {len([f for f in self.findings if f.category == category])} authentication issues",
                duration=duration
            )
            
        except Exception as e:
            logger.error(f"Authentication test error: {e}")
            return TestResult(
                test_name="Authentication Security",
                category=category,
                passed=False,
                message=str(e),
                duration=time.time() - start_time
            )
            
    async def test_authorization(self) -> TestResult:
        """Test authorization and access control"""
        start_time = time.time()
        category = TestCategory.AUTHORIZATION
        
        try:
            # Test for privilege escalation
            test_cases = [
                ('/alerts', 'POST', {'name': 'test', 'severity': 'critical'}),
                ('/alerts/test-id/resolve', 'POST', {}),
                ('/dashboards/deploy/test', 'POST', {}),
            ]
            
            for endpoint, method, data in test_cases:
                async with self.session.request(
                    method,
                    f"{self.base_url}{endpoint}",
                    json=data,
                    headers={'X-User-Role': 'guest'}
                ) as resp:
                    if resp.status in [200, 201]:
                        self.add_finding(SecurityFinding(
                            category=category,
                            level=VulnerabilityLevel.HIGH,
                            title="Insufficient Authorization",
                            description=f"Privileged action allowed without proper authorization: {method} {endpoint}",
                            endpoint=endpoint,
                            remediation="Implement proper role-based access control",
                            evidence={"method": method, "status": resp.status}
                        ))
                        
            duration = time.time() - start_time
            passed = len([f for f in self.findings if f.category == category and f.level in [VulnerabilityLevel.CRITICAL, VulnerabilityLevel.HIGH]]) == 0
            
            return TestResult(
                test_name="Authorization Security",
                category=category,
                passed=passed,
                message=f"Found {len([f for f in self.findings if f.category == category])} authorization issues",
                duration=duration
            )
            
        except Exception as e:
            logger.error(f"Authorization test error: {e}")
            return TestResult(
                test_name="Authorization Security",
                category=category,
                passed=False,
                message=str(e),
                duration=time.time() - start_time
            )
            
    # ============== INJECTION ATTACK TESTS ==============
    
    async def test_injection_attacks(self) -> TestResult:
        """Test for various injection vulnerabilities"""
        start_time = time.time()
        category = TestCategory.INJECTION
        
        injection_payloads = [
            # SQL Injection
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "1' UNION SELECT * FROM information_schema.tables--",
            
            # NoSQL Injection
            '{"$ne": null}',
            '{"$gt": ""}',
            '{"$regex": ".*"}',
            
            # Command Injection
            "; ls -la /",
            "| cat /etc/passwd",
            "$(whoami)",
            "`id`",
            
            # LDAP Injection
            "*)(uid=*",
            "*)(|(uid=*",
            
            # Path Traversal
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            
            # Template Injection
            "{{7*7}}",
            "${7*7}",
            "<%= 7*7 %>",
        ]
        
        try:
            # Test search endpoint
            for payload in injection_payloads:
                async with self.session.get(
                    f"{self.base_url}/logs/search",
                    params={'query': payload}
                ) as resp:
                    content = await resp.text()
                    
                    # Check for signs of successful injection
                    if any(indicator in content.lower() for indicator in ['error', 'syntax', 'unexpected', 'passwd', 'root:', '49']):
                        self.add_finding(SecurityFinding(
                            category=category,
                            level=VulnerabilityLevel.CRITICAL,
                            title="Injection Vulnerability",
                            description=f"Potential injection vulnerability with payload: {payload[:50]}...",
                            endpoint="/logs/search",
                            remediation="Implement proper input sanitization and parameterized queries",
                            evidence={"payload": payload, "response_snippet": content[:200]}
                        ))
                        
            # Test POST endpoints
            for payload in injection_payloads[:5]:  # Test subset for POST
                data = {
                    'message': payload,
                    'component': payload,
                    'level': 'info'
                }
                
                async with self.session.post(
                    f"{self.base_url}/logs",
                    json=data
                ) as resp:
                    if resp.status == 500:
                        content = await resp.text()
                        if 'syntax' in content.lower() or 'error' in content.lower():
                            self.add_finding(SecurityFinding(
                                category=category,
                                level=VulnerabilityLevel.CRITICAL,
                                title="Injection in Log Ingestion",
                                description="Log ingestion vulnerable to injection attacks",
                                endpoint="/logs",
                                remediation="Sanitize all input data before processing",
                                evidence={"payload": payload}
                            ))
                            
            duration = time.time() - start_time
            passed = len([f for f in self.findings if f.category == category and f.level == VulnerabilityLevel.CRITICAL]) == 0
            
            return TestResult(
                test_name="Injection Attack Prevention",
                category=category,
                passed=passed,
                message=f"Tested {len(injection_payloads)} injection payloads",
                duration=duration
            )
            
        except Exception as e:
            logger.error(f"Injection test error: {e}")
            return TestResult(
                test_name="Injection Attack Prevention",
                category=category,
                passed=False,
                message=str(e),
                duration=time.time() - start_time
            )
            
    # ============== XSS TESTS ==============
    
    async def test_xss_vulnerabilities(self) -> TestResult:
        """Test for Cross-Site Scripting vulnerabilities"""
        start_time = time.time()
        category = TestCategory.XSS
        
        xss_payloads = [
            '<script>alert("XSS")</script>',
            '<img src=x onerror=alert("XSS")>',
            '<svg onload=alert("XSS")>',
            'javascript:alert("XSS")',
            '<iframe src="javascript:alert(\'XSS\')">',
            '<body onload=alert("XSS")>',
            '"><script>alert("XSS")</script>',
            '<script>document.location="http://evil.com"</script>',
            '<meta http-equiv="refresh" content="0;url=http://evil.com">',
        ]
        
        try:
            # Test reflected XSS in search
            for payload in xss_payloads:
                async with self.session.get(
                    f"{self.base_url}/logs/search",
                    params={'query': payload}
                ) as resp:
                    content = await resp.text()
                    
                    # Check if payload is reflected without encoding
                    if payload in content or payload.replace('"', '&quot;') not in content:
                        self.add_finding(SecurityFinding(
                            category=category,
                            level=VulnerabilityLevel.HIGH,
                            title="Reflected XSS Vulnerability",
                            description="User input reflected without proper encoding",
                            endpoint="/logs/search",
                            remediation="HTML-encode all user input before displaying",
                            evidence={"payload": payload}
                        ))
                        
            # Test stored XSS in logs
            xss_log = {
                'message': '<script>alert("Stored XSS")</script>',
                'component': 'test',
                'level': 'info'
            }
            
            async with self.session.post(f"{self.base_url}/logs", json=xss_log) as resp:
                if resp.status == 200:
                    # Try to retrieve the log
                    async with self.session.get(
                        f"{self.base_url}/logs/search",
                        params={'query': 'test'}
                    ) as search_resp:
                        content = await search_resp.text()
                        if '<script>' in content:
                            self.add_finding(SecurityFinding(
                                category=category,
                                level=VulnerabilityLevel.CRITICAL,
                                title="Stored XSS Vulnerability",
                                description="Stored data not sanitized, allowing persistent XSS",
                                endpoint="/logs",
                                remediation="Sanitize data on input and encode on output",
                                evidence={"stored_payload": xss_log['message']}
                            ))
                            
            # Test DOM-based XSS in dashboard
            async with self.session.get(f"{self.base_url}/") as resp:
                content = await resp.text()
                
                # Check for unsafe JavaScript patterns
                unsafe_patterns = [
                    'innerHTML =',
                    'document.write(',
                    'eval(',
                    '.html(',
                    'v-html=',
                ]
                
                for pattern in unsafe_patterns:
                    if pattern in content:
                        self.add_finding(SecurityFinding(
                            category=category,
                            level=VulnerabilityLevel.MEDIUM,
                            title="Potential DOM XSS",
                            description=f"Unsafe JavaScript pattern detected: {pattern}",
                            endpoint="/",
                            remediation="Use safe DOM manipulation methods",
                            evidence={"pattern": pattern}
                        ))
                        
            duration = time.time() - start_time
            passed = len([f for f in self.findings if f.category == category and f.level in [VulnerabilityLevel.CRITICAL, VulnerabilityLevel.HIGH]]) == 0
            
            return TestResult(
                test_name="XSS Prevention",
                category=category,
                passed=passed,
                message=f"Tested {len(xss_payloads)} XSS payloads",
                duration=duration
            )
            
        except Exception as e:
            logger.error(f"XSS test error: {e}")
            return TestResult(
                test_name="XSS Prevention",
                category=category,
                passed=False,
                message=str(e),
                duration=time.time() - start_time
            )
            
    # ============== INPUT VALIDATION TESTS ==============
    
    async def test_input_validation(self) -> TestResult:
        """Test input validation and sanitization"""
        start_time = time.time()
        category = TestCategory.INPUT_VALIDATION
        
        try:
            # Test boundary values
            test_cases = [
                # Extremely long strings
                ('message', 'a' * 100000),
                
                # Special characters
                ('component', '!@#$%^&*(){}[]|\\:";\'<>?,./'),
                
                # Unicode and emoji
                ('message', '🔥💀☠️\u0000\uffff'),
                
                # Negative numbers
                ('value', -999999),
                
                # Float overflow
                ('value', float('inf')),
                
                # Null bytes
                ('message', 'test\x00hidden'),
                
                # Format string
                ('message', '%s%s%s%s%s%s%s%s%s%s'),
            ]
            
            for field, value in test_cases:
                data = {field: value, 'level': 'info'}
                
                async with self.session.post(f"{self.base_url}/logs", json=data) as resp:
                    if resp.status == 500:
                        self.add_finding(SecurityFinding(
                            category=category,
                            level=VulnerabilityLevel.MEDIUM,
                            title="Insufficient Input Validation",
                            description=f"Field '{field}' does not handle edge cases properly",
                            endpoint="/logs",
                            remediation="Implement comprehensive input validation",
                            evidence={"field": field, "value_type": type(value).__name__}
                        ))
                        
            # Test integer overflow in SLA measurement
            overflow_data = {
                'slo_name': 'test',
                'value': 2**63,  # Max int64 + 1
                'timestamp': datetime.now().isoformat()
            }
            
            async with self.session.post(f"{self.base_url}/sla/measurement", json=overflow_data) as resp:
                if resp.status == 500:
                    self.add_finding(SecurityFinding(
                        category=category,
                        level=VulnerabilityLevel.MEDIUM,
                        title="Integer Overflow",
                        description="SLA measurement vulnerable to integer overflow",
                        endpoint="/sla/measurement",
                        remediation="Validate numeric inputs are within acceptable ranges"
                    ))
                    
            duration = time.time() - start_time
            passed = len([f for f in self.findings if f.category == category]) < 3
            
            return TestResult(
                test_name="Input Validation",
                category=category,
                passed=passed,
                message=f"Tested {len(test_cases)} input validation cases",
                duration=duration
            )
            
        except Exception as e:
            logger.error(f"Input validation test error: {e}")
            return TestResult(
                test_name="Input Validation",
                category=category,
                passed=False,
                message=str(e),
                duration=time.time() - start_time
            )
            
    # ============== INFORMATION DISCLOSURE TESTS ==============
    
    async def test_information_disclosure(self) -> TestResult:
        """Test for information disclosure vulnerabilities"""
        start_time = time.time()
        category = TestCategory.INFORMATION_DISCLOSURE
        
        try:
            # Test error messages for sensitive information
            async with self.session.get(f"{self.base_url}/nonexistent") as resp:
                if resp.status == 404:
                    content = await resp.text()
                    
                    # Check for stack traces
                    if 'traceback' in content.lower() or 'stack trace' in content.lower():
                        self.add_finding(SecurityFinding(
                            category=category,
                            level=VulnerabilityLevel.MEDIUM,
                            title="Stack Trace Disclosure",
                            description="Error responses contain stack traces",
                            endpoint="/nonexistent",
                            remediation="Use generic error messages in production"
                        ))
                        
            # Test for version disclosure
            async with self.session.get(f"{self.base_url}/health") as resp:
                headers = resp.headers
                
                # Check for server version headers
                sensitive_headers = ['Server', 'X-Powered-By', 'X-AspNet-Version']
                for header in sensitive_headers:
                    if header in headers:
                        self.add_finding(SecurityFinding(
                            category=category,
                            level=VulnerabilityLevel.LOW,
                            title="Version Disclosure",
                            description=f"Server version disclosed in {header} header",
                            endpoint="/health",
                            remediation="Remove or obfuscate version headers",
                            evidence={header: headers[header]}
                        ))
                        
            # Test for directory listing
            test_paths = ['/../', '/scripts/', '/tests/', '/venv/']
            for path in test_paths:
                async with self.session.get(f"{self.base_url}{path}") as resp:
                    content = await resp.text()
                    if 'index of' in content.lower() or '<pre>' in content:
                        self.add_finding(SecurityFinding(
                            category=category,
                            level=VulnerabilityLevel.HIGH,
                            title="Directory Listing Enabled",
                            description=f"Directory listing exposed at {path}",
                            endpoint=path,
                            remediation="Disable directory listing"
                        ))
                        
            # Test for internal IP disclosure
            async with self.session.get(f"{self.base_url}/metrics") as resp:
                content = await resp.text()
                
                # Look for internal IPs
                import re
                ip_pattern = r'\b(?:10|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d{1,3}\.\d{1,3}\b'
                internal_ips = re.findall(ip_pattern, content)
                
                if internal_ips:
                    self.add_finding(SecurityFinding(
                        category=category,
                        level=VulnerabilityLevel.LOW,
                        title="Internal IP Disclosure",
                        description="Internal IP addresses exposed in metrics",
                        endpoint="/metrics",
                        remediation="Sanitize or mask internal IP addresses",
                        evidence={"ips": list(set(internal_ips))}
                    ))
                    
            duration = time.time() - start_time
            passed = len([f for f in self.findings if f.category == category and f.level == VulnerabilityLevel.HIGH]) == 0
            
            return TestResult(
                test_name="Information Disclosure",
                category=category,
                passed=passed,
                message=f"Checked for various information disclosure issues",
                duration=duration
            )
            
        except Exception as e:
            logger.error(f"Information disclosure test error: {e}")
            return TestResult(
                test_name="Information Disclosure",
                category=category,
                passed=False,
                message=str(e),
                duration=time.time() - start_time
            )
            
    # ============== ACCESS CONTROL TESTS ==============
    
    async def test_access_control(self) -> TestResult:
        """Test file and directory access controls"""
        start_time = time.time()
        category = TestCategory.ACCESS_CONTROL
        
        try:
            # Check file permissions
            critical_paths = [
                '/opt/sutazaiapp/scripts/mcp/automation/config.py',
                '/opt/sutazaiapp/scripts/mcp/automation/monitoring/config/',
                '/opt/sutazaiapp/scripts/mcp/automation/venv/',
            ]
            
            for path in critical_paths:
                path_obj = Path(path)
                if path_obj.exists():
                    # Check permissions
                    stat_info = path_obj.stat()
                    mode = oct(stat_info.st_mode)[-3:]
                    
                    # Check for world-writable
                    if mode[-1] in ['2', '3', '6', '7']:
                        self.add_finding(SecurityFinding(
                            category=category,
                            level=VulnerabilityLevel.HIGH,
                            title="World-Writable File/Directory",
                            description=f"Path {path} is world-writable",
                            remediation="Set appropriate file permissions (e.g., 644 for files, 755 for directories)",
                            evidence={"path": path, "permissions": mode}
                        ))
                        
                    # Check for sensitive files readable by others
                    if 'config' in path.lower() and mode[-1] in ['4', '5', '6', '7']:
                        self.add_finding(SecurityFinding(
                            category=category,
                            level=VulnerabilityLevel.MEDIUM,
                            title="Sensitive File Readable",
                            description=f"Configuration file {path} readable by others",
                            remediation="Restrict access to sensitive files (e.g., 600)",
                            evidence={"path": path, "permissions": mode}
                        ))
                        
            # Test directory traversal
            traversal_payloads = [
                '../../../etc/passwd',
                '..\\..\\..\\windows\\system32\\config\\sam',
                'file:///etc/passwd',
                '....//....//....//etc/passwd',
            ]
            
            for payload in traversal_payloads:
                params = {'query': payload, 'component': payload}
                async with self.session.get(f"{self.base_url}/logs/search", params=params) as resp:
                    if resp.status == 200:
                        content = await resp.text()
                        if 'root:' in content or 'administrator' in content.lower():
                            self.add_finding(SecurityFinding(
                                category=category,
                                level=VulnerabilityLevel.CRITICAL,
                                title="Directory Traversal",
                                description="Directory traversal vulnerability detected",
                                endpoint="/logs/search",
                                remediation="Validate and sanitize file paths",
                                evidence={"payload": payload}
                            ))
                            
            duration = time.time() - start_time
            passed = len([f for f in self.findings if f.category == category and f.level in [VulnerabilityLevel.CRITICAL, VulnerabilityLevel.HIGH]]) == 0
            
            return TestResult(
                test_name="Access Control",
                category=category,
                passed=passed,
                message="Checked file permissions and access controls",
                duration=duration
            )
            
        except Exception as e:
            logger.error(f"Access control test error: {e}")
            return TestResult(
                test_name="Access Control",
                category=category,
                passed=False,
                message=str(e),
                duration=time.time() - start_time
            )
            
    # ============== ERROR HANDLING TESTS ==============
    
    async def test_error_handling(self) -> TestResult:
        """Test error handling and graceful degradation"""
        start_time = time.time()
        category = TestCategory.ERROR_HANDLING
        
        try:
            error_scenarios = []
            
            # Test malformed JSON
            async with self.session.post(
                f"{self.base_url}/logs",
                data='{"invalid json}',
                headers={'Content-Type': 'application/json'}
            ) as resp:
                if resp.status == 500:
                    content = await resp.text()
                    if 'internal server error' not in content.lower():
                        error_scenarios.append("Malformed JSON causes unhandled error")
                        
            # Test missing required fields
            async with self.session.post(
                f"{self.base_url}/sla/measurement",
                json={}
            ) as resp:
                if resp.status not in [400, 422]:
                    error_scenarios.append("Missing required fields not validated")
                    
            # Test resource exhaustion - concurrent connections
            tasks = []
            for _ in range(100):
                tasks.append(self.session.get(f"{self.base_url}/health"))
                
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            failures = sum(1 for r in responses if isinstance(r, Exception))
            
            if failures > 10:
                self.add_finding(SecurityFinding(
                    category=category,
                    level=VulnerabilityLevel.MEDIUM,
                    title="Poor Concurrent Request Handling",
                    description=f"{failures}/100 concurrent requests failed",
                    remediation="Implement proper connection pooling and rate limiting",
                    evidence={"failure_rate": f"{failures}%"}
                ))
                
            # Test timeout handling
            # Simulate slow request (if endpoint supports delay parameter)
            timeout = aiohttp.ClientTimeout(total=2)
            try:
                async with self.session.get(
                    f"{self.base_url}/health",
                    timeout=timeout,
                    params={'delay': 5}
                ) as resp:
                    pass
            except asyncio.TimeoutError:
                # This is expected, but check if server handles it gracefully
                pass
                
            # Test invalid HTTP methods
            invalid_methods = ['TRACE', 'CONNECT', 'OPTIONS']
            for method in invalid_methods:
                async with self.session.request(method, f"{self.base_url}/health") as resp:
                    if resp.status == 200:
                        self.add_finding(SecurityFinding(
                            category=category,
                            level=VulnerabilityLevel.LOW,
                            title="Unnecessary HTTP Methods Enabled",
                            description=f"{method} method is enabled",
                            endpoint="/health",
                            remediation="Disable unnecessary HTTP methods"
                        ))
                        
            if error_scenarios:
                self.add_finding(SecurityFinding(
                    category=category,
                    level=VulnerabilityLevel.MEDIUM,
                    title="Inadequate Error Handling",
                    description="Multiple error scenarios not handled properly",
                    remediation="Implement comprehensive error handling",
                    evidence={"scenarios": error_scenarios}
                ))
                
            duration = time.time() - start_time
            passed = len(error_scenarios) == 0
            
            return TestResult(
                test_name="Error Handling",
                category=category,
                passed=passed,
                message=f"Tested {len(error_scenarios) + 5} error scenarios",
                duration=duration
            )
            
        except Exception as e:
            logger.error(f"Error handling test error: {e}")
            return TestResult(
                test_name="Error Handling",
                category=category,
                passed=False,
                message=str(e),
                duration=time.time() - start_time
            )
            
    # ============== CONFIGURATION SECURITY TESTS ==============
    
    async def test_configuration_security(self) -> TestResult:
        """Test for configuration security issues"""
        start_time = time.time()
        category = TestCategory.CONFIGURATION
        
        try:
            # Check for hardcoded secrets in config files
            config_files = [
                '/opt/sutazaiapp/scripts/mcp/automation/config.py',
                '/opt/sutazaiapp/scripts/mcp/automation/monitoring/config/prometheus.yml',
                '/opt/sutazaiapp/scripts/mcp/automation/monitoring/config/alert_rules.yml',
            ]
            
            secret_patterns = [
                r'password\s*=\s*["\'][^"\']+["\']',
                r'api_key\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']+["\']',
                r'token\s*=\s*["\'][^"\']+["\']',
                r'aws_access_key',
                r'private_key',
            ]
            
            import re
            for config_file in config_files:
                if Path(config_file).exists():
                    with open(config_file, 'r') as f:
                        content = f.read()
                        
                    for pattern in secret_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            self.add_finding(SecurityFinding(
                                category=category,
                                level=VulnerabilityLevel.HIGH,
                                title="Hardcoded Secrets",
                                description=f"Potential hardcoded secret in {config_file}",
                                remediation="Use environment variables or secure secret management",
                                evidence={"file": config_file, "pattern": pattern}
                            ))
                            
            # Check CORS configuration
            async with self.session.options(f"{self.base_url}/health") as resp:
                cors_headers = {
                    'Access-Control-Allow-Origin': resp.headers.get('Access-Control-Allow-Origin'),
                    'Access-Control-Allow-Credentials': resp.headers.get('Access-Control-Allow-Credentials'),
                }
                
                if cors_headers['Access-Control-Allow-Origin'] == '*':
                    self.add_finding(SecurityFinding(
                        category=category,
                        level=VulnerabilityLevel.MEDIUM,
                        title="Overly Permissive CORS",
                        description="CORS allows all origins (*)",
                        endpoint="/health",
                        remediation="Restrict CORS to specific trusted origins",
                        evidence=cors_headers
                    ))
                    
            # Check for debug mode
            async with self.session.get(f"{self.base_url}/health") as resp:
                headers = dict(resp.headers)
                content = await resp.text()
                
                if 'debug' in content.lower() or 'DEBUG' in str(headers):
                    self.add_finding(SecurityFinding(
                        category=category,
                        level=VulnerabilityLevel.MEDIUM,
                        title="Debug Mode Potentially Enabled",
                        description="Debug information may be exposed",
                        remediation="Disable debug mode in production"
                    ))
                    
            duration = time.time() - start_time
            passed = len([f for f in self.findings if f.category == category and f.level == VulnerabilityLevel.HIGH]) == 0
            
            return TestResult(
                test_name="Configuration Security",
                category=category,
                passed=passed,
                message="Checked configuration security",
                duration=duration
            )
            
        except Exception as e:
            logger.error(f"Configuration security test error: {e}")
            return TestResult(
                test_name="Configuration Security",
                category=category,
                passed=False,
                message=str(e),
                duration=time.time() - start_time
            )
            
    # ============== MCP PROTECTION TESTS ==============
    
    async def test_mcp_protection(self) -> TestResult:
        """Test MCP infrastructure protection (Rule 20 compliance)"""
        start_time = time.time()
        category = TestCategory.MCP_PROTECTION
        
        try:
            # Check MCP wrapper scripts are not modifiable
            mcp_paths = [
                '/opt/sutazaiapp/scripts/mcp/wrappers/',
                '/opt/sutazaiapp/.mcp.json',
            ]
            
            for path in mcp_paths:
                path_obj = Path(path)
                if path_obj.exists():
                    # Try to check if we can modify (we shouldn't actually modify)
                    if path_obj.is_file():
                        # Check file is not world-writable
                        stat_info = path_obj.stat()
                        mode = oct(stat_info.st_mode)[-3:]
                        
                        if mode[-1] in ['2', '3', '6', '7']:
                            self.add_finding(SecurityFinding(
                                category=category,
                                level=VulnerabilityLevel.CRITICAL,
                                title="MCP Infrastructure Not Protected",
                                description=f"MCP file {path} is modifiable",
                                remediation="Protect MCP infrastructure per Rule 20",
                                evidence={"path": path, "permissions": mode}
                            ))
                            
            # Test staging isolation
            staging_dir = Path('/opt/sutazaiapp/scripts/mcp/automation/staging/')
            if staging_dir.exists():
                # Check staging is properly isolated
                staging_files = list(staging_dir.glob('*'))
                if any('production' in str(f) for f in staging_files):
                    self.add_finding(SecurityFinding(
                        category=category,
                        level=VulnerabilityLevel.HIGH,
                        title="Staging Not Properly Isolated",
                        description="Staging environment may affect production",
                        remediation="Ensure complete staging isolation"
                    ))
                    
            # Verify audit trails exist
            audit_log = Path('/opt/sutazaiapp/scripts/mcp/automation/cleanup/audit.log')
            if not audit_log.exists():
                self.add_finding(SecurityFinding(
                    category=category,
                    level=VulnerabilityLevel.MEDIUM,
                    title="Missing Audit Trail",
                    description="MCP operations audit trail not found",
                    remediation="Implement comprehensive audit logging"
                ))
                
            duration = time.time() - start_time
            passed = len([f for f in self.findings if f.category == category and f.level == VulnerabilityLevel.CRITICAL]) == 0
            
            return TestResult(
                test_name="MCP Infrastructure Protection",
                category=category,
                passed=passed,
                message="Verified MCP protection compliance",
                duration=duration
            )
            
        except Exception as e:
            logger.error(f"MCP protection test error: {e}")
            return TestResult(
                test_name="MCP Infrastructure Protection",
                category=category,
                passed=False,
                message=str(e),
                duration=time.time() - start_time
            )
            
    # ============== REPORT GENERATION ==============
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive security audit report"""
        
        # Categorize findings by severity
        findings_by_severity = {
            VulnerabilityLevel.CRITICAL: [],
            VulnerabilityLevel.HIGH: [],
            VulnerabilityLevel.MEDIUM: [],
            VulnerabilityLevel.LOW: [],
            VulnerabilityLevel.INFO: [],
        }
        
        for finding in self.findings:
            findings_by_severity[finding.level].append({
                'category': finding.category.value,
                'title': finding.title,
                'description': finding.description,
                'endpoint': finding.endpoint,
                'remediation': finding.remediation,
                'evidence': finding.evidence,
                'timestamp': finding.timestamp.isoformat()
            })
            
        # Calculate statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.passed)
        failed_tests = total_tests - passed_tests
        
        # Risk assessment
        risk_score = (
            len(findings_by_severity[VulnerabilityLevel.CRITICAL]) * 10 +
            len(findings_by_severity[VulnerabilityLevel.HIGH]) * 5 +
            len(findings_by_severity[VulnerabilityLevel.MEDIUM]) * 2 +
            len(findings_by_severity[VulnerabilityLevel.LOW]) * 1
        )
        
        if risk_score == 0:
            risk_level = "LOW"
        elif risk_score < 10:
            risk_level = "MEDIUM"
        elif risk_score < 30:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"
            
        report = {
            'report_metadata': {
                'title': 'MCP Automation Security Audit Report',
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0',
                'auditor': 'SecurityAuditor',
            },
            'executive_summary': {
                'risk_level': risk_level,
                'risk_score': risk_score,
                'total_findings': len(self.findings),
                'critical_findings': len(findings_by_severity[VulnerabilityLevel.CRITICAL]),
                'high_findings': len(findings_by_severity[VulnerabilityLevel.HIGH]),
                'tests_passed': passed_tests,
                'tests_failed': failed_tests,
                'compliance_status': 'COMPLIANT' if risk_level in ['LOW', 'MEDIUM'] else 'NON-COMPLIANT'
            },
            'test_results': [
                {
                    'name': r.test_name,
                    'category': r.category.value,
                    'passed': r.passed,
                    'message': r.message,
                    'duration': r.duration
                }
                for r in self.test_results
            ],
            'vulnerabilities': {
                'critical': findings_by_severity[VulnerabilityLevel.CRITICAL],
                'high': findings_by_severity[VulnerabilityLevel.HIGH],
                'medium': findings_by_severity[VulnerabilityLevel.MEDIUM],
                'low': findings_by_severity[VulnerabilityLevel.LOW],
                'info': findings_by_severity[VulnerabilityLevel.INFO],
            },
            'recommendations': {
                'immediate_actions': [
                    f"Fix {len(findings_by_severity[VulnerabilityLevel.CRITICAL])} critical vulnerabilities",
                    f"Address {len(findings_by_severity[VulnerabilityLevel.HIGH])} high-risk issues",
                    "Implement authentication and authorization middleware",
                    "Enable comprehensive input validation",
                ] if findings_by_severity[VulnerabilityLevel.CRITICAL] else [],
                'short_term': [
                    "Implement rate limiting and DDoS protection",
                    "Enable security headers (CSP, HSTS, etc.)",
                    "Set up security monitoring and alerting",
                    "Conduct regular security assessments",
                ],
                'long_term': [
                    "Implement zero-trust architecture",
                    "Deploy Web Application Firewall (WAF)",
                    "Establish security training program",
                    "Implement automated security testing in CI/CD",
                ]
            }
        }
        
        return report
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all security tests"""
        logger.info("Starting comprehensive security audit...")
        
        # Run all test categories
        test_methods = [
            self.test_authentication(),
            self.test_authorization(),
            self.test_injection_attacks(),
            self.test_xss_vulnerabilities(),
            self.test_input_validation(),
            self.test_information_disclosure(),
            self.test_access_control(),
            self.test_error_handling(),
            self.test_configuration_security(),
            self.test_mcp_protection(),
        ]
        
        results = await asyncio.gather(*test_methods, return_exceptions=True)
        
        for result in results:
            if isinstance(result, TestResult):
                self.add_result(result)
            elif isinstance(result, Exception):
                logger.error(f"Test execution error: {result}")
                
        # Generate report
        report = self.generate_report()
        
        # Save report to file
        report_file = Path('security_audit_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Security audit complete. Report saved to {report_file}")
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("SECURITY AUDIT SUMMARY")
        logger.info("="*80)
        logger.info(f"Risk Level: {report['executive_summary']['risk_level']}")
        logger.info(f"Total Findings: {report['executive_summary']['total_findings']}")
        logger.error(f"  - Critical: {report['executive_summary']['critical_findings']}")
        logger.info(f"  - High: {report['executive_summary']['high_findings']}")
        logger.info(f"Tests Passed: {report['executive_summary']['tests_passed']}/{len(self.test_results)}")
        logger.info(f"Compliance: {report['executive_summary']['compliance_status']}")
        logger.info("="*80)
        
        return report


async def main():
    """Main execution function"""
    auditor = SecurityAuditor(base_url="http://localhost:10250")
    
    async with auditor:
        report = await auditor.run_all_tests()
        
        # Return non-zero exit code if critical issues found
        if report['executive_summary']['critical_findings'] > 0:
            sys.exit(1)
        elif report['executive_summary']['risk_level'] == 'CRITICAL':
            sys.exit(2)
            
    return report


if __name__ == "__main__":
    asyncio.run(main())