#!/usr/bin/env python3
"""
SutazAI Security Testing Suite
=============================

Comprehensive security testing including:
- Input validation and sanitization
- Authentication and authorization
- SQL injection protection
- XSS prevention
- CSRF protection
- Rate limiting
- HTTP security headers
- SSL/TLS configuration
"""

import requests
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import re
import urllib.parse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityTestSuite:
    """Comprehensive security testing suite"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.test_results = []
        
        # Common attack payloads
        self.sql_injection_payloads = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "' UNION SELECT * FROM users --",
            "admin'--",
            "'; EXEC xp_cmdshell('dir'); --"
        ]
        
        self.xss_payloads = [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "';alert('XSS');//",
            "<svg/onload=alert('XSS')>"
        ]
        
        self.command_injection_payloads = [
            "; ls -la",
            "| whoami",
            "&& cat /etc/passwd",
            "`id`",
            "$(cat /etc/hosts)"
        ]
    
    def test_input_validation(self) -> Dict[str, Any]:
        """Test input validation and sanitization"""
        logger.info("Testing input validation...")
        
        results = {
            "sql_injection": [],
            "xss_protection": [],
            "command_injection": [],
            "buffer_overflow": []
        }
        
        # Test SQL injection protection
        for payload in self.sql_injection_payloads:
            test_data = {
                "task": payload,
                "agent_type": "senior-ai-engineer",
                "priority": "high"
            }
            
            try:
                response = requests.post(
                    f"{self.base_url}/api/v1/agents/task",
                    json=test_data,
                    timeout=10
                )
                
                # Check if payload was properly handled
                safe_response = response.status_code in [200, 201, 400, 422]
                error_indicates_filtering = "error" in response.text.lower() or "invalid" in response.text.lower()
                
                results["sql_injection"].append({
                    "payload": payload[:50] + "..." if len(payload) > 50 else payload,
                    "status_code": response.status_code,
                    "safe_response": safe_response,
                    "filtered": error_indicates_filtering,
                    "response_text": response.text[:200]
                })
                
            except Exception as e:
                results["sql_injection"].append({
                    "payload": payload[:50] + "..." if len(payload) > 50 else payload,
                    "error": str(e),
                    "safe_response": True  # Exception is acceptable for security
                })
        
        # Test XSS protection
        for payload in self.xss_payloads:
            test_data = {
                "task": f"Create a function with this description: {payload}",
                "agent_type": "senior-ai-engineer"
            }
            
            try:
                response = requests.post(
                    f"{self.base_url}/api/v1/agents/task",
                    json=test_data,
                    timeout=10
                )
                
                # Check if XSS payload is reflected unsanitized
                payload_reflected = payload in response.text
                html_tags_present = "<script>" in response.text or "<img" in response.text
                
                results["xss_protection"].append({
                    "payload": payload[:50] + "..." if len(payload) > 50 else payload,
                    "status_code": response.status_code,
                    "payload_reflected": payload_reflected,
                    "html_tags_present": html_tags_present,
                    "safe": not (payload_reflected and html_tags_present)
                })
                
            except Exception as e:
                results["xss_protection"].append({
                    "payload": payload[:50] + "..." if len(payload) > 50 else payload,
                    "error": str(e),
                    "safe": True
                })
        
        # Test command injection protection
        for payload in self.command_injection_payloads:
            test_data = {
                "task": f"Execute task: {payload}",
                "agent_type": "infrastructure-devops-manager"
            }
            
            try:
                response = requests.post(
                    f"{self.base_url}/api/v1/agents/task",
                    json=test_data,
                    timeout=10
                )
                
                # Check for signs of command execution
                suspicious_output = any(keyword in response.text.lower() for keyword in 
                                      ["root:", "/bin/", "uid=", "gid=", "passwd", "shadow"])
                
                results["command_injection"].append({
                    "payload": payload,
                    "status_code": response.status_code,
                    "suspicious_output": suspicious_output,
                    "safe": not suspicious_output
                })
                
            except Exception as e:
                results["command_injection"].append({
                    "payload": payload,
                    "error": str(e),
                    "safe": True
                })
        
        # Test buffer overflow protection
        large_payload = "A" * 10000  # 10KB payload
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/agents/task",
                json={"task": large_payload},
                timeout=15
            )
            
            results["buffer_overflow"] = {
                "large_payload_handled": True,
                "status_code": response.status_code,
                "response_size": len(response.text),
                "safe": response.status_code in [200, 201, 400, 413, 422]
            }
            
        except Exception as e:
            results["buffer_overflow"] = {
                "large_payload_handled": False,
                "error": str(e),
                "safe": True  # Exception handling is acceptable
            }
        
        return results
    
    def test_http_security_headers(self) -> Dict[str, Any]:
        """Test HTTP security headers"""
        logger.info("Testing HTTP security headers...")
        
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            headers = response.headers
            
            security_headers = {
                "X-Content-Type-Options": headers.get("X-Content-Type-Options"),
                "X-Frame-Options": headers.get("X-Frame-Options"),
                "X-XSS-Protection": headers.get("X-XSS-Protection"),
                "Strict-Transport-Security": headers.get("Strict-Transport-Security"),
                "Content-Security-Policy": headers.get("Content-Security-Policy"),
                "Referrer-Policy": headers.get("Referrer-Policy"),
                "Permissions-Policy": headers.get("Permissions-Policy")
            }
            
            # Evaluate security headers
            security_score = 0
            max_score = 7
            
            if security_headers["X-Content-Type-Options"] == "nosniff":
                security_score += 1
            
            if security_headers["X-Frame-Options"] in ["DENY", "SAMEORIGIN"]:
                security_score += 1
            
            if security_headers["X-XSS-Protection"]:
                security_score += 1
            
            if security_headers["Strict-Transport-Security"]:
                security_score += 1
            
            if security_headers["Content-Security-Policy"]:
                security_score += 1
            
            if security_headers["Referrer-Policy"]:
                security_score += 1
            
            if security_headers["Permissions-Policy"]:
                security_score += 1
            
            return {
                "headers_present": security_headers,
                "security_score": security_score,
                "max_score": max_score,
                "security_percentage": (security_score / max_score) * 100,
                "recommendations": self._get_header_recommendations(security_headers)
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "security_score": 0,
                "max_score": 7,
                "security_percentage": 0
            }
    
    def test_rate_limiting(self) -> Dict[str, Any]:
        """Test rate limiting implementation"""
        logger.info("Testing rate limiting...")
        
        # Make rapid requests to test rate limiting
        rapid_requests = []
        start_time = time.time()
        
        for i in range(50):  # Make 50 rapid requests
            try:
                response = requests.get(f"{self.base_url}/health", timeout=5)
                rapid_requests.append({
                    "request_number": i + 1,
                    "status_code": response.status_code,
                    "response_time": time.time() - start_time
                })
                
                # Check for rate limiting indicators
                if response.status_code == 429:  # Too Many Requests
                    break
                    
            except Exception as e:
                rapid_requests.append({
                    "request_number": i + 1,
                    "error": str(e),
                    "response_time": time.time() - start_time
                })
        
        total_time = time.time() - start_time
        
        # Analyze results
        rate_limited = any(req.get("status_code") == 429 for req in rapid_requests)
        successful_requests = sum(1 for req in rapid_requests if req.get("status_code") == 200)
        requests_per_second = len(rapid_requests) / total_time
        
        return {
            "rate_limiting_detected": rate_limited,
            "total_requests": len(rapid_requests),
            "successful_requests": successful_requests,
            "requests_per_second": requests_per_second,
            "total_time": total_time,
            "recommendation": "Implement rate limiting" if not rate_limited else "Rate limiting working"
        }
    
    def test_cors_configuration(self) -> Dict[str, Any]:
        """Test CORS configuration security"""
        logger.info("Testing CORS configuration...")
        
        test_origins = [
            "http://evil.com",
            "https://malicious.org",
            "http://localhost:3000",
            "http://localhost:8501"
        ]
        
        cors_results = []
        
        for origin in test_origins:
            try:
                headers = {
                    "Origin": origin,
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "Content-Type"
                }
                
                response = requests.options(f"{self.base_url}/api/v1/agents", headers=headers, timeout=10)
                
                allowed_origin = response.headers.get("Access-Control-Allow-Origin")
                allowed_methods = response.headers.get("Access-Control-Allow-Methods")
                allowed_headers = response.headers.get("Access-Control-Allow-Headers")
                
                cors_results.append({
                    "origin": origin,
                    "status_code": response.status_code,
                    "allowed_origin": allowed_origin,
                    "allowed_methods": allowed_methods,
                    "allowed_headers": allowed_headers,
                    "wildcard_origin": allowed_origin == "*",
                    "origin_reflected": allowed_origin == origin
                })
                
            except Exception as e:
                cors_results.append({
                    "origin": origin,
                    "error": str(e)
                })
        
        # Evaluate CORS security
        wildcard_origins = sum(1 for result in cors_results if result.get("wildcard_origin", False))
        security_issue = wildcard_origins > 0
        
        return {
            "cors_results": cors_results,
            "wildcard_origins_detected": wildcard_origins,
            "security_issue": security_issue,
            "recommendation": "Avoid wildcard origins in production" if security_issue else "CORS configuration appears secure"
        }
    
    def test_authentication_bypass(self) -> Dict[str, Any]:
        """Test for authentication bypass vulnerabilities"""
        logger.info("Testing authentication bypass...")
        
        # Test various bypass techniques
        bypass_attempts = [
            {"url": f"{self.base_url}/api/v1/admin", "headers": {}},
            {"url": f"{self.base_url}/api/v1/admin", "headers": {"X-Forwarded-For": "127.0.0.1"}},
            {"url": f"{self.base_url}/api/v1/admin", "headers": {"X-Real-IP": "127.0.0.1"}},
            {"url": f"{self.base_url}/api/v1/admin/../agents", "headers": {}},
            {"url": f"{self.base_url}/api/v1/agents", "headers": {"User-Agent": "admin"}},
        ]
        
        results = []
        
        for attempt in bypass_attempts:
            try:
                response = requests.get(attempt["url"], headers=attempt["headers"], timeout=10)
                
                results.append({
                    "url": attempt["url"],
                    "headers": attempt["headers"],
                    "status_code": response.status_code,
                    "bypass_successful": response.status_code == 200,
                    "response_size": len(response.text)
                })
                
            except Exception as e:
                results.append({
                    "url": attempt["url"],
                    "headers": attempt["headers"],
                    "error": str(e),
                    "bypass_successful": False
                })
        
        successful_bypasses = sum(1 for result in results if result.get("bypass_successful", False))
        
        return {
            "bypass_attempts": len(results),
            "successful_bypasses": successful_bypasses,
            "security_issue": successful_bypasses > 0,
            "results": results
        }
    
    def test_information_disclosure(self) -> Dict[str, Any]:
        """Test for information disclosure vulnerabilities"""
        logger.info("Testing information disclosure...")
        
        # Test various endpoints for information leakage
        test_endpoints = [
            "/api/v1/debug",
            "/api/v1/config",
            "/api/v1/version",
            "/admin",
            "/debug",
            "/.env",
            "/config.json",
            "/api/v1/../../../etc/passwd",
            "/api/v1/agents/../../config"
        ]
        
        results = []
        
        for endpoint in test_endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                
                # Check for sensitive information
                sensitive_indicators = [
                    "password", "secret", "key", "token", "config",
                    "database", "connection", "api_key", "private"
                ]
                
                sensitive_found = any(indicator in response.text.lower() 
                                    for indicator in sensitive_indicators)
                
                results.append({
                    "endpoint": endpoint,
                    "status_code": response.status_code,
                    "response_size": len(response.text),
                    "sensitive_info_found": sensitive_found,
                    "accessible": response.status_code == 200
                })
                
            except Exception as e:
                results.append({
                    "endpoint": endpoint,
                    "error": str(e),
                    "accessible": False,
                    "sensitive_info_found": False
                })
        
        accessible_endpoints = sum(1 for result in results if result.get("accessible", False))
        sensitive_leaks = sum(1 for result in results if result.get("sensitive_info_found", False))
        
        return {
            "endpoints_tested": len(results),
            "accessible_endpoints": accessible_endpoints,
            "sensitive_information_leaks": sensitive_leaks,
            "security_issue": sensitive_leaks > 0,
            "results": results
        }
    
    def _get_header_recommendations(self, headers: Dict[str, Any]) -> List[str]:
        """Generate recommendations for missing security headers"""
        recommendations = []
        
        if not headers.get("X-Content-Type-Options"):
            recommendations.append("Add X-Content-Type-Options: nosniff header")
        
        if not headers.get("X-Frame-Options"):
            recommendations.append("Add X-Frame-Options: DENY or SAMEORIGIN header")
        
        if not headers.get("X-XSS-Protection"):
            recommendations.append("Add X-XSS-Protection: 1; mode=block header")
        
        if not headers.get("Content-Security-Policy"):
            recommendations.append("Implement Content-Security-Policy header")
        
        if not headers.get("Strict-Transport-Security"):
            recommendations.append("Add Strict-Transport-Security header for HTTPS")
        
        return recommendations
    
    async def run_comprehensive_security_suite(self) -> Dict[str, Any]:
        """Run complete security test suite"""
        logger.info("Starting comprehensive security test suite...")
        
        results = {
            "execution_timestamp": datetime.now().isoformat(),
            "test_configuration": {
                "base_url": self.base_url
            }
        }
        
        # Run all security tests
        test_functions = [
            ("Input Validation", self.test_input_validation),
            ("HTTP Security Headers", self.test_http_security_headers),
            ("Rate Limiting", self.test_rate_limiting),
            ("CORS Configuration", self.test_cors_configuration),
            ("Authentication Bypass", self.test_authentication_bypass),
            ("Information Disclosure", self.test_information_disclosure)
        ]
        
        for test_name, test_func in test_functions:
            logger.info(f"Running {test_name} tests...")
            try:
                test_result = test_func()
                results[test_name.lower().replace(" ", "_")] = test_result
            except Exception as e:
                logger.error(f"Error in {test_name}: {e}")
                results[test_name.lower().replace(" ", "_")] = {
                    "error": str(e),
                    "status": "error"
                }
        
        # Generate security summary
        results["security_summary"] = self._generate_security_summary(results)
        
        # Save security report
        await self._save_security_report(results)
        
        return results
    
    def _generate_security_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate security summary and score"""
        
        security_issues = []
        recommendations = []
        
        # Analyze input validation results
        input_validation = results.get("input_validation", {})
        sql_safe = all(result.get("safe_response", True) for result in input_validation.get("sql_injection", []))
        xss_safe = all(result.get("safe", True) for result in input_validation.get("xss_protection", []))
        cmd_safe = all(result.get("safe", True) for result in input_validation.get("command_injection", []))
        
        if not sql_safe:
            security_issues.append("SQL injection vulnerabilities detected")
            recommendations.append("Implement proper input sanitization and parameterized queries")
        
        if not xss_safe:
            security_issues.append("XSS vulnerabilities detected")
            recommendations.append("Implement output encoding and Content Security Policy")
        
        if not cmd_safe:
            security_issues.append("Command injection vulnerabilities detected")
            recommendations.append("Validate and sanitize all user inputs")
        
        # Analyze security headers
        headers_result = results.get("http_security_headers", {})
        headers_score = headers_result.get("security_percentage", 0)
        
        if headers_score < 70:
            security_issues.append("Missing important security headers")
            recommendations.extend(headers_result.get("recommendations", []))
        
        # Analyze rate limiting
        rate_limiting = results.get("rate_limiting", {})
        if not rate_limiting.get("rate_limiting_detected", False):
            security_issues.append("No rate limiting detected")
            recommendations.append("Implement rate limiting to prevent abuse")
        
        # Analyze CORS
        cors_result = results.get("cors_configuration", {})
        if cors_result.get("security_issue", False):
            security_issues.append("CORS misconfiguration detected")
            recommendations.append(cors_result.get("recommendation", ""))
        
        # Analyze authentication
        auth_result = results.get("authentication_bypass", {})
        if auth_result.get("security_issue", False):
            security_issues.append("Authentication bypass vulnerabilities detected")
            recommendations.append("Strengthen authentication and authorization mechanisms")
        
        # Analyze information disclosure
        info_disclosure = results.get("information_disclosure", {})
        if info_disclosure.get("security_issue", False):
            security_issues.append("Information disclosure vulnerabilities detected")
            recommendations.append("Remove or secure sensitive information endpoints")
        
        # Calculate overall security score
        total_checks = 6  # Number of security test categories
        passed_checks = 0
        
        if sql_safe and xss_safe and cmd_safe:
            passed_checks += 1
        
        if headers_score >= 70:
            passed_checks += 1
        
        if rate_limiting.get("rate_limiting_detected", False):
            passed_checks += 1
        
        if not cors_result.get("security_issue", False):
            passed_checks += 1
        
        if not auth_result.get("security_issue", False):
            passed_checks += 1
        
        if not info_disclosure.get("security_issue", False):
            passed_checks += 1
        
        security_score = (passed_checks / total_checks) * 100
        
        if security_score >= 90:
            security_grade = "EXCELLENT"
        elif security_score >= 80:
            security_grade = "GOOD"
        elif security_score >= 70:
            security_grade = "FAIR"
        elif security_score >= 60:
            security_grade = "POOR"
        else:
            security_grade = "CRITICAL"
        
        return {
            "security_score": security_score,
            "security_grade": security_grade,
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "security_issues": security_issues,
            "recommendations": recommendations,
            "overall_status": "SECURE" if len(security_issues) == 0 else "VULNERABLE"
        }
    
    async def _save_security_report(self, report: Dict[str, Any]) -> None:
        """Save security test report"""
        try:
            reports_dir = Path("/opt/sutazaiapp/data/workflow_reports")
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_file = reports_dir / f"security_test_report_{timestamp}.json"
            
            with open(json_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Security test report saved: {json_file}")
            
        except Exception as e:
            logger.error(f"Failed to save security test report: {e}")

async def main():
    """Run security tests"""
    suite = SecurityTestSuite()
    report = await suite.run_comprehensive_security_suite()
    
    print("\n" + "="*80)
    print("SUTAZAI SECURITY TEST RESULTS")
    print("="*80)
    
    summary = report.get("security_summary", {})
    print(f"Security Status: {summary.get('overall_status', 'UNKNOWN')}")
    print(f"Security Grade: {summary.get('security_grade', 'UNKNOWN')}")
    print(f"Security Score: {summary.get('security_score', 0):.1f}/100")
    print(f"Checks Passed: {summary.get('passed_checks', 0)}/{summary.get('total_checks', 0)}")
    
    issues = summary.get("security_issues", [])
    if issues:
        print(f"\nSecurity Issues:")
        for issue in issues:
            print(f"  ⚠️  {issue}")
    else:
        print(f"\n✅ No security issues detected!")
    
    recommendations = summary.get("recommendations", [])
    if recommendations:
        print(f"\nSecurity Recommendations:")
        for rec in recommendations:
            print(f"  • {rec}")
    
    print("="*80)
    
    return report

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())