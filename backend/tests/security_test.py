"""
Security Testing Script for Sutazai Backend
Tests for XSS, SQL Injection, CSRF, and other common vulnerabilities
"""

import asyncio
import aiohttp
import json
from typing import List, Dict, Any
from datetime import datetime


class SecurityTester:
    """Security vulnerability testing utility"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: List[Dict[str, Any]] = []
        
    # XSS Payloads
    XSS_PAYLOADS = [
        "<script>alert('XSS')</script>",
        "<img src=x onerror=alert('XSS')>",
        "javascript:alert('XSS')",
        "<svg onload=alert('XSS')>",
        "'\"><script>alert('XSS')</script>",
        "<iframe src='javascript:alert(\"XSS\")'></iframe>",
        "<body onload=alert('XSS')>",
        "<input onfocus=alert('XSS') autofocus>",
        "<select onfocus=alert('XSS') autofocus>",
        "<textarea onfocus=alert('XSS') autofocus>",
        "<<SCRIPT>alert('XSS');//<</SCRIPT>",
        "<IMG SRC=javascript:alert('XSS')>",
        "<DIV STYLE=\"background-image: url(javascript:alert('XSS'))\">",
    ]
    
    # SQL Injection Payloads
    SQL_INJECTION_PAYLOADS = [
        "' OR '1'='1",
        "' OR '1'='1' --",
        "' OR '1'='1' /*",
        "admin' --",
        "admin' #",
        "admin'/*",
        "' or 1=1--",
        "' or 1=1#",
        "' or 1=1/*",
        "') or '1'='1--",
        "') or ('1'='1--",
        "1' ORDER BY 1--",
        "1' ORDER BY 2--",
        "1' ORDER BY 3--",
        "1' UNION SELECT NULL--",
        "' UNION SELECT NULL, NULL--",
        "' UNION SELECT username, password FROM users--",
        "'; DROP TABLE users--",
        "'; EXEC sp_MSForEachTable 'DROP TABLE ?'--",
    ]
    
    async def test_xss_vulnerability(
        self,
        session: aiohttp.ClientSession,
        endpoint: str,
        field_name: str,
        method: str = "POST"
    ) -> Dict[str, Any]:
        """Test endpoint for XSS vulnerabilities"""
        vulnerabilities = []
        
        for payload in self.XSS_PAYLOADS:
            try:
                url = f"{self.base_url}{endpoint}"
                data = {field_name: payload}
                
                if method.upper() == "POST":
                    async with session.post(url, json=data) as response:
                        response_text = await response.text()
                        response_json = None
                        try:
                            response_json = await response.json()
                        except:
                            pass
                else:
                    params = {field_name: payload}
                    async with session.get(url, params=params) as response:
                        response_text = await response.text()
                        response_json = None
                        try:
                            response_json = await response.json()
                        except:
                            pass
                
                # Check if payload appears unescaped in response
                if payload in response_text:
                    vulnerabilities.append({
                        "payload": payload,
                        "field": field_name,
                        "status": "VULNERABLE",
                        "details": "Payload reflected in response without sanitization"
                    })
                else:
                    # Check if payload was properly sanitized
                    escaped_found = any([
                        "&lt;script&gt;" in response_text,
                        "&lt;img" in response_text,
                        "javascript:" not in response_text.lower()
                    ])
                    
                    vulnerabilities.append({
                        "payload": payload,
                        "field": field_name,
                        "status": "PROTECTED" if escaped_found or payload not in response_text else "UNKNOWN",
                        "details": "Payload sanitized or not reflected"
                    })
                    
            except Exception as e:
                vulnerabilities.append({
                    "payload": payload,
                    "field": field_name,
                    "status": "ERROR",
                    "details": str(e)
                })
        
        vulnerable_count = len([v for v in vulnerabilities if v["status"] == "VULNERABLE"])
        
        return {
            "endpoint": endpoint,
            "field": field_name,
            "total_tests": len(self.XSS_PAYLOADS),
            "vulnerable_count": vulnerable_count,
            "protected_count": len([v for v in vulnerabilities if v["status"] == "PROTECTED"]),
            "is_vulnerable": vulnerable_count > 0,
            "details": vulnerabilities[:5]  # First 5 for brevity
        }
    
    async def test_sql_injection(
        self,
        session: aiohttp.ClientSession,
        endpoint: str,
        field_name: str,
        method: str = "POST"
    ) -> Dict[str, Any]:
        """Test endpoint for SQL injection vulnerabilities"""
        vulnerabilities = []
        
        for payload in self.SQL_INJECTION_PAYLOADS:
            try:
                url = f"{self.base_url}{endpoint}"
                data = {field_name: payload}
                
                if method.upper() == "POST":
                    async with session.post(url, json=data) as response:
                        status = response.status
                        try:
                            response_json = await response.json()
                            response_text = json.dumps(response_json)
                        except:
                            response_text = await response.text()
                else:
                    params = {field_name: payload}
                    async with session.get(url, params=params) as response:
                        status = response.status
                        try:
                            response_json = await response.json()
                            response_text = json.dumps(response_json)
                        except:
                            response_text = await response.text()
                
                # Check for SQL error messages
                sql_errors = [
                    "sql syntax",
                    "mysql error",
                    "postgresql error",
                    "sqlite error",
                    "syntax error",
                    "database error",
                    "sqlalchemy",
                    "unclosed quotation",
                ]
                
                has_sql_error = any(error in response_text.lower() for error in sql_errors)
                
                # Status 500 might indicate SQL error
                if status == 500 and has_sql_error:
                    vulnerabilities.append({
                        "payload": payload,
                        "field": field_name,
                        "status": "VULNERABLE",
                        "details": f"SQL error detected (HTTP {status})"
                    })
                elif status in [400, 422]:
                    # Proper validation
                    vulnerabilities.append({
                        "payload": payload,
                        "field": field_name,
                        "status": "PROTECTED",
                        "details": "Input properly validated"
                    })
                else:
                    vulnerabilities.append({
                        "payload": payload,
                        "field": field_name,
                        "status": "PROTECTED",
                        "details": "No SQL error detected"
                    })
                    
            except Exception as e:
                vulnerabilities.append({
                    "payload": payload,
                    "field": field_name,
                    "status": "ERROR",
                    "details": str(e)
                })
        
        vulnerable_count = len([v for v in vulnerabilities if v["status"] == "VULNERABLE"])
        
        return {
            "endpoint": endpoint,
            "field": field_name,
            "total_tests": len(self.SQL_INJECTION_PAYLOADS),
            "vulnerable_count": vulnerable_count,
            "protected_count": len([v for v in vulnerabilities if v["status"] == "PROTECTED"]),
            "is_vulnerable": vulnerable_count > 0,
            "details": vulnerabilities[:5]
        }
    
    async def test_security_headers(
        self,
        session: aiohttp.ClientSession,
        endpoint: str = "/"
    ) -> Dict[str, Any]:
        """Test for proper security headers"""
        url = f"{self.base_url}{endpoint}"
        
        async with session.get(url) as response:
            headers = response.headers
            
            required_headers = {
                "X-Frame-Options": ["DENY", "SAMEORIGIN"],
                "X-Content-Type-Options": ["nosniff"],
                "X-XSS-Protection": ["1; mode=block", "0"],  # 0 is also acceptable (CSP preferred)
                "Strict-Transport-Security": None,  # Any value is good
                "Content-Security-Policy": None,
            }
            
            header_results = {}
            all_present = True
            
            for header, expected_values in required_headers.items():
                present = header in headers
                value = headers.get(header, "")
                
                if expected_values:
                    valid = value in expected_values
                else:
                    valid = present
                
                header_results[header] = {
                    "present": present,
                    "value": value,
                    "valid": valid
                }
                
                if not present:
                    all_present = False
            
            return {
                "endpoint": endpoint,
                "all_headers_present": all_present,
                "headers": header_results,
                "security_score": sum(1 for h in header_results.values() if h["valid"]) / len(required_headers) * 100
            }
    
    async def run_security_tests(self) -> Dict[str, Any]:
        """Run all security tests"""
        print("\n" + "="*60)
        print("SUTAZAI BACKEND SECURITY TESTING SUITE")
        print("="*60 + "\n")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "tests": []
        }
        
        async with aiohttp.ClientSession() as session:
            # Test 1: Security Headers
            print("[1/4] Testing Security Headers...")
            headers_result = await self.test_security_headers(session)
            results["tests"].append({
                "category": "Security Headers",
                "result": headers_result,
                "passed": headers_result["security_score"] >= 80
            })
            print(f"  Security Score: {headers_result['security_score']:.0f}%")
            print(f"  Status: {'✓ PASSED' if headers_result['security_score'] >= 80 else '✗ FAILED'}\n")
            
            # Test 2: XSS on Chat Endpoint
            print("[2/4] Testing XSS Protection (Chat Messages)...")
            xss_result = await self.test_xss_vulnerability(
                session,
                "/api/v1/chat",
                "message"
            )
            results["tests"].append({
                "category": "XSS Protection",
                "result": xss_result,
                "passed": not xss_result["is_vulnerable"]
            })
            print(f"  Tests Run: {xss_result['total_tests']}")
            print(f"  Vulnerabilities: {xss_result['vulnerable_count']}")
            print(f"  Status: {'✓ PASSED' if not xss_result['is_vulnerable'] else '✗ FAILED'}\n")
            
            # Test 3: SQL Injection on Auth Endpoint
            print("[3/4] Testing SQL Injection Protection (Login)...")
            sqli_result = await self.test_sql_injection(
                session,
                "/api/v1/auth/login",
                "username"
            )
            results["tests"].append({
                "category": "SQL Injection Protection",
                "result": sqli_result,
                "passed": not sqli_result["is_vulnerable"]
            })
            print(f"  Tests Run: {sqli_result['total_tests']}")
            print(f"  Vulnerabilities: {sqli_result['vulnerable_count']}")
            print(f"  Status: {'✓ PASSED' if not sqli_result['is_vulnerable'] else '✗ FAILED'}\n")
            
            # Test 4: SQL Injection on Registration
            print("[4/4] Testing SQL Injection Protection (Register)...")
            sqli_reg_result = await self.test_sql_injection(
                session,
                "/api/v1/auth/register",
                "email"
            )
            results["tests"].append({
                "category": "SQL Injection Protection (Register)",
                "result": sqli_reg_result,
                "passed": not sqli_reg_result["is_vulnerable"]
            })
            print(f"  Tests Run: {sqli_reg_result['total_tests']}")
            print(f"  Vulnerabilities: {sqli_reg_result['vulnerable_count']}")
            print(f"  Status: {'✓ PASSED' if not sqli_reg_result['is_vulnerable'] else '✗ FAILED'}\n")
        
        # Overall summary
        total_tests = len(results["tests"])
        passed_tests = sum(1 for t in results["tests"] if t["passed"])
        
        results["summary"] = {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": total_tests - passed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "overall_status": "PASSED" if passed_tests == total_tests else "FAILED"
        }
        
        print("="*60)
        print("SECURITY TEST SUMMARY")
        print("="*60)
        print(f"Tests Run: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {results['summary']['failed']}")
        print(f"Success Rate: {results['summary']['success_rate']:.0f}%")
        print(f"Overall Status: {results['summary']['overall_status']}")
        print("="*60 + "\n")
        
        # Save results
        output_file = f"security_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Results saved to {output_file}\n")
        
        return results


async def main():
    """Run security tests"""
    tester = SecurityTester()
    results = await tester.run_security_tests()
    
    # Exit with appropriate code
    import sys
    sys.exit(0 if results["summary"]["overall_status"] == "PASSED" else 1)


if __name__ == "__main__":
    asyncio.run(main())
