"""
import logging

logger = logging.getLogger(__name__)
Comprehensive XSS Protection Test Suite for SutazAI
Tests all aspects of XSS protection including frontend, backend, and API security
"""

import pytest
import sys
import os
import json
import requests
import time
from typing import List, Dict, Any

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'frontend'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

# Import XSS protection modules
from frontend.utils.xss_protection import XSSProtection, xss_protection
from frontend.utils.secure_components import SecureStreamlit, get_csp_header
from app.utils.validation import validate_model_name, sanitize_string_input
from app.core.xss_tester import SimpleXSSValidator

class TestXSSProtection:
    """Comprehensive XSS protection test suite"""
    
    def setup_method(self):
        """Set up test environment"""
        self.xss_protection = XSSProtection()
        self.xss_validator = SimpleXSSValidator()
        
        # XSS payloads for testing
        self.xss_payloads = [
            # Script injection
            '<script>alert("XSS")</script>',
            '<ScRiPt>alert("XSS")</ScRiPt>',
            '<script type="text/javascript">alert("XSS")</script>',
            
            # Event handlers
            '<img src=x onerror=alert("XSS")>',
            '<div onmouseover=alert("XSS")>',
            '<body onload=alert("XSS")>',
            '<input onfocus=alert("XSS")>',
            
            # JavaScript URLs
            'javascript:alert("XSS")',
            'JAVASCRIPT:alert("XSS")',
            'javascript://alert("XSS")',
            
            # Data URLs
            'data:text/html,<script>alert("XSS")</script>',
            'data:text/html;base64,PHNjcmlwdD5hbGVydCgiWFNTIik8L3NjcmlwdD4=',
            
            # HTML entities
            '&lt;script&gt;alert("XSS")&lt;/script&gt;',
            '&#60;script&#62;alert("XSS")&#60;/script&#62;',
            
            # SVG attacks
            '<svg onload=alert("XSS")>',
            '<svg><script>alert("XSS")</script></svg>',
            
            # Style attacks
            '<style>@import "javascript:alert(\'XSS\')";</style>',
            '<div style="background:url(javascript:alert(\'XSS\'))">',
            
            # Object/embed attacks
            '<object data="javascript:alert(\'XSS\')">',
            '<embed src="javascript:alert(\'XSS\')">',
            
            # Form attacks
            '<form action="javascript:alert(\'XSS\')">',
            '<input type="image" src="javascript:alert(\'XSS\')">',
            
            # Advanced attacks
            '<iframe onload=alert("XSS")>',
            '<meta http-equiv="refresh" content="0;url=javascript:alert(\'XSS\')">',
            '<link rel="stylesheet" href="javascript:alert(\'XSS\')">',
            
            # Encoding attacks
            '%3Cscript%3Ealert(%22XSS%22)%3C/script%3E',
            '\\x3Cscript\\x3Ealert(\\x22XSS\\x22)\\x3C/script\\x3E',
            '\\u003Cscript\\u003Ealert(\\u0022XSS\\u0022)\\u003C/script\\u003E',
            
            # Mixed case and whitespace
            '<ScRiPt   >alert("XSS")</ScRiPt   >',
            '<script\n>alert("XSS")</script>',
            '<script\t>alert("XSS")</script>',
        ]
        
        # Safe content that should pass
        self.safe_content = [
            "Hello world",
            "This is a normal message",
            "Email: user@example.com",
            "Price: $99.99",
            "Date: 2025-01-01",
            "Normal <brackets> content",
            "Math: 2 + 2 = 4",
        ]
    
    def test_frontend_xss_protection(self):
        """Test frontend XSS protection"""
        logger.info("\n=== Frontend XSS Protection Tests ===")
        
        blocked_count = 0
        
        for i, payload in enumerate(self.xss_payloads, 1):
            try:
                result = self.xss_protection.sanitize_string(payload)
                logger.info(f"   {i:2d}. FAIL: XSS payload was not blocked: {payload[:50]}...")
            except ValueError as e:
                logger.info(f"   {i:2d}. PASS: XSS payload blocked - {str(e)[:60]}...")
                blocked_count += 1
        
        logger.info(f"\n   Frontend Result: {blocked_count}/{len(self.xss_payloads)} XSS payloads blocked")
        
        # Test safe content
        safe_passed = 0
        for content in self.safe_content:
            try:
                result = self.xss_protection.sanitize_string(content)
                safe_passed += 1
            except ValueError:
                logger.info(f"   FAIL: Safe content blocked: {content}")
        
        logger.info(f"   Safe Content: {safe_passed}/{len(self.safe_content)} passed")
        
        # Frontend should block all XSS payloads
        assert blocked_count == len(self.xss_payloads), f"Frontend only blocked {blocked_count}/{len(self.xss_payloads)} XSS payloads"
        assert safe_passed == len(self.safe_content), f"Frontend blocked {len(self.safe_content) - safe_passed} safe contents"
    
    def test_backend_xss_protection(self):
        """Test backend XSS protection"""
        logger.info("\n=== Backend XSS Protection Tests ===")
        
        blocked_count = 0
        
        for i, payload in enumerate(self.xss_payloads, 1):
            try:
                result = self.xss_validator.validate_input(payload, "text")
                logger.info(f"   {i:2d}. FAIL: XSS payload was not blocked: {payload[:50]}...")
            except ValueError as e:
                logger.info(f"   {i:2d}. PASS: XSS payload blocked - {str(e)[:60]}...")
                blocked_count += 1
        
        logger.info(f"\n   Backend Result: {blocked_count}/{len(self.xss_payloads)} XSS payloads blocked")
        
        # Backend should block most XSS payloads
        assert blocked_count >= len(self.xss_payloads) * 0.9, f"Backend only blocked {blocked_count}/{len(self.xss_payloads)} XSS payloads"
    
    def test_model_name_validation(self):
        """Test model name validation against injection attacks"""
        logger.info("\n=== Model Name Validation Tests ===")
        
        malicious_model_names = [
            "../../../etc/passwd",
            "model; rm -rf /",
            "model && curl evil.com",
            "model | cat /etc/passwd",
            "model`whoami`",
            "model$(id)",
            "javascript:alert('XSS')",
            "data:text/html,<script>alert('XSS')</script>",
            "<script>alert('XSS')</script>",
            "model\\x41\\x42",  # Hex encoding
            "model\\u0041\\u0042",  # Unicode encoding
        ]
        
        blocked_count = 0
        
        for i, malicious_name in enumerate(malicious_model_names, 1):
            try:
                result = validate_model_name(malicious_name)
                logger.info(f"   {i:2d}. FAIL: Malicious model name allowed: {malicious_name}")
            except ValueError as e:
                logger.info(f"   {i:2d}. PASS: Malicious model name blocked - {str(e)[:50]}...")
                blocked_count += 1
        
        logger.info(f"\n   Model Validation Result: {blocked_count}/{len(malicious_model_names)} malicious names blocked")
        
        # Test valid model names
        valid_models = ["tinyllama", "llama2:7b", "codellama", "mistral:7b"]
        valid_passed = 0
        
        for model in valid_models:
            try:
                result = validate_model_name(model)
                if result == model or result is None:  # None is valid (uses default)
                    valid_passed += 1
                    logger.info(f"   PASS: Valid model name: {model}")
                else:
                    logger.info(f"   FAIL: Valid model name rejected: {model}")
            except ValueError:
                logger.info(f"   FAIL: Valid model name rejected: {model}")
        
        logger.info(f"   Valid Models: {valid_passed}/{len(valid_models)} passed")
        
        assert blocked_count == len(malicious_model_names), f"Model validation only blocked {blocked_count}/{len(malicious_model_names)} malicious names"
        assert valid_passed == len(valid_models), f"Model validation rejected {len(valid_models) - valid_passed} valid models"
    
    def test_csp_header_generation(self):
        """Test Content Security Policy header generation"""
        logger.info("\n=== CSP Header Tests ===")
        
        csp_header = get_csp_header()
        
        # Check for required CSP directives
        required_directives = [
            "default-src 'self'",
            "object-src 'none'",
            "base-uri 'self'",
            "form-action 'self'",
            "frame-ancestors 'none'",
        ]
        
        passed_directives = 0
        for directive in required_directives:
            if directive in csp_header:
                logger.info(f"   PASS: CSP directive present: {directive}")
                passed_directives += 1
            else:
                logger.info(f"   FAIL: CSP directive missing: {directive}")
        
        logger.info(f"\n   CSP Result: {passed_directives}/{len(required_directives)} required directives present")
        logger.info(f"   Full CSP: {csp_header}")
        
        assert passed_directives == len(required_directives), f"CSP missing {len(required_directives) - passed_directives} required directives"
    
    def test_json_sanitization(self):
        """Test JSON response sanitization"""
        logger.info("\n=== JSON Sanitization Tests ===")
        
        malicious_json = {
            "message": "<script>alert('XSS')</script>",
            "user": {
                "name": "<img src=x onerror=alert('XSS')>",
                "comments": [
                    "Safe comment",
                    "<svg onload=alert('XSS')>",
                    "javascript:alert('XSS')"
                ]
            },
            "data": "<iframe onload=alert('XSS')>",
            "safe_field": "This is perfectly safe content"
        }
        
        try:
            sanitized = self.xss_protection.sanitize_dict(malicious_json)
            
            # Check that dangerous content is removed/escaped
            json_str = json.dumps(sanitized, indent=2)
            
            dangerous_patterns = [
                '<script', 'onerror', 'onload', 'alert(', 'javascript:', '<iframe', '<svg'
            ]
            
            dangerous_found = any(pattern in json_str.lower() for pattern in dangerous_patterns)
            
            if not dangerous_found:
                logger.info("   PASS: JSON sanitization removed all dangerous content")
                logger.info(f"   Sanitized JSON: {json_str}")
            else:
                logger.info("   FAIL: JSON sanitization did not remove all dangerous content")
                logger.info(f"   Result: {json_str}")
                
        except Exception as e:
            logger.error(f"   FAIL: JSON sanitization failed: {e}")
            dangerous_found = True
        
        assert not dangerous_found, "JSON sanitization failed to remove dangerous content"
    
    def test_html_encoding(self):
        """Test HTML encoding functionality"""
        logger.info("\n=== HTML Encoding Tests ===")
        
        test_cases = [
            ("Hello <world>", "&lt;", "&gt;"),
            ("Test & example", "&amp;"),
            ('Quote "test"', "&quot;"),
            ("Apostrophe's test", "&#x27;"),
            ("Forward/slash", "&#x2F;"),
            ("Backslash\\test", "&#x5C;"),
            ("Equal=sign", "&#x3D;"),
        ]
        
        encoding_passed = 0
        
        for i, test_case in enumerate(test_cases, 1):
            input_text = test_case[0]
            expected_encodings = test_case[1:]
            
            try:
                result = self.xss_protection.sanitize_string(input_text)
                
                # Check if all expected encodings are present
                all_encoded = all(encoding in result for encoding in expected_encodings)
                
                if all_encoded:
                    logger.info(f"   {i}. PASS: '{input_text}' -> '{result}'")
                    encoding_passed += 1
                else:
                    logger.info(f"   {i}. FAIL: '{input_text}' -> '{result}' (not properly encoded)")
                    
            except ValueError as e:
                logger.info(f"   {i}. FAIL: '{input_text}' was blocked: {e}")
        
        logger.info(f"\n   Encoding Result: {encoding_passed}/{len(test_cases)} encoding tests passed")
        
        assert encoding_passed == len(test_cases), f"HTML encoding failed {len(test_cases) - encoding_passed} tests"
    
    def test_url_validation(self):
        """Test URL validation for security"""
        logger.info("\n=== URL Validation Tests ===")
        
        malicious_urls = [
            "javascript:alert('XSS')",
            "vbscript:alert('XSS')",
            "data:text/html,<script>alert('XSS')</script>",
            "file:///etc/passwd",
            "ftp://example.com/evil.exe",
        ]
        
        safe_urls = [
            "https://example.com",
            "http://localhost:3000",
            "https://sutazai.com/docs",
        ]
        
        malicious_blocked = 0
        for url in malicious_urls:
            if not self.xss_protection.validate_url(url):
                malicious_blocked += 1
                logger.info(f"   PASS: Malicious URL blocked: {url}")
            else:
                logger.info(f"   FAIL: Malicious URL allowed: {url}")
        
        safe_allowed = 0
        for url in safe_urls:
            if self.xss_protection.validate_url(url):
                safe_allowed += 1
                logger.info(f"   PASS: Safe URL allowed: {url}")
            else:
                logger.info(f"   FAIL: Safe URL blocked: {url}")
        
        logger.info(f"\n   URL Validation Result: {malicious_blocked}/{len(malicious_urls)} malicious URLs blocked")
        logger.info(f"   Safe URLs: {safe_allowed}/{len(safe_urls)} allowed")
        
        assert malicious_blocked == len(malicious_urls), f"URL validation only blocked {malicious_blocked}/{len(malicious_urls)} malicious URLs"
        assert safe_allowed == len(safe_urls), f"URL validation blocked {len(safe_urls) - safe_allowed} safe URLs"
    
    def test_integration_with_backend(self):
        """Test integration between frontend and backend security"""
        logger.info("\n=== Frontend-Backend Integration Tests ===")
        
        # Test that both frontend and backend reject the same XSS payloads
        integration_passed = 0
        
        for payload in self.xss_payloads[:5]:  # Test first 5 payloads
            frontend_blocked = False
            backend_blocked = False
            
            try:
                self.xss_protection.sanitize_string(payload)
            except ValueError:
                frontend_blocked = True
                
            try:
                self.xss_validator.validate_input(payload, "text")
            except ValueError:
                backend_blocked = True
            
            if frontend_blocked and backend_blocked:
                logger.info(f"   PASS: Both frontend and backend blocked: {payload[:30]}...")
                integration_passed += 1
            elif not frontend_blocked and not backend_blocked:
                logger.info(f"   FAIL: Neither frontend nor backend blocked: {payload[:30]}...")
            else:
                logger.info(f"   PARTIAL: Only {'frontend' if frontend_blocked else 'backend'} blocked: {payload[:30]}...")
        
        logger.info(f"\n   Integration Result: {integration_passed}/5 payloads consistently handled")
        
        assert integration_passed >= 4, f"Frontend-Backend integration only consistent for {integration_passed}/5 payloads"

def test_comprehensive_xss_suite():
    """Run comprehensive XSS protection test suite"""
    logger.info("=" * 60)
    logger.info("COMPREHENSIVE XSS PROTECTION TEST SUITE")
    logger.info("=" * 60)
    
    test_instance = TestXSSProtection()
    test_instance.setup_method()
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Frontend XSS Protection", test_instance.test_frontend_xss_protection),
        ("Backend XSS Protection", test_instance.test_backend_xss_protection),
        ("Model Name Validation", test_instance.test_model_name_validation),
        ("CSP Header Generation", test_instance.test_csp_header_generation),
        ("JSON Sanitization", test_instance.test_json_sanitization),
        ("HTML Encoding", test_instance.test_html_encoding),
        ("URL Validation", test_instance.test_url_validation),
        ("Frontend-Backend Integration", test_instance.test_integration_with_backend),
    ]
    
    passed_tests = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            test_results.append((test_name, "PASSED"))
            passed_tests += 1
        except AssertionError as e:
            test_results.append((test_name, f"FAILED: {str(e)}"))
        except Exception as e:
            test_results.append((test_name, f"ERROR: {str(e)}"))
    
    # Print final results
    logger.info("\n" + "=" * 60)
    logger.info("FINAL TEST RESULTS")
    logger.info("=" * 60)
    
    for test_name, result in test_results:
        status = "✅" if result == "PASSED" else "❌"
        logger.info(f"{status} {test_name}: {result}")
    
    logger.info(f"\nOverall Result: {passed_tests}/{len(tests)} tests passed")
    success_rate = (passed_tests / len(tests)) * 100
    logger.info(f"Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 95:
        logger.info("\n🛡️  XSS PROTECTION IS EXCELLENT!")
    elif success_rate >= 85:
        logger.info("\n✅ XSS PROTECTION IS GOOD (minor issues)")
    elif success_rate >= 70:
        logger.info("\n⚠️  XSS PROTECTION NEEDS IMPROVEMENT")
    else:
        logger.error("\n❌ XSS PROTECTION HAS CRITICAL ISSUES")
    
    return success_rate >= 85

if __name__ == "__main__":
    test_comprehensive_xss_suite()