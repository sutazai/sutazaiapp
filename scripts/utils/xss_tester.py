"""
import logging

logger = logging.getLogger(__name__)
Simple XSS Protection Tester
Tests the core XSS protection functionality without external dependencies
"""

import re
import html
import json
class SimpleXSSValidator:
    """Simplified XSS validator for testing"""
    
    def __init__(self):
        self.max_input_length = 10000
        
        # XSS patterns to detect
        self.xss_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'alert\s*\(',
            r'eval\s*\(',
            r'document\.cookie',
            r'document\.write',
            r'window\.location',
            r'innerHTML',
            r'<iframe[^>]*>',
            r'<object[^>]*>',
            r'<embed[^>]*>',
            r'<svg[^>]*>.*?</svg>',
            r'<img[^>]+onerror[^>]*>',
            r'<style[^>]*>.*?</style>',
            r'expression\s*\(',
            r'vbscript:',
            r'data:.*base64',
        ]
        
        # Suspicious patterns in chat messages
        self.suspicious_patterns = [
            r'<.*?>',  # Any HTML tags
            r'javascript:',
            r'data:',
            r'vbscript:',
            r'file://',
            r'\\x[0-9a-fA-F]{2}',  # Hex encoded characters
            r'\\u[0-9a-fA-F]{4}',  # Unicode escapes
        ]
    
    def validate_input(self, input_data: str, input_type: str = "text") -> str:
        """Validate and sanitize input with XSS protection"""
        if not input_data:
            return input_data
            
        # Check length
        if len(input_data) > self.max_input_length:
            raise ValueError(f"Input exceeds maximum length of {self.max_input_length}")
            
        # Check for XSS patterns
        for pattern in self.xss_patterns:
            if re.search(pattern, input_data, re.IGNORECASE | re.DOTALL):
                raise ValueError(f"Cross-site scripting (XSS) content detected: {pattern}")
        
        # Chat message specific validation
        if input_type == "chat_message":
            for pattern in self.suspicious_patterns:
                if re.search(pattern, input_data, re.IGNORECASE):
                    # Remove suspicious content instead of raising error
                    input_data = re.sub(pattern, '', input_data, flags=re.IGNORECASE)
        
        # Sanitize the input
        sanitized = self._sanitize_input(input_data, input_type)
        return sanitized
        
    def _sanitize_input(self, input_data: str, input_type: str) -> str:
        """Sanitize input with HTML escaping"""
        # Remove null bytes and control characters
        sanitized = input_data.replace('\x00', '')
        sanitized = ''.join(char for char in sanitized if ord(char) >= 32 or char in '\n\r\t')
        
        # HTML escape dangerous characters
        sanitized = html.escape(sanitized, quote=True)
        
        # Additional encoding for extra safety
        sanitized = sanitized.replace('&', '&amp;')
        sanitized = sanitized.replace('<', '&lt;')
        sanitized = sanitized.replace('>', '&gt;')
        sanitized = sanitized.replace('"', '&quot;')
        sanitized = sanitized.replace("'", '&#x27;')
        sanitized = sanitized.replace('/', '&#x2F;')
        
        return sanitized.strip()
    
    def sanitize_json_response(self, data: Any) -> Any:
        """Sanitize data before JSON serialization"""
        if isinstance(data, str):
            try:
                return self.validate_input(data, "text")
            except ValueError:
                return "[Content filtered for security]"
        elif isinstance(data, dict):
            return {key: self.sanitize_json_response(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.sanitize_json_response(item) for item in data]
        else:
            return data

def test_xss_protection():
    """Test the XSS protection functionality"""
    
    validator = SimpleXSSValidator()
    
    # XSS payloads that should be blocked
    xss_payloads = [
        '<script>alert("XSS")</script>',
        '<ScRiPt>alert("XSS")</ScRiPt>',
        '<img src=x onerror=alert("XSS")>',
        '<svg onload=alert("XSS")>',
        'javascript:alert("XSS")',
        '<iframe onload=alert("XSS")>',
        '<div onmouseover=alert("XSS")>',
        '<style>@import "javascript:alert(\'XSS\')";</style>',
        '<object data="data:text/html,<script>alert(\'XSS\')</script>"></object>',
        'eval("alert(\'XSS\')")',
        'document.cookie',
        'window.location="evil.com"',
    ]
    
    # Safe messages that should pass
    safe_messages = [
        "Hello, how are you?",
        "What is the weather like today?",
        "Can you help me with coding?",
        "I need assistance with Python programming.",
        "Tell me about machine learning.",
        "How do I deploy a web application?",
        "What are the best practices for security?",
        "Can you explain RESTful APIs?",
    ]
    
    logger.info("=== XSS Protection Test Results ===\n")
    
    # Test XSS payload blocking
    logger.info("1. Testing XSS Payload Detection:")
    xss_blocked = 0
    for i, payload in enumerate(xss_payloads, 1):
        try:
            result = validator.validate_input(payload, "chat_message")
            logger.info(f"   {i:2d}. FAIL: Payload was not blocked: {payload[:40]}...")
        except ValueError as e:
            logger.info(f"   {i:2d}. PASS: Payload blocked - {str(e)[:60]}...")
            xss_blocked += 1
    
    logger.info(f"\n   Result: {xss_blocked}/{len(xss_payloads)} XSS payloads blocked")
    
    # Test safe message processing
    logger.info("\n2. Testing Safe Message Processing:")
    safe_passed = 0
    for i, message in enumerate(safe_messages, 1):
        try:
            result = validator.validate_input(message, "chat_message")
            logger.info(f"   {i:2d}. PASS: Safe message processed: {message[:40]}...")
            safe_passed += 1
        except ValueError as e:
            logger.info(f"   {i:2d}. FAIL: Safe message blocked: {message[:40]}... - {e}")
    
    logger.info(f"\n   Result: {safe_passed}/{len(safe_messages)} safe messages processed")
    
    # Test HTML escaping
    logger.info("\n3. Testing HTML Escaping:")
    test_cases = [
        ("Hello <world>", "Hello &lt;world&gt;"),
        ("Test & example", "Test &amp; example"),
        ('Quote "test"', "Quote &quot;test&quot;"),
        ("Apostrophe's test", "Apostrophe&#x27;s test"),
        ("Forward/slash", "Forward&#x2F;slash"),
    ]
    
    escaping_passed = 0
    for i, (input_text, expected_pattern) in enumerate(test_cases, 1):
        try:
            result = validator.validate_input(input_text, "text")
            if "&lt;" in result or "&amp;" in result or "&quot;" in result:
                logger.info(f"   {i}. PASS: '{input_text}' -> '{result}'")
                escaping_passed += 1
            else:
                logger.info(f"   {i}. FAIL: '{input_text}' -> '{result}' (not properly escaped)")
        except ValueError as e:
            logger.info(f"   {i}. FAIL: '{input_text}' was blocked: {e}")
    
    logger.info(f"\n   Result: {escaping_passed}/{len(test_cases)} escaping tests passed")
    
    # Test JSON sanitization
    logger.info("\n4. Testing JSON Response Sanitization:")
    test_data = {
        "message": "<script>alert('XSS')</script>",
        "user": {
            "name": "<img src=x onerror=alert('XSS')>",
            "comments": [
                "Safe comment",
                "<svg onload=alert('XSS')>"
            ]
        },
        "safe_field": "This is perfectly safe content"
    }
    
    try:
        sanitized = validator.sanitize_json_response(test_data)
        json_str = json.dumps(sanitized, indent=2)
        
        # Check that dangerous content is removed/escaped
        dangerous_found = any(pattern in json_str.lower() for pattern in [
            '<script', 'onerror', 'onload', 'alert('
        ])
        
        if not dangerous_found:
            logger.info("   PASS: JSON sanitization removed all dangerous content")
            logger.info(f"   Sanitized JSON: {json_str}")
        else:
            logger.info("   FAIL: JSON sanitization did not remove all dangerous content")
            logger.info(f"   Result: {json_str}")
            
    except Exception as e:
        logger.error(f"   FAIL: JSON sanitization failed: {e}")
    
    # Overall results
    logger.info("\n=== Summary ===")
    total_tests = len(xss_payloads) + len(safe_messages) + len(test_cases) + 1
    passed_tests = xss_blocked + safe_passed + escaping_passed + (1 if not dangerous_found else 0)
    
    logger.info(f"Total tests: {total_tests}")
    logger.info(f"Passed tests: {passed_tests}")
    logger.info(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        logger.info("\n🛡️  XSS PROTECTION IS WORKING CORRECTLY!")
    elif passed_tests >= total_tests * 0.8:
        logger.info("\n⚠️  XSS PROTECTION IS MOSTLY WORKING (some issues detected)")
    else:
        logger.info("\n❌ XSS PROTECTION HAS SIGNIFICANT ISSUES")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    test_xss_protection()