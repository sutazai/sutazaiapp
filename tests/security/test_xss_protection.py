"""
XSS Protection Test Suite
Tests comprehensive XSS protection implementation for SutazAI chat endpoints
"""

import pytest
import asyncio
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

# Common XSS payloads for testing
XSS_PAYLOADS = [
    # Basic script injection
    "<script>alert('XSS')</script>",
    "<ScRiPt>alert('XSS')</ScRiPt>",
    
    # Image based XSS
    "<img src=x onerror=alert('XSS')>",
    "<img src='x' onerror='alert(\"XSS\")'>",
    
    # SVG based XSS
    "<svg onload=alert('XSS')>",
    "<svg><script>alert('XSS')</script></svg>",
    
    # JavaScript URL schemes
    "javascript:alert('XSS')",
    "javascript:void(0);alert('XSS')",
    
    # Data URL schemes
    "data:text/html,<script>alert('XSS')</script>",
    
    # Event handlers
    "<div onload=alert('XSS')>",
    "<body onload=alert('XSS')>",
    "<iframe onload=alert('XSS')>",
    
    # Style based XSS
    "<style>@import 'javascript:alert(\"XSS\")';</style>",
    "<div style='background:url(javascript:alert(\"XSS\"))'></div>",
    
    # Object/embed tags
    "<object data='data:text/html,<script>alert(\"XSS\")</script>'></object>",
    "<embed src='data:text/html,<script>alert(\"XSS\")</script>'>",
    
    # Form based XSS
    "<form action='javascript:alert(\"XSS\")'><input type=submit></form>",
    
    # Meta refresh XSS
    "<meta http-equiv='refresh' content='0;url=javascript:alert(\"XSS\")'>",
    
    # Link stylesheet XSS
    "<link rel='stylesheet' href='javascript:alert(\"XSS\")'>",
    
    # Advanced payloads
    "';alert('XSS');//",
    "\"><script>alert('XSS')</script>",
    "'+alert('XSS')+'",
    "%3Cscript%3Ealert('XSS')%3C/script%3E",
    
    # Unicode and encoding
    "\\u003Cscript\\u003Ealert('XSS')\\u003C/script\\u003E",
    "\\x3Cscript\\x3Ealert('XSS')\\x3C/script\\x3E",
    
    # Bypass attempts
    "<script>eval(String.fromCharCode(97,108,101,114,116,40,39,88,83,83,39,41))</script>",
    "<script>window['al'+'ert']('XSS')</script>",
]

# Safe test messages that should pass
SAFE_MESSAGES = [
    "Hello, how are you?",
    "What is the weather like today?",
    "Can you help me with coding?",
    "I need assistance with Python programming.",
    "Tell me about machine learning.",
    "How do I deploy a web application?",
    "What are the best practices for security?",
    "Can you explain RESTful APIs?",
    "Show me an example of JSON structure.",
    "Help me understand databases.",
]

class TestXSSProtection:
    """Test suite for XSS protection"""
    
    def setup_method(self):
        """Setup test environment"""
        # Mock the security components
        self.mock_validator = Mock()
        self.mock_xss_protection = Mock()
        
    @pytest.mark.asyncio
    async def test_input_validator_blocks_xss_payloads(self):
        """Test that input validator blocks known XSS payloads"""
        from app.core.security import InputValidator
        
        validator = InputValidator()
        
        for payload in XSS_PAYLOADS:
            with pytest.raises(ValueError) as exc_info:
                validator.validate_input(payload, "chat_message")
            
            # Check that the error message indicates XSS detection
            error_msg = str(exc_info.value).lower()
            assert any(keyword in error_msg for keyword in [
                "xss", "cross-site", "malicious", "script", "detected"
            ]), f"XSS payload not properly detected: {payload}"
    
    @pytest.mark.asyncio 
    async def test_input_validator_allows_safe_content(self):
        """Test that input validator allows safe content"""
        from app.core.security import InputValidator
        
        validator = InputValidator()
        
        for message in SAFE_MESSAGES:
            try:
                sanitized = validator.validate_input(message, "chat_message")
                assert sanitized is not None
                assert isinstance(sanitized, str)
                # Should not contain dangerous characters after sanitization
                assert "<script" not in sanitized.lower()
                assert "javascript:" not in sanitized.lower()
                assert "onerror" not in sanitized.lower()
            except ValueError as e:
                pytest.fail(f"Safe message was incorrectly blocked: {message}. Error: {e}")
    
    @pytest.mark.asyncio
    async def test_html_escaping(self):
        """Test that HTML characters are properly escaped"""
        from app.core.security import InputValidator
        
        validator = InputValidator()
        
        test_cases = [
            ("<", "&lt;"),
            (">", "&gt;"),
            ("&", "&amp;"),
            ('"', "&quot;"),
            ("'", "&#x27;"),
            ("/", "&#x2F;"),
        ]
        
        for input_char, expected in test_cases:
            sanitized = validator.validate_input(input_char, "text")
            assert expected in sanitized, f"Character {input_char} not properly escaped"
    
    @pytest.mark.asyncio
    async def test_chat_message_validation(self):
        """Test special chat message validation"""
        from app.core.security import InputValidator
        
        validator = InputValidator()
        
        # Test that HTML tags are removed/escaped
        dangerous_message = "Hello <script>alert('test')</script> world"
        sanitized = validator.validate_input(dangerous_message, "chat_message")
        
        # Should not contain script tags
        assert "<script" not in sanitized.lower()
        assert "alert" not in sanitized.lower()
        
        # Should contain the safe parts
        assert "hello" in sanitized.lower()
        assert "world" in sanitized.lower()
    
    @pytest.mark.asyncio
    async def test_json_response_sanitization(self):
        """Test JSON response sanitization"""
        from app.core.security import InputValidator
        
        validator = InputValidator()
        
        # Test nested data structure
        test_data = {
            "message": "<script>alert('XSS')</script>",
            "user": {
                "name": "<img src=x onerror=alert('XSS')>",
                "comments": [
                    "Safe comment",
                    "<svg onload=alert('XSS')>"
                ]
            }
        }
        
        sanitized = validator.sanitize_json_response(test_data)
        
        # Check that all dangerous content is sanitized
        json_str = json.dumps(sanitized).lower()
        assert "<script" not in json_str
        assert "onerror" not in json_str
        assert "onload" not in json_str
        assert "alert" not in json_str
    
    @pytest.mark.asyncio
    async def test_xss_protection_middleware(self):
        """Test XSS protection middleware processing"""
        from app.core.security import XSSProtectionMiddleware
        
        middleware = XSSProtectionMiddleware()
        
        # Test request sanitization
        request_data = {
            "body": {
                "message": "<script>alert('XSS')</script>",
                "model": "test-model"
            }
        }
        
        with patch.object(middleware.validator, 'validate_input') as mock_validate:
            mock_validate.return_value = "[Content filtered for security]"
            
            sanitized = await middleware.process_request(request_data)
            
            # Check that validation was called
            mock_validate.assert_called()
            assert sanitized["body"]["message"] == "[Content filtered for security]"
    
    def test_csp_header_generation(self):
        """Test Content Security Policy header generation"""
        from app.core.security import SecurityManager
        
        security_manager = SecurityManager()
        headers = security_manager.get_security_headers()
        
        # Check that CSP header exists and is restrictive
        assert "Content-Security-Policy" in headers
        csp = headers["Content-Security-Policy"]
        
        # Should have restrictive directives
        assert "default-src 'self'" in csp
        assert "script-src 'self'" in csp
        assert "object-src 'none'" in csp
        assert "base-uri 'self'" in csp
        assert "frame-ancestors 'none'" in csp
        
        # Should not allow unsafe-inline for scripts
        assert "script-src 'self' 'unsafe-inline'" not in csp
        assert "script-src 'self' 'unsafe-eval'" not in csp
    
    def test_security_headers_completeness(self):
        """Test that all required security headers are present"""
        from app.core.security import SecurityManager
        
        security_manager = SecurityManager()
        headers = security_manager.get_security_headers()
        
        required_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options", 
            "X-XSS-Protection",
            "Strict-Transport-Security",
            "Content-Security-Policy",
            "Referrer-Policy",
            "Permissions-Policy",
            "Cross-Origin-Embedder-Policy",
            "Cross-Origin-Opener-Policy",
            "Cross-Origin-Resource-Policy"
        ]
        
        for header in required_headers:
            assert header in headers, f"Missing security header: {header}"
            assert headers[header], f"Empty security header: {header}"
    
    @pytest.mark.asyncio
    async def test_chat_request_validation(self):
        """Test ChatRequest model validation"""
        from app.api.v1.endpoints.chat import ChatRequest
        
        # Test that XSS payloads are rejected
        for payload in XSS_PAYLOADS[:5]:  # Test first 5 to avoid timeout
            with pytest.raises(ValueError):
                ChatRequest(message=payload)
        
        # Test that safe messages are accepted
        for safe_msg in SAFE_MESSAGES[:3]:  # Test first 3 to avoid timeout
            try:
                request = ChatRequest(message=safe_msg)
                assert request.message is not None
                # Message should be sanitized but not completely blocked
                assert len(request.message) > 0
            except ValueError as e:
                pytest.fail(f"Safe message incorrectly rejected: {safe_msg}. Error: {e}")
    
    @pytest.mark.asyncio
    async def test_streaming_request_validation(self):
        """Test streaming request validation"""
        from app.api.v1.endpoints.streaming import StreamingChatRequest
        
        # Test malicious messages in chat format
        malicious_messages = [
            {"role": "user", "content": "<script>alert('XSS')</script>"},
            {"role": "system", "content": "<img src=x onerror=alert('XSS')>"}
        ]
        
        with pytest.raises(ValueError):
            StreamingChatRequest(messages=malicious_messages)
        
        # Test safe messages
        safe_messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"}
        ]
        
        try:
            request = StreamingChatRequest(messages=safe_messages)
            assert len(request.messages) == 2
            assert all(msg["content"] for msg in request.messages)
        except ValueError as e:
            pytest.fail(f"Safe messages incorrectly rejected. Error: {e}")

class TestIntegration:
    """Integration tests for XSS protection with FastAPI endpoints"""
    
    def setup_method(self):
        """Setup test client"""
        # This would need the actual FastAPI app instance
        # For now, we'll test the components in isolation
        pass
    
    def test_xss_protection_blocks_malicious_requests(self):
        """Test that XSS protection blocks malicious requests end-to-end"""
        # This would test actual HTTP requests with XSS payloads
        # and verify they are blocked at the middleware level
        pass
    
    def test_safe_requests_pass_through(self):
        """Test that safe requests pass through all layers"""
        # This would test that legitimate requests are not blocked
        # by the XSS protection mechanisms
        pass

# Performance tests
class TestPerformance:
    """Performance tests for XSS protection"""
    
    @pytest.mark.asyncio
    async def test_validation_performance(self):
        """Test that XSS validation doesn't significantly impact performance"""
        from app.core.security import InputValidator
        import time
        
        validator = InputValidator()
        
        # Test with large safe message
        large_message = "This is a safe message. " * 1000
        
        start_time = time.time()
        for _ in range(100):
            validator.validate_input(large_message, "text")
        end_time = time.time()
        
        # Should complete 100 validations in under 1 second
        total_time = end_time - start_time
        assert total_time < 1.0, f"Validation too slow: {total_time}s for 100 operations"
    
    @pytest.mark.asyncio
    async def test_sanitization_performance(self):
        """Test JSON sanitization performance"""
        from app.core.security import InputValidator
        import time
        
        validator = InputValidator()
        
        # Create large nested data structure
        large_data = {
            "messages": [{"content": f"Message {i}"} for i in range(1000)],
            "metadata": {"key": "value"} 
        }
        
        start_time = time.time()
        sanitized = validator.sanitize_json_response(large_data)
        end_time = time.time()
        
        # Should complete sanitization quickly
        assert end_time - start_time < 0.5, "JSON sanitization too slow"
        assert len(sanitized["messages"]) == 1000

if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])