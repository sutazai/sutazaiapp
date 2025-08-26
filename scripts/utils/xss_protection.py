"""
XSS Protection Utilities for SutazAI Frontend
Centralized XSS prevention and input sanitization for Streamlit components
"""

import re
import html
import logging
from typing import Optional, Dict, Any, List, Union
import urllib.parse

logger = logging.getLogger(__name__)

class XSSProtection:
    """Centralized XSS protection for frontend components"""
    
    def __init__(self):
        # XSS patterns to detect and block
        self.xss_patterns = [
            r'<script[^>]*>.*?</script>',
            r'<iframe[^>]*>.*?</iframe>',
            r'<object[^>]*>.*?</object>',
            r'<embed[^>]*>.*?</embed>',
            r'<svg[^>]*>.*?</svg>',
            r'<style[^>]*>.*?</style>',
            r'javascript:',
            r'vbscript:',
            r'data:.*base64',
            r'on\w+\s*=',
            r'expression\s*\(',
            r'url\s*\(',
            r'alert\s*\(',
            r'eval\s*\(',
            r'document\.',
            r'window\.',
            r'location\.',
            r'innerHTML',
            r'outerHTML',
        ]
        
        # Safe HTML tags for limited markup (if needed)
        self.safe_tags = ['b', 'i', 'em', 'strong', 'u', 'br', 'p', 'span']
        
    def sanitize_string(self, input_str: Optional[str], max_length: int = 10000) -> str:
        """
        Sanitize string input for safe HTML rendering
        
        Args:
            input_str: Input string to sanitize
            max_length: Maximum allowed string length
            
        Returns:
            Sanitized string safe for HTML rendering
            
        Raises:
            ValueError: If input contains XSS patterns
        """
        if not input_str:
            return ""
            
        # Convert to string and normalize
        sanitized = str(input_str).strip()
        
        # Check length limit
        if len(sanitized) > max_length:
            logger.warning(f"Input truncated from {len(sanitized)} to {max_length} characters")
            sanitized = sanitized[:max_length] + "..."
            
        # Check for XSS patterns
        for pattern in self.xss_patterns:
            if re.search(pattern, sanitized, re.IGNORECASE | re.DOTALL):
                logger.warning(f"XSS pattern detected and blocked: {pattern}")
                raise ValueError(f"Input contains potentially malicious content: {pattern}")
        
        # HTML escape all content
        sanitized = html.escape(sanitized, quote=True)
        
        # Additional encoding for extra safety
        sanitized = self._deep_encode(sanitized)
        
        return sanitized
        
    def _deep_encode(self, text: str) -> str:
        """Apply additional encoding for maximum safety"""
        # Encode potentially dangerous characters
        replacements = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#x27;',
            '/': '&#x2F;',
            '\\': '&#x5C;',
            '`': '&#x60;',
            '=': '&#x3D;'
        }
        
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
            
        return text
    
    def sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively sanitize dictionary data
        
        Args:
            data: Dictionary to sanitize
            
        Returns:
            Sanitized dictionary
        """
        sanitized = {}
        
        for key, value in data.items():
            # Sanitize keys
            safe_key = self.sanitize_string(str(key), max_length=100)
            
            # Sanitize values based on type
            if isinstance(value, str):
                try:
                    safe_value = self.sanitize_string(value)
                except ValueError:
                    safe_value = "[Content blocked for security]"
            elif isinstance(value, dict):
                safe_value = self.sanitize_dict(value)
            elif isinstance(value, list):
                safe_value = self.sanitize_list(value)
            else:
                safe_value = value
                
            sanitized[safe_key] = safe_value
            
        return sanitized
    
    def sanitize_list(self, data: List[Any]) -> List[Any]:
        """
        Sanitize list data
        
        Args:
            data: List to sanitize
            
        Returns:
            Sanitized list
        """
        sanitized = []
        
        for item in data:
            if isinstance(item, str):
                try:
                    safe_item = self.sanitize_string(item)
                except ValueError:
                    safe_item = "[Content blocked for security]"
            elif isinstance(item, dict):
                safe_item = self.sanitize_dict(item)
            elif isinstance(item, list):
                safe_item = self.sanitize_list(item)
            else:
                safe_item = item
                
            sanitized.append(safe_item)
            
        return sanitized
    
    def safe_html_render(self, content: str, allow_basic_formatting: bool = False) -> str:
        """
        Safely render HTML content with optional basic formatting
        
        Args:
            content: Content to render
            allow_basic_formatting: Whether to allow basic HTML tags
            
        Returns:
            Safe HTML content
        """
        if not content:
            return ""
            
        # First sanitize the content
        try:
            safe_content = self.sanitize_string(content)
        except ValueError:
            return "[Content blocked for security]"
        
        # If basic formatting is allowed, restore safe tags
        if allow_basic_formatting:
            # Only allow very basic tags
            for tag in self.safe_tags:
                safe_content = safe_content.replace(f'&lt;{tag}&gt;', f'<{tag}>')
                safe_content = safe_content.replace(f'&lt;/{tag}&gt;', f'</{tag}>')
                
        return safe_content
    
    def validate_url(self, url: str) -> bool:
        """
        Validate URL for safety
        
        Args:
            url: URL to validate
            
        Returns:
            True if URL is safe
        """
        if not url:
            return False
            
        # Check for dangerous protocols
        dangerous_protocols = [
            'javascript:', 'vbscript:', 'data:', 'file:', 'ftp:'
        ]
        
        url_lower = url.lower().strip()
        
        for protocol in dangerous_protocols:
            if url_lower.startswith(protocol):
                return False
                
        # Only allow http and https
        if not (url_lower.startswith('http://') or url_lower.startswith('https://')):
            return False
            
        try:
            parsed = urllib.parse.urlparse(url)
            # Basic validation
            if not parsed.netloc:
                return False
        except Exception:
            return False
            
        return True


# Global instance
xss_protection = XSSProtection()

# Convenience functions
def safe_render(content: str, allow_formatting: bool = False) -> str:
    """Safely render content for HTML display"""
    return xss_protection.safe_html_render(content, allow_formatting)

def sanitize_user_input(input_str: str) -> str:
    """Sanitize user input for safe processing"""
    return xss_protection.sanitize_string(input_str)

def sanitize_api_response(data: Union[Dict, List, str]) -> Union[Dict, List, str]:
    """Sanitize API response data"""
    if isinstance(data, dict):
        return xss_protection.sanitize_dict(data)
    elif isinstance(data, list):
        return xss_protection.sanitize_list(data)
    elif isinstance(data, str):
        try:
            return xss_protection.sanitize_string(data)
        except ValueError:
            return "[Content blocked for security]"
    return data

def is_safe_url(url: str) -> bool:
    """Check if URL is safe to use"""
    return xss_protection.validate_url(url)