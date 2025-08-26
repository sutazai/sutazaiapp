"""
Secure Streamlit Components with Automatic XSS Protection
Wrapper functions for Streamlit components that automatically sanitize content
"""

import streamlit as st
from typing import Optional, Dict, Any, Union, List
import logging

from .xss_protection import xss_protection, safe_render

logger = logging.getLogger(__name__)

class SecureStreamlit:
    """Secure wrappers for Streamlit components with automatic XSS protection"""
    
    @staticmethod
    def markdown(body: str, unsafe_allow_html: bool = False, **kwargs):
        """
        Secure markdown rendering with XSS protection
        
        Args:
            body: Markdown content
            unsafe_allow_html: Whether to allow HTML (always sanitized)
            **kwargs: Additional arguments for st.markdown
        """
        if not body:
            return st.markdown("", **kwargs)
            
        if unsafe_allow_html:
            # Always sanitize HTML content
            try:
                safe_body = safe_render(body, allow_formatting=True)
                logger.debug(f"Sanitized HTML content for markdown: {len(body)} -> {len(safe_body)} chars")
            except Exception as e:
                logger.warning(f"Failed to sanitize markdown content: {e}")
                safe_body = "[Content blocked for security]"
        else:
            # For non-HTML markdown, still check for basic safety
            safe_body = xss_protection.sanitize_string(body)
            
        return st.markdown(safe_body, unsafe_allow_html=unsafe_allow_html, **kwargs)
    
    @staticmethod
    def html(body: str, **kwargs):
        """
        Secure HTML rendering with mandatory sanitization
        
        Args:
            body: HTML content
            **kwargs: Additional arguments
        """
        if not body:
            return st.html("")
            
        try:
            safe_body = safe_render(body, allow_formatting=True)
            logger.debug(f"Sanitized HTML content: {len(body)} -> {len(safe_body)} chars")
        except Exception as e:
            logger.warning(f"Failed to sanitize HTML content: {e}")
            safe_body = "<div>Content blocked for security</div>"
            
        return st.html(safe_body, **kwargs)
    
    @staticmethod
    def text(body: str, **kwargs):
        """
        Secure text display
        
        Args:
            body: Text content
            **kwargs: Additional arguments
        """
        if not body:
            return st.text("")
            
        safe_body = xss_protection.sanitize_string(body)
        return st.text(safe_body, **kwargs)
    
    @staticmethod
    def write(*args, **kwargs):
        """
        Secure write function with content sanitization
        
        Args:
            *args: Content to write
            **kwargs: Additional arguments
        """
        safe_args = []
        
        for arg in args:
            if isinstance(arg, str):
                try:
                    safe_arg = xss_protection.sanitize_string(arg)
                except ValueError:
                    safe_arg = "[Content blocked for security]"
            elif isinstance(arg, dict):
                safe_arg = xss_protection.sanitize_dict(arg)
            elif isinstance(arg, list):
                safe_arg = xss_protection.sanitize_list(arg)
            else:
                safe_arg = arg
                
            safe_args.append(safe_arg)
        
        return st.write(*safe_args, **kwargs)
    
    @staticmethod
    def error(body: str, **kwargs):
        """
        Secure error message display
        
        Args:
            body: Error message
            **kwargs: Additional arguments
        """
        if not body:
            return st.error("Unknown error")
            
        safe_body = xss_protection.sanitize_string(body, max_length=1000)
        return st.error(safe_body, **kwargs)
    
    @staticmethod
    def warning(body: str, **kwargs):
        """
        Secure warning message display
        
        Args:
            body: Warning message
            **kwargs: Additional arguments
        """
        if not body:
            return st.warning("Unknown warning")
            
        safe_body = xss_protection.sanitize_string(body, max_length=1000)
        return st.warning(safe_body, **kwargs)
    
    @staticmethod
    def info(body: str, **kwargs):
        """
        Secure info message display
        
        Args:
            body: Info message
            **kwargs: Additional arguments
        """
        if not body:
            return st.info("No information")
            
        safe_body = xss_protection.sanitize_string(body, max_length=1000)
        return st.info(safe_body, **kwargs)
    
    @staticmethod
    def success(body: str, **kwargs):
        """
        Secure success message display
        
        Args:
            body: Success message
            **kwargs: Additional arguments
        """
        if not body:
            return st.success("Success")
            
        safe_body = xss_protection.sanitize_string(body, max_length=1000)
        return st.success(safe_body, **kwargs)
    
    @staticmethod
    def metric(label: str, value: Union[str, int, float], delta: Optional[str] = None, **kwargs):
        """
        Secure metric display
        
        Args:
            label: Metric label
            value: Metric value
            delta: Delta value
            **kwargs: Additional arguments
        """
        safe_label = xss_protection.sanitize_string(str(label), max_length=100)
        
        # Convert value to string and sanitize if it's a string
        if isinstance(value, str):
            safe_value = xss_protection.sanitize_string(value, max_length=100)
        else:
            safe_value = value
        
        safe_delta = None
        if delta is not None:
            safe_delta = xss_protection.sanitize_string(str(delta), max_length=100)
            
        return st.metric(safe_label, safe_value, safe_delta, **kwargs)
    
    @staticmethod 
    def json(obj: Union[Dict, List, str], **kwargs):
        """
        Secure JSON display with sanitization
        
        Args:
            obj: Object to display as JSON
            **kwargs: Additional arguments
        """
        if isinstance(obj, dict):
            safe_obj = xss_protection.sanitize_dict(obj)
        elif isinstance(obj, list):
            safe_obj = xss_protection.sanitize_list(obj)
        elif isinstance(obj, str):
            try:
                import json
                parsed = json.loads(obj)
                if isinstance(parsed, dict):
                    safe_obj = xss_protection.sanitize_dict(parsed)
                elif isinstance(parsed, list):
                    safe_obj = xss_protection.sanitize_list(parsed)
                else:
                    safe_obj = parsed
            except Exception:
                safe_obj = xss_protection.sanitize_string(obj)
        else:
            safe_obj = obj
            
        return st.json(safe_obj, **kwargs)
    
    @staticmethod
    def code(body: str, language: Optional[str] = None, **kwargs):
        """
        Secure code display
        
        Args:
            body: Code content
            language: Programming language
            **kwargs: Additional arguments
        """
        if not body:
            return st.code("", language=language, **kwargs)
            
        # Code should be sanitized but preserve formatting
        safe_body = xss_protection.sanitize_string(body, max_length=50000)
        safe_language = None
        
        if language:
            safe_language = xss_protection.sanitize_string(language, max_length=20)
            
        return st.code(safe_body, language=safe_language, **kwargs)


# Content Security Policy helper
def get_csp_header() -> str:
    """
    Generate Content Security Policy header for maximum XSS protection
    
    Returns:
        CSP header string
    """
    csp_directives = [
        "default-src 'self'",
        "script-src 'self' 'unsafe-inline' 'unsafe-eval' localhost:* 127.0.0.1:*",  # Streamlit needs some unsafe directives
        "style-src 'self' 'unsafe-inline' fonts.googleapis.com",
        "font-src 'self' fonts.gstatic.com",
        "img-src 'self' data: https:",
        "connect-src 'self' localhost:* 127.0.0.1:* ws: wss:",
        "object-src 'none'",
        "base-uri 'self'",
        "form-action 'self'",
        "frame-ancestors 'none'",
        "upgrade-insecure-requests"
    ]
    
    return "; ".join(csp_directives)

def apply_security_headers():
    """Apply security headers to Streamlit app"""
    # or through middleware in a production deployment
    headers = {
        'Content-Security-Policy': get_csp_header(),
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
        'Referrer-Policy': 'strict-origin-when-cross-origin'
    }
    
    return headers


# Create global instance for easy importing
secure = SecureStreamlit()

# Export commonly used functions
__all__ = [
    'SecureStreamlit',
    'secure',
    'get_csp_header', 
    'apply_security_headers'
