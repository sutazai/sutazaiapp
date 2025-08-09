"""
Security configuration for frontend
Emergency fix for XSS vulnerabilities
"""

# Disable unsafe HTML globally
ALLOW_UNSAFE_HTML = False

# Content Security Policy
CSP_HEADER = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';"

# Input sanitization required
REQUIRE_INPUT_SANITIZATION = True

# Session security
SESSION_COOKIE_SECURE = True
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = 'Strict'
