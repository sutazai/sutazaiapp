"""
HTML sanitization utilities for XSS prevention
Uses bleach library to clean potentially malicious HTML/JS
"""

import bleach
import logging
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)


# Allowed HTML tags for rich text content
ALLOWED_TAGS = [
    'p', 'br', 'strong', 'em', 'u', 'a', 'ul', 'ol', 'li',
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'blockquote', 'code', 'pre',
    'span', 'div'
]

# Allowed HTML attributes
ALLOWED_ATTRIBUTES = {
    'a': ['href', 'title', 'target'],
    'span': ['class'],
    'div': ['class'],
    'code': ['class'],
}

# Allowed URL protocols for links
ALLOWED_PROTOCOLS = ['http', 'https', 'mailto']


def sanitize_html(
    text: str,
    strip: bool = False,
    allowed_tags: Optional[List[str]] = None,
    allowed_attributes: Optional[Dict] = None
) -> str:
    """
    Sanitize HTML content to prevent XSS attacks
    
    Args:
        text: Input text that may contain HTML
        strip: If True, strip all HTML tags. If False, clean allowed tags
        allowed_tags: Custom list of allowed HTML tags (default: ALLOWED_TAGS)
        allowed_attributes: Custom dict of allowed attributes (default: ALLOWED_ATTRIBUTES)
        
    Returns:
        Sanitized HTML string safe for display
    """
    if not text:
        return text
    
    try:
        if strip:
            # Strip all HTML tags, leaving only text
            return bleach.clean(text, tags=[], strip=True)
        else:
            # Clean HTML, allowing only safe tags and attributes
            tags = allowed_tags if allowed_tags is not None else ALLOWED_TAGS
            attrs = allowed_attributes if allowed_attributes is not None else ALLOWED_ATTRIBUTES
            
            cleaned = bleach.clean(
                text,
                tags=tags,
                attributes=attrs,
                protocols=ALLOWED_PROTOCOLS,
                strip=True  # Strip disallowed tags instead of escaping
            )
            
            # Also linkify URLs for convenience
            cleaned = bleach.linkify(
                cleaned,
                parse_email=True,
                callbacks=[],
                skip_tags=['pre', 'code']  # Don't linkify in code blocks
            )
            
            return cleaned
    except Exception as e:
        logger.error(f"HTML sanitization error: {e}", exc_info=True)
        # On error, strip all HTML as safe fallback
        return bleach.clean(text, tags=[], strip=True)


def sanitize_text(text: str) -> str:
    """
    Sanitize plain text by stripping all HTML
    
    Args:
        text: Input text
        
    Returns:
        Plain text with HTML stripped
    """
    return sanitize_html(text, strip=True)


def escape_html(text: str) -> str:
    """
    Escape HTML special characters
    
    Args:
        text: Input text
        
    Returns:
        Text with HTML characters escaped
    """
    if not text:
        return text
    
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def sanitize_markdown(text: str) -> str:
    """
    Sanitize markdown content
    Allows basic markdown but prevents XSS
    
    Args:
        text: Markdown text
        
    Returns:
        Sanitized markdown
    """
    # For markdown, we allow a broader set of tags
    markdown_tags = ALLOWED_TAGS + [
        'table', 'thead', 'tbody', 'tr', 'th', 'td',
        'img', 'hr', 'del', 'ins'
    ]
    
    markdown_attrs = {
        **ALLOWED_ATTRIBUTES,
        'img': ['src', 'alt', 'title'],
        'th': ['align'],
        'td': ['align'],
    }
    
    return sanitize_html(text, allowed_tags=markdown_tags, allowed_attributes=markdown_attrs)


def is_safe_url(url: str) -> bool:
    """
    Check if URL is safe (no javascript: or data: URIs)
    
    Args:
        url: URL to check
        
    Returns:
        True if URL is safe, False otherwise
    """
    if not url:
        return True
    
    url_lower = url.lower().strip()
    
    # Block dangerous protocols
    dangerous_protocols = ['javascript:', 'data:', 'vbscript:', 'file:']
    for protocol in dangerous_protocols:
        if url_lower.startswith(protocol):
            return False
    
    # Allow http, https, mailto
    safe_protocols = ['http://', 'https://', 'mailto:', '//', '/']
    return any(url_lower.startswith(p) for p in safe_protocols)


def sanitize_user_input(
    data: Dict,
    text_fields: List[str],
    html_fields: Optional[List[str]] = None
) -> Dict:
    """
    Sanitize multiple fields in a dictionary
    
    Args:
        data: Input dictionary
        text_fields: Fields to sanitize as plain text
        html_fields: Fields to sanitize as HTML (optional)
        
    Returns:
        Dictionary with sanitized fields
    """
    result = data.copy()
    
    # Sanitize text fields (strip all HTML)
    for field in text_fields:
        if field in result and isinstance(result[field], str):
            result[field] = sanitize_text(result[field])
    
    # Sanitize HTML fields (clean but allow safe HTML)
    if html_fields:
        for field in html_fields:
            if field in result and isinstance(result[field], str):
                result[field] = sanitize_html(result[field])
    
    return result


# Example usage:
# from app.core.sanitization import sanitize_html, sanitize_text
#
# # For plain text (chat messages, usernames, etc.)
# clean_message = sanitize_text(user_message)
#
# # For rich content (descriptions, posts with formatting)
# clean_html = sanitize_html(user_content)
#
# # For markdown content
# clean_markdown = sanitize_markdown(user_markdown)
