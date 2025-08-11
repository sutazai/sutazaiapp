"""
API Client Utilities - Extracted from monolith
Centralized API communication functions
"""

import asyncio
import httpx
import streamlit as st
import logging
from typing import Dict, Optional

# Configure logging
logger = logging.getLogger(__name__)

# API Configuration
API_BASE_URL = "http://127.0.0.1:10010"
DEFAULT_TIMEOUT = 10.0

async def call_api(endpoint: str, method: str = "GET", data: Dict = None, timeout: float = None) -> Optional[Dict]:
    """
    Make async API call to backend with error handling
    
    Args:
        endpoint: API endpoint path
        method: HTTP method (GET, POST, PUT, DELETE)
        data: Request data for POST/PUT requests
        timeout: Request timeout in seconds
        
    Returns:
        Response data dict or None if error
    """
    if timeout is None:
        timeout = DEFAULT_TIMEOUT
        
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            if method.upper() == "GET":
                response = await client.get(url)
            elif method.upper() == "POST":
                response = await client.post(url, json=data)
            elif method.upper() == "PUT":
                response = await client.put(url, json=data)
            elif method.upper() == "DELETE":
                response = await client.delete(url)
            else:
                logger.error(f"Unsupported HTTP method: {method}")
                return None
                
            response.raise_for_status()
            return response.json()
            
    except httpx.TimeoutException:
        logger.warning(f"API call timeout: {endpoint}")
        st.warning(f"Request timeout for {endpoint}")
        return None
        
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error {e.response.status_code} for {endpoint}: {e.response.text}")
        st.error(f"API error ({e.response.status_code}): {endpoint}")
        return None
        
    except httpx.RequestError as e:
        logger.error(f"Request error for {endpoint}: {str(e)}")
        st.error(f"Connection error: Unable to reach API")
        return None
        
    except Exception as e:
        logger.error(f"Unexpected error for {endpoint}: {str(e)}")
        st.error(f"Unexpected error: {str(e)}")
        return None

def handle_api_error(response: dict, context: str = "operation") -> bool:
    """
    Handle API response and show appropriate error messages
    
    Args:
        response: API response dictionary
        context: Context description for error messages
        
    Returns:
        True if response is valid, False if error
    """
    if not response:
        st.error(f"No response received for {context}")
        return False
        
    if "error" in response:
        st.error(f"API Error ({context}): {response['error']}")
        return False
        
    if "status" in response and response["status"] == "error":
        error_msg = response.get("message", "Unknown error")
        st.error(f"Error ({context}): {error_msg}")
        return False
        
    return True

async def check_service_health(url: str, timeout: float = 2.0) -> bool:
    """
    Check if a service is healthy
    
    Args:
        url: Service health check URL
        timeout: Request timeout
        
    Returns:
        True if service is healthy
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            return response.status_code == 200
    except (ValueError, TypeError, KeyError, AttributeError) as e:
        logger.warning(f"Exception caught, returning: {e}")
        return False

# Synchronous wrapper for Streamlit compatibility
def sync_call_api(endpoint: str, method: str = "GET", data: Dict = None, timeout: float = None) -> Optional[Dict]:
    """Synchronous wrapper for API calls"""
    return asyncio.run(call_api(endpoint, method, data, timeout))

def sync_check_service_health(url: str, timeout: float = 2.0) -> bool:
    """Synchronous wrapper for health checks"""
    return asyncio.run(check_service_health(url, timeout))