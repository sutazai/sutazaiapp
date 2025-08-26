#!/usr/bin/env python3
"""
Health check script for MCP HTTP services
"""

import os
import sys
import requests
import time

def health_check():
    """Perform health check"""
    port = int(os.getenv('MCP_SERVICE_PORT', '11100'))
    timeout = int(os.getenv('HEALTH_CHECK_TIMEOUT', '5'))
    
    try:
        response = requests.get(
            f"http://localhost:{port}/health",
            timeout=timeout
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'healthy':
                print(f"Health check passed: {data}")
                return 0
            else:
                print(f"Health check failed: {data}")
                return 1
        else:
            print(f"Health check failed with status {response.status_code}")
            return 1
            
    except requests.exceptions.RequestException as e:
        print(f"Health check failed: {e}")
        return 1
    except Exception as e:
        print(f"Health check error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(health_check())