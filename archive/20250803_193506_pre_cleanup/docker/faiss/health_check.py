#!/usr/bin/env python3
"""Health check script for FAISS service"""

import sys
import urllib.request
import urllib.error
import json

def check_health():
    """Check if FAISS service is healthy"""
    try:
        # Try to connect to the health endpoint on correct port
        with urllib.request.urlopen('http://localhost:8000/health', timeout=10) as response:
            if response.getcode() == 200:
                try:
                    data = json.loads(response.read().decode('utf-8'))
                    if data.get("status") == "healthy":
                        return True
                except json.JSONDecodeError:
                    # If no JSON response, just check status code
                    return True
        return False
    except (urllib.error.URLError, OSError, Exception) as e:
        print(f"Health check failed: {e}", file=sys.stderr)
        return False

if __name__ == "__main__":
    if check_health():
        sys.exit(0)
    else:
        sys.exit(1)