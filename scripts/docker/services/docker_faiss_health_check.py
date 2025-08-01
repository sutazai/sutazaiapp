#!/usr/bin/env python3
"""Health check script for FAISS service"""

import requests
import sys
import time

def check_health():
    try:
        response = requests.get("http://localhost:8088/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "healthy":
                print("FAISS service is healthy")
                return True
        return False
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

if __name__ == "__main__":
    if check_health():
        sys.exit(0)
    else:
        sys.exit(1)