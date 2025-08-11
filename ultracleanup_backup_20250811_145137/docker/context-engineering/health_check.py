import logging

# Configure logger for exception handling
logger = logging.getLogger(__name__)

import requests
import sys

def check_health():
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            return True
    except Exception as e:
        # Suppressed exception (was bare except)
        logger.debug(f"Suppressed exception: {e}")
        pass
    return False

if __name__ == "__main__":
    if check_health():
        sys.exit(0)
    else:
        sys.exit(1)