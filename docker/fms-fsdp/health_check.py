import requests
import sys

def check_health():
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            return True
    except:
        pass
    return False

if __name__ == "__main__":
    if check_health():
        sys.exit(0)
    else:
        sys.exit(1)