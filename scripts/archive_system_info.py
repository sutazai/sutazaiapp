import psutil
import platform
import sys
import json
from datetime import datetime

def get_system_info():
    return {
        "timestamp": datetime.now().isoformat(),
        "system": {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "python_version": sys.version
        },
        "cpu": {
            "physical_cores": psutil.cpu_count(logical=False),
            "total_cores": psutil.cpu_count(logical=True),
            "cpu_usage": psutil.cpu_percent(interval=1)
        },
        "memory": {
            "total": psutil.virtual_memory().total,
            "available": psutil.virtual_memory().available,
            "percent": psutil.virtual_memory().percent
        },
        "disk": {
            "total": psutil.disk_usage('/').total,
            "used": psutil.disk_usage('/').used,
            "free": psutil.disk_usage('/').free,
            "percent": psutil.disk_usage('/').percent
        },
        "processes": {
            "backend": [p.info for p in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']) if 'simple_backend' in p.info['name']],
            "frontend": [p.info for p in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']) if 'streamlit' in p.info['name']]
        }
    }

if __name__ == "__main__":
    info = get_system_info()
    print(json.dumps(info, indent=2))
