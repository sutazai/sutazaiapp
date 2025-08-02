#!/usr/bin/env python3
"""Quick system status check for SutazAI"""

import subprocess
import requests
import psutil
from datetime import datetime

def check_containers():
    """Check Docker containers"""
    print("\n🐳 Docker Containers:")
    result = subprocess.run(['docker', 'ps', '--format', 'table {{.Names}}\t{{.Status}}'], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        lines = result.stdout.strip().split('\n')
        for line in lines:
            if 'sutazai' in line:
                print(f"  {line}")
    print()

def check_models():
    """Check Ollama models"""
    print("🤖 AI Models:")
    try:
        result = subprocess.run(['docker', 'exec', 'sutazai-ollama-minimal', 'ollama', 'list'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
    except:
        print("  Unable to retrieve models")
    print()

def check_apis():
    """Check API endpoints"""
    print("🌐 API Endpoints:")
    endpoints = {
        'Backend API': 'http://localhost:8000/health',
        'Frontend UI': 'http://localhost:8501',
        'Ollama API': 'http://localhost:11434/api/tags'
    }
    
    for name, url in endpoints.items():
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print(f"  ✅ {name}: {url} (OK)")
            else:
                print(f"  ⚠️  {name}: {url} (Status: {response.status_code})")
        except:
            print(f"  ❌ {name}: {url} (Not responding)")
    print()

def check_resources():
    """Check system resources"""
    print("💻 System Resources:")
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    print(f"  CPU Usage: {cpu}%")
    print(f"  Memory: {mem.percent}% ({mem.used/1024/1024/1024:.1f}GB / {mem.total/1024/1024/1024:.1f}GB)")
    print(f"  Disk: {disk.percent}% used ({disk.free/1024/1024/1024:.1f}GB free)")
    print()

def main():
    print(f"\n{'='*60}")
    print(f"SutazAI System Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    check_containers()
    check_models()
    check_apis()
    check_resources()
    
    print("\n✨ Status check complete!")
    print("\nFor continuous monitoring, run: python3 scripts/static_monitor.py")
    print("To access the system:")
    print("  - Frontend: http://localhost:8501")
    print("  - API Docs: http://localhost:8000/docs")

if __name__ == "__main__":
    main()