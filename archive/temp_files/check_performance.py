#!/usr/bin/env python3
"""
Quick Performance Check for SutazAI System
"""

import psutil
import requests
import json
from datetime import datetime

def check_system_performance():
    """Check current system performance"""
    print("🚀 SutazAI System Performance Report")
    print("=" * 60)
    print(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # System metrics
    print("🖥️  SYSTEM METRICS")
    print("-" * 20)
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    print(f"CPU Usage:      {cpu_percent:.1f}% ({psutil.cpu_count()} cores)")
    print(f"Memory Usage:   {memory.percent:.1f}% ({memory.used/(1024**3):.1f}/{memory.total/(1024**3):.1f} GB)")
    print(f"Disk Usage:     {(disk.used/disk.total)*100:.1f}% ({disk.used/(1024**3):.1f}/{disk.total/(1024**3):.1f} GB)")
    print(f"Processes:      {len(psutil.pids())}")
    
    # Health status
    print()
    print("🔋 HEALTH STATUS")
    print("-" * 20)
    
    health_status = "🟢 HEALTHY"
    alerts = []
    
    if cpu_percent > 90:
        health_status = "🔴 CRITICAL"
        alerts.append(f"CPU usage critically high: {cpu_percent:.1f}%")
    elif cpu_percent > 70:
        health_status = "🟡 WARNING"
        alerts.append(f"CPU usage high: {cpu_percent:.1f}%")
    
    if memory.percent > 95:
        health_status = "🔴 CRITICAL"
        alerts.append(f"Memory usage critically high: {memory.percent:.1f}%")
    elif memory.percent > 80:
        if health_status == "🟢 HEALTHY":
            health_status = "🟡 WARNING"
        alerts.append(f"Memory usage high: {memory.percent:.1f}%")
    
    if (disk.used/disk.total)*100 > 90:
        health_status = "🔴 CRITICAL"
        alerts.append(f"Disk usage critically high: {(disk.used/disk.total)*100:.1f}%")
    elif (disk.used/disk.total)*100 > 75:
        if health_status == "🟢 HEALTHY":
            health_status = "🟡 WARNING"
        alerts.append(f"Disk usage high: {(disk.used/disk.total)*100:.1f}%")
    
    print(f"Overall Status: {health_status}")
    
    if alerts:
        print()
        print("⚠️  ALERTS")
        print("-" * 20)
        for alert in alerts:
            print(f"• {alert}")
    
    # Service checks
    print()
    print("🔧 SERVICE STATUS")
    print("-" * 20)
    
    # Check SutazAI Backend
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print(f"✅ SutazAI Backend:  Healthy ({response.elapsed.total_seconds():.3f}s)")
        else:
            print(f"❌ SutazAI Backend:  Unhealthy (HTTP {response.status_code})")
    except:
        print("❌ SutazAI Backend:  Offline")
    
    # Check Ollama
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        if response.status_code == 200:
            version = response.json().get("version", "unknown")
            print(f"✅ Ollama:           Healthy (v{version})")
        else:
            print(f"❌ Ollama:           Unhealthy")
    except:
        print("❌ Ollama:           Offline")
    
    # Check Docker services
    services = {
        "PostgreSQL": 5432,
        "Redis": 6379,
        "Qdrant": 6333,
        "ChromaDB": 8001
    }
    
    for service, port in services.items():
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            result = sock.connect_ex(("localhost", port))
            sock.close()
            if result == 0:
                print(f"✅ {service:<12}: Healthy")
            else:
                print(f"❌ {service:<12}: Unhealthy")
        except:
            print(f"❌ {service:<12}: Unknown")
    
    print()
    print("=" * 60)
    print("📊 Performance monitoring system is operational!")

if __name__ == "__main__":
    check_system_performance()