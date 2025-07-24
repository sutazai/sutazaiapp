#!/usr/bin/env python3
"""
Test Live Performance Monitoring
"""

import time
import requests

def test_live_performance():
    """Test the live performance monitoring system"""
    print("🧪 Testing Live Performance Monitoring")
    print("=" * 50)
    
    # Test 1: Check if psutil is working
    try:
        import psutil
        print("✅ psutil available")
        print(f"   CPU: {psutil.cpu_percent(interval=0.1):.1f}%")
        print(f"   Memory: {psutil.virtual_memory().percent:.1f}%")
        print(f"   Processes: {len(psutil.pids())}")
    except ImportError:
        print("❌ psutil not available")
        return False
    
    # Test 2: Check backend connectivity
    try:
        response = requests.get("http://localhost:8000/health", timeout=3)
        if response.status_code == 200:
            print("✅ Backend healthy")
            print(f"   Response time: {response.elapsed.total_seconds():.3f}s")
        else:
            print(f"⚠️ Backend unhealthy: {response.status_code}")
    except:
        print("❌ Backend offline")
    
    # Test 3: Check Ollama
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=3)
        if response.status_code == 200:
            version = response.json().get("version", "unknown")
            print(f"✅ Ollama healthy (v{version})")
        else:
            print("⚠️ Ollama unhealthy")
    except:
        print("❌ Ollama offline")
    
    # Test 4: Generate some load and measure changes
    print("\n🔄 Generating load for 5 seconds...")
    initial_cpu = psutil.cpu_percent(interval=1)
    
    # Create some CPU load
    start_time = time.time()
    counter = 0
    while time.time() - start_time < 3:
        counter += 1
        _ = [i**2 for i in range(1000)]
    
    final_cpu = psutil.cpu_percent(interval=1)
    cpu_change = final_cpu - initial_cpu
    
    print(f"   Initial CPU: {initial_cpu:.1f}%")
    print(f"   Final CPU: {final_cpu:.1f}%")
    print(f"   Change: {cpu_change:+.1f}%")
    
    if abs(cpu_change) > 1:
        print("✅ CPU monitoring detecting changes")
    else:
        print("⚠️ CPU change detection may be limited")
    
    print("\n" + "=" * 50)
    print("🎯 Live Performance Monitoring Test Complete")
    print("💡 Enable 'Show Performance Metrics' in Streamlit to see live data!")
    
    return True

if __name__ == "__main__":
    test_live_performance()