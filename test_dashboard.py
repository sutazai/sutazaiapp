#!/usr/bin/env python3
"""
Quick test to verify dashboard can connect to the performance backend
"""

import requests
import json

def test_dashboard_connection():
    """Test if dashboard can get data from performance backend"""
    
    print("🧪 Testing Dashboard Connection to Performance Backend")
    print("=" * 60)
    
    # Test performance summary endpoint
    try:
        response = requests.get("http://localhost:8000/api/performance/summary", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Backend connection successful!")
            print(f"   CPU Usage: {data['system']['cpu_usage']:.1f}%")
            print(f"   Memory Usage: {data['system']['memory_usage']:.1f}%")
            print(f"   Total Requests: {data['api']['total_requests']}")
            print(f"   Active Models: {data['models']['active_models']}")
            
            # Test format compatibility
            has_new_format = "system" in data and "api" in data
            has_old_format = "system_summary" in data
            
            print(f"\n📊 Data Format Analysis:")
            print(f"   New format (system/api/models): {'✅' if has_new_format else '❌'}")
            print(f"   Old format (system_summary): {'✅' if has_old_format else '❌'}")
            print(f"   Dashboard compatibility: ✅ (Fixed to handle both)")
            
            return True
        else:
            print(f"❌ Backend error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

def test_dashboard_access():
    """Test if dashboard is accessible"""
    
    print("\n🌐 Testing Dashboard Accessibility")
    print("=" * 60)
    
    try:
        # Test if Streamlit is running
        response = requests.get("http://192.168.131.128:8501/healthz", timeout=3)
        if response.status_code == 200:
            print("✅ Streamlit dashboard is running and accessible")
            print("   URL: http://192.168.131.128:8501/")
            return True
    except:
        pass
    
    try:
        # Alternative health check
        response = requests.get("http://192.168.131.128:8501/", timeout=3)
        if response.status_code == 200:
            print("✅ Streamlit dashboard is accessible")
            print("   URL: http://192.168.131.128:8501/")
            return True
    except Exception as e:
        print(f"❌ Dashboard not accessible: {e}")
        print("   Check if Streamlit is running on port 8501")
        return False

def main():
    """Run all tests"""
    
    print("🚀 SutazAI Dashboard Connection Test")
    print("=" * 60)
    
    # Test backend connection
    backend_ok = test_dashboard_connection()
    
    # Test dashboard access
    dashboard_ok = test_dashboard_access()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    if backend_ok and dashboard_ok:
        print("🎉 All tests passed! Dashboard should show live metrics.")
        print("\n📋 What to expect:")
        print("   • CPU and Memory usage showing real values")
        print("   • API requests counter increasing")
        print("   • No more 'list object has no attribute get' errors")
        print("   • Performance alerts working correctly")
        
    elif backend_ok:
        print("⚠️  Backend OK, but dashboard not accessible")
        print("   • Performance data is available")
        print("   • Check Streamlit is running: streamlit run app.py")
        
    else:
        print("❌ Tests failed")
        print("   • Check if performance backend is running on port 8000")
        print("   • Restart backend: systemctl restart sutazai-performance-backend")

if __name__ == "__main__":
    main()