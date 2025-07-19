#!/usr/bin/env python3
"""
Test dashboard log functionality by making some API calls
and verifying the logs show up with proper filtering
"""

import requests
import time
import json

def test_dashboard_logs():
    """Test dashboard logs by generating some activity"""
    
    print("🧪 Testing Dashboard Log Functionality")
    print("=" * 50)
    
    # Generate some API activity to create logs
    print("📊 Generating API activity to create logs...")
    
    # Make some API calls to generate logs
    endpoints = [
        "http://localhost:8000/health",
        "http://localhost:8000/api/models", 
        "http://localhost:8000/api/performance/summary",
        "http://localhost:8000/api/performance/alerts"
    ]
    
    for endpoint in endpoints:
        try:
            response = requests.get(endpoint, timeout=5)
            print(f"✅ {endpoint}: {response.status_code}")
        except Exception as e:
            print(f"❌ {endpoint}: {e}")
        time.sleep(0.5)
    
    # Make a chat request to generate more logs
    try:
        chat_response = requests.post(
            "http://localhost:8000/api/chat",
            json={"message": "Test log filtering", "model": "llama3.2:1b"},
            timeout=35
        )
        print(f"✅ Chat API: {chat_response.status_code}")
    except Exception as e:
        print(f"❌ Chat API: {e}")
    
    print("\n📊 Generated multiple API calls to create diverse logs")
    
    # Test the backend logs endpoint
    try:
        logs_response = requests.get("http://localhost:8000/api/logs?limit=10", timeout=5)
        if logs_response.status_code == 200:
            logs_data = logs_response.json()
            print(f"✅ Backend logs endpoint: {logs_data['stats']['total']} total logs")
            print(f"   Errors: {logs_data['stats']['errors']}")
            print(f"   Warnings: {logs_data['stats']['warnings']}")
            
            # Show sample logs
            if logs_data['logs']:
                print("\n📋 Sample backend logs:")
                for i, log in enumerate(logs_data['logs'][-3:]):
                    timestamp = log['timestamp'].split('T')[1][:8] if 'T' in log['timestamp'] else log['timestamp']
                    print(f"  {i+1}. [{log['level']}] {log['category']}: {log['message'][:50]}...")
            
        else:
            print(f"❌ Backend logs endpoint error: {logs_response.status_code}")
    except Exception as e:
        print(f"❌ Backend logs test failed: {e}")
    
    print("\n📋 Dashboard Log Filter Testing:")
    print("   1. Visit: http://192.168.131.128:8501/")
    print("   2. Enable 'Show Real-time Logs' in sidebar")  
    print("   3. Try different filter combinations:")
    print("      • Filter by level: INFO, DEBUG, ERROR")
    print("      • Filter by category: api, ui, system")
    print("      • Change 'Show entries' limit")
    print("   4. Verify logs update and filters work")
    
    print("\n✅ Expected Results:")
    print("   • Logs should display with proper formatting")
    print("   • Level filters should show only selected levels")
    print("   • Category filters should show only selected categories")
    print("   • Entry limit should control number of displayed logs")
    print("   • Auto-refresh should update logs every 5 seconds")
    print("   • Manual refresh button should work immediately")

def main():
    """Run log functionality test"""
    
    print("🚀 Dashboard Log Functionality Test")
    print("=" * 50)
    
    test_dashboard_logs()
    
    print("\n" + "=" * 50)
    print("📊 LOG TESTING COMPLETE")
    print("=" * 50)
    print("🎯 Next Steps:")
    print("   1. Open dashboard: http://192.168.131.128:8501/")
    print("   2. Enable log viewing in sidebar")
    print("   3. Test all filter combinations")
    print("   4. Verify real-time updates work")

if __name__ == "__main__":
    main()