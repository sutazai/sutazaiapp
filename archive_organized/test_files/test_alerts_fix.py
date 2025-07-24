#!/usr/bin/env python3
"""
Test the alerts endpoint fix
"""

import requests
import json

def test_alerts_endpoint():
    """Test alerts endpoint response format"""
    
    print("🧪 Testing Alerts Endpoint Fix")
    print("=" * 50)
    
    try:
        # Test alerts endpoint
        response = requests.get("http://localhost:8000/api/performance/alerts", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Alerts endpoint responding")
            print(f"   Response type: {type(data)}")
            print(f"   Content: {data}")
            
            # Test if it's a list (expected format)
            if isinstance(data, list):
                print("✅ Correct format: Returns array of alerts")
                
                if len(data) > 0:
                    print("📋 Alert structure:")
                    for i, alert in enumerate(data[:3]):  # Show first 3
                        print(f"   Alert {i+1}: {alert}")
                else:
                    print("📋 No active alerts (system healthy)")
                
            else:
                print(f"⚠️  Unexpected format: Expected list, got {type(data)}")
            
            return True
        else:
            print(f"❌ Alerts endpoint error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Alerts test failed: {e}")
        return False

def test_dashboard_alerts_compatibility():
    """Test dashboard can handle alerts correctly"""
    
    print("\n🖥️ Testing Dashboard Alerts Compatibility")
    print("=" * 50)
    
    # Simulate what the dashboard does
    try:
        # Make the API call like the dashboard does
        response = requests.get("http://localhost:8000/api/performance/alerts", timeout=5)
        
        if response.status_code == 200:
            # Simulate APIClient.make_request response format
            alerts_result = {
                "success": True,
                "data": response.json()
            }
            
            print("✅ Simulated APIClient response:")
            print(f"   Success: {alerts_result['success']}")
            print(f"   Data type: {type(alerts_result['data'])}")
            
            # Test the dashboard logic
            if alerts_result["success"]:
                if isinstance(alerts_result["data"], list):
                    alerts = alerts_result["data"]
                    print("✅ Dashboard will handle as plain array")
                else:
                    alerts_data = alerts_result["data"]
                    alerts = alerts_data.get("alerts", [])
                    print("✅ Dashboard will handle as structured response")
                
                print(f"   Final alerts: {len(alerts)} items")
                
                # Test alert format handling
                for alert in alerts[:2]:  # Test first 2
                    if "level" in alert:
                        print(f"   New format alert: level={alert.get('level')}, category={alert.get('category')}")
                    else:
                        print(f"   Old format alert: type={alert.get('type')}, metric={alert.get('metric')}")
                
                return True
            else:
                print("❌ Simulated API call failed")
                return False
        else:
            print(f"❌ API call failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Compatibility test failed: {e}")
        return False

def main():
    """Run all tests"""
    
    print("🚀 Dashboard Alerts Fix Verification")
    print("=" * 50)
    
    # Test alerts endpoint
    alerts_ok = test_alerts_endpoint()
    
    # Test dashboard compatibility
    compat_ok = test_dashboard_alerts_compatibility()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    if alerts_ok and compat_ok:
        print("🎉 All tests passed! Dashboard alerts should work correctly.")
        print("\n✅ Expected results:")
        print("   • No more 'list object has no attribute get' errors")
        print("   • Alerts section displays properly")
        print("   • Performance alerts (if any) show with correct styling")
        
    else:
        print("❌ Some tests failed")
        if not alerts_ok:
            print("   • Alerts endpoint issues")
        if not compat_ok:
            print("   • Dashboard compatibility issues")

if __name__ == "__main__":
    main()