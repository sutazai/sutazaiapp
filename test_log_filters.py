#!/usr/bin/env python3
"""
Test the log filtering functionality
"""

import sys
sys.path.append('/opt/sutazaiapp')

from enhanced_logging_system import sutazai_logger, info, debug, warning, error

def test_log_filtering():
    """Test that log filtering works correctly"""
    
    print("🧪 Testing Log Filtering Functionality")
    print("=" * 50)
    
    # Clear existing logs
    sutazai_logger.log_history.clear()
    
    # Generate test logs with different levels and categories
    debug("This is a debug message", category="system")
    info("This is an info message", category="api") 
    warning("This is a warning message", category="ui")
    error("This is an error message", category="app")
    info("Another info message", category="system")
    debug("Another debug message", category="api")
    
    print("✅ Generated 6 test log entries")
    
    # Wait for async log processing
    import time
    time.sleep(1)
    print("⏳ Waited for log processing...")
    
    # Test 1: Get all logs
    all_logs = sutazai_logger.get_recent_logs(limit=100)
    print(f"📊 Total logs: {len(all_logs)}")
    
    # Test 2: Filter by level - INFO only
    info_logs = sutazai_logger.get_recent_logs(limit=100, level_filter="INFO")
    print(f"📊 INFO logs: {len(info_logs)}")
    
    # Test 3: Filter by level - DEBUG only  
    debug_logs = sutazai_logger.get_recent_logs(limit=100, level_filter="DEBUG")
    print(f"📊 DEBUG logs: {len(debug_logs)}")
    
    # Test 4: Filter by category - system
    system_logs = [log for log in all_logs if log["category"] == "system"]
    print(f"📊 System category logs: {len(system_logs)}")
    
    # Test 5: Filter by category - api
    api_logs = [log for log in all_logs if log["category"] == "api"]
    print(f"📊 API category logs: {len(api_logs)}")
    
    # Display some sample logs
    print("\n📋 Sample log entries:")
    for i, log in enumerate(all_logs[:3]):
        print(f"  {i+1}. [{log['level']}] {log['category']}: {log['message']}")
    
    # Test stats
    stats = sutazai_logger.get_log_stats()
    print(f"\n📊 Log Statistics:")
    print(f"  Total: {stats['total']}")
    print(f"  By level: {stats['by_level']}")
    print(f"  By category: {stats['by_category']}")
    
    # Verify expected results
    expected_info = 2
    expected_debug = 2  
    expected_system = 2
    expected_api = 2
    
    success = True
    if len(info_logs) != expected_info:
        print(f"❌ INFO filter failed: expected {expected_info}, got {len(info_logs)}")
        success = False
    else:
        print("✅ INFO level filter working")
        
    if len(debug_logs) != expected_debug:
        print(f"❌ DEBUG filter failed: expected {expected_debug}, got {len(debug_logs)}")
        success = False
    else:
        print("✅ DEBUG level filter working")
        
    if len(system_logs) != expected_system:
        print(f"❌ System category filter failed: expected {expected_system}, got {len(system_logs)}")
        success = False
    else:
        print("✅ System category filter working")
        
    if len(api_logs) != expected_api:
        print(f"❌ API category filter failed: expected {expected_api}, got {len(api_logs)}")
        success = False
    else:
        print("✅ API category filter working")
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All log filtering tests passed!")
        print("   The dashboard log filters should now work correctly.")
    else:
        print("❌ Some log filtering tests failed.")
    
    return success

if __name__ == "__main__":
    test_log_filtering()