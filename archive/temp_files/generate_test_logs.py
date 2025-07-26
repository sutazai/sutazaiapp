#!/usr/bin/env python3
"""
Generate test logs of different levels to test filtering
"""

import sys
sys.path.append('/opt/sutazaiapp')

from enhanced_logging_system import sutazai_logger, info, debug, warning, error, critical
import time

def generate_test_logs():
    """Generate various log levels for testing filters"""
    
    print("🧪 Generating Test Logs for Filter Testing")
    print("=" * 50)
    
    # Generate logs of different levels
    error("This is a test ERROR log - should appear in ERROR filter", category="test")
    error("Another ERROR log for testing filters", category="api") 
    warning("This is a test WARNING log", category="ui")
    warning("Another WARNING for filter testing", category="system")
    critical("CRITICAL test log - highest priority", category="error")
    info("Test INFO log - informational", category="app")
    debug("Test DEBUG log - detailed debugging", category="system")
    
    # Wait for processing
    time.sleep(1)
    
    # Check what was actually logged
    all_logs = sutazai_logger.get_recent_logs(limit=100)
    error_logs = sutazai_logger.get_recent_logs(limit=100, level_filter="ERROR")
    warning_logs = sutazai_logger.get_recent_logs(limit=100, level_filter="WARNING")
    critical_logs = sutazai_logger.get_recent_logs(limit=100, level_filter="CRITICAL")
    
    print(f"📊 Generated logs summary:")
    print(f"   Total logs: {len(all_logs)}")
    print(f"   ERROR logs: {len(error_logs)}")
    print(f"   WARNING logs: {len(warning_logs)}")
    print(f"   CRITICAL logs: {len(critical_logs)}")
    
    # Show sample error logs
    if error_logs:
        print("\n❌ ERROR logs found:")
        for log in error_logs[-3:]:
            print(f"   - [{log['level']}] {log['category']}: {log['message'][:50]}...")
    
    if critical_logs:
        print("\n🚨 CRITICAL logs found:")
        for log in critical_logs:
            print(f"   - [{log['level']}] {log['category']}: {log['message'][:50]}...")
    
    print(f"\n✅ Test logs generated! Now test the dashboard filters:")
    print(f"   1. Set filter to 'ERROR' - should show {len(error_logs)} logs")
    print(f"   2. Set filter to 'WARNING' - should show {len(warning_logs)} logs")
    print(f"   3. Set filter to 'CRITICAL' - should show {len(critical_logs)} logs")

if __name__ == "__main__":
    generate_test_logs()