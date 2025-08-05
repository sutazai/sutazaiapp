#!/usr/bin/env python3
"""
Test Monitoring Integration
Purpose: Comprehensive testing of the real-time monitoring system
Author: AI Observability and Monitoring Engineer
Version: 1.0.0 - Integration Testing
"""

import asyncio
import json
import os
import requests
import subprocess
import sys
import time
import websockets
from pathlib import Path

class MonitoringIntegrationTester:
    """Test the complete monitoring integration"""
    
    def __init__(self):
        self.project_root = Path("/opt/sutazaiapp")
        self.api_base = "http://localhost:8080/api"
        self.websocket_url = "ws://localhost:8080/ws"
        self.test_results = []
        
    def log(self, message, level="INFO"):
        """Log test messages"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")
        
    def test_result(self, test_name, success, details=""):
        """Record test result"""
        self.test_results.append({
            'test': test_name,
            'success': success,
            'details': details,
            'timestamp': time.time()
        })
        
        status = "PASS" if success else "FAIL"
        self.log(f"{test_name}: {status} - {details}", "TEST")

    def test_backend_health(self):
        """Test if the backend is running and responding"""
        try:
            response = requests.get(f"{self.api_base}/hygiene/status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.test_result("Backend Health Check", True, 
                               f"Status: {data.get('systemStatus', 'Unknown')}")
                return True
            else:
                self.test_result("Backend Health Check", False, 
                               f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.test_result("Backend Health Check", False, str(e))
            return False

    def test_system_metrics_api(self):
        """Test system metrics API endpoint"""
        try:
            response = requests.get(f"{self.api_base}/system/metrics", timeout=5)
            if response.status_code == 200:
                data = response.json()
                # Check for required metrics fields
                required_fields = ['cpu_usage', 'memory_percentage', 'disk_percentage']
                missing_fields = [field for field in required_fields if field not in data]
                
                if not missing_fields:
                    self.test_result("System Metrics API", True, 
                                   f"CPU: {data.get('cpu_usage')}%, Memory: {data.get('memory_percentage')}%")
                    return True
                else:
                    self.test_result("System Metrics API", False, 
                                   f"Missing fields: {missing_fields}")
                    return False
            else:
                self.test_result("System Metrics API", False, 
                               f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.test_result("System Metrics API", False, str(e))
            return False

    def test_violations_api(self):
        """Test violations API endpoint"""
        try:
            response = requests.get(f"{self.api_base}/hygiene/violations", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.test_result("Violations API", True, 
                               f"Retrieved {len(data)} violations")
                return True
            else:
                self.test_result("Violations API", False, 
                               f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.test_result("Violations API", False, str(e))
            return False

    def test_agents_api(self):
        """Test agents API endpoint"""
        try:
            response = requests.get(f"{self.api_base}/hygiene/agents", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.test_result("Agents API", True, 
                               f"Found {len(data)} agents")
                return True
            else:
                self.test_result("Agents API", False, 
                               f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.test_result("Agents API", False, str(e))
            return False

    def test_trigger_scan_api(self):
        """Test triggering a violation scan"""
        try:
            response = requests.post(f"{self.api_base}/hygiene/scan", timeout=10)
            if response.status_code == 200:
                data = response.json()
                violations_found = data.get('violations_found', 0)
                self.test_result("Trigger Scan API", True, 
                               f"Scan completed, found {violations_found} violations")
                return True
            else:
                self.test_result("Trigger Scan API", False, 
                               f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.test_result("Trigger Scan API", False, str(e))
            return False

    async def test_websocket_connection(self):
        """Test WebSocket real-time connection"""
        try:
            async with websockets.connect(self.websocket_url) as websocket:
                # Wait for initial data message
                message = await asyncio.wait_for(websocket.recv(), timeout=10)
                data = json.loads(message)
                
                if data.get('type') == 'initial_data':
                    self.test_result("WebSocket Connection", True, 
                                   "Received initial data")
                    
                    # Wait for a real-time update (system metrics)
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=5)
                        update_data = json.loads(message)
                        self.test_result("WebSocket Real-time Updates", True, 
                                       f"Received {update_data.get('type')} update")
                    except asyncio.TimeoutError:
                        self.test_result("WebSocket Real-time Updates", False, 
                                       "No real-time updates received")
                    
                    return True
                else:
                    self.test_result("WebSocket Connection", False, 
                                   f"Unexpected message type: {data.get('type')}")
                    return False
                    
        except Exception as e:
            self.test_result("WebSocket Connection", False, str(e))
            return False

    def test_log_files_creation(self):
        """Test that log files are being created"""
        logs_dir = self.project_root / "logs"
        expected_logs = [
            "hygiene-monitor.log",
            "hygiene-monitor-backend.log",
            "agent-orchestrator.log"
        ]
        
        all_logs_exist = True
        missing_logs = []
        
        for log_file in expected_logs:
            log_path = logs_dir / log_file
            if log_path.exists():
                file_size = log_path.stat().st_size
                self.log(f"Found log file: {log_file} ({file_size} bytes)")
            else:
                all_logs_exist = False
                missing_logs.append(log_file)
        
        if all_logs_exist:
            self.test_result("Log Files Creation", True, 
                           f"All {len(expected_logs)} log files exist")
        else:
            self.test_result("Log Files Creation", False, 
                           f"Missing log files: {missing_logs}")
        
        return all_logs_exist

    def test_database_creation(self):
        """Test that SQLite databases are created"""
        monitoring_dir = self.project_root / "monitoring"
        expected_dbs = [
            "hygiene.db",
            "logs.db"
        ]
        
        all_dbs_exist = True
        missing_dbs = []
        
        for db_file in expected_dbs:
            db_path = monitoring_dir / db_file
            if db_path.exists():
                file_size = db_path.stat().st_size
                self.log(f"Found database: {db_file} ({file_size} bytes)")
            else:
                all_dbs_exist = False
                missing_dbs.append(db_file)
        
        if all_dbs_exist:
            self.test_result("Database Creation", True, 
                           f"All {len(expected_dbs)} databases exist")
        else:
            self.test_result("Database Creation", False, 
                           f"Missing databases: {missing_dbs}")
        
        return all_dbs_exist

    def test_dashboard_accessibility(self):
        """Test that dashboard files are accessible"""
        dashboard_path = self.project_root / "dashboard" / "hygiene-monitor" / "index.html"
        
        if dashboard_path.exists():
            # Check that the updated JavaScript file has WebSocket code
            js_path = self.project_root / "dashboard" / "hygiene-monitor" / "app.js"
            if js_path.exists():
                with open(js_path, 'r') as f:
                    js_content = f.read()
                    if 'websocketEndpoint' in js_content and 'handleWebSocketMessage' in js_content:
                        self.test_result("Dashboard Accessibility", True, 
                                       "Dashboard files exist with WebSocket support")
                        return True
                    else:
                        self.test_result("Dashboard Accessibility", False, 
                                       "Dashboard missing WebSocket support")
                        return False
            else:
                self.test_result("Dashboard Accessibility", False, 
                               "app.js file missing")
                return False
        else:
            self.test_result("Dashboard Accessibility", False, 
                           "index.html file missing")
            return False

    def create_test_violations(self):
        """Create some test files with violations for testing"""
        test_files_dir = self.project_root / "test_violations"
        test_files_dir.mkdir(exist_ok=True)
        
        # Create a file with fantasy elements (rule 1 violation)
        fantasy_file = test_files_dir / "fantasy_code.py"
        with open(fantasy_file, 'w') as f:
            f.write("""
# This file contains fantasy elements for testing
def magic_function():
    wizard_result = teleport_data()
    return wizard_result

def teleport_data():
    # TODO: add telekinesis here
    return "magic result"
""")
        
        # Create a temp file (rule 13 violation)
        temp_file = test_files_dir / "temp_file.tmp"
        with open(temp_file, 'w') as f:
            f.write("This is a temporary file that should be cleaned up")
        
        # Create a backup file (rule 13 violation)
        backup_file = test_files_dir / "script_backup.py"
        with open(backup_file, 'w') as f:
            f.write("# This is a backup file")
        
        self.log(f"Created test violation files in {test_files_dir}")
        return test_files_dir

    def cleanup_test_files(self, test_dir):
        """Clean up test files"""
        try:
            import shutil
            if test_dir.exists():
                shutil.rmtree(test_dir)
                self.log(f"Cleaned up test files in {test_dir}")
        except Exception as e:
            self.log(f"Failed to cleanup test files: {e}", "WARN")

    async def run_all_tests(self):
        """Run all integration tests"""
        self.log("Starting Hygiene Monitoring Integration Tests")
        self.log("=" * 60)
        
        # Create test violations for scanning
        test_dir = self.create_test_violations()
        
        try:
            # Test individual components
            tests = [
                ("Backend Health", self.test_backend_health),
                ("System Metrics API", self.test_system_metrics_api),
                ("Violations API", self.test_violations_api),
                ("Agents API", self.test_agents_api),
                ("Trigger Scan API", self.test_trigger_scan_api),
                ("Log Files Creation", self.test_log_files_creation),
                ("Database Creation", self.test_database_creation),
                ("Dashboard Accessibility", self.test_dashboard_accessibility),
            ]
            
            # Run synchronous tests
            for test_name, test_func in tests:
                self.log(f"Running {test_name} test...")
                test_func()
                time.sleep(0.5)  # Brief pause between tests
            
            # Run WebSocket test (async)
            self.log("Running WebSocket Connection test...")
            await self.test_websocket_connection()
            
        finally:
            # Cleanup test files
            self.cleanup_test_files(test_dir)
        
        # Print summary
        self.print_test_summary()

    def print_test_summary(self):
        """Print test results summary"""
        self.log("=" * 60)
        self.log("TEST RESULTS SUMMARY")
        self.log("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['success'])
        failed_tests = total_tests - passed_tests
        
        self.log(f"Total Tests: {total_tests}")
        self.log(f"Passed: {passed_tests}")
        self.log(f"Failed: {failed_tests}")
        self.log(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            self.log("\nFAILED TESTS:")
            for result in self.test_results:
                if not result['success']:
                    self.log(f"  ❌ {result['test']}: {result['details']}")
        
        self.log("\nPASSED TESTS:")
        for result in self.test_results:
            if result['success']:
                self.log(f"  ✅ {result['test']}: {result['details']}")
        
        self.log("=" * 60)
        
        if failed_tests == 0:
            self.log("🎉 ALL TESTS PASSED! Monitoring system is working correctly.")
            return True
        else:
            self.log("⚠️  Some tests failed. Please check the monitoring system.")
            return False

def check_monitoring_system_running():
    """Check if the monitoring system is running"""
    try:
        result = subprocess.run(['pgrep', '-f', 'hygiene-monitor-backend.py'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return True
        else:
            print("❌ Monitoring system is not running.")
            print("Please start it first with: ./scripts/start-hygiene-monitoring.sh start")
            return False
    except Exception as e:
        print(f"Error checking system status: {e}")
        return False

async def main():
    """Main test execution"""
    print("🔍 Hygiene Monitoring System Integration Test")
    print("=" * 60)
    
    # Check if monitoring system is running
    if not check_monitoring_system_running():
        sys.exit(1)
    
    # Wait a moment for system to be fully ready
    print("⏳ Waiting for system to be ready...")
    time.sleep(5)
    
    # Run integration tests
    tester = MonitoringIntegrationTester()
    success = await tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())