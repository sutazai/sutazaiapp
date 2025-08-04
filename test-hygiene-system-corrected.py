#!/usr/bin/env python3
"""
Comprehensive Hygiene Monitor Test Suite
Tests all system components with correct port configuration
"""
import asyncio
import aiohttp
import time
import json
from datetime import datetime
import sys

# Correct API endpoints based on actual deployment
API_BASE = "http://localhost:10420/api/hygiene"
RULE_API_BASE = "http://localhost:10421/api"
DASHBOARD_URL = "http://localhost:10422"
WEBSOCKET_URL = "ws://localhost:10420/ws"

class HygieneSystemTester:
    def __init__(self):
        self.test_results = []
        self.total_tests = 0
        self.passed_tests = 0
        
    def log_test(self, test_name, passed, details=""):
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
            print(f"✅ {test_name}: PASSED {details}")
        else:
            print(f"❌ {test_name}: FAILED {details}")
        
        self.test_results.append({
            'test': test_name,
            'passed': passed,
            'details': details,
            'timestamp': datetime.now().isoformat()
        })

    async def test_backend_api_endpoints(self):
        """Test all backend API endpoints"""
        print("\n🔧 Testing Backend API Endpoints...")
        
        async with aiohttp.ClientSession() as session:
            # Test status endpoint
            try:
                async with session.get(f"{API_BASE}/status") as response:
                    if response.status == 200:
                        data = await response.json()
                        total_violations = data.get("totalViolations", 0)
                        self.log_test("Backend Status API", True, f"- {total_violations} violations found")
                    else:
                        self.log_test("Backend Status API", False, f"- HTTP {response.status}")
            except Exception as e:
                self.log_test("Backend Status API", False, f"- {str(e)}")
            
            # Test scan endpoint
            try:
                start_time = time.time()
                async with session.post(f"{API_BASE}/scan") as response:
                    elapsed = time.time() - start_time
                    if response.status == 200:
                        data = await response.json()
                        violations_found = data.get("violations_found", 0)
                        self.log_test("Backend Scan API", True, 
                                    f"- Found {violations_found} violations in {elapsed:.2f}s")
                    else:
                        self.log_test("Backend Scan API", False, f"- HTTP {response.status}")
            except Exception as e:
                self.log_test("Backend Scan API", False, f"- {str(e)}")
            
            # Test system metrics endpoint
            try:
                async with session.get("http://localhost:10420/api/system/metrics") as response:
                    if response.status == 200:
                        data = await response.json()
                        cpu_usage = data.get("cpu_usage", 0)
                        memory_percentage = data.get("memory_percentage", 0)
                        self.log_test("System Metrics API", True, 
                                    f"- CPU: {cpu_usage}%, Memory: {memory_percentage}%")
                    else:
                        self.log_test("System Metrics API", False, f"- HTTP {response.status}")
            except Exception as e:
                self.log_test("System Metrics API", False, f"- {str(e)}")

    async def test_rule_control_api(self):
        """Test rule control API"""
        print("\n⚙️ Testing Rule Control API...")
        
        async with aiohttp.ClientSession() as session:
            # Test health endpoint
            try:
                async with session.get(f"{RULE_API_BASE}/health/live") as response:
                    if response.status == 200:
                        data = await response.json()
                        status = data.get("status", "unknown")
                        self.log_test("Rule Control Health", True, f"- Status: {status}")
                    else:
                        self.log_test("Rule Control Health", False, f"- HTTP {response.status}")
            except Exception as e:
                self.log_test("Rule Control Health", False, f"- {str(e)}")

    async def test_dashboard_ui(self):
        """Test dashboard UI functionality"""
        print("\n🌐 Testing Dashboard UI...")
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(DASHBOARD_URL) as response:
                    if response.status == 200:
                        content = await response.text()
                        if "Sutazai Hygiene" in content and "<html" in content:
                            self.log_test("Dashboard UI", True, "- HTML content served successfully")
                        else:
                            self.log_test("Dashboard UI", False, "- Invalid HTML content")
                    else:
                        self.log_test("Dashboard UI", False, f"- HTTP {response.status}")
            except Exception as e:
                self.log_test("Dashboard UI", False, f"- {str(e)}")

    async def test_websocket_connection(self):
        """Test WebSocket real-time updates"""
        print("\n🔌 Testing WebSocket Connection...")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(WEBSOCKET_URL) as ws:
                    # Send ping message
                    await ws.send_json({"type": "ping"})
                    
                    # Wait for response with timeout
                    try:
                        msg = await asyncio.wait_for(ws.receive(), timeout=5.0)
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            msg_type = data.get('type', 'unknown')
                            self.log_test("WebSocket Connection", True, f"- Received: {msg_type}")
                        else:
                            self.log_test("WebSocket Connection", True, "- Connection established")
                    except asyncio.TimeoutError:
                        self.log_test("WebSocket Connection", True, "- Connected (no immediate response)")
                    
                    await ws.close()
                    
        except Exception as e:
            self.log_test("WebSocket Connection", False, f"- {str(e)}")

    async def test_rapid_api_calls(self, count=10):
        """Test rapid API calls to check for throttling/errors"""
        print(f"\n⚡ Testing {count} Rapid API Calls...")
        
        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            tasks = []
            
            for i in range(count):
                task = session.get(f"{API_BASE}/status")
                tasks.append(task)
                
            try:
                responses = await asyncio.gather(*tasks)
                elapsed = time.time() - start_time
                
                success_count = sum(1 for r in responses if r.status == 200)
                
                for response in responses:
                    response.close()
                
                if success_count == count:
                    self.log_test("Rapid API Calls", True, 
                                f"- {success_count}/{count} successful in {elapsed:.2f}s")
                else:
                    self.log_test("Rapid API Calls", False, 
                                f"- Only {success_count}/{count} successful")
                    
            except Exception as e:
                self.log_test("Rapid API Calls", False, f"- {str(e)}")

    async def test_memory_usage(self):
        """Test memory usage and performance"""
        print("\n💾 Testing Memory Usage...")
        
        async with aiohttp.ClientSession() as session:
            try:
                # Get initial metrics
                async with session.get("http://localhost:10420/api/system/metrics") as response:
                    if response.status == 200:
                        data = await response.json()
                        memory_usage = data.get("memory_percentage", 0)
                        
                        if memory_usage > 90:
                            self.log_test("Memory Usage", False, f"- High memory usage: {memory_usage}%")
                        elif memory_usage > 75:
                            self.log_test("Memory Usage", True, f"- Moderate memory usage: {memory_usage}%")
                        else:
                            self.log_test("Memory Usage", True, f"- Good memory usage: {memory_usage}%")
                    else:
                        self.log_test("Memory Usage", False, "- Could not retrieve metrics")
            except Exception as e:
                self.log_test("Memory Usage", False, f"- {str(e)}")

    async def test_database_persistence(self):
        """Test database operations and data persistence"""
        print("\n🗄️ Testing Database Persistence...")
        
        async with aiohttp.ClientSession() as session:
            try:
                # Trigger a scan to generate data
                async with session.post(f"{API_BASE}/scan") as response:
                    if response.status == 200:
                        scan_data = await response.json()
                        
                        # Wait a moment for data to be stored
                        await asyncio.sleep(1)
                        
                        # Check if data persists by getting status
                        async with session.get(f"{API_BASE}/status") as status_response:
                            if status_response.status == 200:
                                status_data = await status_response.json()
                                total_violations = status_data.get("totalViolations", 0)
                                
                                if total_violations > 0:
                                    self.log_test("Database Persistence", True, 
                                                f"- {total_violations} violations persisted")
                                else:
                                    self.log_test("Database Persistence", False, 
                                                "- No violations found in database")
                            else:
                                self.log_test("Database Persistence", False, 
                                            "- Could not retrieve status after scan")
                    else:
                        self.log_test("Database Persistence", False, 
                                    "- Could not trigger scan")
                        
            except Exception as e:
                self.log_test("Database Persistence", False, f"- {str(e)}")

    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*60)
        print("🧪 HYGIENE MONITORING SYSTEM - TEST SUMMARY")
        print("="*60)
        print(f"📊 Total Tests: {self.total_tests}")
        print(f"✅ Passed: {self.passed_tests}")
        print(f"❌ Failed: {self.total_tests - self.passed_tests}")
        print(f"📈 Success Rate: {(self.passed_tests/self.total_tests)*100:.1f}%" if self.total_tests > 0 else "No tests run")
        
        if self.passed_tests == self.total_tests:
            print("\n🎉 ALL TESTS PASSED! System is functioning correctly.")
        else:
            print(f"\n⚠️  {self.total_tests - self.passed_tests} test(s) failed. Review details above.")
        
        return self.passed_tests == self.total_tests

    async def run_all_tests(self):
        """Run comprehensive test suite"""
        print("🧪 Sutazai Hygiene Monitoring System - Comprehensive Test Suite")
        print("="*70)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Backend API: {API_BASE}")
        print(f"Dashboard URL: {DASHBOARD_URL}")
        print(f"WebSocket URL: {WEBSOCKET_URL}")
        
        # Run all tests
        await self.test_backend_api_endpoints()
        await self.test_rule_control_api()
        await self.test_dashboard_ui()
        await self.test_websocket_connection()
        await self.test_rapid_api_calls(10)
        await self.test_memory_usage()
        await self.test_database_persistence()
        
        # Print summary
        return self.print_summary()

async def main():
    """Main test execution"""
    tester = HygieneSystemTester()
    success = await tester.run_all_tests()
    return success

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite failed with exception: {e}")
        sys.exit(1)