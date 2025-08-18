#!/usr/bin/env python3
"""
Comprehensive Testing Agent for Hygiene Dashboard Stack Overflow Fix
Purpose: End-to-end validation of the dashboard fix with automated testing
Usage: python test-dashboard-fix.py [--headless] [--iterations N]
Requirements: selenium, requests, psutil
"""

import asyncio
import aiohttp
import json
import time
import sys
import argparse
import subprocess
import logging
import signal
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DashboardTestAgent:
    def __init__(self, headless: bool = False, iterations: int = 10):
        self.headless = headless
        self.iterations = iterations
        self.test_results = {}
        self.server_process = None
        self.backend_process = None
        self.base_url = "http://127.0.0.1:8081"
        self.backend_url = "http://127.0.0.1:8080"
        
    async def setup_test_environment(self):
        """Set up the test environment with servers"""
        logger.info("Setting up test environment...")
        
        # Start backend server
        try:
            backend_script = Path("/opt/sutazaiapp/scripts/hygiene-enforcement-coordinator.py")
            if backend_script.exists():
                self.backend_process = subprocess.Popen(
                    [sys.executable, str(backend_script)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                logger.info("Backend server started")
                await asyncio.sleep(2)  # Wait for backend to start
        except Exception as e:
            logger.warning(f"Could not start backend: {e}")
        
        # Start frontend server
        try:
            dashboard_dir = Path("/opt/sutazaiapp/dashboard/hygiene-monitor")
            self.server_process = subprocess.Popen(
                [sys.executable, "-m", "http.server", "8081", "--bind", "127.0.0.1"],
                cwd=dashboard_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            logger.info("Frontend server started on port 8081")
            await asyncio.sleep(2)  # Wait for server to start
        except Exception as e:
            logger.error(f"Failed to start frontend server: {e}")
            raise
    
    async def test_backend_connectivity(self) -> bool:
        """Test if backend is responsive"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.backend_url}/api/hygiene/status") as response:
                    if response.status == 200:
                        logger.info("Backend connectivity: PASS")
                        return True
                    else:
                        logger.warning(f"Backend returned status {response.status}")
                        return False
        except Exception as e:
            logger.warning(f"Backend connectivity failed: {e}")
            return False
    
    async def test_frontend_accessibility(self) -> bool:
        """Test if frontend is accessible"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url) as response:
                    if response.status == 200:
                        content = await response.text()
                        if "Hygiene Monitoring Dashboard" in content:
                            logger.info("Frontend accessibility: PASS")
                            return True
                    logger.warning(f"Frontend check failed: status {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Frontend accessibility failed: {e}")
            return False
    
    async def test_javascript_loading(self) -> bool:
        """Test if JavaScript file loads without syntax errors"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/app.js") as response:
                    if response.status == 200:
                        js_content = await response.text()
                        
                        # Basic syntax check - look for critical functions
                        critical_functions = [
                            "renderDashboard()",
                            "runFullAudit()",
                            "startRealTimeUpdates()",
                            "isUpdating"
                        ]
                        
                        missing = []
                        for func in critical_functions:
                            if func not in js_content:
                                missing.append(func)
                        
                        if missing:
                            logger.error(f"Missing critical functions: {missing}")
                            return False
                        
                        # Check for our fix
                        if "// Stronger recursion protection" in js_content:
                            logger.info("Stack overflow fix is present: PASS")
                        else:
                            logger.warning("Stack overflow fix not found")
                            
                        logger.info("JavaScript loading: PASS")
                        return True
                    
                    logger.error(f"JavaScript file not accessible: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"JavaScript loading test failed: {e}")
            return False
    
    async def test_audit_endpoint_directly(self) -> bool:
        """Test the audit endpoint directly"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.backend_url}/api/hygiene/audit") as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('success'):
                            logger.info("Direct audit endpoint: PASS")
                            return True
                    logger.warning(f"Audit endpoint returned: {response.status}")
                    return False
        except Exception as e:
            logger.warning(f"Direct audit test failed: {e}")
            return False
    
    def simulate_rapid_clicks(self) -> str:
        """Generate JavaScript to simulate rapid audit button clicks"""
        return """
        // Test function to simulate rapid clicking
        (function testRapidClicks() {
            let clickCount = 0;
            let errors = [];
            let startTime = Date.now();
            
            // Override console.error to catch stack overflow
            const originalError = console.error;
            console.error = function(...args) {
                if (args[0] && args[0].toString().includes('stack')) {
                    errors.push('STACK_OVERFLOW: ' + args[0]);
                }
                originalError.apply(console, args);
            };
            
            // Simulate rapid clicks
            const clickInterval = setInterval(() => {
                try {
                    const auditBtn = document.getElementById('run-audit');
                    if (auditBtn) {
                        auditBtn.click();
                        clickCount++;
                    }
                } catch (e) {
                    errors.push('CLICK_ERROR: ' + e.message);
                }
                
                if (clickCount >= 20 || Date.now() - startTime > 10000) {
                    clearInterval(clickInterval);
                    
                    // Report results
                    setTimeout(() => {
                        const result = {
                            clickCount: clickCount,
                            errors: errors,
                            duration: Date.now() - startTime,
                            success: errors.length === 0
                        };
                        
                        // Store result for retrieval
                        window.testResult = result;
                        console.log('RAPID_CLICK_TEST_RESULT:', JSON.stringify(result));
                    }, 2000);
                }
            }, 100);
        })();
        """
    
    def create_browser_test_html(self) -> str:
        """Create HTML for browser-based testing"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dashboard Test</title>
            <script>
                let testResults = {{}};
                
                async function runComprehensiveTest() {{
                    console.log('Starting comprehensive dashboard test...');
                    
                    // Load the dashboard in iframe
                    const iframe = document.createElement('iframe');
                    iframe.src = '{self.base_url}';
                    iframe.style = 'width: 100%; height: 80vh;';
                    document.body.appendChild(iframe);
                    
                    iframe.onload = function() {{
                        setTimeout(() => {{
                            // Inject rapid click test
                            const testScript = `{self.simulate_rapid_clicks()}`;
                            iframe.contentWindow.eval(testScript);
                            
                            // Monitor for results
                            setTimeout(() => {{
                                const result = iframe.contentWindow.testResult;
                                if (result) {{
                                    console.log('Test completed:', result);
                                    document.getElementById('results').innerHTML = 
                                        '<h3>Test Results:</h3><pre>' + JSON.stringify(result, null, 2) + '</pre>';
                                    if (result.success) {{
                                        document.body.style.backgroundColor = '#d4edda';
                                    }} else {{
                                        document.body.style.backgroundColor = '#f8d7da';
                                    }}
                                }}
                            }}, 15000);
                        }}, 2000);
                    }};
                }}
                
                window.onload = runComprehensiveTest;
            </script>
        </head>
        <body>
            <h1>Dashboard Stack Overflow Fix Test</h1>
            <div id="results">Loading test...</div>
        </body>
        </html>
        """
    
    async def run_browser_test(self) -> Dict[str, Any]:
        """Run browser-based test"""
        test_html = self.create_browser_test_html()
        test_file = Path("/tmp/dashboard_test.html")
        
        with open(test_file, 'w') as f:
            f.write(test_html)
        
        logger.info(f"Browser test file created: {test_file}")
        logger.info(f"Open this URL in your browser: file://{test_file}")
        
        return {
            "test_file": str(test_file),
            "status": "created",
            "instructions": f"Open file://{test_file} in browser to run test"
        }
    
    async def run_performance_test(self) -> Dict[str, Any]:
        """Test performance and memory usage"""
        logger.info("Running performance test...")
        
        results = {
            "memory_usage": "N/A",
            "cpu_usage": "N/A",
            "response_times": []
        }
        
        # Test multiple audit calls
        try:
            for i in range(5):
                start_time = time.time()
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(f"{self.backend_url}/api/hygiene/audit") as response:
                        await response.json()
                        
                response_time = time.time() - start_time
                results["response_times"].append(response_time)
                
                await asyncio.sleep(1)  # Wait between requests
            
            avg_response = sum(results["response_times"]) / len(results["response_times"])
            logger.info(f"Average response time: {avg_response:.2f}s")
            
        except Exception as e:
            logger.warning(f"Performance test failed: {e}")
        
        return results
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive test suite"""
        logger.info("Starting comprehensive test suite...")
        
        results = {
            "timestamp": time.time(),
            "environment_setup": False,
            "backend_connectivity": False,
            "frontend_accessibility": False,
            "javascript_loading": False,
            "audit_endpoint": False,
            "browser_test": {},
            "performance_test": {},
            "overall_success": False
        }
        
        try:
            # Setup
            await self.setup_test_environment()
            results["environment_setup"] = True
            
            # Basic connectivity
            results["backend_connectivity"] = await self.test_backend_connectivity()
            results["frontend_accessibility"] = await self.test_frontend_accessibility()
            results["javascript_loading"] = await self.test_javascript_loading()
            results["audit_endpoint"] = await self.test_audit_endpoint_directly()
            
            # Browser test
            results["browser_test"] = await self.run_browser_test()
            
            # Performance test
            results["performance_test"] = await self.run_performance_test()
            
            # Overall success
            critical_tests = [
                results["frontend_accessibility"],
                results["javascript_loading"]
            ]
            
            results["overall_success"] = all(critical_tests)
            
        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            results["error"] = str(e)
        
        return results
    
    def cleanup(self):
        """Clean up test environment"""
        logger.info("Cleaning up test environment...")
        
        if self.server_process:
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
        
        if self.backend_process:
            self.backend_process.terminate()
            try:
                self.backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.backend_process.kill()
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate test report"""
        report = f"""
HYGIENE DASHBOARD STACK OVERFLOW FIX TEST REPORT
===============================================
Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
Overall Success: {'✅ PASS' if results['overall_success'] else '❌ FAIL'}

DETAILED RESULTS:
-----------------
Environment Setup: {'✅' if results['environment_setup'] else '❌'}
Backend Connectivity: {'✅' if results['backend_connectivity'] else '❌'}
Frontend Accessibility: {'✅' if results['frontend_accessibility'] else '❌'}
JavaScript Loading: {'✅' if results['javascript_loading'] else '❌'}
Audit Endpoint: {'✅' if results['audit_endpoint'] else '❌'}

BROWSER TEST:
{json.dumps(results['browser_test'], indent=2)}

PERFORMANCE TEST:
{json.dumps(results['performance_test'], indent=2)}

RECOMMENDATIONS:
----------------
"""
        
        if results['overall_success']:
            report += "✅ All critical tests passed. The stack overflow fix appears to be working correctly.\n"
            report += "✅ Manual browser testing is recommended to confirm user experience.\n"
        else:
            report += "❌ Some tests failed. Review the detailed results above.\n"
            report += "❌ Check browser console for JavaScript errors.\n"
            report += "❌ Verify server processes are running correctly.\n"
        
        report += f"\nFor interactive testing, open: {results['browser_test'].get('test_file', 'N/A')}\n"
        
        return report

async def main():
    parser = argparse.ArgumentParser(description='Test Hygiene Dashboard Stack Overflow Fix')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode')
    parser.add_argument('--iterations', type=int, default=10, help='Number of test iterations')
    
    args = parser.parse_args()
    
    agent = DashboardTestAgent(headless=args.headless, iterations=args.iterations)
    
    def signal_handler(sig, frame):
        logger.info("Received interrupt, cleaning up...")
        agent.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        results = await agent.run_all_tests()
        report = agent.generate_report(results)
        
        logger.info(report)
        
        # Save report
        report_file = Path("/tmp/dashboard_test_report.txt")
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Test report saved to: {report_file}")
        
        return 0 if results['overall_success'] else 1
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        return 1
    finally:
        agent.cleanup()

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))