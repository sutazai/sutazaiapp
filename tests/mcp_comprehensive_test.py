#!/usr/bin/env python3
"""
Comprehensive MCP (Model Context Protocol) Test Suite
Tests ALL MCP endpoints and wrapper scripts systematically
Reports REAL results, not assumptions
"""

import subprocess
import json
import time
import requests
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime, timezone
import pytest
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/tests/results/mcp_test_results.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MCPTester:
    """Comprehensive MCP testing framework"""
    
    def __init__(self):
        self.base_url = "http://localhost:10010"
        self.wrapper_dir = Path("/opt/sutazaiapp/scripts/mcp/wrappers")
        self.results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "api_tests": {},
            "wrapper_tests": {},
            "summary": {}
        }
    
    def test_backend_connectivity(self) -> Dict[str, Any]:
        """Test if backend API is responding"""
        logger.info("Testing backend connectivity...")
        
        test_result = {
            "endpoint": self.base_url,
            "status": "unknown",
            "response_time": None,
            "error": None
        }
        
        try:
            start_time = time.time()
            response = requests.get(f"{self.base_url}/health", timeout=10)
            response_time = time.time() - start_time
            
            test_result.update({
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "response_time": response_time,
                "status_code": response.status_code,
                "response_body": response.text[:200] if response.text else None
            })
            
        except requests.exceptions.ConnectionError:
            test_result.update({
                "status": "connection_refused",
                "error": "Connection refused - backend not running"
            })
        except requests.exceptions.Timeout:
            test_result.update({
                "status": "timeout",
                "error": "Request timeout after 10 seconds"
            })
        except Exception as e:
            test_result.update({
                "status": "error",
                "error": str(e)
            })
        
        self.results["api_tests"]["backend_health"] = test_result
        return test_result
    
    def test_mcp_status_endpoint(self) -> Dict[str, Any]:
        """Test GET /api/v1/mcp/status"""
        logger.info("Testing MCP status endpoint...")
        
        test_result = {
            "endpoint": "/api/v1/mcp/status",
            "status": "unknown",
            "response_time": None,
            "error": None
        }
        
        try:
            start_time = time.time()
            response = requests.get(f"{self.base_url}/api/v1/mcp/status", timeout=10)
            response_time = time.time() - start_time
            
            test_result.update({
                "status": "success" if response.status_code == 200 else "failed",
                "response_time": response_time,
                "status_code": response.status_code,
                "response_body": response.text[:500] if response.text else None
            })
            
            if response.status_code == 200:
                try:
                    json_data = response.json()
                    test_result["parsed_json"] = json_data
                except json.JSONDecodeError:
                    test_result["json_error"] = "Response not valid JSON"
                    
        except Exception as e:
            test_result.update({
                "status": "error",
                "error": str(e)
            })
        
        self.results["api_tests"]["mcp_status"] = test_result
        return test_result
    
    def test_mcp_servers_endpoint(self) -> Dict[str, Any]:
        """Test GET /api/v1/mcp/servers"""
        logger.info("Testing MCP servers endpoint...")
        
        test_result = {
            "endpoint": "/api/v1/mcp/servers",
            "status": "unknown",
            "response_time": None,
            "error": None
        }
        
        try:
            start_time = time.time()
            response = requests.get(f"{self.base_url}/api/v1/mcp/servers", timeout=10)
            response_time = time.time() - start_time
            
            test_result.update({
                "status": "success" if response.status_code == 200 else "failed",
                "response_time": response_time,
                "status_code": response.status_code,
                "response_body": response.text[:500] if response.text else None
            })
            
            if response.status_code == 200:
                try:
                    json_data = response.json()
                    test_result["parsed_json"] = json_data
                    test_result["server_count"] = len(json_data) if isinstance(json_data, list) else 0
                except json.JSONDecodeError:
                    test_result["json_error"] = "Response not valid JSON"
                    
        except Exception as e:
            test_result.update({
                "status": "error",
                "error": str(e)
            })
        
        self.results["api_tests"]["mcp_servers"] = test_result
        return test_result
    
    def test_mcp_execute_endpoint(self, server: str, method: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Test POST /api/v1/mcp/execute"""
        logger.info(f"Testing MCP execute endpoint with server={server}, method={method}...")
        
        test_result = {
            "endpoint": "/api/v1/mcp/execute",
            "server": server,
            "method": method,
            "params": params,
            "status": "unknown",
            "response_time": None,
            "error": None
        }
        
        payload = {
            "server": server,
            "method": method
        }
        if params:
            payload["params"] = params
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/api/v1/mcp/execute",
                json=payload,
                timeout=15,
                headers={"Content-Type": "application/json"}
            )
            response_time = time.time() - start_time
            
            test_result.update({
                "status": "success" if response.status_code == 200 else "failed",
                "response_time": response_time,
                "status_code": response.status_code,
                "response_body": response.text[:500] if response.text else None
            })
            
            if response.status_code == 200:
                try:
                    json_data = response.json()
                    test_result["parsed_json"] = json_data
                except json.JSONDecodeError:
                    test_result["json_error"] = "Response not valid JSON"
                    
        except Exception as e:
            test_result.update({
                "status": "error",
                "error": str(e)
            })
        
        test_key = f"mcp_execute_{server}_{method}"
        self.results["api_tests"][test_key] = test_result
        return test_result
    
    def test_wrapper_script(self, script_path: Path) -> Dict[str, Any]:
        """Test MCP wrapper script directly"""
        script_name = script_path.stem
        logger.info(f"Testing wrapper script: {script_name}")
        
        test_result = {
            "script": script_name,
            "path": str(script_path),
            "status": "unknown",
            "execution_time": None,
            "error": None
        }
        
        if not script_path.exists():
            test_result.update({
                "status": "not_found",
                "error": f"Script not found at {script_path}"
            })
            return test_result
        
        if not script_path.is_file() or not os.access(script_path, os.X_OK):
            test_result.update({
                "status": "not_executable",
                "error": f"Script not executable: {script_path}"
            })
            return test_result
        
        # Test with basic ping/list method
        test_input = '{"method":"list"}'
        
        try:
            start_time = time.time()
            process = subprocess.run(
                [str(script_path)],
                input=test_input,
                text=True,
                capture_output=True,
                timeout=10
            )
            execution_time = time.time() - start_time
            
            test_result.update({
                "status": "success" if process.returncode == 0 else "failed",
                "execution_time": execution_time,
                "return_code": process.returncode,
                "stdout": process.stdout[:500] if process.stdout else None,
                "stderr": process.stderr[:200] if process.stderr else None
            })
            
            # Try to parse stdout as JSON
            if process.stdout:
                try:
                    json_output = json.loads(process.stdout)
                    test_result["parsed_output"] = json_output
                except json.JSONDecodeError:
                    test_result["output_type"] = "text"
                    
        except subprocess.TimeoutExpired:
            test_result.update({
                "status": "timeout",
                "error": "Script execution timeout after 10 seconds"
            })
        except Exception as e:
            test_result.update({
                "status": "error",
                "error": str(e)
            })
        
        self.results["wrapper_tests"][script_name] = test_result
        return test_result
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all MCP tests comprehensively"""
        logger.info("Starting comprehensive MCP test suite...")
        
        # Test backend connectivity first
        backend_result = self.test_backend_connectivity()
        
        # Test MCP API endpoints
        if backend_result["status"] in ["healthy", "unhealthy"]:
            self.test_mcp_status_endpoint()
            self.test_mcp_servers_endpoint()
            
            # Test execute endpoint with known servers
            mcp_servers = ["files", "ddg", "github", "http", "language-server", "mcp_ssh"]
            for server in mcp_servers:
                self.test_mcp_execute_endpoint(server, "list")
        
        # Test all wrapper scripts
        if self.wrapper_dir.exists():
            for script_path in self.wrapper_dir.glob("*.sh"):
                self.test_wrapper_script(script_path)
        
        # Generate summary
        self.generate_summary()
        
        return self.results
    
    def generate_summary(self):
        """Generate test summary statistics"""
        api_tests = self.results["api_tests"]
        wrapper_tests = self.results["wrapper_tests"]
        
        # API test summary
        api_total = len(api_tests)
        api_success = sum(1 for test in api_tests.values() 
                         if test.get("status") in ["success", "healthy"])
        api_failed = api_total - api_success
        
        # Wrapper test summary
        wrapper_total = len(wrapper_tests)
        wrapper_success = sum(1 for test in wrapper_tests.values() 
                             if test.get("status") == "success")
        wrapper_failed = wrapper_total - wrapper_success
        
        self.results["summary"] = {
            "total_tests": api_total + wrapper_total,
            "api_tests": {
                "total": api_total,
                "success": api_success,
                "failed": api_failed,
                "success_rate": api_success / api_total * 100 if api_total > 0 else 0
            },
            "wrapper_tests": {
                "total": wrapper_total,
                "success": wrapper_success,
                "failed": wrapper_failed,
                "success_rate": wrapper_success / wrapper_total * 100 if wrapper_total > 0 else 0
            },
            "overall_success_rate": (api_success + wrapper_success) / (api_total + wrapper_total) * 100 if (api_total + wrapper_total) > 0 else 0
        }
    
    def save_results(self, output_path: str = "/opt/sutazaiapp/tests/results/mcp_comprehensive_test_results.json"):
        """Save test results to JSON file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Test results saved to: {output_file}")
        return str(output_file)

def main():
    """Main test execution function"""
    tester = MCPTester()
    results = tester.run_comprehensive_test()
    output_file = tester.save_results()
    
    # Print summary
    summary = results["summary"]
    print("\n" + "="*80)
    print("MCP COMPREHENSIVE TEST RESULTS")
    print("="*80)
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Overall Success Rate: {summary['overall_success_rate']:.1f}%")
    print()
    print(f"API Tests: {summary['api_tests']['success']}/{summary['api_tests']['total']} passed ({summary['api_tests']['success_rate']:.1f}%)")
    print(f"Wrapper Tests: {summary['wrapper_tests']['success']}/{summary['wrapper_tests']['total']} passed ({summary['wrapper_tests']['success_rate']:.1f}%)")
    print()
    print(f"Detailed results saved to: {output_file}")
    print("="*80)
    
    return results

if __name__ == "__main__":
    main()