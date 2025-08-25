#!/usr/bin/env python3
"""
Simple test runner for SutazAI system without Unicode issues
"""

import requests
import time
import json
from datetime import datetime
from pathlib import Path

class SimpleSutazAITester:
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "summary": {
                "total": 0,
                "passed": 0,
                "failed": 0
            }
        }
        
    def test_service(self, name, url, expected_status=200):
        """Test a service endpoint"""
        print(f"\n[TEST] {name}")
        print("-" * 40)
        
        try:
            response = requests.get(url, timeout=5)
            success = response.status_code == expected_status
            
            self.results["tests"][name] = {
                "status": "PASS" if success else "FAIL",
                "status_code": response.status_code,
                "response_time": response.elapsed.total_seconds()
            }
            
            if success:
                print(f"[PASS] {name} - Status: {response.status_code} - Time: {response.elapsed.total_seconds():.2f}s")
                self.results["summary"]["passed"] += 1
            else:
                print(f"[FAIL] {name} - Expected: {expected_status}, Got: {response.status_code}")
                self.results["summary"]["failed"] += 1
                
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] {name} - Connection failed: {str(e)}")
            self.results["tests"][name] = {
                "status": "ERROR",
                "error": str(e)
            }
            self.results["summary"]["failed"] += 1
            
        self.results["summary"]["total"] += 1
        
    def run_all_tests(self):
        """Run all system tests"""
        print("=" * 60)
        print("SUTAZAI SYSTEM TESTING")
        print("=" * 60)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Test core services
        print("\n[PHASE 1] Core Infrastructure")
        print("=" * 40)
        
        # Backend API
        self.test_service(
            "Backend API Health",
            "http://localhost:10010/health"
        )
        
        self.test_service(
            "Backend API Agents",
            "http://localhost:10010/api/v1/agents"
        )
        
        # Frontend
        self.test_service(
            "Frontend UI",
            "http://localhost:10011"
        )
        
        # Databases
        print("\n[PHASE 2] Database Services")
        print("=" * 40)
        
        # Note: Direct database testing would require psycopg2/redis clients
        # For now, we'll test through the API
        self.test_service(
            "Backend Database Connection",
            "http://localhost:10010/api/v1/status"
        )
        
        # AI Services
        print("\n[PHASE 3] AI Services")
        print("=" * 40)
        
        self.test_service(
            "Ollama Model Server",
            "http://localhost:10104/api/tags"
        )
        
        self.test_service(
            "ChromaDB Vector Store",
            "http://localhost:10100/api/v1/heartbeat"
        )
        
        self.test_service(
            "Qdrant Vector Search",
            "http://localhost:10101/health"
        )
        
        # Service Mesh
        print("\n[PHASE 4] Service Mesh")
        print("=" * 40)
        
        self.test_service(
            "Consul Service Discovery",
            "http://localhost:10006/v1/status/leader"
        )
        
        self.test_service(
            "Kong API Gateway Admin",
            "http://localhost:10015/status"
        )
        
        # Monitoring
        print("\n[PHASE 5] Monitoring Stack")
        print("=" * 40)
        
        self.test_service(
            "Prometheus Metrics",
            "http://localhost:10200/-/healthy"
        )
        
        self.test_service(
            "Grafana Dashboard",
            "http://localhost:10201/api/health"
        )
        
        # Print summary
        self.print_summary()
        
        # Save results
        self.save_results()
        
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        summary = self.results["summary"]
        total = summary["total"]
        passed = summary["passed"]
        failed = summary["failed"]
        
        if total > 0:
            success_rate = (passed / total) * 100
        else:
            success_rate = 0
            
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        # System readiness assessment
        if success_rate >= 80:
            print("\n[READY] System is OPERATIONAL")
        elif success_rate >= 60:
            print("\n[PARTIAL] System is PARTIALLY OPERATIONAL")
        else:
            print("\n[NOT READY] System requires attention")
            
        # Show failed tests
        if failed > 0:
            print("\nFailed Tests:")
            for test_name, result in self.results["tests"].items():
                if result["status"] in ["FAIL", "ERROR"]:
                    error_msg = result.get("error", f"Status {result.get('status_code', 'unknown')}")
                    print(f"  - {test_name}: {error_msg}")
                    
    def save_results(self):
        """Save test results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_results_{timestamp}.json"
        
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2)
            
        print(f"\nResults saved to: {filename}")

def main():
    """Main execution"""
    tester = SimpleSutazAITester()
    
    try:
        tester.run_all_tests()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Testing cancelled by user")
    except Exception as e:
        print(f"\n[ERROR] Testing failed: {str(e)}")
        
    return tester.results["summary"]["failed"] == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)