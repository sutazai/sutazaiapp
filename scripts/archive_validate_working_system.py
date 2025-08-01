#!/usr/bin/env python3
"""
Validate SutazAI v8 Working System
Test all deployed components and confirm functionality
"""

import requests
import time
import sys
import json
from typing import Dict, Any

class SutazAIValidator:
    def __init__(self):
        self.results = {
            "timestamp": time.time(),
            "version": "SutazAI v8 (2.0.0)",
            "tests": {},
            "summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "success_rate": 0.0
            }
        }
    
    def run_test(self, test_name: str, test_func):
        """Run a test and record results"""
        print(f"🧪 Running {test_name}...")
        self.results["summary"]["total_tests"] += 1
        
        try:
            result = test_func()
            if result:
                print(f"✅ {test_name}: PASSED")
                self.results["tests"][test_name] = {"status": "PASSED", "details": result}
                self.results["summary"]["passed_tests"] += 1
            else:
                print(f"❌ {test_name}: FAILED")
                self.results["tests"][test_name] = {"status": "FAILED", "details": "Test returned False"}
                self.results["summary"]["failed_tests"] += 1
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {str(e)}")
            self.results["tests"][test_name] = {"status": "ERROR", "details": str(e)}
            self.results["summary"]["failed_tests"] += 1
    
    def test_backend_health(self):
        """Test backend health endpoint"""
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("status") == "healthy"
        return False
    
    def test_faiss_functionality(self):
        """Test FAISS vector search"""
        response = requests.get("http://localhost:8000/test/faiss", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("status") == "success"
        return False
    
    def test_chromadb_functionality(self):
        """Test ChromaDB integration"""
        response = requests.get("http://localhost:8000/test/chromadb", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("status") == "success"
        return False
    
    def test_system_status(self):
        """Test overall system status"""
        response = requests.get("http://localhost:8000/system/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("status") == "operational"
        return False
    
    def test_frontend_accessibility(self):
        """Test frontend accessibility"""
        response = requests.get("http://localhost:8501/healthz", timeout=5)
        return response.status_code == 200
    
    def test_qdrant_service(self):
        """Test Qdrant vector database"""
        response = requests.get("http://localhost:6333/healthz", timeout=5)
        return response.status_code == 200
    
    def test_environment_setup(self):
        """Test Python environment and dependencies"""
        try:
            import faiss
            import chromadb
            import fastapi
            import streamlit
            import uvicorn
            return True
        except ImportError:
            return False
    
    def run_all_tests(self):
        """Run all validation tests"""
        print("🚀 SutazAI v8 Working System Validation")
        print("="*50)
        
        # Run all tests
        self.run_test("Python Environment Setup", self.test_environment_setup)
        self.run_test("Backend Health Check", self.test_backend_health)
        self.run_test("FAISS Vector Search", self.test_faiss_functionality)
        self.run_test("ChromaDB Integration", self.test_chromadb_functionality)
        self.run_test("System Status", self.test_system_status)
        self.run_test("Frontend Accessibility", self.test_frontend_accessibility)
        self.run_test("Qdrant Service", self.test_qdrant_service)
        
        # Calculate success rate
        if self.results["summary"]["total_tests"] > 0:
            self.results["summary"]["success_rate"] = (
                self.results["summary"]["passed_tests"] / 
                self.results["summary"]["total_tests"] * 100
            )
        
        # Print summary
        self.print_summary()
        
        return self.results["summary"]["success_rate"]
    
    def print_summary(self):
        """Print validation summary"""
        print("\n" + "="*50)
        print("📊 VALIDATION SUMMARY")
        print("="*50)
        
        summary = self.results["summary"]
        print(f"📈 Total Tests: {summary['total_tests']}")
        print(f"✅ Passed: {summary['passed_tests']}")
        print(f"❌ Failed: {summary['failed_tests']}")
        print(f"📊 Success Rate: {summary['success_rate']:.1f}%")
        
        if summary['success_rate'] >= 80:
            print("\n🎉 VALIDATION RESULT: SYSTEM IS WORKING!")
            print("✅ SutazAI v8 is operational and ready for use")
        elif summary['success_rate'] >= 60:
            print("\n⚠️ VALIDATION RESULT: PARTIAL SUCCESS")
            print("🔧 Some components need attention")
        else:
            print("\n❌ VALIDATION RESULT: SYSTEM NEEDS FIXES")
            print("🛠️ Multiple issues need to be resolved")
        
        print("\n📋 Detailed Results:")
        for test_name, result in self.results["tests"].items():
            status_icon = "✅" if result["status"] == "PASSED" else "❌"
            print(f"   {status_icon} {test_name}: {result['status']}")
        
        print("\n" + "="*50)

def main():
    """Main validation function"""
    validator = SutazAIValidator()
    success_rate = validator.run_all_tests()
    
    # Save results
    with open("validation_results.json", "w") as f:
        json.dump(validator.results, f, indent=2)
    
    print(f"\n📋 Results saved to: validation_results.json")
    
    # Return appropriate exit code
    sys.exit(0 if success_rate >= 80 else 1)

if __name__ == "__main__":
    main()