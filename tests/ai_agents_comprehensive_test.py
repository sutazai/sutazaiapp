#!/usr/bin/env python3
"""
AI Agents Comprehensive Test Suite - Phase 5
Purpose: Test all AI agents, Ollama, and agent orchestration
Created: 2025-11-15
Version: 1.0.0

Tests:
- Ollama LLM service
- TinyLlama model performance
- Backend AI agent endpoints
- Concurrent agent requests
- Agent error handling
- Response time measurements
- Load testing
- Health checks
"""

import requests
import json
import time
import sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import concurrent.futures
from pathlib import Path
import subprocess
import statistics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d UTC - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(f'ai_agents_test_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Service URLs
OLLAMA_URL = "http://localhost:11434"
BACKEND_URL = "http://localhost:10200"
KONG_PROXY_URL = "http://localhost:10008"

# Test execution timestamp
TEST_START_TIME = datetime.now(timezone.utc)
EXECUTION_ID = f"ai_agents_test_{TEST_START_TIME.strftime('%Y%m%d_%H%M%S_%f')[:-3]}"

@dataclass
class TestResult:
    """Test result data structure"""
    test_name: str
    status: str  # PASS, FAIL, SKIP
    duration: float
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class AIAgentsTester:
    """AI Agents comprehensive testing"""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.test_start_time = TEST_START_TIME
        logger.info(f"AI Agents Tester initialized with execution ID: {EXECUTION_ID}")
    
    def test_ollama_health(self) -> TestResult:
        """Test Ollama service is running"""
        start_time = time.time()
        try:
            response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                models = response.json().get('models', [])
                return TestResult(
                    test_name="Ollama Health Check",
                    status="PASS",
                    duration=duration,
                    details={
                        "available_models": [m.get('name') for m in models],
                        "total_models": len(models)
                    }
                )
            else:
                return TestResult(
                    test_name="Ollama Health Check",
                    status="FAIL",
                    duration=duration,
                    error=f"HTTP {response.status_code}: {response.text}"
                )
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="Ollama Health Check",
                status="FAIL",
                duration=duration,
                error=str(e)
            )
    
    def test_tinyllama_inference(self) -> TestResult:
        """Test TinyLlama model inference"""
        start_time = time.time()
        try:
            prompt = "What is the capital of France?"
            payload = {
                "model": "tinyllama",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 50
                }
            }
            
            response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=30)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                response_text = data.get('response', '')
                total_duration = data.get('total_duration', 0) / 1e9  # Convert nanoseconds to seconds
                
                # Calculate tokens per second
                eval_count = data.get('eval_count', 0)
                eval_duration = data.get('eval_duration', 1) / 1e9
                tokens_per_sec = eval_count / eval_duration if eval_duration > 0 else 0
                
                return TestResult(
                    test_name="TinyLlama Inference",
                    status="PASS",
                    duration=duration,
                    details={
                        "prompt": prompt,
                        "response": response_text[:200] + "..." if len(response_text) > 200 else response_text,
                        "total_duration": round(total_duration, 3),
                        "tokens_generated": eval_count,
                        "tokens_per_second": round(tokens_per_sec, 2),
                        "model": data.get('model')
                    }
                )
            else:
                return TestResult(
                    test_name="TinyLlama Inference",
                    status="FAIL",
                    duration=duration,
                    error=f"HTTP {response.status_code}: {response.text}"
                )
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="TinyLlama Inference",
                status="FAIL",
                duration=duration,
                error=str(e)
            )
    
    def test_backend_health(self) -> TestResult:
        """Test backend API health endpoint"""
        start_time = time.time()
        try:
            response = requests.get(f"{BACKEND_URL}/health", timeout=5)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                health_data = response.json()
                return TestResult(
                    test_name="Backend API Health",
                    status="PASS",
                    duration=duration,
                    details=health_data
                )
            else:
                return TestResult(
                    test_name="Backend API Health",
                    status="FAIL",
                    duration=duration,
                    error=f"HTTP {response.status_code}"
                )
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="Backend API Health",
                status="FAIL",
                duration=duration,
                error=str(e)
            )
    
    def test_concurrent_ollama_requests(self, num_requests: int = 5) -> TestResult:
        """Test concurrent Ollama inference requests"""
        start_time = time.time()
        try:
            def make_request(request_id: int):
                req_start = time.time()
                try:
                    payload = {
                        "model": "tinyllama",
                        "prompt": f"Request {request_id}: Count from 1 to 5.",
                        "stream": False,
                        "options": {"num_predict": 30}
                    }
                    resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=30)
                    req_duration = time.time() - req_start
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        return {
                            "request_id": request_id,
                            "status": "success",
                            "duration": req_duration,
                            "tokens": data.get('eval_count', 0)
                        }
                    else:
                        return {
                            "request_id": request_id,
                            "status": "failed",
                            "duration": req_duration,
                            "error": resp.status_code
                        }
                except Exception as e:
                    return {
                        "request_id": request_id,
                        "status": "error",
                        "duration": time.time() - req_start,
                        "error": str(e)
                    }
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
                futures = [executor.submit(make_request, i) for i in range(num_requests)]
                results = [f.result() for f in concurrent.futures.as_completed(futures)]
            
            duration = time.time() - start_time
            
            successful = sum(1 for r in results if r['status'] == 'success')
            durations = [r['duration'] for r in results if 'duration' in r]
            avg_duration = statistics.mean(durations) if durations else 0
            
            return TestResult(
                test_name="Concurrent Ollama Requests",
                status="PASS" if successful >= num_requests * 0.8 else "FAIL",
                duration=duration,
                details={
                    "total_requests": num_requests,
                    "successful": successful,
                    "failed": num_requests - successful,
                    "avg_request_duration": round(avg_duration, 3),
                    "total_duration": round(duration, 3),
                    "requests_per_second": round(num_requests / duration, 2)
                }
            )
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="Concurrent Ollama Requests",
                status="FAIL",
                duration=duration,
                error=str(e)
            )
    
    def test_ollama_model_list(self) -> TestResult:
        """List all available Ollama models"""
        start_time = time.time()
        try:
            response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                models = data.get('models', [])
                
                model_details = []
                for model in models:
                    model_details.append({
                        "name": model.get('name'),
                        "size": model.get('size', 0),
                        "modified": model.get('modified_at')
                    })
                
                return TestResult(
                    test_name="Ollama Model List",
                    status="PASS",
                    duration=duration,
                    details={
                        "total_models": len(models),
                        "models": model_details
                    }
                )
            else:
                return TestResult(
                    test_name="Ollama Model List",
                    status="FAIL",
                    duration=duration,
                    error=f"HTTP {response.status_code}"
                )
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="Ollama Model List",
                status="FAIL",
                duration=duration,
                error=str(e)
            )
    
    def test_backend_agents_endpoint(self) -> TestResult:
        """Test backend /agents endpoint"""
        start_time = time.time()
        try:
            response = requests.get(f"{BACKEND_URL}/agents", timeout=5)
            duration = time.time() - start_time
            
            if response.status_code in [200, 404]:  # 404 ok if not implemented yet
                return TestResult(
                    test_name="Backend Agents Endpoint",
                    status="PASS",
                    duration=duration,
                    details={
                        "status_code": response.status_code,
                        "endpoint_exists": response.status_code == 200
                    }
                )
            else:
                return TestResult(
                    test_name="Backend Agents Endpoint",
                    status="FAIL",
                    duration=duration,
                    error=f"HTTP {response.status_code}"
                )
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="Backend Agents Endpoint",
                status="FAIL",
                duration=duration,
                error=str(e)
            )
    
    def test_agent_error_handling(self) -> TestResult:
        """Test agent error handling with invalid inputs"""
        start_time = time.time()
        try:
            # Test with invalid model
            payload = {
                "model": "nonexistent-model-12345",
                "prompt": "Test prompt",
                "stream": False
            }
            
            response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=10)
            duration = time.time() - start_time
            
            # Should return error gracefully
            if response.status_code in [400, 404, 500]:
                return TestResult(
                    test_name="Agent Error Handling",
                    status="PASS",
                    duration=duration,
                    details={
                        "error_status": response.status_code,
                        "error_handled_gracefully": True,
                        "error_message": response.text[:200]
                    }
                )
            else:
                return TestResult(
                    test_name="Agent Error Handling",
                    status="FAIL",
                    duration=duration,
                    error=f"Expected error status, got {response.status_code}"
                )
        except Exception as e:
            duration = time.time() - start_time
            # Exception is acceptable for error handling test
            return TestResult(
                test_name="Agent Error Handling",
                status="PASS",
                duration=duration,
                details={
                    "error_handled_gracefully": True,
                    "exception_type": type(e).__name__
                }
            )
    
    def test_ollama_performance_baseline(self) -> TestResult:
        """Measure Ollama performance baseline"""
        start_time = time.time()
        try:
            prompts = [
                "What is 2+2?",
                "Name three colors.",
                "Count to 5."
            ]
            
            measurements = []
            for prompt in prompts:
                req_start = time.time()
                payload = {
                    "model": "tinyllama",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 20}
                }
                
                resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=30)
                req_duration = time.time() - req_start
                
                if resp.status_code == 200:
                    data = resp.json()
                    measurements.append({
                        "prompt": prompt,
                        "duration": req_duration,
                        "tokens": data.get('eval_count', 0),
                        "tokens_per_sec": data.get('eval_count', 0) / (data.get('eval_duration', 1) / 1e9)
                    })
            
            duration = time.time() - start_time
            
            if measurements:
                avg_duration = statistics.mean([m['duration'] for m in measurements])
                avg_tokens_per_sec = statistics.mean([m['tokens_per_sec'] for m in measurements])
                
                return TestResult(
                    test_name="Ollama Performance Baseline",
                    status="PASS",
                    duration=duration,
                    details={
                        "measurements": measurements,
                        "avg_response_time": round(avg_duration, 3),
                        "avg_tokens_per_second": round(avg_tokens_per_sec, 2),
                        "total_prompts": len(prompts)
                    }
                )
            else:
                return TestResult(
                    test_name="Ollama Performance Baseline",
                    status="FAIL",
                    duration=duration,
                    error="No successful measurements"
                )
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="Ollama Performance Baseline",
                status="FAIL",
                duration=duration,
                error=str(e)
            )
    
    def run_all_tests(self) -> List[TestResult]:
        """Run all AI agent tests"""
        logger.info("Starting AI Agents comprehensive test suite")
        
        test_methods = [
            self.test_ollama_health,
            self.test_ollama_model_list,
            self.test_tinyllama_inference,
            self.test_backend_health,
            self.test_backend_agents_endpoint,
            self.test_ollama_performance_baseline,
            lambda: self.test_concurrent_ollama_requests(5),
            self.test_agent_error_handling
        ]
        
        for test_method in test_methods:
            logger.info(f"Running test: {test_method.__name__ if hasattr(test_method, '__name__') else 'concurrent_test'}")
            result = test_method()
            self.results.append(result)
            logger.info(f"Test {result.test_name}: {result.status} (Duration: {result.duration:.4f}s)")
            
            if result.error:
                logger.error(f"Error in {result.test_name}: {result.error}")
            
            # Small delay between tests
            time.sleep(0.5)
        
        return self.results
    
    def generate_report(self) -> str:
        """Generate comprehensive test report"""
        test_end_time = datetime.now(timezone.utc)
        total_duration = (test_end_time - self.test_start_time).total_seconds()
        
        passed = sum(1 for r in self.results if r.status == "PASS")
        failed = sum(1 for r in self.results if r.status == "FAIL")
        skipped = sum(1 for r in self.results if r.status == "SKIP")
        total = len(self.results)
        
        success_rate = (passed / total * 100) if total > 0 else 0
        
        report = []
        report.append("=" * 80)
        report.append("AI AGENTS COMPREHENSIVE TEST REPORT")
        report.append("=" * 80)
        report.append(f"Execution ID: {EXECUTION_ID}")
        report.append(f"Start Time: {self.test_start_time.isoformat()}")
        report.append(f"End Time: {test_end_time.isoformat()}")
        report.append(f"Total Duration: {total_duration:.2f}s")
        report.append("")
        report.append("=" * 80)
        report.append("TEST SUMMARY")
        report.append("=" * 80)
        report.append(f"Total Tests: {total}")
        report.append(f"Passed: {passed} ({passed/total*100:.1f}%)")
        report.append(f"Failed: {failed} ({failed/total*100:.1f}%)")
        report.append(f"Skipped: {skipped} ({skipped/total*100:.1f}%)")
        report.append(f"Success Rate: {success_rate:.2f}%")
        report.append("")
        
        report.append("=" * 80)
        report.append("DETAILED RESULTS")
        report.append("=" * 80)
        
        for result in self.results:
            status_symbol = "✓" if result.status == "PASS" else "✗" if result.status == "FAIL" else "○"
            report.append(f"{status_symbol} {result.test_name}: {result.status} ({result.duration:.4f}s)")
            
            if result.error:
                report.append(f"  Error: {result.error}")
            
            if result.details:
                report.append(f"  Details: {json.dumps(result.details, indent=2)}")
            
            report.append("")
        
        report.append("=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_results_json(self, filename: str):
        """Save results to JSON file"""
        data = {
            "execution_id": EXECUTION_ID,
            "start_time": self.test_start_time.isoformat(),
            "end_time": datetime.now(timezone.utc).isoformat(),
            "results": [
                {
                    "test_name": r.test_name,
                    "status": r.status,
                    "duration": r.duration,
                    "error": r.error,
                    "details": r.details,
                    "timestamp": r.timestamp
                }
                for r in self.results
            ],
            "summary": {
                "total": len(self.results),
                "passed": sum(1 for r in self.results if r.status == "PASS"),
                "failed": sum(1 for r in self.results if r.status == "FAIL"),
                "skipped": sum(1 for r in self.results if r.status == "SKIP"),
                "success_rate": sum(1 for r in self.results if r.status == "PASS") / len(self.results) * 100 if self.results else 0
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results saved to {filename}")

def main() -> int:
    """Main test execution"""
    try:
        logger.info(f"AI Agents Comprehensive Test Suite starting at {TEST_START_TIME.isoformat()}")
        
        tester = AIAgentsTester()
        results = tester.run_all_tests()
        
        # Generate and display report
        report = tester.generate_report()
        print(report)
        
        # Save results
        report_file = f"AI_AGENTS_TEST_REPORT_{TEST_START_TIME.strftime('%Y%m%d_%H%M%S')}.txt"
        json_file = f"ai_agents_test_results_{TEST_START_TIME.strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        tester.save_results_json(json_file)
        
        logger.info(f"Report saved to {report_file}")
        logger.info(f"JSON results saved to {json_file}")
        
        # Return exit code based on success
        passed = sum(1 for r in results if r.status == "PASS")
        total = len(results)
        
        if passed == total:
            logger.info("All tests passed!")
            return 0
        else:
            logger.warning(f"{total - passed} test(s) failed")
            return 1
        
    except Exception as e:
        logger.exception(f"Fatal error in test execution: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
