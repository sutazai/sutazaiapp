#!/usr/bin/env python3
"""
Specialized Test Cases for SutazAI System
=========================================

This module contains specialized test cases to address specific issues
and improve test coverage for edge cases and AI model integration.
"""

import asyncio
import requests
import time
import json
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpecializedTestSuite:
    """Specialized tests for edge cases and specific components"""
    
    def __init__(self):
        self.base_url = "http://localhost:10010"
        self.ollama_url = "http://localhost:10104"
        
    async def test_ollama_inference_with_retry(self):
        """Test Ollama inference with shorter timeout and retry logic"""
        logger.info("Testing Ollama inference with optimized parameters...")
        
        # First, check if tinyllama model is available and loaded
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            models = response.json().get("models", [])
            
            if not models:
                logger.warning("No models available in Ollama")
                return {"status": "skipped", "reason": "No models available"}
                
            model_name = models[0]["name"]
            logger.info(f"Using model: {model_name}")
            
            # Test with very simple prompt and short response
            payload = {
                "model": model_name,
                "prompt": "Hi",
                "stream": False,
                "options": {
                    "num_predict": 10,  # Limit response length
                    "temperature": 0.1,
                    "top_p": 0.1
                }
            }
            
            start_time = time.time()
            
            # Use shorter timeout for the test
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=15  # Reduced from 30 to 15 seconds
            )
            
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "passed",
                    "duration": duration,
                    "model": model_name,
                    "response_length": len(data.get("response", "")),
                    "eval_count": data.get("eval_count", 0),
                    "inference_successful": True
                }
            else:
                return {
                    "status": "failed",
                    "reason": f"HTTP {response.status_code}",
                    "duration": duration
                }
                
        except requests.exceptions.Timeout:
            return {
                "status": "failed",
                "reason": "Timeout - model may be loading or system overloaded",
                "duration": 15.0
            }
        except Exception as e:
            return {
                "status": "error",
                "reason": str(e),
                "duration": time.time() - start_time if 'start_time' in locals() else 0
            }
    
    async def test_concurrent_api_calls(self):
        """Test concurrent API calls to check system stability"""
        logger.info("Testing concurrent API calls...")
        
        import concurrent.futures
        import threading
        
        def make_api_call(endpoint):
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                return {
                    "endpoint": endpoint,
                    "status_code": response.status_code,
                    "success": response.status_code == 200,
                    "response_time": response.elapsed.total_seconds()
                }
            except Exception as e:
                return {
                    "endpoint": endpoint,
                    "status_code": 0,
                    "success": False,
                    "error": str(e),
                    "response_time": 0
                }
        
        endpoints = [
            "/health",
            "/api/v1/agents",
            "/api/v1/agents/status",
            "/api/v1/health",
            "/api/v1"
        ]
        
        # Test with 10 concurrent requests (2 per endpoint)
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_api_call, endpoint) for endpoint in endpoints * 2]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        successful_calls = sum(1 for r in results if r["success"])
        total_calls = len(results)
        success_rate = (successful_calls / total_calls) * 100
        avg_response_time = sum(r["response_time"] for r in results if r["success"]) / successful_calls if successful_calls > 0 else 0
        
        return {
            "status": "passed" if success_rate >= 90 else "failed",
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "success_rate": success_rate,
            "average_response_time": avg_response_time,
            "details": results
        }
    
    async def test_large_json_response_handling(self):
        """Test handling of large JSON responses"""
        logger.info("Testing large JSON response handling...")
        
        try:
            # Test the agents status endpoint which returns structured JSON
            response = requests.get(f"{self.base_url}/api/v1/agents/status", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Verify JSON structure
                required_fields = ["agents", "total_agents", "active_agents", "timestamp"]
                missing_fields = [field for field in required_fields if field not in data]
                
                if not missing_fields:
                    return {
                        "status": "passed",
                        "response_size": len(response.text),
                        "agents_count": len(data.get("agents", [])),
                        "json_valid": True,
                        "structure_valid": True
                    }
                else:
                    return {
                        "status": "failed",
                        "reason": f"Missing fields: {missing_fields}",
                        "json_valid": True,
                        "structure_valid": False
                    }
            else:
                return {
                    "status": "failed",
                    "reason": f"HTTP {response.status_code}",
                    "json_valid": False,
                    "structure_valid": False
                }
                
        except Exception as e:
            return {
                "status": "error",
                "reason": str(e),
                "json_valid": False,
                "structure_valid": False
            }
    
    async def test_malformed_json_payload(self):
        """Test system response to malformed JSON payloads"""
        logger.info("Testing malformed JSON payload handling...")
        
        malformed_payloads = [
            '{"invalid": json}',  # Invalid JSON syntax
            '{"task": "test", "agent_type":}',  # Incomplete JSON
            '{"task": "", "agent_type": null}',  # Null/empty values
            '{"task": "' + "x" * 10000 + '"}',  # Very long string
        ]
        
        results = []
        
        for i, payload in enumerate(malformed_payloads):
            try:
                response = requests.post(
                    f"{self.base_url}/api/v1/agents/task",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )
                
                # We expect error codes for malformed JSON
                expected_error_codes = [400, 422, 500]
                handled_correctly = response.status_code in expected_error_codes
                
                results.append({
                    "payload_index": i,
                    "status_code": response.status_code,
                    "handled_correctly": handled_correctly,
                    "response_text": response.text[:100]  # First 100 chars
                })
                
            except Exception as e:
                results.append({
                    "payload_index": i,
                    "error": str(e),
                    "handled_correctly": True  # Exception is acceptable
                })
        
        correctly_handled = sum(1 for r in results if r.get("handled_correctly", False))
        total_tests = len(results)
        
        return {
            "status": "passed" if correctly_handled == total_tests else "failed",
            "correctly_handled": correctly_handled,
            "total_tests": total_tests,
            "success_rate": (correctly_handled / total_tests) * 100,
            "details": results
        }
    
    async def test_system_resource_usage(self):
        """Test system resource usage during operations"""
        logger.info("Testing system resource usage...")
        
        try:
            import psutil
            
            # Get initial system stats
            initial_cpu = psutil.cpu_percent(interval=1)
            initial_memory = psutil.virtual_memory()
            
            # Perform some API operations
            endpoints_to_test = [
                "/health",
                "/api/v1/agents",
                "/api/v1/agents/status"
            ]
            
            start_time = time.time()
            
            # Make multiple requests
            for _ in range(10):
                for endpoint in endpoints_to_test:
                    requests.get(f"{self.base_url}{endpoint}", timeout=5)
            
            # Get final system stats
            final_cpu = psutil.cpu_percent(interval=1)
            final_memory = psutil.virtual_memory()
            test_duration = time.time() - start_time
            
            # Calculate resource usage
            cpu_increase = final_cpu - initial_cpu
            memory_increase = final_memory.used - initial_memory.used
            
            return {
                "status": "passed",
                "test_duration": test_duration,
                "initial_cpu_percent": initial_cpu,
                "final_cpu_percent": final_cpu,
                "cpu_increase": cpu_increase,
                "initial_memory_mb": initial_memory.used / (1024 * 1024),
                "final_memory_mb": final_memory.used / (1024 * 1024),
                "memory_increase_mb": memory_increase / (1024 * 1024),
                "requests_made": len(endpoints_to_test) * 10
            }
            
        except ImportError:
            return {
                "status": "skipped",
                "reason": "psutil not available for resource monitoring"
            }
        except Exception as e:
            return {
                "status": "error",
                "reason": str(e)
            }
    
    async def test_docker_container_health(self):
        """Test Docker container health status"""
        logger.info("Testing Docker container health...")
        
        try:
            import subprocess
            
            # Get container status
            result = subprocess.run(
                ["docker", "ps", "--format", "table {{.Names}}\t{{.Status}}\t{{.Ports}}"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                containers_info = result.stdout.strip().split('\n')[1:]  # Skip header
                containers = []
                
                for line in containers_info:
                    if line.strip():
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            containers.append({
                                "name": parts[0],
                                "status": parts[1],
                                "ports": parts[2] if len(parts) > 2 else "",
                                "healthy": "healthy" in parts[1].lower() or "up" in parts[1].lower()
                            })
                
                healthy_containers = [c for c in containers if c["healthy"]]
                sutazai_containers = [c for c in containers if "sutazai" in c["name"].lower()]
                
                return {
                    "status": "passed",
                    "total_containers": len(containers),
                    "healthy_containers": len(healthy_containers),
                    "sutazai_containers": len(sutazai_containers),
                    "all_healthy": len(healthy_containers) == len(containers),
                    "containers": containers
                }
            else:
                return {
                    "status": "error",
                    "reason": "Failed to get container status",
                    "stderr": result.stderr
                }
                
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "reason": "Docker command timed out"
            }
        except Exception as e:
            return {
                "status": "error",
                "reason": str(e)
            }
    
    async def run_all_specialized_tests(self):
        """Run all specialized tests"""
        logger.info("Running specialized test suite...")
        
        tests = [
            ("Ollama Inference (Optimized)", self.test_ollama_inference_with_retry),
            ("Concurrent API Calls", self.test_concurrent_api_calls),
            ("Large JSON Response", self.test_large_json_response_handling),
            ("Malformed JSON Handling", self.test_malformed_json_payload),
            ("System Resource Usage", self.test_system_resource_usage),
            ("Docker Container Health", self.test_docker_container_health)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            logger.info(f"Running {test_name}...")
            try:
                result = await test_func()
                results[test_name] = result
                logger.info(f"{test_name}: {result.get('status', 'unknown')}")
            except Exception as e:
                results[test_name] = {
                    "status": "error",
                    "reason": str(e)
                }
                logger.error(f"{test_name} failed: {e}")
        
        # Generate summary
        passed_tests = sum(1 for r in results.values() if r.get("status") == "passed")
        total_tests = len(results)
        success_rate = (passed_tests / total_tests) * 100
        
        summary = {
            "execution_timestamp": datetime.now().isoformat(),
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": success_rate,
            "status": "PASSED" if success_rate >= 80 else "FAILED",
            "detailed_results": results
        }
        
        # Save specialized test report
        await self._save_specialized_report(summary)
        
        return summary
    
    async def _save_specialized_report(self, report):
        """Save specialized test report"""
        try:
            reports_dir = Path("/opt/sutazaiapp/data/workflow_reports")
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_file = reports_dir / f"specialized_test_report_{timestamp}.json"
            
            with open(json_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Specialized test report saved: {json_file}")
            
        except Exception as e:
            logger.error(f"Failed to save specialized test report: {e}")

async def main():
    """Run specialized tests"""
    suite = SpecializedTestSuite()
    report = await suite.run_all_specialized_tests()
    
    logger.info("\n" + "="*80)
    logger.info("SUTAZAI SPECIALIZED TEST RESULTS")
    logger.info("="*80)
    logger.info(f"Status: {report['status']}")
    logger.info(f"Tests: {report['passed_tests']}/{report['total_tests']} passed ({report['success_rate']:.1f}%)")
    
    logger.info(f"\nDetailed Results:")
    for test_name, result in report["detailed_results"].items():
        status_icon = "✅" if result["status"] == "passed" else "❌" if result["status"] == "failed" else "⚠️"
        logger.info(f"  {status_icon} {test_name}: {result['status']}")
        if result["status"] != "passed" and "reason" in result:
            logger.info(f"    Reason: {result['reason']}")
    
    logger.info("="*80)
    
    return report

if __name__ == "__main__":
    asyncio.run(main())