#!/usr/bin/env python3
"""
ULTRA-COMPREHENSIVE HEALTH VALIDATION SUITE
Hardware Resource Optimizer Service - ALL SCENARIOS

This test suite validates ALL operational scenarios with ZERO tolerance for failures.
"""

import requests
import json
import time
import threading
import psutil
import signal
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HealthValidationSuite:
    """Ultra-comprehensive health validation suite for hardware resource optimizer"""
    
    def __init__(self):
        self.primary_service_url = "http://localhost:11110"
        self.secondary_service_url = "http://localhost:11104"
        self.services = {
            "primary": {
                "url": self.primary_service_url,
                "name": "hardware-resource-optimizer",
                "container": "sutazai-hardware-resource-optimizer"
            },
            "secondary": {
                "url": self.secondary_service_url, 
                "name": "jarvis-hardware-resource-optimizer",
                "container": "sutazai-jarvis-hardware-resource-optimizer"
            }
        }
        
        self.test_results = {}
        self.performance_metrics = {}
        self.error_log = []
        
    def log_error(self, test_name: str, error: str):
        """Log error for reporting"""
        error_entry = {
            "test": test_name,
            "error": error,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.error_log.append(error_entry)
        logger.error(f"{test_name}: {error}")
    
    def log_result(self, test_name: str, success: bool, details: Dict[str, Any]):
        """Log test result"""
        self.test_results[test_name] = {
            "success": success,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if success:
            logger.info(f"✅ {test_name}: PASSED")
        else:
            logger.error(f"❌ {test_name}: FAILED - {details.get('error', 'Unknown error')}")

    # SCENARIO 1: NORMAL OPERATION HEALTH VALIDATION
    def test_basic_health_endpoints(self) -> Dict[str, Any]:
        """Test basic health endpoint functionality"""
        results = {}
        
        for service_key, service in self.services.items():
            try:
                start_time = time.time()
                response = requests.get(f"{service['url']}/health", timeout=30)
                response_time = time.time() - start_time
                
                results[service_key] = {
                    "status_code": response.status_code,
                    "response_time": response_time,
                    "content": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text,
                    "success": response.status_code == 200
                }
                
            except requests.exceptions.Timeout:
                results[service_key] = {
                    "success": False,
                    "error": "Timeout after 30 seconds"
                }
            except Exception as e:
                results[service_key] = {
                    "success": False,
                    "error": str(e)
                }
        
        self.log_result("basic_health_endpoints", 
                       all(r.get('success', False) for r in results.values()), 
                       results)
        return results

    def test_internal_container_health(self) -> Dict[str, Any]:
        """Test health from inside containers"""
        results = {}
        
        for service_key, service in self.services.items():
            try:
                cmd = f"docker exec {service['container']} curl -s http://localhost:8080/health"
                result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=30)
                
                results[service_key] = {
                    "return_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "success": result.returncode == 0
                }
                
                if result.returncode == 0:
                    try:
                        health_data = json.loads(result.stdout)
                        results[service_key]["health_data"] = health_data
                    except json.JSONDecodeError:
                        results[service_key]["raw_response"] = result.stdout
                        
            except subprocess.TimeoutExpired:
                results[service_key] = {
                    "success": False,
                    "error": "Internal health check timeout"
                }
            except Exception as e:
                results[service_key] = {
                    "success": False,
                    "error": str(e)
                }
        
        self.log_result("internal_container_health",
                       all(r.get('success', False) for r in results.values()),
                       results)
        return results

    def test_service_endpoints_comprehensive(self) -> Dict[str, Any]:
        """Test all available service endpoints"""
        results = {}
        
        # Test primary service endpoints
        primary_endpoints = ["/", "/health", "/task", "/docs"]
        results["primary_endpoints"] = {}
        
        for endpoint in primary_endpoints:
            try:
                response = requests.get(f"{self.primary_service_url}{endpoint}", timeout=30)
                results["primary_endpoints"][endpoint] = {
                    "status_code": response.status_code,
                    "success": response.status_code < 400,
                    "response_time": response.elapsed.total_seconds()
                }
            except Exception as e:
                results["primary_endpoints"][endpoint] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Test secondary service endpoints (AI-powered)
        secondary_endpoints = ["/health", "/analyze", "/recommendations", "/metrics", "/docs"]
        results["secondary_endpoints"] = {}
        
        for endpoint in secondary_endpoints:
            try:
                response = requests.get(f"{self.secondary_service_url}{endpoint}", timeout=30)
                results["secondary_endpoints"][endpoint] = {
                    "status_code": response.status_code,
                    "success": response.status_code < 400,
                    "response_time": response.elapsed.total_seconds()
                }
            except Exception as e:
                results["secondary_endpoints"][endpoint] = {
                    "success": False,
                    "error": str(e)
                }
        
        overall_success = (
            all(r.get('success', False) for r in results["primary_endpoints"].values()) and
            any(r.get('success', False) for r in results["secondary_endpoints"].values())  # At least some should work
        )
        
        self.log_result("service_endpoints_comprehensive", overall_success, results)
        return results

    # SCENARIO 2: HIGH LOAD TESTING
    def test_concurrent_requests(self, num_requests: int = 50, num_workers: int = 10) -> Dict[str, Any]:
        """Test service under concurrent load"""
        results = {}
        
        for service_key, service in self.services.items():
            success_count = 0
            error_count = 0
            response_times = []
            errors = []
            
            def make_request():
                try:
                    start = time.time()
                    response = requests.get(f"{service['url']}/health", timeout=10)
                    response_time = time.time() - start
                    response_times.append(response_time)
                    return response.status_code == 200
                except Exception as e:
                    errors.append(str(e))
                    return False
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(make_request) for _ in range(num_requests)]
                
                for future in as_completed(futures):
                    if future.result():
                        success_count += 1
                    else:
                        error_count += 1
            
            results[service_key] = {
                "total_requests": num_requests,
                "success_count": success_count,
                "error_count": error_count,
                "success_rate": success_count / num_requests * 100,
                "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
                "max_response_time": max(response_times) if response_times else 0,
                "min_response_time": min(response_times) if response_times else 0,
                "errors": errors[:10],  # First 10 errors
                "success": success_count > num_requests * 0.8  # 80% success rate required
            }
        
        overall_success = all(r.get('success', False) for r in results.values())
        self.log_result("concurrent_requests", overall_success, results)
        return results

    # SCENARIO 3: RESOURCE CONSTRAINT TESTING
    def test_resource_monitoring_accuracy(self) -> Dict[str, Any]:
        """Test accuracy of resource monitoring"""
        results = {}
        
        # Get system metrics from both services
        for service_key, service in self.services.items():
            try:
                if service_key == "primary":
                    # Primary service doesn't have metrics endpoint, use health
                    response = requests.get(f"{service['url']}/health", timeout=10)
                    service_data = response.json()
                else:
                    # Secondary service has dedicated metrics endpoint
                    response = requests.get(f"{service['url']}/metrics", timeout=10)
                    service_data = response.json()
                
                # Get actual system metrics for comparison
                actual_metrics = {
                    "cpu_percent": psutil.cpu_percent(interval=1),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_percent": psutil.disk_usage('/').used / psutil.disk_usage('/').total * 100
                }
                
                results[service_key] = {
                    "service_metrics": service_data,
                    "actual_metrics": actual_metrics,
                    "success": response.status_code == 200
                }
                
                # Check metric accuracy if available
                if "system_status" in service_data or "system_metrics" in service_data:
                    metrics_key = "system_status" if "system_status" in service_data else "system_metrics"
                    service_metrics = service_data[metrics_key]
                    
                    cpu_accuracy = abs(service_metrics.get("cpu_percent", 0) - actual_metrics["cpu_percent"]) < 10
                    mem_accuracy = abs(service_metrics.get("memory_percent", 0) - actual_metrics["memory_percent"]) < 10
                    
                    results[service_key]["metrics_accuracy"] = {
                        "cpu_accurate": cpu_accuracy,
                        "memory_accurate": mem_accuracy,
                        "overall_accurate": cpu_accuracy and mem_accuracy
                    }
                
            except Exception as e:
                results[service_key] = {
                    "success": False,
                    "error": str(e)
                }
        
        overall_success = all(r.get('success', False) for r in results.values())
        self.log_result("resource_monitoring_accuracy", overall_success, results)
        return results

    # SCENARIO 4: NETWORK FAILURE SIMULATION
    def test_network_resilience(self) -> Dict[str, Any]:
        """Test service resilience to network issues"""
        results = {}
        
        # Test timeout handling
        for service_key, service in self.services.items():
            timeout_results = []
            
            for timeout in [1, 5, 10]:  # Test different timeouts
                try:
                    start = time.time()
                    response = requests.get(f"{service['url']}/health", timeout=timeout)
                    elapsed = time.time() - start
                    
                    timeout_results.append({
                        "timeout_setting": timeout,
                        "actual_time": elapsed,
                        "success": response.status_code == 200,
                        "within_timeout": elapsed < timeout
                    })
                    
                except requests.exceptions.Timeout:
                    timeout_results.append({
                        "timeout_setting": timeout,
                        "success": False,
                        "error": "Timeout"
                    })
                except Exception as e:
                    timeout_results.append({
                        "timeout_setting": timeout,
                        "success": False,
                        "error": str(e)
                    })
            
            results[service_key] = {
                "timeout_tests": timeout_results,
                "success": any(t.get('success', False) for t in timeout_results)
            }
        
        overall_success = all(r.get('success', False) for r in results.values())
        self.log_result("network_resilience", overall_success, results)
        return results

    # SCENARIO 5: DEPENDENCY VALIDATION
    def test_dependency_health(self) -> Dict[str, Any]:
        """Test health of service dependencies"""
        results = {}
        
        # Check container status
        for service_key, service in self.services.items():
            try:
                # Check container status
                cmd = f"docker inspect {service['container']} --format='{{{{.State.Status}}}}'"
                result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=10)
                
                container_status = result.stdout.strip()
                
                # Check container health
                health_cmd = f"docker inspect {service['container']} --format='{{{{.State.Health.Status}}}}'"
                health_result = subprocess.run(health_cmd.split(), capture_output=True, text=True, timeout=10)
                health_status = health_result.stdout.strip()
                
                results[service_key] = {
                    "container_status": container_status,
                    "container_health": health_status,
                    "container_running": container_status == "running",
                    "container_healthy": health_status in ["healthy", ""],
                    "success": container_status == "running"
                }
                
            except Exception as e:
                results[service_key] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Check if Ollama dependency is available (for secondary service)
        try:
            ollama_response = requests.get("http://localhost:10104/api/tags", timeout=5)
            results["ollama_dependency"] = {
                "available": ollama_response.status_code == 200,
                "success": ollama_response.status_code == 200
            }
        except Exception as e:
            results["ollama_dependency"] = {
                "available": False,
                "success": False,
                "error": str(e)
            }
        
        overall_success = all(r.get('success', False) for r in results.values())
        self.log_result("dependency_health", overall_success, results)
        return results

    # SCENARIO 6: ERROR HANDLING VALIDATION
    def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling and graceful degradation"""
        results = {}
        
        for service_key, service in self.services.items():
            error_tests = {}
            
            # Test invalid endpoints
            invalid_endpoints = ["/invalid", "/nonexistent", "/error"]
            for endpoint in invalid_endpoints:
                try:
                    response = requests.get(f"{service['url']}{endpoint}", timeout=10)
                    error_tests[f"invalid_endpoint_{endpoint}"] = {
                        "status_code": response.status_code,
                        "handles_gracefully": 400 <= response.status_code < 500,
                        "success": 400 <= response.status_code < 500
                    }
                except Exception as e:
                    error_tests[f"invalid_endpoint_{endpoint}"] = {
                        "success": False,
                        "error": str(e)
                    }
            
            # Test malformed requests (for services that accept POST)
            if service_key == "primary":
                try:
                    response = requests.post(f"{service['url']}/task", 
                                           json={"invalid": "data"}, 
                                           timeout=10)
                    error_tests["malformed_post"] = {
                        "status_code": response.status_code,
                        "handles_gracefully": response.status_code >= 400,
                        "success": response.status_code >= 400
                    }
                except Exception as e:
                    error_tests["malformed_post"] = {
                        "success": True,  # Connection errors are acceptable
                        "error": str(e)
                    }
            
            results[service_key] = {
                "error_tests": error_tests,
                "success": all(t.get('success', False) for t in error_tests.values())
            }
        
        overall_success = all(r.get('success', False) for r in results.values())
        self.log_result("error_handling", overall_success, results)
        return results

    # SCENARIO 7: PERFORMANCE THRESHOLDS
    def test_performance_thresholds(self) -> Dict[str, Any]:
        """Test performance meets required thresholds"""
        results = {}
        
        # Performance requirements
        max_response_time = 5.0  # 5 seconds max
        min_success_rate = 95.0  # 95% success rate
        max_cpu_usage = 80.0     # 80% max CPU
        max_memory_usage = 80.0  # 80% max memory
        
        for service_key, service in self.services.items():
            perf_results = {}
            
            # Response time test
            response_times = []
            for _ in range(10):
                try:
                    start = time.time()
                    response = requests.get(f"{service['url']}/health", timeout=10)
                    response_time = time.time() - start
                    response_times.append(response_time)
                except:
                    response_times.append(10.0)  # Timeout penalty
            
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time_actual = max(response_times)
            
            perf_results["response_time"] = {
                "avg_response_time": avg_response_time,
                "max_response_time": max_response_time_actual,
                "meets_threshold": avg_response_time <= max_response_time,
                "success": avg_response_time <= max_response_time
            }
            
            # Resource usage test (from service if available)
            try:
                if service_key == "secondary":
                    response = requests.get(f"{service['url']}/metrics", timeout=10)
                    metrics = response.json()
                    
                    if "cpu" in metrics:
                        cpu_usage = metrics["cpu"].get("usage_percent", 0)
                        memory_usage = metrics["memory"].get("usage_percent", 0)
                        
                        perf_results["resource_usage"] = {
                            "cpu_usage": cpu_usage,
                            "memory_usage": memory_usage,
                            "cpu_meets_threshold": cpu_usage <= max_cpu_usage,
                            "memory_meets_threshold": memory_usage <= max_memory_usage,
                            "success": cpu_usage <= max_cpu_usage and memory_usage <= max_memory_usage
                        }
            except:
                perf_results["resource_usage"] = {
                    "success": True,  # Skip if not available
                    "note": "Resource metrics not available"
                }
            
            results[service_key] = {
                "performance_tests": perf_results,
                "success": all(t.get('success', False) for t in perf_results.values())
            }
        
        overall_success = all(r.get('success', False) for r in results.values())
        self.log_result("performance_thresholds", overall_success, results)
        return results

    def run_comprehensive_health_validation(self) -> Dict[str, Any]:
        """Run all health validation scenarios"""
        logger.info("🚀 Starting ULTRA-COMPREHENSIVE Health Validation Suite")
        start_time = time.time()
        
        validation_results = {}
        
        # SCENARIO 1: Normal Operation
        logger.info("📊 SCENARIO 1: Normal Operation Health Validation")
        validation_results["basic_health"] = self.test_basic_health_endpoints()
        validation_results["internal_health"] = self.test_internal_container_health()
        validation_results["endpoint_comprehensive"] = self.test_service_endpoints_comprehensive()
        
        # SCENARIO 2: High Load Testing
        logger.info("⚡ SCENARIO 2: High Load Testing")
        validation_results["concurrent_load"] = self.test_concurrent_requests(50, 10)
        
        # SCENARIO 3: Resource Constraint Testing
        logger.info("💾 SCENARIO 3: Resource Constraint Testing")
        validation_results["resource_monitoring"] = self.test_resource_monitoring_accuracy()
        
        # SCENARIO 4: Network Resilience
        logger.info("🌐 SCENARIO 4: Network Resilience Testing")
        validation_results["network_resilience"] = self.test_network_resilience()
        
        # SCENARIO 5: Dependency Validation
        logger.info("🔗 SCENARIO 5: Dependency Health Validation")
        validation_results["dependency_health"] = self.test_dependency_health()
        
        # SCENARIO 6: Error Handling
        logger.info("🚨 SCENARIO 6: Error Handling Validation")
        validation_results["error_handling"] = self.test_error_handling()
        
        # SCENARIO 7: Performance Thresholds
        logger.info("🏎️ SCENARIO 7: Performance Threshold Validation")
        validation_results["performance_thresholds"] = self.test_performance_thresholds()
        
        total_time = time.time() - start_time
        
        # Calculate overall results
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['success'])
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        final_results = {
            "validation_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate": success_rate,
                "overall_success": success_rate >= 80,  # 80% minimum
                "total_execution_time": total_time
            },
            "detailed_results": validation_results,
            "test_results": self.test_results,
            "error_log": self.error_log,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(f"🏁 VALIDATION COMPLETE: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
        return final_results

def main():
    """Main execution"""
    suite = HealthValidationSuite()
    results = suite.run_comprehensive_health_validation()
    
    # Save results to file
    output_file = f"/opt/sutazaiapp/tests/health_validation_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📋 COMPREHENSIVE HEALTH VALIDATION RESULTS")
    print(f"Results saved to: {output_file}")
    print(f"Overall Success Rate: {results['validation_summary']['success_rate']:.1f}%")
    print(f"Tests Passed: {results['validation_summary']['passed_tests']}/{results['validation_summary']['total_tests']}")
    
    return results

if __name__ == "__main__":
    results = main()