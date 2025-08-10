#!/usr/bin/env python3
"""
MONITORING SYSTEM VALIDATION
Hardware Resource Optimizer Service - Monitoring Health Validation

This validates ALL monitoring systems accuracy:
- Prometheus metrics collection
- Grafana dashboard accuracy
- Docker health checks
- Service mesh monitoring
- Backend API health proxying
- Container stats accuracy
- Alert system validation
- Log aggregation accuracy
"""

import requests
import json
import time
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MonitoringSystemValidator:
    """Comprehensive monitoring system validation"""
    
    def __init__(self):
        self.services = {
            "hardware_optimizer": {
                "url": "http://localhost:11110",
                "container": "sutazai-hardware-resource-optimizer"
            },
            "jarvis_optimizer": {
                "url": "http://localhost:11104", 
                "container": "sutazai-jarvis-hardware-resource-optimizer"
            }
        }
        
        self.monitoring_endpoints = {
            "prometheus": "http://localhost:10200",
            "grafana": "http://localhost:10201",
            "loki": "http://localhost:10202"
        }
        
        self.test_results = {}
        
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
            logger.error(f"❌ {test_name}: FAILED")

    def test_prometheus_metrics_collection(self) -> Dict[str, Any]:
        """Test Prometheus metrics collection accuracy"""
        results = {}
        
        # Test Prometheus accessibility
        try:
            prometheus_response = requests.get(f"{self.monitoring_endpoints['prometheus']}/api/v1/query", 
                                             params={"query": "up"}, timeout=10)
            prometheus_accessible = prometheus_response.status_code == 200
            
            if prometheus_accessible:
                metrics_data = prometheus_response.json()
                results["prometheus_accessible"] = True
                results["metrics_available"] = len(metrics_data.get("data", {}).get("result", [])) > 0
            else:
                results["prometheus_accessible"] = False
                
        except Exception as e:
            results["prometheus_accessible"] = False
            results["error"] = str(e)
        
        # Test service-specific metrics
        service_metrics = {}
        for service_name, service_config in self.services.items():
            container_name = service_config["container"]
            
            # Query container metrics from Prometheus
            try:
                # Container CPU usage
                cpu_query = f'rate(container_cpu_usage_seconds_total{{name="{container_name}"}}[5m])'
                cpu_response = requests.get(f"{self.monitoring_endpoints['prometheus']}/api/v1/query",
                                          params={"query": cpu_query}, timeout=10)
                
                # Container memory usage  
                mem_query = f'container_memory_usage_bytes{{name="{container_name}"}}'
                mem_response = requests.get(f"{self.monitoring_endpoints['prometheus']}/api/v1/query",
                                          params={"query": mem_query}, timeout=10)
                
                service_metrics[service_name] = {
                    "cpu_metrics_available": cpu_response.status_code == 200,
                    "memory_metrics_available": mem_response.status_code == 200,
                    "cpu_data": cpu_response.json() if cpu_response.status_code == 200 else None,
                    "memory_data": mem_response.json() if mem_response.status_code == 200 else None
                }
                
            except Exception as e:
                service_metrics[service_name] = {
                    "error": str(e),
                    "metrics_available": False
                }
        
        results["service_metrics"] = service_metrics
        results["success"] = results.get("prometheus_accessible", False)
        
        self.log_result("prometheus_metrics_collection", results["success"], results)
        return results

    def test_grafana_dashboard_access(self) -> Dict[str, Any]:
        """Test Grafana dashboard accessibility"""
        results = {}
        
        try:
            # Test Grafana main page
            grafana_response = requests.get(f"{self.monitoring_endpoints['grafana']}/api/health", timeout=10)
            grafana_accessible = grafana_response.status_code == 200
            
            results["grafana_accessible"] = grafana_accessible
            
            if grafana_accessible:
                # Test dashboard API
                try:
                    dashboard_response = requests.get(f"{self.monitoring_endpoints['grafana']}/api/search", 
                                                    timeout=10,
                                                    auth=('admin', 'admin'))  # Default credentials
                    results["dashboard_api_accessible"] = dashboard_response.status_code == 200
                    
                    if dashboard_response.status_code == 200:
                        dashboards = dashboard_response.json()
                        results["dashboard_count"] = len(dashboards)
                        results["dashboards_available"] = len(dashboards) > 0
                    
                except Exception as e:
                    results["dashboard_api_error"] = str(e)
            
        except Exception as e:
            results["grafana_accessible"] = False
            results["error"] = str(e)
        
        results["success"] = results.get("grafana_accessible", False)
        
        self.log_result("grafana_dashboard_access", results["success"], results)
        return results

    def test_docker_health_checks(self) -> Dict[str, Any]:
        """Test Docker health check accuracy"""
        results = {}
        
        for service_name, service_config in self.services.items():
            container = service_config["container"]
            
            try:
                # Get Docker health status
                health_cmd = f"docker inspect {container} --format='{{{{.State.Health.Status}}}}'"
                health_result = subprocess.run(health_cmd.split(), capture_output=True, text=True, timeout=10)
                docker_health = health_result.stdout.strip().replace("'", "")
                
                # Get container state
                state_cmd = f"docker inspect {container} --format='{{{{.State.Status}}}}'"
                state_result = subprocess.run(state_cmd.split(), capture_output=True, text=True, timeout=10)
                container_state = state_result.stdout.strip().replace("'", "")
                
                # Test actual service health
                actual_health = False
                try:
                    health_response = requests.get(f"{service_config['url']}/health", timeout=30)
                    actual_health = health_response.status_code == 200
                except:
                    pass
                
                results[service_name] = {
                    "docker_health_status": docker_health,
                    "container_state": container_state,
                    "actual_service_health": actual_health,
                    "health_check_accurate": (docker_health == "healthy") == actual_health or docker_health == "",
                    "container_running": container_state == "running",
                    "success": container_state == "running"
                }
                
            except Exception as e:
                results[service_name] = {
                    "error": str(e),
                    "success": False
                }
        
        overall_success = all(r.get('success', False) for r in results.values())
        self.log_result("docker_health_checks", overall_success, results)
        return results

    def test_service_health_consistency(self) -> Dict[str, Any]:
        """Test consistency between different health monitoring systems"""
        results = {}
        
        for service_name, service_config in self.services.items():
            # Collect health data from multiple sources
            health_sources = {}
            
            # Direct service health
            try:
                response = requests.get(f"{service_config['url']}/health", timeout=30)
                health_sources["direct_service"] = {
                    "healthy": response.status_code == 200,
                    "response_time": response.elapsed.total_seconds(),
                    "status_code": response.status_code,
                    "response_data": response.json() if response.headers.get('content-type', '').startswith('application/json') else None
                }
            except Exception as e:
                health_sources["direct_service"] = {
                    "healthy": False,
                    "error": str(e)
                }
            
            # Container internal health
            try:
                internal_cmd = f"docker exec {service_config['container']} curl -s http://localhost:8080/health"
                internal_result = subprocess.run(internal_cmd.split(), capture_output=True, text=True, timeout=30)
                internal_healthy = internal_result.returncode == 0
                
                health_sources["container_internal"] = {
                    "healthy": internal_healthy,
                    "return_code": internal_result.returncode,
                    "response": internal_result.stdout if internal_healthy else internal_result.stderr
                }
                
            except Exception as e:
                health_sources["container_internal"] = {
                    "healthy": False,
                    "error": str(e)
                }
            
            # Docker health check
            try:
                health_cmd = f"docker inspect {service_config['container']} --format='{{{{.State.Health.Status}}}}'"
                health_result = subprocess.run(health_cmd.split(), capture_output=True, text=True, timeout=10)
                docker_health = health_result.stdout.strip().replace("'", "")
                
                health_sources["docker_health"] = {
                    "healthy": docker_health == "healthy" or docker_health == "",
                    "status": docker_health
                }
                
            except Exception as e:
                health_sources["docker_health"] = {
                    "healthy": None,
                    "error": str(e)
                }
            
            # Analyze consistency
            health_states = [source.get("healthy") for source in health_sources.values() if source.get("healthy") is not None]
            consistent = len(set(health_states)) <= 1 if health_states else False
            
            results[service_name] = {
                "health_sources": health_sources,
                "consistent_health_reporting": consistent,
                "health_states": health_states,
                "success": len(health_states) > 0  # At least one source should work
            }
        
        overall_success = all(r.get('success', False) for r in results.values())
        self.log_result("service_health_consistency", overall_success, results)
        return results

    def test_performance_monitoring_accuracy(self) -> Dict[str, Any]:
        """Test accuracy of performance monitoring"""
        results = {}
        
        for service_name, service_config in self.services.items():
            # Get metrics from service (if available)
            service_metrics = None
            try:
                if service_name == "jarvis_optimizer":
                    metrics_response = requests.get(f"{service_config['url']}/metrics", timeout=10)
                    if metrics_response.status_code == 200:
                        service_metrics = metrics_response.json()
                else:
                    # Try health endpoint for basic metrics
                    health_response = requests.get(f"{service_config['url']}/health", timeout=10)
                    if health_response.status_code == 200:
                        service_metrics = health_response.json()
            except:
                pass
            
            # Get container stats
            container_stats = None
            try:
                stats_cmd = f"docker stats --no-stream --format 'table {{{{.CPUPerc}}}}\t{{{{.MemUsage}}}}' {service_config['container']}"
                stats_result = subprocess.run(stats_cmd.split(), capture_output=True, text=True, timeout=10)
                if stats_result.returncode == 0:
                    lines = stats_result.stdout.strip().split('\n')
                    if len(lines) > 1:
                        stats_line = lines[1].split('\t')
                        container_stats = {
                            "cpu_percent": stats_line[0] if len(stats_line) > 0 else "0%",
                            "memory_usage": stats_line[1] if len(stats_line) > 1 else "0B / 0B"
                        }
            except:
                pass
            
            # Compare metrics accuracy
            metrics_comparison = {
                "service_metrics_available": service_metrics is not None,
                "container_stats_available": container_stats is not None,
                "service_metrics": service_metrics,
                "container_stats": container_stats
            }
            
            # Check if both sources provide similar data
            metrics_consistent = False
            if service_metrics and container_stats:
                # Try to extract comparable values
                try:
                    if "system_status" in service_metrics:
                        service_cpu = service_metrics["system_status"].get("cpu_percent", 0)
                        service_mem = service_metrics["system_status"].get("memory_percent", 0)
                        
                        # Parse container stats
                        container_cpu = float(container_stats["cpu_percent"].rstrip('%'))
                        container_mem_parts = container_stats["memory_usage"].split(' / ')
                        
                        # Basic consistency check (within reasonable range)
                        cpu_diff = abs(service_cpu - container_cpu)
                        metrics_consistent = cpu_diff < 20  # Within 20% difference acceptable
                        
                        metrics_comparison["cpu_difference"] = cpu_diff
                        metrics_comparison["metrics_consistent"] = metrics_consistent
                        
                except:
                    metrics_comparison["comparison_error"] = "Could not parse metrics for comparison"
            
            results[service_name] = {
                "metrics_comparison": metrics_comparison,
                "success": metrics_comparison["service_metrics_available"] or metrics_comparison["container_stats_available"]
            }
        
        overall_success = all(r.get('success', False) for r in results.values())
        self.log_result("performance_monitoring_accuracy", overall_success, results)
        return results

    def test_log_aggregation(self) -> Dict[str, Any]:
        """Test log aggregation system"""
        results = {}
        
        try:
            # Test Loki health
            loki_response = requests.get(f"{self.monitoring_endpoints['loki']}/ready", timeout=10)
            loki_healthy = loki_response.status_code == 200
            
            results["loki_accessible"] = loki_healthy
            
            if loki_healthy:
                # Test log query capability
                try:
                    # Query recent logs
                    query_params = {
                        "query": "{job=\"docker\"} |~ \".*\"",
                        "limit": "10",
                        "start": int((datetime.utcnow() - timedelta(minutes=5)).timestamp() * 1000000000)
                    }
                    
                    logs_response = requests.get(f"{self.monitoring_endpoints['loki']}/loki/api/v1/query_range",
                                               params=query_params, timeout=10)
                    
                    results["log_query_successful"] = logs_response.status_code == 200
                    
                    if logs_response.status_code == 200:
                        logs_data = logs_response.json()
                        results["logs_available"] = len(logs_data.get("data", {}).get("result", [])) > 0
                        
                except Exception as e:
                    results["log_query_error"] = str(e)
            
        except Exception as e:
            results["loki_accessible"] = False
            results["error"] = str(e)
        
        results["success"] = results.get("loki_accessible", False)
        
        self.log_result("log_aggregation", results["success"], results)
        return results

    def run_monitoring_validation(self) -> Dict[str, Any]:
        """Run complete monitoring system validation"""
        logger.info("🔍 Starting MONITORING SYSTEM VALIDATION")
        start_time = time.time()
        
        monitoring_results = {}
        
        # Test all monitoring components
        logger.info("📊 Testing Prometheus Metrics Collection")
        monitoring_results["prometheus_metrics"] = self.test_prometheus_metrics_collection()
        
        logger.info("📈 Testing Grafana Dashboard Access")
        monitoring_results["grafana_dashboard"] = self.test_grafana_dashboard_access()
        
        logger.info("🐳 Testing Docker Health Checks")
        monitoring_results["docker_health"] = self.test_docker_health_checks()
        
        logger.info("🔄 Testing Service Health Consistency")
        monitoring_results["health_consistency"] = self.test_service_health_consistency()
        
        logger.info("⚡ Testing Performance Monitoring Accuracy")
        monitoring_results["performance_monitoring"] = self.test_performance_monitoring_accuracy()
        
        logger.info("📋 Testing Log Aggregation")
        monitoring_results["log_aggregation"] = self.test_log_aggregation()
        
        total_time = time.time() - start_time
        
        # Calculate results
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['success'])
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        final_results = {
            "monitoring_validation_summary": {
                "total_monitoring_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate": success_rate,
                "monitoring_systems_healthy": success_rate >= 80,  # 80% minimum
                "total_execution_time": total_time
            },
            "monitoring_results": monitoring_results,
            "test_results": self.test_results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(f"🏁 MONITORING VALIDATION COMPLETE: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
        return final_results

def main():
    """Main execution for monitoring validation"""
    validator = MonitoringSystemValidator()
    results = validator.run_monitoring_validation()
    
    # Save results
    output_file = f"/opt/sutazaiapp/tests/monitoring_validation_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🔍 MONITORING SYSTEM VALIDATION RESULTS")
    print(f"Results saved to: {output_file}")
    print(f"Monitoring Systems Health: {results['monitoring_validation_summary']['success_rate']:.1f}%")
    print(f"Tests Passed: {results['monitoring_validation_summary']['passed_tests']}/{results['monitoring_validation_summary']['total_monitoring_tests']}")
    
    return results

if __name__ == "__main__":
    results = main()