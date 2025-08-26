"""
Load test runner for SutazAI system
Automates load testing with different scenarios and reporting
"""

import subprocess
import json
import csv
import time
import os
import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import requests
import psutil


class LoadTestRunner:
    """Automated load test runner with reporting."""

    def __init__(self, base_url: str = "http://localhost:8000", results_dir: str = "load_test_results"):
        self.base_url = base_url
        self.results_dir = results_dir
        self.test_start_time = None
        self.test_end_time = None
        self.system_metrics = []
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{results_dir}/load_test.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def check_system_prerequisites(self) -> bool:
        """Check if system is ready for load testing."""
        self.logger.info("Checking system prerequisites...")
        
        # Check if services are running
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code != 200:
                self.logger.error(f"Backend health check failed: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Cannot connect to backend: {str(e)}")
            return False
        
        # Check system resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        if cpu_percent > 80:
            self.logger.warning(f"High CPU usage before test: {cpu_percent}%")
        
        if memory.percent > 85:
            self.logger.warning(f"High memory usage before test: {memory.percent}%")
        
        # Check if Locust is available
        try:
            result = subprocess.run(["locust", "--version"], capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error("Locust is not installed or not working")
                return False
        except FileNotFoundError:
            self.logger.error("Locust command not found")
            return False
        
        self.logger.info("System prerequisites check passed")
        return True

    def run_load_test(
        self, 
        scenario: str = "medium_load",
        users: int = 10,
        spawn_rate: int = 2,
        run_time: str = "300s",
        additional_args: List[str] = None
    ) -> Dict[str, Any]:
        """Run load test with specified parameters."""
        
        if not self.check_system_prerequisites():
            raise RuntimeError("System prerequisites not met")
        
        self.test_start_time = datetime.now()
        test_id = self.test_start_time.strftime("%Y%m%d_%H%M%S")
        
        self.logger.info(f"Starting load test {test_id}")
        self.logger.info(f"Scenario: {scenario}, Users: {users}, Spawn Rate: {spawn_rate}, Duration: {run_time}")
        
        # Prepare Locust command
        cmd = [
            "locust",
            "-f", "locustfile.py",
            "--headless",
            f"--users={users}",
            f"--spawn-rate={spawn_rate}",
            f"--run-time={run_time}",
            f"--host={self.base_url}",
            f"--html={self.results_dir}/report_{test_id}.html",
            f"--csv={self.results_dir}/results_{test_id}"
        ]
        
        if additional_args:
            cmd.extend(additional_args)
        
        # Start system monitoring
        monitoring_process = self._start_system_monitoring()
        
        try:
            # Run load test
            self.logger.info(f"Executing command: {' '.join(cmd)}")
            
            # Change to tests/load directory
            original_cwd = os.getcwd()
            test_dir = os.path.join(os.path.dirname(__file__))
            os.chdir(test_dir)
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._parse_time_to_seconds(run_time) + 60  # Add buffer
            )
            
            os.chdir(original_cwd)
            
            self.test_end_time = datetime.now()
            
            # Stop system monitoring
            self._stop_system_monitoring(monitoring_process)
            
            # Process results
            test_results = self._process_test_results(test_id, result)
            
            # Generate report
            self._generate_comprehensive_report(test_id, test_results)
            
            return test_results
            
        except subprocess.TimeoutExpired:
            self.logger.error("Load test timed out")
            self._stop_system_monitoring(monitoring_process)
            raise
        except Exception as e:
            self.logger.error(f"Load test failed: {str(e)}")
            self._stop_system_monitoring(monitoring_process)
            raise

    def _start_system_monitoring(self):
        """Start system resource monitoring."""
        import threading
        
        self.system_metrics = []
        self.monitoring_active = True
        
        def monitor_system():
            while self.monitoring_active:
                try:
                    cpu_percent = psutil.cpu_percent()
                    memory = psutil.virtual_memory()
                    disk = psutil.disk_usage('/')
                    network = psutil.net_io_counters()
                    
                    metrics = {
                        "timestamp": datetime.now().isoformat(),
                        "cpu_percent": cpu_percent,
                        "memory_percent": memory.percent,
                        "memory_used_gb": memory.used / (1024**3),
                        "disk_percent": (disk.used / disk.total) * 100,
                        "network_bytes_sent": network.bytes_sent,
                        "network_bytes_recv": network.bytes_recv
                    }
                    
                    self.system_metrics.append(metrics)
                    time.sleep(5)  # Monitor every 5 seconds
                    
                except Exception as e:
                    self.logger.error(f"System monitoring error: {str(e)}")
        
        monitoring_thread = threading.Thread(target=monitor_system)
        monitoring_thread.daemon = True
        monitoring_thread.start()
        
        return monitoring_thread

    def _stop_system_monitoring(self, monitoring_process):
        """Stop system resource monitoring."""
        self.monitoring_active = False
        time.sleep(1)  # Allow thread to finish

    def _process_test_results(self, test_id: str, subprocess_result) -> Dict[str, Any]:
        """Process load test results."""
        results = {
            "test_id": test_id,
            "start_time": self.test_start_time.isoformat(),
            "end_time": self.test_end_time.isoformat(),
            "duration": (self.test_end_time - self.test_start_time).total_seconds(),
            "subprocess_result": {
                "returncode": subprocess_result.returncode,
                "stdout": subprocess_result.stdout,
                "stderr": subprocess_result.stderr
            },
            "system_metrics": self.system_metrics
        }
        
        # Parse Locust CSV results if available
        stats_file = f"{self.results_dir}/results_{test_id}_stats.csv"
        if os.path.exists(stats_file):
            results["locust_stats"] = self._parse_locust_stats(stats_file)
        
        failures_file = f"{self.results_dir}/results_{test_id}_failures.csv"
        if os.path.exists(failures_file):
            results["locust_failures"] = self._parse_locust_failures(failures_file)
        
        exceptions_file = f"{self.results_dir}/results_{test_id}_exceptions.csv"
        if os.path.exists(exceptions_file):
            results["locust_exceptions"] = self._parse_locust_exceptions(exceptions_file)
        
        return results

    def _parse_locust_stats(self, stats_file: str) -> List[Dict]:
        """Parse Locust statistics CSV file."""
        stats = []
        try:
            with open(stats_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    stats.append(dict(row))
        except Exception as e:
            self.logger.error(f"Error parsing Locust stats: {str(e)}")
        
        return stats

    def _parse_locust_failures(self, failures_file: str) -> List[Dict]:
        """Parse Locust failures CSV file."""
        failures = []
        try:
            with open(failures_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    failures.append(dict(row))
        except Exception as e:
            self.logger.error(f"Error parsing Locust failures: {str(e)}")
        
        return failures

    def _parse_locust_exceptions(self, exceptions_file: str) -> List[Dict]:
        """Parse Locust exceptions CSV file."""
        exceptions = []
        try:
            with open(exceptions_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    exceptions.append(dict(row))
        except Exception as e:
            self.logger.error(f"Error parsing Locust exceptions: {str(e)}")
        
        return exceptions

    def _generate_comprehensive_report(self, test_id: str, results: Dict[str, Any]):
        """Generate comprehensive test report."""
        report_file = f"{self.results_dir}/comprehensive_report_{test_id}.json"
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(results)
        results["performance_metrics"] = performance_metrics
        
        # Generate system health analysis
        system_analysis = self._analyze_system_metrics(results.get("system_metrics", []))
        results["system_analysis"] = system_analysis
        
        # Save comprehensive report
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate summary report
        self._generate_summary_report(test_id, results)
        
        self.logger.info(f"Comprehensive report saved: {report_file}")

    def _calculate_performance_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance metrics from test results."""
        metrics = {}
        
        # Analyze Locust stats
        locust_stats = results.get("locust_stats", [])
        if locust_stats:
            for stat in locust_stats:
                if stat.get("Type") == "GET" or stat.get("Type") == "POST":
                    endpoint = stat.get("Name", "unknown")
                    metrics[f"{endpoint}_avg_response_time"] = float(stat.get("Average", 0))
                    metrics[f"{endpoint}_min_response_time"] = float(stat.get("Min", 0))
                    metrics[f"{endpoint}_max_response_time"] = float(stat.get("Max", 0))
                    metrics[f"{endpoint}_requests_per_sec"] = float(stat.get("Requests/s", 0))
                    metrics[f"{endpoint}_failure_rate"] = float(stat.get("Failure %", 0))
        
        # Analyze failures
        locust_failures = results.get("locust_failures", [])
        metrics["total_failures"] = len(locust_failures)
        
        if locust_failures:
            failure_types = {}
            for failure in locust_failures:
                failure_type = failure.get("Error", "unknown")
                failure_types[failure_type] = failure_types.get(failure_type, 0) + 1
            metrics["failure_breakdown"] = failure_types
        
        return metrics

    def _analyze_system_metrics(self, system_metrics: List[Dict]) -> Dict[str, Any]:
        """Analyze system metrics during load test."""
        if not system_metrics:
            return {}
        
        analysis = {}
        
        # CPU analysis
        cpu_values = [m["cpu_percent"] for m in system_metrics]
        analysis["cpu"] = {
            "avg": sum(cpu_values) / len(cpu_values),
            "max": max(cpu_values),
            "min": min(cpu_values)
        }
        
        # Memory analysis
        memory_values = [m["memory_percent"] for m in system_metrics]
        analysis["memory"] = {
            "avg": sum(memory_values) / len(memory_values),
            "max": max(memory_values),
            "min": min(memory_values)
        }
        
        # Disk analysis
        disk_values = [m["disk_percent"] for m in system_metrics]
        analysis["disk"] = {
            "avg": sum(disk_values) / len(disk_values),
            "max": max(disk_values),
            "min": min(disk_values)
        }
        
        # Network analysis
        if len(system_metrics) > 1:
            bytes_sent_diff = system_metrics[-1]["network_bytes_sent"] - system_metrics[0]["network_bytes_sent"]
            bytes_recv_diff = system_metrics[-1]["network_bytes_recv"] - system_metrics[0]["network_bytes_recv"]
            duration = len(system_metrics) * 5  # 5 seconds between measurements
            
            analysis["network"] = {
                "avg_bytes_sent_per_sec": bytes_sent_diff / duration,
                "avg_bytes_recv_per_sec": bytes_recv_diff / duration,
                "total_bytes_sent": bytes_sent_diff,
                "total_bytes_recv": bytes_recv_diff
            }
        
        return analysis

    def _generate_summary_report(self, test_id: str, results: Dict[str, Any]):
        """Generate human-readable summary report."""
        summary_file = f"{self.results_dir}/summary_{test_id}.txt"
        
        with open(summary_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write(f"SutazAI Load Test Summary - {test_id}\n")
            f.write("=" * 60 + "\n\n")
            
            # Test information
            f.write("Test Information:\n")
            f.write(f"  Start Time: {results['start_time']}\n")
            f.write(f"  End Time: {results['end_time']}\n")
            f.write(f"  Duration: {results['duration']:.2f} seconds\n")
            f.write(f"  Exit Code: {results['subprocess_result']['returncode']}\n\n")
            
            # Performance metrics
            if "performance_metrics" in results:
                f.write("Performance Metrics:\n")
                metrics = results["performance_metrics"]
                
                for key, value in metrics.items():
                    if isinstance(value, dict):
                        f.write(f"  {key}:\n")
                        for subkey, subvalue in value.items():
                            f.write(f"    {subkey}: {subvalue}\n")
                    else:
                        f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            # System analysis
            if "system_analysis" in results:
                f.write("System Resource Analysis:\n")
                analysis = results["system_analysis"]
                
                for resource, metrics in analysis.items():
                    f.write(f"  {resource.upper()}:\n")
                    if isinstance(metrics, dict):
                        for metric, value in metrics.items():
                            if isinstance(value, float):
                                f.write(f"    {metric}: {value:.2f}\n")
                            else:
                                f.write(f"    {metric}: {value}\n")
                f.write("\n")
            
            # Failures summary
            if "locust_failures" in results and results["locust_failures"]:
                f.write("Failures Summary:\n")
                f.write(f"  Total Failures: {len(results['locust_failures'])}\n")
                
                # Group failures by type
                failure_types = {}
                for failure in results["locust_failures"]:
                    error = failure.get("Error", "Unknown")
                    failure_types[error] = failure_types.get(error, 0) + 1
                
                for error, count in failure_types.items():
                    f.write(f"  {error}: {count}\n")
                f.write("\n")
            
            # Recommendations
            f.write("Recommendations:\n")
            recommendations = self._generate_recommendations(results)
            for rec in recommendations:
                f.write(f"  - {rec}\n")
        
        self.logger.info(f"Summary report saved: {summary_file}")

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations based on test results."""
        recommendations = []
        
        # Analyze system metrics for recommendations
        system_analysis = results.get("system_analysis", {})
        
        if "cpu" in system_analysis:
            cpu_max = system_analysis["cpu"]["max"]
            if cpu_max > 90:
                recommendations.append(f"High CPU usage detected ({cpu_max:.1f}%). Consider scaling horizontally or optimizing CPU-intensive operations.")
            elif cpu_max > 70:
                recommendations.append(f"Moderate CPU usage ({cpu_max:.1f}%). Monitor CPU usage under higher loads.")
        
        if "memory" in system_analysis:
            memory_max = system_analysis["memory"]["max"]
            if memory_max > 85:
                recommendations.append(f"High memory usage detected ({memory_max:.1f}%). Consider increasing memory or optimizing memory usage.")
            elif memory_max > 70:
                recommendations.append(f"Moderate memory usage ({memory_max:.1f}%). Monitor memory usage patterns.")
        
        # Analyze performance metrics
        performance_metrics = results.get("performance_metrics", {})
        
        if "total_failures" in performance_metrics and performance_metrics["total_failures"] > 0:
            recommendations.append(f"Test had {performance_metrics['total_failures']} failures. Investigate error handling and system stability.")
        
        # Response time recommendations
        for key, value in performance_metrics.items():
            if key.endswith("_avg_response_time") and isinstance(value, (int, float)):
                if value > 5000:  # 5 seconds
                    endpoint = key.replace("_avg_response_time", "")
                    recommendations.append(f"Slow response time for {endpoint} ({value:.2f}ms). Consider caching or optimization.")
                elif value > 2000:  # 2 seconds
                    endpoint = key.replace("_avg_response_time", "")
                    recommendations.append(f"Moderate response time for {endpoint} ({value:.2f}ms). Monitor performance under higher loads.")
        
        if not recommendations:
            recommendations.append("System performed well under test conditions. Consider testing with higher loads.")
        
        return recommendations

    def _parse_time_to_seconds(self, time_str: str) -> int:
        """Parse time string to seconds."""
        if time_str.endswith('s'):
            return int(time_str[:-1])
        elif time_str.endswith('m'):
            return int(time_str[:-1]) * 60
        elif time_str.endswith('h'):
            return int(time_str[:-1]) * 3600
        else:
            return int(time_str)

    def run_scenario_suite(self, scenarios: Dict[str, Dict]) -> Dict[str, Any]:
        """Run multiple load test scenarios."""
        suite_results = {}
        suite_start_time = datetime.now()
        
        self.logger.info(f"Starting load test scenario suite with {len(scenarios)} scenarios")
        
        for scenario_name, scenario_config in scenarios.items():
            self.logger.info(f"Running scenario: {scenario_name}")
            
            try:
                result = self.run_load_test(
                    scenario=scenario_name,
                    users=scenario_config.get("users", 10),
                    spawn_rate=scenario_config.get("spawn_rate", 2),
                    run_time=scenario_config.get("run_time", "300s"),
                    additional_args=scenario_config.get("additional_args", [])
                )
                
                suite_results[scenario_name] = {
                    "status": "success",
                    "result": result
                }
                
                # Wait between scenarios
                time.sleep(30)
                
            except Exception as e:
                self.logger.error(f"Scenario {scenario_name} failed: {str(e)}")
                suite_results[scenario_name] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        suite_end_time = datetime.now()
        
        # Generate suite summary
        suite_summary = {
            "start_time": suite_start_time.isoformat(),
            "end_time": suite_end_time.isoformat(),
            "duration": (suite_end_time - suite_start_time).total_seconds(),
            "scenarios": suite_results,
            "summary": {
                "total_scenarios": len(scenarios),
                "successful_scenarios": sum(1 for r in suite_results.values() if r["status"] == "success"),
                "failed_scenarios": sum(1 for r in suite_results.values() if r["status"] == "failed")
            }
        }
        
        # Save suite results
        suite_file = f"{self.results_dir}/suite_results_{suite_start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(suite_file, 'w') as f:
            json.dump(suite_summary, f, indent=2, default=str)
        
        self.logger.info(f"Scenario suite completed. Results saved: {suite_file}")
        return suite_summary


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="SutazAI Load Test Runner")
    parser.add_argument("--host", default="http://localhost:8000", help="Target host URL")
    parser.add_argument("--users", type=int, default=10, help="Number of users")
    parser.add_argument("--spawn-rate", type=int, default=2, help="Spawn rate")
    parser.add_argument("--run-time", default="300s", help="Run time")
    parser.add_argument("--scenario", default="medium_load", help="Load test scenario")
    parser.add_argument("--results-dir", default="load_test_results", help="Results directory")
    parser.add_argument("--suite", action="store_true", help="Run full scenario suite")
    
    args = parser.parse_args()
    
    runner = LoadTestRunner(base_url=args.host, results_dir=args.results_dir)
    
    if args.suite:
        # Run full scenario suite
        scenarios = {
            "light_load": {"users": 5, "spawn_rate": 1, "run_time": "180s"},
            "medium_load": {"users": 10, "spawn_rate": 2, "run_time": "300s"},
            "heavy_load": {"users": 20, "spawn_rate": 4, "run_time": "300s"},
            "spike_test": {"users": 50, "spawn_rate": 10, "run_time": "120s"}
        }
        
        suite_results = runner.run_scenario_suite(scenarios)
        logger.info(f"Suite completed. {suite_results['summary']['successful_scenarios']}/{suite_results['summary']['total_scenarios']} scenarios successful.")
    else:
        # Run single scenario
        results = runner.run_load_test(
            scenario=args.scenario,
            users=args.users,
            spawn_rate=args.spawn_rate,
            run_time=args.run_time
        )
        
        logger.info(f"Load test completed. Test ID: {results['test_id']}")
        if results['subprocess_result']['returncode'] == 0:
            logger.info("Test executed successfully.")
        else:
            logger.info("Test had issues. Check the logs for details.")


if __name__ == "__main__":
    main()