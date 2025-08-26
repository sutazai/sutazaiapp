#!/usr/bin/env python3
"""
Continuous Test Runner for Hardware Resource Optimizer

Purpose: Automated continuous testing with monitoring and alerting
Usage: python continuous_validator.py [--interval MINUTES] [--dashboard]
Requirements: Integration test suite, agent running on port 8116
"""

import os
import sys
import json
import time
import schedule
import threading
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import requests
import smtplib
from email.mime.text import MIMEText
from collections import deque, defaultdict
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socketserver

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('continuous_validator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ContinuousValidator:
    """Continuous validation and monitoring for hardware optimizer"""
    
    def __init__(self, interval_minutes: int = 60):
        self.interval_minutes = interval_minutes
        self.base_url = "http://localhost:8116"
        self.test_history = deque(maxlen=168)  # Keep 1 week of hourly results
        self.metrics_history = defaultdict(lambda: deque(maxlen=168))
        self.alert_thresholds = {
            "failure_rate": 20.0,  # Alert if >20% tests fail
            "response_time": 5.0,  # Alert if responses >5 seconds
            "consecutive_failures": 3  # Alert after 3 consecutive failures
        }
        self.consecutive_failures = 0
        self.dashboard_server = None
        self.monitoring_active = True
        
        # Ensure reports directory exists
        self.reports_dir = Path("continuous_test_reports")
        self.reports_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized continuous validator with {interval_minutes} minute interval")
    
    def start(self, with_dashboard: bool = False):
        """Start continuous validation"""
        logger.info("Starting continuous validation...")
        
        # Start dashboard if requested
        if with_dashboard:
            self._start_dashboard_server()
        
        # Run initial test
        self._run_validation_cycle()
        
        # Schedule regular tests
        schedule.every(self.interval_minutes).minutes.do(self._run_validation_cycle)
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self._monitor_agent_health, daemon=True)
        monitor_thread.start()
        
        # Run scheduler
        try:
            while self.monitoring_active:
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds
        except KeyboardInterrupt:
            logger.info("Stopping continuous validation...")
            self.monitoring_active = False
            if self.dashboard_server:
                self.dashboard_server.shutdown()
    
    def _run_validation_cycle(self):
        """Run a complete validation cycle"""
        cycle_start = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting validation cycle at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'='*60}")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "cycle_id": f"cycle_{int(time.time())}",
            "tests": {},
            "metrics": {},
            "alerts": []
        }
        
        # Run quick health check
        health_ok, health_response = self._check_agent_health()
        results["agent_healthy"] = health_ok
        
        if not health_ok:
            logger.error("Agent health check failed!")
            results["alerts"].append({
                "type": "agent_down",
                "message": "Hardware optimizer agent is not responding",
                "severity": "critical"
            })
            self.consecutive_failures += 1
        else:
            # Run test scenarios
            test_results = self._run_test_scenarios()
            results["tests"] = test_results
            
            # Collect performance metrics
            metrics = self._collect_performance_metrics()
            results["metrics"] = metrics
            
            # Analyze results
            analysis = self._analyze_results(test_results, metrics)
            results["analysis"] = analysis
            
            # Check for alerts
            alerts = self._check_alert_conditions(test_results, metrics, analysis)
            results["alerts"] = alerts
            
            # Update success/failure tracking
            if analysis["overall_success"]:
                self.consecutive_failures = 0
            else:
                self.consecutive_failures += 1
        
        # Record cycle duration
        results["cycle_duration_seconds"] = time.time() - cycle_start
        
        # Save results
        self._save_results(results)
        
        # Update history
        self.test_history.append(results)
        self._update_metrics_history(results)
        
        # Send alerts if needed
        if results["alerts"]:
            self._send_alerts(results["alerts"])
        
        # Generate summary report
        self._generate_summary_report(results)
        
        logger.info(f"Validation cycle completed in {results['cycle_duration_seconds']:.1f} seconds")
    
    def _check_agent_health(self) -> Tuple[bool, Optional[Dict]]:
        """Quick health check of the agent"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                return True, response.json()
            return False, None
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False, None
    
    def _run_test_scenarios(self) -> Dict[str, Any]:
        """Run integration test scenarios"""
        logger.info("Running test scenarios...")
        
        # Use the integration test suite
        test_script = Path(__file__).parent / "integration_test_suite.py"
        
        scenarios = ["full_system", "storage", "pressure", "docker", "concurrent", "errors"]
        scenario_results = {}
        
        for scenario in scenarios:
            logger.info(f"Running scenario: {scenario}")
            
            try:
                # Run scenario via subprocess to isolate it
                cmd = [sys.executable, str(test_script), "--scenario", scenario]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                # Parse results from output
                success = result.returncode == 0
                output_lines = result.stdout.split('\n')
                
                # Look for pass/fail in output
                passed = any("PASSED" in line and scenario.upper() in line for line in output_lines)
                
                scenario_results[scenario] = {
                    "success": passed,
                    "duration": time.time() - time.time(),  # Will be updated from logs
                    "output": result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout,
                    "errors": result.stderr if not success else None
                }
                
            except subprocess.TimeoutExpired:
                logger.error(f"Scenario {scenario} timed out")
                scenario_results[scenario] = {
                    "success": False,
                    "error": "Timeout after 5 minutes"
                }
            except Exception as e:
                logger.error(f"Scenario {scenario} failed: {str(e)}")
                scenario_results[scenario] = {
                    "success": False,
                    "error": str(e)
                }
        
        return scenario_results
    
    def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance metrics from the agent"""
        metrics = {}
        
        try:
            # Get system status
            response = requests.get(f"{self.base_url}/status", timeout=5)
            if response.status_code == 200:
                status = response.json()
                metrics["system_status"] = status.get("system_status", {})
                metrics["agent_status"] = status.get("agent_status", {})
                metrics["docker_status"] = status.get("docker_status", {})
            
            # Test endpoint response times
            endpoints = [
                "/health",
                "/status", 
                "/analyze/storage?path=/tmp",
                "/analyze/storage/report?days=1"
            ]
            
            response_times = {}
            for endpoint in endpoints:
                start = time.time()
                try:
                    response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                    response_times[endpoint] = {
                        "time_ms": (time.time() - start) * 1000,
                        "status_code": response.status_code
                    }
                except (AssertionError, Exception) as e:
                    logger.error(f"Unexpected exception: {e}", exc_info=True)
                    response_times[endpoint] = {
                        "time_ms": -1,
                        "status_code": -1
                    }
            
            metrics["response_times"] = response_times
            
            # Calculate average response time
            valid_times = [rt["time_ms"] for rt in response_times.values() if rt["time_ms"] > 0]
            metrics["avg_response_time_ms"] = sum(valid_times) / len(valid_times) if valid_times else -1
            
        except Exception as e:
            logger.error(f"Failed to collect metrics: {str(e)}")
            metrics["error"] = str(e)
        
        return metrics
    
    def _analyze_results(self, test_results: Dict, metrics: Dict) -> Dict[str, Any]:
        """Analyze test results and metrics"""
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "scenarios_total": len(test_results),
            "scenarios_passed": sum(1 for r in test_results.values() if r.get("success", False)),
            "scenarios_failed": sum(1 for r in test_results.values() if not r.get("success", False)),
            "success_rate": 0.0,
            "overall_success": False,
            "performance_grade": "Unknown"
        }
        
        # Calculate success rate
        if analysis["scenarios_total"] > 0:
            analysis["success_rate"] = (analysis["scenarios_passed"] / analysis["scenarios_total"]) * 100
            analysis["overall_success"] = analysis["success_rate"] >= 80.0  # 80% threshold
        
        # Grade performance
        avg_response = metrics.get("avg_response_time_ms", -1)
        if avg_response > 0:
            if avg_response < 100:
                analysis["performance_grade"] = "Excellent"
            elif avg_response < 500:
                analysis["performance_grade"] = "Good"
            elif avg_response < 1000:
                analysis["performance_grade"] = "Fair"
            else:
                analysis["performance_grade"] = "Poor"
        
        # Add system health
        if "system_status" in metrics:
            analysis["system_health"] = {
                "cpu_ok": metrics["system_status"].get("cpu_percent", 100) < 80,
                "memory_ok": metrics["system_status"].get("memory_percent", 100) < 85,
                "disk_ok": metrics["system_status"].get("disk_percent", 100) < 90
            }
        
        return analysis
    
    def _check_alert_conditions(self, test_results: Dict, metrics: Dict, 
                               analysis: Dict) -> List[Dict]:
        """Check for alert conditions"""
        alerts = []
        
        # Check failure rate
        if analysis["success_rate"] < (100 - self.alert_thresholds["failure_rate"]):
            alerts.append({
                "type": "high_failure_rate",
                "message": f"Test failure rate is {100 - analysis['success_rate']:.1f}%",
                "severity": "warning" if analysis["success_rate"] > 50 else "critical",
                "value": 100 - analysis["success_rate"]
            })
        
        # Check response times
        avg_response = metrics.get("avg_response_time_ms", 0) / 1000  # Convert to seconds
        if avg_response > self.alert_thresholds["response_time"]:
            alerts.append({
                "type": "slow_response",
                "message": f"Average response time is {avg_response:.1f} seconds",
                "severity": "warning",
                "value": avg_response
            })
        
        # Check consecutive failures
        if self.consecutive_failures >= self.alert_thresholds["consecutive_failures"]:
            alerts.append({
                "type": "consecutive_failures",
                "message": f"{self.consecutive_failures} consecutive test failures",
                "severity": "critical",
                "value": self.consecutive_failures
            })
        
        # Check system health
        if "system_health" in analysis:
            health = analysis["system_health"]
            if not health.get("cpu_ok", True):
                alerts.append({
                    "type": "high_cpu",
                    "message": "System CPU usage is above 80%",
                    "severity": "warning"
                })
            if not health.get("memory_ok", True):
                alerts.append({
                    "type": "high_memory",
                    "message": "System memory usage is above 85%",
                    "severity": "warning"
                })
            if not health.get("disk_ok", True):
                alerts.append({
                    "type": "high_disk",
                    "message": "System disk usage is above 90%",
                    "severity": "critical"
                })
        
        return alerts
    
    def _save_results(self, results: Dict):
        """Save test results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.reports_dir / f"validation_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Also save latest results for dashboard
        latest_file = self.reports_dir / "latest_results.json"
        with open(latest_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    def _update_metrics_history(self, results: Dict):
        """Update metrics history for trending"""
        timestamp = results["timestamp"]
        
        # Track key metrics over time
        if "analysis" in results:
            self.metrics_history["success_rate"].append({
                "timestamp": timestamp,
                "value": results["analysis"]["success_rate"]
            })
        
        if "metrics" in results and "avg_response_time_ms" in results["metrics"]:
            self.metrics_history["response_time"].append({
                "timestamp": timestamp,
                "value": results["metrics"]["avg_response_time_ms"]
            })
        
        if "metrics" in results and "system_status" in results["metrics"]:
            sys_status = results["metrics"]["system_status"]
            self.metrics_history["cpu_usage"].append({
                "timestamp": timestamp,
                "value": sys_status.get("cpu_percent", 0)
            })
            self.metrics_history["memory_usage"].append({
                "timestamp": timestamp,
                "value": sys_status.get("memory_percent", 0)
            })
    
    def _send_alerts(self, alerts: List[Dict]):
        """Send alerts (log for now, can add email/webhook later)"""
        for alert in alerts:
            level = logging.CRITICAL if alert["severity"] == "critical" else logging.WARNING
            logger.log(level, f"ALERT: {alert['type']} - {alert['message']}")
        
        # Write alerts to dedicated file
        alert_file = self.reports_dir / "alerts.log"
        with open(alert_file, 'a') as f:
            for alert in alerts:
                f.write(f"{datetime.now().isoformat()} - {alert['severity'].upper()} - "
                       f"{alert['type']} - {alert['message']}\n")
    
    def _generate_summary_report(self, results: Dict):
        """Generate human-readable summary report"""
        report_lines = [
            "\nVALIDATION SUMMARY",
            "=" * 50,
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Agent Status: {'Healthy' if results.get('agent_healthy', False) else 'UNHEALTHY'}",
            ""
        ]
        
        if "analysis" in results:
            analysis = results["analysis"]
            report_lines.extend([
                "Test Results:",
                f"  Total Scenarios: {analysis['scenarios_total']}",
                f"  Passed: {analysis['scenarios_passed']}",
                f"  Failed: {analysis['scenarios_failed']}",
                f"  Success Rate: {analysis['success_rate']:.1f}%",
                f"  Performance: {analysis['performance_grade']}",
                ""
            ])
        
        if results.get("alerts"):
            report_lines.extend([
                "ALERTS:",
                *[f"  - [{alert['severity'].upper()}] {alert['message']}" 
                  for alert in results["alerts"]],
                ""
            ])
        else:
            report_lines.append("No alerts - all systems normal\n")
        
        report_lines.append("=" * 50)
        
        # Print to console
        for line in report_lines:
            logger.info(line)
        
        # Save to summary file
        summary_file = self.reports_dir / "summary.txt"
        with open(summary_file, 'a') as f:
            f.write('\n'.join(report_lines) + '\n\n')
    
    def _monitor_agent_health(self):
        """Background thread to monitor agent health"""
        while self.monitoring_active:
            try:
                # Quick health check every minute
                health_ok, _ = self._check_agent_health()
                
                if not health_ok:
                    logger.warning("Agent health check failed in monitor thread")
                    
                    # Try to restart agent if it's down for too long
                    if self.consecutive_failures > 5:
                        logger.error("Agent has been down for multiple cycles")
                        # Could implement auto-restart here if needed
                
            except Exception as e:
                logger.error(f"Health monitor error: {str(e)}")
            
            time.sleep(60)  # Check every minute
    
    def _start_dashboard_server(self):
        """Start HTTP server for dashboard"""
        logger.info("Starting dashboard server on port 8117...")
        
        class DashboardHandler(SimpleHTTPRequestHandler):
            def __init__(self, *args, validator=None, **kwargs):
                self.validator = validator
                super().__init__(*args, **kwargs)
            
            def do_GET(self):
                if self.path == "/api/latest":
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    
                    # Send latest results
                    latest_file = Path("continuous_test_reports/latest_results.json")
                    if latest_file.exists():
                        with open(latest_file, 'r') as f:
                            self.wfile.write(f.read().encode())
                    else:
                        self.wfile.write(json.dumps({"error": "No results yet"}).encode())
                
                elif self.path == "/api/history":
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    
                    # Convert deques to lists for JSON serialization
                    history = {
                        "test_history": list(self.validator.test_history)[-24:],  # Last 24 hours
                        "metrics_history": {
                            k: list(v)[-24:] for k, v in self.validator.metrics_history.items()
                        }
                    }
                    self.wfile.write(json.dumps(history).encode())
                
                else:
                    # Serve dashboard HTML
                    self.path = '/test_dashboard.html'
                    super().do_GET()
        
        # Create handler with validator reference
        handler = lambda *args, **kwargs: DashboardHandler(*args, validator=self, **kwargs)
        
        # Start server in thread
        server_thread = threading.Thread(
            target=lambda: HTTPServer(('', 8117), handler).serve_forever(),
            daemon=True
        )
        server_thread.start()
        
        logger.info("Dashboard available at http://localhost:8117")
    
    def generate_trends_report(self) -> Dict[str, Any]:
        """Generate trends report from historical data"""
        if len(self.test_history) < 2:
            return {"error": "Not enough data for trends"}
        
        trends = {
            "period": f"Last {len(self.test_history)} runs",
            "success_rate_trend": [],
            "performance_trend": [],
            "alert_frequency": defaultdict(int)
        }
        
        # Calculate trends
        for result in self.test_history:
            if "analysis" in result:
                trends["success_rate_trend"].append(result["analysis"]["success_rate"])
            
            if "metrics" in result and "avg_response_time_ms" in result["metrics"]:
                trends["performance_trend"].append(result["metrics"]["avg_response_time_ms"])
            
            for alert in result.get("alerts", []):
                trends["alert_frequency"][alert["type"]] += 1
        
        # Calculate averages and changes
        if trends["success_rate_trend"]:
            trends["avg_success_rate"] = sum(trends["success_rate_trend"]) / len(trends["success_rate_trend"])
            trends["success_rate_change"] = trends["success_rate_trend"][-1] - trends["success_rate_trend"][0]
        
        if trends["performance_trend"]:
            trends["avg_response_time_ms"] = sum(trends["performance_trend"]) / len(trends["performance_trend"])
            trends["performance_change_ms"] = trends["performance_trend"][-1] - trends["performance_trend"][0]
        
        return trends


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Continuous Hardware Optimizer Validator")
    parser.add_argument("--interval", type=int, default=60,
                       help="Test interval in minutes (default: 60)")
    parser.add_argument("--dashboard", action="store_true",
                       help="Start web dashboard on port 8117")
    parser.add_argument("--once", action="store_true",
                       help="Run validation once and exit")
    
    args = parser.parse_args()
    
    validator = ContinuousValidator(interval_minutes=args.interval)
    
    try:
        if args.once:
            # Run single validation cycle
            validator._run_validation_cycle()
            
            # Generate trends report
            trends = validator.generate_trends_report()
            logger.info(f"\nTrends Report: {json.dumps(trends, indent=2)}")
        else:
            # Start continuous validation
            validator.start(with_dashboard=args.dashboard)
    
    except KeyboardInterrupt:
        logger.info("Validation stopped by user")
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        raise


if __name__ == "__main__":
