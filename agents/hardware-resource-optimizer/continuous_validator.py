#!/usr/bin/env python3
"""
Continuous Validation System for Hardware Resource Optimizer
Runs automated tests and monitors agent health continuously
"""

import os
import sys
import time
import json
import schedule
import requests
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import logging
import smtplib
from email.mime.text import MIMEText

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/agents/hardware-resource-optimizer/continuous_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ContinuousValidator')

BASE_URL = "http://localhost:8080"
RESULTS_DIR = "/opt/sutazaiapp/agents/hardware-resource-optimizer/validation_results"
ALERT_THRESHOLD = 0.8  # Alert if pass rate drops below 80%

class ContinuousValidator:
    """Continuous validation and monitoring system"""
    
    def __init__(self):
        self.history = []
        self.alerts = []
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
    def check_agent_health(self) -> Dict[str, Any]:
        """Check if agent is healthy and responding"""
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "data": response.json(),
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            
        return {
            "status": "unhealthy",
            "error": str(e) if 'e' in locals() else "Unknown error",
            "timestamp": datetime.now().isoformat()
        }
        
    def run_quick_tests(self) -> Dict[str, Any]:
        """Run quick endpoint tests"""
        endpoints = [
            ("GET", "/health", None),
            ("GET", "/status", None),
            ("POST", "/optimize/memory", {"dry_run": "true"}),
            ("GET", "/analyze/storage", {"path": "/tmp"}),
            ("POST", "/optimize/storage/cache", {})
        ]
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "tests": [],
            "passed": 0,
            "failed": 0
        }
        
        for method, endpoint, params in endpoints:
            try:
                url = f"{BASE_URL}{endpoint}"
                if method == "GET":
                    response = requests.get(url, params=params if params else None, timeout=10)
                else:
                    response = requests.post(url, json=params if params else None, timeout=10)
                    
                success = response.status_code == 200
                results["tests"].append({
                    "endpoint": endpoint,
                    "method": method,
                    "status": "pass" if success else "fail",
                    "response_time": response.elapsed.total_seconds(),
                    "status_code": response.status_code
                })
                
                if success:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                    
            except Exception as e:
                results["tests"].append({
                    "endpoint": endpoint,
                    "method": method,
                    "status": "error",
                    "error": str(e)
                })
                results["failed"] += 1
                
        results["pass_rate"] = results["passed"] / len(endpoints) if endpoints else 0
        results["status"] = "healthy" if results["pass_rate"] >= ALERT_THRESHOLD else "degraded"
        
        return results
        
    def run_full_integration_tests(self) -> Dict[str, Any]:
        """Run complete integration test suite"""
        logger.info("Running full integration tests...")
        
        try:
            # Run the integration test suite
            result = subprocess.run(
                [sys.executable, "/opt/sutazaiapp/agents/hardware-resource-optimizer/integration_test_suite.py"],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            # Parse results from output
            output_lines = result.stdout.split('\n')
            summary = {}
            
            for line in output_lines:
                if "Pass Rate:" in line:
                    summary["pass_rate"] = line.split(":")[1].strip()
                elif "Overall Status:" in line:
                    summary["status"] = line.split(":")[1].strip()
                elif "Duration:" in line:
                    summary["duration"] = line.split(":")[1].strip()
                    
            return {
                "timestamp": datetime.now().isoformat(),
                "type": "full_integration",
                "success": result.returncode == 0,
                "summary": summary,
                "stdout": result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout,
                "stderr": result.stderr
            }
            
        except subprocess.TimeoutExpired:
            return {
                "timestamp": datetime.now().isoformat(),
                "type": "full_integration",
                "success": False,
                "error": "Test suite timeout (>5 minutes)"
            }
        except Exception as e:
            return {
                "timestamp": datetime.now().isoformat(),
                "type": "full_integration",
                "success": False,
                "error": str(e)
            }
            
    def check_performance_metrics(self) -> Dict[str, Any]:
        """Check performance metrics and trends"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "response_times": {},
            "optimization_effectiveness": {}
        }
        
        # Test response times
        endpoints = ["/health", "/status", "/analyze/storage"]
        for endpoint in endpoints:
            try:
                start = time.time()
                response = requests.get(f"{BASE_URL}{endpoint}", timeout=5)
                elapsed = time.time() - start
                metrics["response_times"][endpoint] = {
                    "time": elapsed,
                    "status": "ok" if elapsed < 1.0 else "slow"
                }
            except Exception as e:
                logger.error(f"Unexpected exception: {e}", exc_info=True)
                metrics["response_times"][endpoint] = {"status": "error"}
                
        # Test optimization effectiveness
        try:
            # Memory optimization test
            response = requests.post(f"{BASE_URL}/optimize/memory", timeout=10)
            if response.status_code == 200:
                data = response.json()
                metrics["optimization_effectiveness"]["memory"] = {
                    "freed_mb": data.get("memory_freed_mb", 0),
                    "effective": data.get("memory_freed_mb", 0) > 0
                }
        except Exception as e:
            # Suppressed exception (was bare except)
            logger.debug(f"Suppressed exception: {e}")
            pass
            
        return metrics
        
    def detect_anomalies(self, current_results: Dict[str, Any]) -> List[str]:
        """Detect anomalies in test results"""
        anomalies = []
        
        # Check for sudden drop in pass rate
        if self.history:
            recent_pass_rates = [r.get("pass_rate", 1.0) for r in self.history[-5:]]
            avg_pass_rate = sum(recent_pass_rates) / len(recent_pass_rates)
            current_pass_rate = current_results.get("pass_rate", 1.0)
            
            if current_pass_rate < avg_pass_rate - 0.2:
                anomalies.append(f"Pass rate dropped significantly: {current_pass_rate:.1%} vs avg {avg_pass_rate:.1%}")
                
        # Check for slow response times
        if "response_times" in current_results:
            slow_endpoints = [ep for ep, data in current_results["response_times"].items() 
                            if data.get("status") == "slow"]
            if slow_endpoints:
                anomalies.append(f"Slow endpoints detected: {', '.join(slow_endpoints)}")
                
        # Check for failed health checks
        if current_results.get("status") == "unhealthy":
            anomalies.append("Agent health check failed")
            
        return anomalies
        
    def save_results(self, results: Dict[str, Any], test_type: str):
        """Save test results to file"""
        filename = f"{RESULTS_DIR}/{test_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
            
        # Keep only last 100 result files
        result_files = sorted(Path(RESULTS_DIR).glob(f"{test_type}_*.json"))
        if len(result_files) > 100:
            for old_file in result_files[:-100]:
                old_file.unlink()
                
    def send_alert(self, message: str, severity: str = "warning"):
        """Send alert (placeholder - implement email/Slack/etc)"""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "severity": severity,
            "message": message
        }
        
        self.alerts.append(alert)
        logger.warning(f"ALERT [{severity.upper()}]: {message}")
        
        # For now, just log to file
        alert_file = f"{RESULTS_DIR}/alerts.log"
        with open(alert_file, 'a') as f:
            f.write(f"{json.dumps(alert)}\n")
            
    def hourly_validation(self):
        """Run hourly validation tasks"""
        logger.info("Starting hourly validation...")
        
        # Quick health check
        health = self.check_agent_health()
        if health["status"] != "healthy":
            self.send_alert("Agent health check failed", "critical")
            
        # Run quick tests
        quick_results = self.run_quick_tests()
        self.save_results(quick_results, "quick_test")
        
        # Check for anomalies
        anomalies = self.detect_anomalies(quick_results)
        for anomaly in anomalies:
            self.send_alert(anomaly, "warning")
            
        # Update history
        self.history.append(quick_results)
        if len(self.history) > 168:  # Keep 1 week of hourly data
            self.history = self.history[-168:]
            
        logger.info(f"Hourly validation complete. Pass rate: {quick_results['pass_rate']:.1%}")
        
    def daily_validation(self):
        """Run daily comprehensive validation"""
        logger.info("Starting daily comprehensive validation...")
        
        # Run full integration tests
        integration_results = self.run_full_integration_tests()
        self.save_results(integration_results, "integration_test")
        
        # Check performance metrics
        perf_metrics = self.check_performance_metrics()
        self.save_results(perf_metrics, "performance")
        
        # Generate daily summary
        summary = self.generate_daily_summary()
        self.save_results(summary, "daily_summary")
        
        # Alert on any failures
        if not integration_results.get("success", False):
            self.send_alert("Daily integration tests failed", "critical")
            
        logger.info("Daily validation complete")
        
    def generate_daily_summary(self) -> Dict[str, Any]:
        """Generate daily summary report"""
        # Load recent test results
        recent_tests = []
        for result_file in sorted(Path(RESULTS_DIR).glob("quick_test_*.json"))[-24:]:
            with open(result_file) as f:
                recent_tests.append(json.load(f))
                
        if not recent_tests:
            return {"error": "No test data available"}
            
        # Calculate statistics
        pass_rates = [t.get("pass_rate", 0) for t in recent_tests]
        avg_pass_rate = sum(pass_rates) / len(pass_rates)
        min_pass_rate = min(pass_rates)
        max_pass_rate = max(pass_rates)
        
        # Count endpoint failures
        endpoint_failures = {}
        for test in recent_tests:
            for t in test.get("tests", []):
                if t["status"] != "pass":
                    endpoint = f"{t['method']} {t['endpoint']}"
                    endpoint_failures[endpoint] = endpoint_failures.get(endpoint, 0) + 1
                    
        return {
            "date": datetime.now().date().isoformat(),
            "tests_run": len(recent_tests),
            "average_pass_rate": avg_pass_rate,
            "min_pass_rate": min_pass_rate,
            "max_pass_rate": max_pass_rate,
            "alerts_today": len([a for a in self.alerts if a["timestamp"].startswith(datetime.now().date().isoformat())]),
            "problematic_endpoints": sorted(endpoint_failures.items(), key=lambda x: x[1], reverse=True)[:5]
        }
        
    def start(self):
        """Start continuous validation"""
        logger.info("Starting Continuous Validation System")
        
        # Schedule tasks
        schedule.every().hour.do(self.hourly_validation)
        schedule.every().day.at("02:00").do(self.daily_validation)
        
        # Run initial validation
        self.hourly_validation()
        
        # Main loop
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except KeyboardInterrupt:
                logger.info("Continuous validation stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(60)
                

def main():
    """Run continuous validator"""
    validator = ContinuousValidator()
    
    # Check if running as daemon or one-time
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        # Run validation once
        print("Running validation once...")
        validator.hourly_validation()
        
        # Also run daily if requested
        if len(sys.argv) > 2 and sys.argv[2] == "--daily":
            validator.daily_validation()
    else:
        # Run continuously
        validator.start()
        

if __name__ == "__main__":
