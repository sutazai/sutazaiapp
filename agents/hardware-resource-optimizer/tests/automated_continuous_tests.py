#!/usr/bin/env python3
"""
Automated Continuous Testing Suite for Hardware Resource Optimizer
=================================================================

This script provides automated continuous testing capabilities:
- Automated test execution on schedule
- Real-time monitoring and alerting
- Regression detection
- Performance trend analysis
- CI/CD integration support
- Automated rollback triggers

Author: QA Team Lead  
Version: 1.0.0
"""

import os
import sys
import json
import time
import asyncio
import schedule
import threading
import subprocess
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests

# Add the tests directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from comprehensive_e2e_test_framework import E2ETestFramework

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('automated_tests.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('AutomatedTests')

class ContinuousTestingOrchestrator:
    """Orchestrates continuous automated testing"""
    
    def __init__(self, config_file: str = "test_config.json"):
        self.config = self.load_config(config_file)
        self.test_history = []
        self.performance_baselines = {}
        self.alert_thresholds = self.config.get("alert_thresholds", {})
        self.is_running = False
        
        # Initialize test framework
        self.framework = E2ETestFramework(
            base_url=self.config.get("agent_url", "http://localhost:8116")
        )
        
        logger.info("Continuous Testing Orchestrator initialized")
    
    def load_config(self, config_file: str) -> Dict[str, Any]:
        """Load testing configuration"""
        default_config = {
            "agent_url": "http://localhost:8116",
            "test_schedules": {
                "health_check": "*/5 * * * *",  # Every 5 minutes
                "quick_smoke": "*/15 * * * *",  # Every 15 minutes
                "comprehensive": "0 */2 * * *",  # Every 2 hours
                "performance": "0 8,20 * * *"   # 8 AM and 8 PM
            },
            "alert_thresholds": {
                "success_rate_min": 95.0,
                "response_time_max": 5.0,
                "failure_count_max": 3
            },
            "retention_hours": 168,  # 1 week
            "notifications": {
                "enabled": False,
                "email": {
                    "smtp_server": "localhost",
                    "smtp_port": 587,
                    "username": "",
                    "password": "",
                    "recipients": []
                },
                "webhook": {
                    "url": "",
                    "headers": {}
                }
            }
        }
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Could not load config file {config_file}: {e}")
        
        return default_config
    
    def save_test_result(self, test_type: str, result: Dict[str, Any]):
        """Save test result with timestamp"""
        timestamped_result = {
            "timestamp": datetime.now().isoformat(),
            "test_type": test_type,
            "result": result
        }
        
        self.test_history.append(timestamped_result)
        
        # Clean up old results
        cutoff_time = datetime.now() - timedelta(hours=self.config["retention_hours"])
        self.test_history = [
            r for r in self.test_history 
            if datetime.fromisoformat(r["timestamp"]) > cutoff_time
        ]
        
        # Save to file
        self.save_history_to_file()
    
    def save_history_to_file(self):
        """Save test history to file"""
        history_file = f"test_history_{datetime.now().strftime('%Y%m')}.json"
        try:
            with open(history_file, 'w') as f:
                json.dump(self.test_history, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save test history: {e}")
    
    def run_health_check(self):
        """Quick health check test"""
        logger.info("Running automated health check...")
        
        try:
            result = self.framework.test_endpoint("GET", "/health")
            
            test_result = {
                "success": result["success"],
                "response_time": result["duration"],
                "status_code": result.get("status_code"),
                "error": result.get("error")
            }
            
            self.save_test_result("health_check", test_result)
            
            # Check for alerts
            self.check_health_alerts(test_result)
            
            return test_result
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            error_result = {"success": False, "error": str(e)}
            self.save_test_result("health_check", error_result)
            return error_result
    
    def run_smoke_tests(self):
        """Quick smoke tests for critical functionality"""
        logger.info("Running automated smoke tests...")
        
        critical_tests = [
            ("GET", "/health"),
            ("GET", "/status"),
            ("POST", "/optimize/memory"),
            ("GET", "/analyze/storage", {"path": "/tmp"})
        ]
        
        results = []
        for method, endpoint, *params in critical_tests:
            params_dict = params[0] if params else None
            result = self.framework.test_endpoint(method, endpoint, params_dict)
            results.append(result)
        
        # Analyze results
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r["success"])
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        avg_response_time = sum(r["duration"] for r in results if r["success"]) / successful_tests if successful_tests > 0 else 0
        
        smoke_result = {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": success_rate,
            "average_response_time": avg_response_time,
            "details": results
        }
        
        self.save_test_result("smoke_tests", smoke_result)
        
        # Check for alerts
        self.check_smoke_alerts(smoke_result)
        
        return smoke_result
    
    def run_comprehensive_tests(self):
        """Run full comprehensive test suite"""
        logger.info("Running comprehensive automated tests...")
        
        try:
            # Create new framework instance to avoid state issues
            framework = E2ETestFramework(self.config.get("agent_url", "http://localhost:8116"))
            result = framework.run_comprehensive_test_suite()
            
            self.save_test_result("comprehensive", result)
            
            # Update performance baselines
            self.update_performance_baselines(result)
            
            # Check for alerts
            self.check_comprehensive_alerts(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Comprehensive tests failed: {e}")
            error_result = {"success": False, "error": str(e)}
            self.save_test_result("comprehensive", error_result)
            return error_result
    
    def run_performance_tests(self):
        """Run performance-focused tests"""
        logger.info("Running automated performance tests...")
        
        try:
            framework = E2ETestFramework(self.config.get("agent_url", "http://localhost:8116"))
            
            # Run performance tests multiple times for better accuracy
            performance_results = []
            for i in range(3):
                logger.info(f"Performance test run {i+1}/3...")
                perf_result = framework.run_performance_tests()
                performance_results.append(perf_result)
                time.sleep(10)  # Brief pause between runs
            
            # Aggregate results
            aggregated_results = self.aggregate_performance_results(performance_results)
            
            self.save_test_result("performance", aggregated_results)
            
            # Check for performance regressions
            self.check_performance_regressions(aggregated_results)
            
            framework.cleanup_test_environment()
            return aggregated_results
            
        except Exception as e:
            logger.error(f"Performance tests failed: {e}")
            error_result = {"success": False, "error": str(e)}
            self.save_test_result("performance", error_result)
            return error_result
    
    def aggregate_performance_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate multiple performance test runs"""
        if not results:
            return {}
        
        aggregated = {}
        
        # Get all endpoint keys from first result
        if results[0]:
            for endpoint in results[0].keys():
                times = []
                for result in results:
                    if endpoint in result and 'avg_response_time' in result[endpoint]:
                        times.append(result[endpoint]['avg_response_time'])
                
                if times:
                    aggregated[endpoint] = {
                        'avg_response_time': sum(times) / len(times),
                        'min_response_time': min(times),  
                        'max_response_time': max(times),
                        'runs': len(times)
                    }
        
        return aggregated
    
    def update_performance_baselines(self, result: Dict[str, Any]):
        """Update performance baselines based on recent results"""
        if 'performance_metrics' in result:
            current_time = datetime.now()
            
            for endpoint, metrics in result['performance_metrics'].items():
                if endpoint not in self.performance_baselines:
                    self.performance_baselines[endpoint] = {
                        'baseline_response_time': metrics.get('avg_response_time', 0),
                        'last_updated': current_time.isoformat(),
                        'samples': 1
                    }
                else:
                    # Update baseline with exponential moving average
                    current_baseline = self.performance_baselines[endpoint]['baseline_response_time']
                    new_time = metrics.get('avg_response_time', current_baseline)
                    alpha = 0.1  # Smoothing factor
                    
                    self.performance_baselines[endpoint]['baseline_response_time'] = (
                        alpha * new_time + (1 - alpha) * current_baseline
                    )
                    self.performance_baselines[endpoint]['last_updated'] = current_time.isoformat()
                    self.performance_baselines[endpoint]['samples'] += 1
    
    def check_health_alerts(self, result: Dict[str, Any]):
        """Check health test results for alert conditions"""
        if not result["success"]:
            self.send_alert(
                "CRITICAL",
                "Health Check Failed",
                f"Agent health check failed: {result.get('error', 'Unknown error')}"
            )
        elif result.get("response_time", 0) > self.alert_thresholds.get("response_time_max", 5.0):
            self.send_alert(
                "WARNING", 
                "Slow Health Check Response",
                f"Health check response time: {result['response_time']:.3f}s (threshold: {self.alert_thresholds['response_time_max']}s)"
            )
    
    def check_smoke_alerts(self, result: Dict[str, Any]):
        """Check smoke test results for alert conditions"""
        success_rate = result.get("success_rate", 0)
        min_success_rate = self.alert_thresholds.get("success_rate_min", 95.0)
        
        if success_rate < min_success_rate:
            self.send_alert(
                "CRITICAL",
                "Smoke Tests Failing",
                f"Smoke test success rate: {success_rate:.1f}% (threshold: {min_success_rate}%)"
            )
    
    def check_comprehensive_alerts(self, result: Dict[str, Any]):
        """Check comprehensive test results for alert conditions"""
        if 'test_summary' in result:
            summary = result['test_summary']
            success_rate = summary.get('success_rate', 0)
            min_success_rate = self.alert_thresholds.get('success_rate_min', 95.0)
            
            if success_rate < min_success_rate:
                self.send_alert(
                    "CRITICAL",
                    "Comprehensive Tests Failing",
                    f"Overall success rate: {success_rate:.1f}% (threshold: {min_success_rate}%)\n"
                    f"Failed tests: {summary.get('failed_tests', 0)}/{summary.get('total_tests', 0)}"
                )
    
    def check_performance_regressions(self, result: Dict[str, Any]):
        """Check for performance regressions"""
        for endpoint, metrics in result.items():
            if endpoint in self.performance_baselines:
                baseline = self.performance_baselines[endpoint]['baseline_response_time']
                current = metrics.get('avg_response_time', 0)
                
                # Alert if current performance is 50% worse than baseline
                if current > baseline * 1.5 and baseline > 0:
                    self.send_alert(
                        "WARNING",
                        "Performance Regression Detected",
                        f"Endpoint {endpoint} performance regression:\n"
                        f"Current: {current:.3f}s, Baseline: {baseline:.3f}s "
                        f"({((current/baseline - 1) * 100):.1f}% slower)"
                    )
    
    def send_alert(self, severity: str, subject: str, message: str):
        """Send alert notification"""
        logger.warning(f"ALERT [{severity}] {subject}: {message}")
        
        if not self.config.get("notifications", {}).get("enabled", False):
            return
        
        # Email notification
        email_config = self.config.get("notifications", {}).get("email", {})
        if email_config.get("recipients"):
            self.send_email_alert(severity, subject, message, email_config)
        
        # Webhook notification
        webhook_config = self.config.get("notifications", {}).get("webhook", {})
        if webhook_config.get("url"):
            self.send_webhook_alert(severity, subject, message, webhook_config)
    
    def send_email_alert(self, severity: str, subject: str, message: str, email_config: Dict[str, Any]):
        """Send email alert"""
        try:
            msg = MIMEMultipart()
            msg['From'] = email_config.get('username', 'noreply@testing.local')
            msg['To'] = ', '.join(email_config['recipients'])
            msg['Subject'] = f"[{severity}] Hardware Optimizer Test Alert: {subject}"
            
            body = f"""
Hardware Resource Optimizer Test Alert

Severity: {severity}
Subject: {subject}
Timestamp: {datetime.now().isoformat()}

Details:
{message}

This is an automated alert from the continuous testing system.
"""
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            if email_config.get('username') and email_config.get('password'):
                server.starttls()
                server.login(email_config['username'], email_config['password'])
            
            server.sendmail(msg['From'], email_config['recipients'], msg.as_string())
            server.quit()
            
            logger.info(f"Email alert sent for: {subject}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def send_webhook_alert(self, severity: str, subject: str, message: str, webhook_config: Dict[str, Any]):
        """Send webhook alert"""
        try:
            payload = {
                "severity": severity,
                "subject": subject,
                "message": message,
                "timestamp": datetime.now().isoformat(),
                "source": "hardware-optimizer-tests"
            }
            
            headers = webhook_config.get('headers', {})
            headers['Content-Type'] = 'application/json'
            
            response = requests.post(
                webhook_config['url'],
                json=payload,
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Webhook alert sent for: {subject}")
            else:
                logger.warning(f"Webhook alert failed with status {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
    
    def setup_scheduled_tests(self):
        """Setup scheduled test execution"""
        schedules = self.config.get("test_schedules", {})
        
        if "health_check" in schedules:
            schedule.every(5).minutes.do(self.run_health_check)
            logger.info("Scheduled health checks every 5 minutes")
        
        if "quick_smoke" in schedules:
            schedule.every(15).minutes.do(self.run_smoke_tests)
            logger.info("Scheduled smoke tests every 15 minutes")
        
        if "comprehensive" in schedules:
            schedule.every(2).hours.do(self.run_comprehensive_tests)
            logger.info("Scheduled comprehensive tests every 2 hours")
        
        if "performance" in schedules:
            schedule.every().day.at("08:00").do(self.run_performance_tests)
            schedule.every().day.at("20:00").do(self.run_performance_tests)
            logger.info("Scheduled performance tests at 8 AM and 8 PM")
    
    def generate_status_dashboard(self) -> Dict[str, Any]:
        """Generate current status dashboard"""
        if not self.test_history:
            return {"status": "no_data", "message": "No test history available"}
        
        # Get recent test results (last 4 hours)
        recent_cutoff = datetime.now() - timedelta(hours=4)
        recent_tests = [
            t for t in self.test_history
            if datetime.fromisoformat(t["timestamp"]) > recent_cutoff
        ]
        
        # Analyze recent health
        health_checks = [t for t in recent_tests if t["test_type"] == "health_check"]
        health_success_rate = (
            sum(1 for h in health_checks if h["result"]["success"]) / len(health_checks) * 100
            if health_checks else 0
        )
        
        # Get latest comprehensive test
        comprehensive_tests = [t for t in self.test_history if t["test_type"] == "comprehensive"]
        latest_comprehensive = comprehensive_tests[-1] if comprehensive_tests else None
        
        dashboard = {
            "overall_status": "healthy" if health_success_rate > 90 else "degraded" if health_success_rate > 50 else "unhealthy",
            "health_success_rate_4h": health_success_rate,
            "total_tests_4h": len(recent_tests),
            "recent_health_checks": len(health_checks),
            "latest_comprehensive_test": latest_comprehensive["timestamp"] if latest_comprehensive else "never",
            "performance_baselines": self.performance_baselines,
            "last_updated": datetime.now().isoformat()
        }
        
        if latest_comprehensive:
            comp_result = latest_comprehensive["result"]
            if "test_summary" in comp_result:
                dashboard["latest_comprehensive_summary"] = comp_result["test_summary"]
        
        return dashboard
    
    def start_continuous_testing(self):
        """Start the continuous testing service"""
        logger.info("Starting continuous testing service...")
        self.is_running = True
        
        # Setup scheduled tests
        self.setup_scheduled_tests()
        
        # Run initial health check
        self.run_health_check()
        
        # Start scheduler thread
        def run_scheduler():
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        
        logger.info("Continuous testing service started")
        return scheduler_thread
    
    def stop_continuous_testing(self):
        """Stop the continuous testing service"""
        logger.info("Stopping continuous testing service...")
        self.is_running = False
        schedule.clear()
        logger.info("Continuous testing service stopped")

def main():
    """Main entry point for automated testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hardware Optimizer Continuous Testing")
    parser.add_argument("--config", default="test_config.json", help="Config file path")
    parser.add_argument("--mode", choices=["continuous", "once", "dashboard"], 
                       default="once", help="Testing mode")
    parser.add_argument("--test-type", choices=["health", "smoke", "comprehensive", "performance"],
                       default="comprehensive", help="Test type for 'once' mode")
    
    args = parser.parse_args()
    
    orchestrator = ContinuousTestingOrchestrator(args.config)
    
    if args.mode == "continuous":
        try:
            scheduler_thread = orchestrator.start_continuous_testing()
            
            # Keep running until interrupted
            while orchestrator.is_running:
                time.sleep(10)
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            orchestrator.stop_continuous_testing()
    
    elif args.mode == "once":
        if args.test_type == "health":
            result = orchestrator.run_health_check()
        elif args.test_type == "smoke":
            result = orchestrator.run_smoke_tests()
        elif args.test_type == "comprehensive":
            result = orchestrator.run_comprehensive_tests()
        elif args.test_type == "performance":
            result = orchestrator.run_performance_tests()
        
        print(json.dumps(result, indent=2))
    
    elif args.mode == "dashboard":
        dashboard = orchestrator.generate_status_dashboard()
        print(json.dumps(dashboard, indent=2))

if __name__ == "__main__":
    main()