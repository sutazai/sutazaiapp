#!/usr/bin/env python3
"""
🔧 CONTINUOUS RULE MONITORING SYSTEM
Real-time monitoring and enforcement of all 20 Fundamental Rules
Provides continuous compliance tracking and automatic remediation
"""

import os
import sys
import time
import json
import threading
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from comprehensive_rule_enforcer import ComprehensiveRuleEnforcer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s UTC - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class RuleMonitor(FileSystemEventHandler):
    """Continuous monitoring of rule compliance"""
    
    def __init__(self, root_path: str):
        self.root = Path(root_path)
        self.enforcer = ComprehensiveRuleEnforcer(root_path, auto_fix=False)
        self.last_check = time.time()
        self.check_interval = 60  # Minimum seconds between checks
        self.violation_history = []
        self.compliance_metrics = {
            "checks_performed": 0,
            "violations_found": 0,
            "auto_fixes_applied": 0,
            "critical_violations": 0,
            "compliance_trend": []
        }
        
    def on_modified(self, event):
        """Handle file modification events"""
        if event.is_directory:
            return
            
        # Skip certain files
        skip_patterns = ['.git', '__pycache__', '.pyc', '.log']
        if any(pattern in event.src_path for pattern in skip_patterns):
            return
        
        # Throttle checks
        if time.time() - self.last_check < self.check_interval:
            return
        
        logger.info(f"File modified: {event.src_path}")
        self.run_validation_check()
    
    def on_created(self, event):
        """Handle file creation events"""
        if not event.is_directory:
            logger.info(f"New file created: {event.src_path}")
            self.run_validation_check()
    
    def run_validation_check(self):
        """Run comprehensive validation check"""
        self.last_check = time.time()
        self.compliance_metrics["checks_performed"] += 1
        
        logger.info("Running compliance validation...")
        report = self.enforcer.validate_all_rules()
        
        # Update metrics
        self.compliance_metrics["violations_found"] += report["total_violations"]
        self.compliance_metrics["critical_violations"] += report["violations_by_severity"].get("CRITICAL", 0)
        self.compliance_metrics["compliance_trend"].append({
            "timestamp": report["timestamp"],
            "score": report["compliance_score"],
            "violations": report["total_violations"]
        })
        
        # Keep only last 100 trend points
        if len(self.compliance_metrics["compliance_trend"]) > 100:
            self.compliance_metrics["compliance_trend"] = self.compliance_metrics["compliance_trend"][-100:]
        
        # Handle critical violations
        if report["violations_by_severity"].get("CRITICAL", 0) > 0:
            self.handle_critical_violations(report)
        
        # Log summary
        logger.info(f"Compliance Score: {report['compliance_score']}%")
        logger.info(f"Total Violations: {report['total_violations']}")
        
        # Save report
        self.save_monitoring_report(report)
    
    def handle_critical_violations(self, report: Dict[str, Any]):
        """Handle critical violations with alerts and remediation"""
        critical_violations = [
            v for v in report["violations"] 
            if v["severity"] == "CRITICAL"
        ]
        
        logger.critical(f"⚠️  CRITICAL VIOLATIONS DETECTED: {len(critical_violations)}")
        
        for violation in critical_violations[:5]:  # Log first 5
            logger.critical(f"  Rule {violation['rule']}: {violation['description']}")
            logger.critical(f"  File: {violation['file']}:{violation['line']}")
            
        # Send alert (could integrate with Slack, email, etc.)
        self.send_alert(critical_violations)
        
        # Attempt auto-remediation for fixable violations
        fixable = [v for v in critical_violations if v.get("auto_fixable")]
        if fixable:
            logger.info(f"Attempting auto-remediation for {len(fixable)} violations...")
            self.apply_auto_fixes(fixable)
    
    def apply_auto_fixes(self, violations: List[Dict[str, Any]]):
        """Apply automatic fixes for violations"""
        fixed = 0
        for violation in violations:
            if violation.get("fix_command"):
                try:
                    result = subprocess.run(
                        violation["fix_command"],
                        shell=True,
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        fixed += 1
                        logger.info(f"Auto-fixed: {violation['description']}")
                        self.compliance_metrics["auto_fixes_applied"] += 1
                except Exception as e:
                    logger.error(f"Failed to apply fix: {e}")
        
        if fixed > 0:
            logger.info(f"Successfully applied {fixed} automatic fixes")
    
    def send_alert(self, violations: List[Dict[str, Any]]):
        """Send alert for critical violations"""
        # This could be extended to send emails, Slack messages, etc.
        alert_file = self.root / "logs" / "critical_violations.json"
        alert_file.parent.mkdir(exist_ok=True)
        
        alert_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "critical_violations": len(violations),
            "violations": violations[:10]  # First 10
        }
        
        with open(alert_file, 'w') as f:
            json.dump(alert_data, f, indent=2)
        
        logger.critical(f"Critical violations logged to {alert_file}")
    
    def save_monitoring_report(self, report: Dict[str, Any]):
        """Save monitoring report to file"""
        reports_dir = self.root / "reports" / "enforcement"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_file = reports_dir / f"compliance_report_{timestamp}.json"
        
        # Add monitoring metrics to report
        report["monitoring_metrics"] = self.compliance_metrics
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Also save latest report
        latest_file = reports_dir / "latest_compliance_report.json"
        with open(latest_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Report saved to {report_file}")
    
    def get_compliance_summary(self) -> Dict[str, Any]:
        """Get current compliance summary"""
        return {
            "metrics": self.compliance_metrics,
            "last_check": datetime.fromtimestamp(self.last_check, timezone.utc).isoformat(),
            "trend": self.get_compliance_trend()
        }
    
    def get_compliance_trend(self) -> str:
        """Analyze compliance trend"""
        if len(self.compliance_metrics["compliance_trend"]) < 2:
            return "insufficient_data"
        
        recent = self.compliance_metrics["compliance_trend"][-10:]
        scores = [t["score"] for t in recent]
        
        if all(s >= 90 for s in scores):
            return "excellent"
        elif scores[-1] > scores[0]:
            return "improving"
        elif scores[-1] < scores[0]:
            return "degrading"
        else:
            return "stable"


class ContinuousMonitor:
    """Main continuous monitoring system"""
    
    def __init__(self, root_path: str):
        self.root = Path(root_path)
        self.monitor = RuleMonitor(root_path)
        self.observer = Observer()
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        
    def start(self):
        """Start continuous monitoring"""
        logger.info("🔧 STARTING CONTINUOUS RULE MONITORING")
        logger.info("=" * 60)
        logger.info(f"Monitoring: {self.root}")
        
        # Initial compliance check
        self.monitor.run_validation_check()
        
        # Set up file system monitoring
        self.observer.schedule(self.monitor, str(self.root), recursive=True)
        self.observer.start()
        
        # Start periodic checks
        self.monitoring_thread = threading.Thread(target=self.periodic_check)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info("Monitoring system started successfully")
        
    def periodic_check(self):
        """Run periodic compliance checks"""
        while not self.stop_event.is_set():
            # Wait 5 minutes between periodic checks
            self.stop_event.wait(300)
            
            if not self.stop_event.is_set():
                logger.info("Running periodic compliance check...")
                self.monitor.run_validation_check()
                
                # Log summary
                summary = self.monitor.get_compliance_summary()
                logger.info(f"Compliance Trend: {summary['trend']}")
                logger.info(f"Total Checks: {summary['metrics']['checks_performed']}")
                logger.info(f"Auto-fixes Applied: {summary['metrics']['auto_fixes_applied']}")
    
    def stop(self):
        """Stop monitoring"""
        logger.info("Stopping monitoring system...")
        self.stop_event.set()
        self.observer.stop()
        self.observer.join()
        if self.monitoring_thread:
            self.monitoring_thread.join()
        logger.info("Monitoring system stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get monitoring status"""
        return {
            "running": self.observer.is_alive(),
            "summary": self.monitor.get_compliance_summary()
        }


def main():
    """Main monitoring entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Continuous Rule Monitoring System"
    )
    parser.add_argument("--root", default="/opt/sutazaiapp", help="Root directory to monitor")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    
    args = parser.parse_args()
    
    # Create monitor
    monitor = ContinuousMonitor(args.root)
    
    try:
        # Start monitoring
        monitor.start()
        
        if args.daemon:
            # Run as daemon
            while True:
                time.sleep(60)
                status = monitor.get_status()
                if not status["running"]:
                    logger.error("Monitor stopped unexpectedly, restarting...")
                    monitor.start()
        else:
            # Interactive mode
            logger.info("\nMonitoring active. Press Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(60)
                    # Print periodic status
                    status = monitor.get_status()
                    summary = status["summary"]
                    logger.info(f"\r[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] "
                          f"Checks: {summary['metrics']['checks_performed']} | "
                          f"Trend: {summary['trend']} | "
                          f"Critical: {summary['metrics']['critical_violations']}", end="")
            except KeyboardInterrupt:
                logger.info("\n\nStopping monitor...")
    except Exception as e:
        logger.error(f"Monitor error: {e}")
    finally:
        monitor.stop()


if __name__ == "__main__":
    main()