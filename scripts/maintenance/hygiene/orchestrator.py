#!/usr/bin/env python3
"""
Main hygiene orchestrator - reduced from 1,278 to manageable size
Coordinates detection, fixing, and monitoring
"""

import os
import sys
import json
import time
import asyncio
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set, Any
from concurrent.futures import ThreadPoolExecutor
import psutil
import docker

from .core import ViolationPattern, HygieneMetrics, SystemHealth, HygieneConfig
from .detectors import DetectorRegistry
from .fixers import FixerRegistry

logger = logging.getLogger(__name__)

class HygieneOrchestrator:
    """
    Main orchestrator for codebase hygiene operations
    Reduced from 1,278 lines to focused responsibility
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = HygieneConfig(config_path)
        self.detectors = DetectorRegistry(self.config.config)
        self.fixers = FixerRegistry(self.config.config)
        self.metrics = HygieneMetrics()
        self.system_health = SystemHealth()
        self._running = False
        self._stop_event = threading.Event()
        
    def scan_directory(self, directory: str) -> List[ViolationPattern]:
        """Scan a directory for hygiene violations"""
        violations = []
        scan_start = time.time()
        
        logger.info(f"Starting hygiene scan of {directory}")
        
        try:
            for root, dirs, files in os.walk(directory):
                # Skip excluded directories
                dirs[:] = [d for d in dirs if not self._should_exclude(d)]
                
                for file in files:
                    if self._should_scan_file(file):
                        file_path = os.path.join(root, file)
                        file_violations = self.detectors.detect_all(file_path)
                        violations.extend(file_violations)
                        self.metrics.total_files_scanned += 1
                        
        except Exception as e:
            logger.error(f"Error scanning directory {directory}: {e}")
            
        self.metrics.scan_duration = time.time() - scan_start
        self.metrics.violations_found = len(violations)
        
        logger.info(f"Scan complete: {len(violations)} violations found in {self.metrics.scan_duration:.2f}s")
        return violations
    
    def fix_violations(self, violations: List[ViolationPattern]) -> Dict[str, int]:
        """Fix auto-fixable violations"""
        if not self.config.get('auto_fix_enabled', False):
            logger.info("Auto-fix disabled, skipping violation fixes")
            return {"fixed": 0, "failed": 0, "skipped": len(violations)}
            
        fix_start = time.time()
        results = self.fixers.fix_all(violations)
        self.metrics.fix_duration = time.time() - fix_start
        self.metrics.violations_fixed = results["fixed"]
        
        logger.info(f"Fix complete: {results['fixed']} fixed, {results['failed']} failed, {results['skipped']} skipped")
        return results
    
    def generate_report(self, violations: List[ViolationPattern]) -> Dict[str, Any]:
        """Generate comprehensive hygiene report"""
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        type_counts = {}
        auto_fixable_count = 0
        
        for violation in violations:
            severity_counts[violation.severity] += 1
            type_counts[violation.pattern_type] = type_counts.get(violation.pattern_type, 0) + 1
            if violation.auto_fixable:
                auto_fixable_count += 1
                
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_violations": len(violations),
                "auto_fixable": auto_fixable_count,
                "files_scanned": self.metrics.total_files_scanned,
                "scan_duration": self.metrics.scan_duration,
                "fix_duration": self.metrics.fix_duration
            },
            "severity_breakdown": severity_counts,
            "type_breakdown": type_counts,
            "violations": [
                {
                    "type": v.pattern_type,
                    "severity": v.severity,
                    "file": v.file_path,
                    "line": v.line_number,
                    "description": v.description,
                    "suggested_fix": v.suggested_fix,
                    "auto_fixable": v.auto_fixable
                }
                for v in violations
            ]
        }
        
        return report
    
    def run_continuous_monitoring(self, interval_minutes: int = 60):
        """Run continuous hygiene monitoring"""
        self._running = True
        logger.info(f"Starting continuous monitoring (interval: {interval_minutes} minutes)")
        
        while self._running and not self._stop_event.is_set():
            try:
                self._monitor_cycle()
                self._stop_event.wait(interval_minutes * 60)
            except Exception as e:
                logger.error(f"Error in monitoring cycle: {e}")
                self._stop_event.wait(60)  # Wait 1 minute before retry
                
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self._running = False
        self._stop_event.set()
        logger.info("Stopping hygiene monitoring")
        
    def _monitor_cycle(self):
        """Single monitoring cycle"""
        logger.info("Starting hygiene monitoring cycle")
        
        # Update system health
        self._update_system_health()
        
        # Scan configured directories
        all_violations = []
        for directory in self.config.get('scan_directories', []):
            if os.path.exists(directory):
                violations = self.scan_directory(directory)
                all_violations.extend(violations)
                
        # Fix violations if enabled
        if all_violations:
            self.fix_violations(all_violations)
            
        # Generate and save report
        report = self.generate_report(all_violations)
        self._save_report(report)
        
        # Check alert thresholds
        self._check_alert_thresholds(all_violations)
        
    def _update_system_health(self):
        """Update system health metrics"""
        try:
            process = psutil.Process()
            self.metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
            self.metrics.cpu_usage_percent = process.cpu_percent()
            
            # Check Docker health
            try:
                client = docker.from_env()
                containers = client.containers.list()
                self.system_health.components['docker'] = True
                self.system_health.components['containers_running'] = len(containers) > 0
            except:
                self.system_health.components['docker'] = False
                
            self.system_health.status = "healthy"
            self.system_health.timestamp = datetime.now()
            
        except Exception as e:
            logger.warning(f"Failed to update system health: {e}")
            self.system_health.status = "degraded"
            
    def _save_report(self, report: Dict[str, Any]):
        """Save hygiene report to file"""
        try:
            report_dir = Path("reports/hygiene")
            report_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = report_dir / f"hygiene_report_{timestamp}.json"
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
                
            logger.info(f"Hygiene report saved to {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
            
    def _check_alert_thresholds(self, violations: List[ViolationPattern]):
        """Check if violations exceed alert thresholds"""
        thresholds = self.config.get('violation_thresholds', {})
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        
        for violation in violations:
            severity_counts[violation.severity] += 1
            
        alerts = []
        for severity, count in severity_counts.items():
            threshold = thresholds.get(severity, float('inf'))
            if count > threshold:
                alerts.append(f"{severity.upper()}: {count} violations (threshold: {threshold})")
                
        if alerts:
            self.system_health.alerts = alerts
            logger.warning(f"Hygiene alert thresholds exceeded: {', '.join(alerts)}")
            
    def _should_exclude(self, directory: str) -> bool:
        """Check if directory should be excluded from scanning"""
        exclude_patterns = self.config.get('exclude_patterns', [])
        return any(pattern in directory for pattern in exclude_patterns)
        
    def _should_scan_file(self, filename: str) -> bool:
        """Check if file should be scanned"""
        scan_extensions = ['.py', '.js', '.ts', '.jsx', '.tsx', '.go', '.java', '.cpp', '.c']
        return any(filename.endswith(ext) for ext in scan_extensions)


def main():
    """CLI entry point for hygiene orchestrator"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Codebase Hygiene Orchestrator")
    parser.add_argument("--scan", type=str, help="Directory to scan")
    parser.add_argument("--fix", action="store_true", help="Enable auto-fixing")
    parser.add_argument("--monitor", action="store_true", help="Start continuous monitoring")
    parser.add_argument("--config", type=str, help="Configuration file path")
    
    args = parser.parse_args()
    
    orchestrator = HygieneOrchestrator(args.config)
    
    if args.fix:
        orchestrator.config.update({"auto_fix_enabled": True})
        
    if args.scan:
        violations = orchestrator.scan_directory(args.scan)
        if violations:
            orchestrator.fix_violations(violations)
        report = orchestrator.generate_report(violations)
        print(json.dumps(report, indent=2))
        
    elif args.monitor:
        try:
            orchestrator.run_continuous_monitoring()
        except KeyboardInterrupt:
            orchestrator.stop_monitoring()


if __name__ == "__main__":
    main()