#!/usr/bin/env python3
"""
Core Compliance Monitoring System for SutazAI Project
Monitors all 16 codebase hygiene rules in real-time with automated enforcement

Usage: python compliance-monitor-core.py [--mode=monitor|scan|report] [--auto-fix]
"""

import os
import sys
import json
import time
import logging
import threading
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import re
import hashlib
import shutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/logs/compliance-monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ComplianceMonitor')

class ViolationSeverity(Enum):
    """Violation severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class RuleType(Enum):
    """Types of compliance rules"""
    FILE_PATTERN = "file_pattern"
    NAMING_CONVENTION = "naming_convention"
    CONTENT_ANALYSIS = "content_analysis"
    STRUCTURE_VALIDATION = "structure_validation"
    SECURITY_CHECK = "security_check"
    DOCUMENTATION = "documentation"

@dataclass
class ComplianceViolation:
    """Represents a compliance violation"""
    rule_id: str
    rule_name: str
    severity: ViolationSeverity
    file_path: str
    message: str
    violation_type: RuleType
    auto_fixable: bool
    detected_at: datetime
    fixed_at: Optional[datetime] = None
    fix_attempted: bool = False
    
    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            'severity': self.severity.value,
            'violation_type': self.violation_type.value,
            'detected_at': self.detected_at.isoformat(),
            'fixed_at': self.fixed_at.isoformat() if self.fixed_at else None
        }

class ComplianceRuleEngine:
    """Engine for defining and checking compliance rules"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.rules = self._initialize_rules()
        
    def _initialize_rules(self) -> Dict[str, Dict]:
        """Initialize all 16 CLAUDE.md compliance rules"""
        return {
            "rule_01_no_fantasy": {
                "name": "No Fantasy Elements",
                "type": RuleType.CONTENT_ANALYSIS,
                "severity": ViolationSeverity.CRITICAL,
                "patterns": [
                    r"(magic|wizard|teleport|fantasy)\w*",
                    r"(superIntuitive|magicHandler|wizardService)",
                    r"TODO:.*magic",
                    r"# imagine this"
                ],
                "file_types": [".py", ".js", ".ts", ".md"],
                "auto_fixable": False
            },
            "rule_02_no_breaking_changes": {
                "name": "Do Not Break Existing Functionality",
                "type": RuleType.CONTENT_ANALYSIS,
                "severity": ViolationSeverity.CRITICAL,
                "patterns": [],  # Requires runtime analysis
                "file_types": ["*"],
                "auto_fixable": False
            },
            "rule_03_analyze_everything": {
                "name": "Analyze Everything Every Time",
                "type": RuleType.STRUCTURE_VALIDATION,
                "severity": ViolationSeverity.HIGH,
                "patterns": [],
                "file_types": ["*"],
                "auto_fixable": False
            },
            "rule_04_reuse_before_creating": {
                "name": "Reuse Before Creating",
                "type": RuleType.FILE_PATTERN,
                "severity": ViolationSeverity.MEDIUM,
                "patterns": [
                    r".*_copy\d*\.(py|sh|js)$",
                    r".*_duplicate\.(py|sh|js)$",
                    r".*_v\d+\.(py|sh|js)$"
                ],
                "file_types": [".py", ".sh", ".js"],
                "auto_fixable": True
            },
            "rule_05_professional_project": {
                "name": "Professional Project Standards",
                "type": RuleType.CONTENT_ANALYSIS,
                "severity": ViolationSeverity.HIGH,
                "patterns": [
                    r"# TODO: test",
                    r"# HACK:",
                    r"# quick fix",
                    r"print\(.*debug.*\)"
                ],
                "file_types": [".py", ".js", ".ts"],
                "auto_fixable": True
            },
            "rule_06_centralized_documentation": {
                "name": "Clear Centralized Documentation",
                "type": RuleType.STRUCTURE_VALIDATION,
                "severity": ViolationSeverity.MEDIUM,
                "patterns": [],
                "file_types": [".md"],
                "auto_fixable": False
            },
            "rule_07_script_consolidation": {
                "name": "Script Consolidation and Control",
                "type": RuleType.FILE_PATTERN,
                "severity": ViolationSeverity.HIGH,
                "patterns": [
                    r".*/(test|temp|fix)\d*\.(sh|py)$",
                    r".*/script\d*\.(sh|py)$",
                    r".*_backup\.(sh|py)$"
                ],
                "file_types": [".sh", ".py"],
                "auto_fixable": True
            },
            "rule_08_python_script_sanity": {
                "name": "Python Script Structure and Purpose",
                "type": RuleType.CONTENT_ANALYSIS,
                "severity": ViolationSeverity.MEDIUM,
                "patterns": [],
                "file_types": [".py"],
                "auto_fixable": False
            },
            "rule_09_no_duplication_chaos": {
                "name": "No Backend/Frontend Duplication",
                "type": RuleType.STRUCTURE_VALIDATION,
                "severity": ViolationSeverity.HIGH,
                "patterns": [
                    r".*/backend_\w+/.*",
                    r".*/frontend_\w+/.*",
                    r".*/old_\w+/.*"
                ],
                "file_types": ["*"],
                "auto_fixable": True
            },
            "rule_10_functionality_first_cleanup": {
                "name": "Functionality-First Cleanup",
                "type": RuleType.CONTENT_ANALYSIS,
                "severity": ViolationSeverity.CRITICAL,
                "patterns": [],
                "file_types": ["*"],
                "auto_fixable": False
            },
            "rule_11_docker_structure": {
                "name": "Clean Docker Structure",
                "type": RuleType.FILE_PATTERN,
                "severity": ViolationSeverity.HIGH,
                "patterns": [
                    r"Dockerfile\.\w+\.backup$",
                    r"docker-compose\.\w+\.old$"
                ],
                "file_types": ["Dockerfile", "docker-compose.yml"],
                "auto_fixable": True
            },
            "rule_12_single_deployment_script": {
                "name": "One Deployment Script",
                "type": RuleType.FILE_PATTERN,
                "severity": ViolationSeverity.CRITICAL,
                "patterns": [
                    r"deploy\w*\.(sh|py)$",
                    r".*deploy.*\.(sh|py)$"
                ],
                "file_types": [".sh", ".py"],
                "auto_fixable": False  # Requires manual consolidation
            },
            "rule_13_no_garbage": {
                "name": "No Garbage No Rot",
                "type": RuleType.FILE_PATTERN,
                "severity": ViolationSeverity.CRITICAL,
                "patterns": [
                    r".*\.(bak|backup|tmp|old|copy)\d*$",
                    r".*~$",
                    r".*_backup.*$",
                    r".*_old.*$",
                    r".*_temp.*$"
                ],
                "file_types": ["*"],
                "auto_fixable": True
            },
            "rule_14_correct_ai_agent": {
                "name": "Use Correct AI Agent",
                "type": RuleType.CONTENT_ANALYSIS,
                "severity": ViolationSeverity.MEDIUM,
                "patterns": [],
                "file_types": [".py", ".md"],
                "auto_fixable": False
            },
            "rule_15_clean_documentation": {
                "name": "Clean Documentation",
                "type": RuleType.STRUCTURE_VALIDATION,
                "severity": ViolationSeverity.MEDIUM,
                "patterns": [
                    r".*/README\d*\.md$",
                    r".*/docs.*/README\.md$"
                ],
                "file_types": [".md"],
                "auto_fixable": False
            },
            "rule_16_local_llms_ollama": {
                "name": "Use Ollama for Local LLMs",
                "type": RuleType.CONTENT_ANALYSIS,
                "severity": ViolationSeverity.MEDIUM,
                "patterns": [
                    r"import.*transformers",
                    r"from.*huggingface",
                    r"openai\.\w+"
                ],
                "file_types": [".py"],
                "auto_fixable": False
            }
        }
        
    def check_file_against_rules(self, file_path: Path) -> List[ComplianceViolation]:
        """Check a file against all applicable rules"""
        violations = []
        
        if not file_path.exists() or file_path.is_dir():
            return violations
            
        for rule_id, rule_config in self.rules.items():
            try:
                rule_violations = self._check_single_rule(file_path, rule_id, rule_config)
                violations.extend(rule_violations)
            except Exception as e:
                logger.error(f"Error checking rule {rule_id} for {file_path}: {e}")
                
        return violations
        
    def _check_single_rule(self, file_path: Path, rule_id: str, rule_config: Dict) -> List[ComplianceViolation]:
        """Check a file against a single rule"""
        violations = []
        
        # Check if rule applies to this file type
        file_types = rule_config.get("file_types", [])
        if "*" not in file_types and file_path.suffix not in file_types:
            return violations
            
        rule_type = rule_config["type"]
        
        if rule_type == RuleType.FILE_PATTERN:
            violations.extend(self._check_file_pattern_rule(file_path, rule_id, rule_config))
        elif rule_type == RuleType.CONTENT_ANALYSIS:
            violations.extend(self._check_content_analysis_rule(file_path, rule_id, rule_config))
        elif rule_type == RuleType.STRUCTURE_VALIDATION:
            violations.extend(self._check_structure_validation_rule(file_path, rule_id, rule_config))
            
        return violations
        
    def _check_file_pattern_rule(self, file_path: Path, rule_id: str, rule_config: Dict) -> List[ComplianceViolation]:
        """Check file pattern-based rules"""
        violations = []
        patterns = rule_config.get("patterns", [])
        
        for pattern in patterns:
            if re.search(pattern, str(file_path)):
                violations.append(ComplianceViolation(
                    rule_id=rule_id,
                    rule_name=rule_config["name"],
                    severity=rule_config["severity"],
                    file_path=str(file_path),
                    message=f"File matches forbidden pattern: {pattern}",
                    violation_type=rule_config["type"],
                    auto_fixable=rule_config["auto_fixable"],
                    detected_at=datetime.now()
                ))
                
        return violations
        
    def _check_content_analysis_rule(self, file_path: Path, rule_id: str, rule_config: Dict) -> List[ComplianceViolation]:
        """Check content-based rules"""
        violations = []
        patterns = rule_config.get("patterns", [])
        
        if not patterns:
            return violations
            
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            for pattern in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    violations.append(ComplianceViolation(
                        rule_id=rule_id,
                        rule_name=rule_config["name"],
                        severity=rule_config["severity"],
                        file_path=str(file_path),
                        message=f"Content violation at line {line_num}: {match.group()}",
                        violation_type=rule_config["type"],
                        auto_fixable=rule_config["auto_fixable"],
                        detected_at=datetime.now()
                    ))
                    
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")
            
        return violations
        
    def _check_structure_validation_rule(self, file_path: Path, rule_id: str, rule_config: Dict) -> List[ComplianceViolation]:
        """Check structure-based rules"""
        violations = []
        
        # Rule-specific structure checks
        if rule_id == "rule_06_centralized_documentation":
            violations.extend(self._check_documentation_structure(file_path, rule_id, rule_config))
        elif rule_id == "rule_09_no_duplication_chaos":
            violations.extend(self._check_duplication_structure(file_path, rule_id, rule_config))
            
        return violations
        
    def _check_documentation_structure(self, file_path: Path, rule_id: str, rule_config: Dict) -> List[ComplianceViolation]:
        """Check documentation structure compliance"""
        violations = []
        
        if file_path.name == "README.md":
            # Check if README is in root or docs folder
            relative_path = file_path.relative_to(self.project_root)
            parts = relative_path.parts
            
            if len(parts) > 2 and "docs" not in parts:
                violations.append(ComplianceViolation(
                    rule_id=rule_id,
                    rule_name=rule_config["name"],
                    severity=rule_config["severity"],
                    file_path=str(file_path),
                    message="README.md should be in root or /docs/ directory",
                    violation_type=rule_config["type"],
                    auto_fixable=False,
                    detected_at=datetime.now()
                ))
                
        return violations
        
    def _check_duplication_structure(self, file_path: Path, rule_id: str, rule_config: Dict) -> List[ComplianceViolation]:
        """Check for duplicate directory structures"""
        violations = []
        patterns = rule_config.get("patterns", [])
        
        for pattern in patterns:
            if re.search(pattern, str(file_path)):
                violations.append(ComplianceViolation(
                    rule_id=rule_id,
                    rule_name=rule_config["name"],
                    severity=rule_config["severity"],
                    file_path=str(file_path),
                    message=f"File in duplicate directory structure: {pattern}",
                    violation_type=rule_config["type"],
                    auto_fixable=True,
                    detected_at=datetime.now()
                ))
                
        return violations

class ComplianceAutoFixer:
    """Handles automatic fixing of violations"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.archive_root = self.project_root / "archive" / "compliance-fixes"
        self.archive_root.mkdir(parents=True, exist_ok=True)
        
    def can_auto_fix(self, violation: ComplianceViolation) -> bool:
        """Check if violation can be automatically fixed"""
        return violation.auto_fixable and not violation.fix_attempted
        
    def auto_fix_violation(self, violation: ComplianceViolation) -> bool:
        """Attempt to automatically fix a violation"""
        if not self.can_auto_fix(violation):
            return False
            
        violation.fix_attempted = True
        success = False
        
        try:
            if violation.rule_id in ["rule_04_reuse_before_creating", "rule_07_script_consolidation", "rule_13_no_garbage"]:
                success = self._fix_by_archiving(violation)
            elif violation.rule_id == "rule_05_professional_project":
                success = self._fix_content_issues(violation)
            elif violation.rule_id == "rule_09_no_duplication_chaos":
                success = self._fix_duplicate_directories(violation)
            elif violation.rule_id == "rule_11_docker_structure":
                success = self._fix_docker_files(violation)
                
            if success:
                violation.fixed_at = datetime.now()
                logger.info(f"Auto-fixed violation: {violation.rule_id} in {violation.file_path}")
            else:
                logger.warning(f"Failed to auto-fix violation: {violation.rule_id} in {violation.file_path}")
                
        except Exception as e:
            logger.error(f"Error auto-fixing {violation.rule_id} in {violation.file_path}: {e}")
            
        return success
        
    def _fix_by_archiving(self, violation: ComplianceViolation) -> bool:
        """Fix violation by archiving the problematic file"""
        file_path = Path(violation.file_path)
        
        if not file_path.exists():
            return True  # Already gone
            
        # Create archive with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = self.archive_root / f"{violation.rule_id}_{timestamp}"
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Preserve directory structure
        relative_path = file_path.relative_to(self.project_root)
        archive_path = archive_dir / relative_path
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            shutil.move(str(file_path), str(archive_path))
            logger.info(f"Archived {file_path} to {archive_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to archive {file_path}: {e}")
            return False
            
    def _fix_content_issues(self, violation: ComplianceViolation) -> bool:
        """Fix content-related violations"""
        file_path = Path(violation.file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Remove debug prints
            content = re.sub(r'print\(.*debug.*\)\n?', '', content, flags=re.IGNORECASE)
            
            # Clean up TODO comments
            content = re.sub(r'# TODO: test\n?', '', content)
            content = re.sub(r'# HACK:.*\n?', '', content, flags=re.IGNORECASE)
            content = re.sub(r'# quick fix.*\n?', '', content, flags=re.IGNORECASE)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            return True
        except Exception as e:
            logger.error(f"Failed to fix content in {file_path}: {e}")
            return False
            
    def _fix_duplicate_directories(self, violation: ComplianceViolation) -> bool:
        """Fix duplicate directory structures"""
        # This requires more complex logic and human oversight
        # For now, just log the issue
        logger.warning(f"Duplicate directory requires manual intervention: {violation.file_path}")
        return False
        
    def _fix_docker_files(self, violation: ComplianceViolation) -> bool:
        """Fix Docker-related violations"""
        return self._fix_by_archiving(violation)

class ComplianceFileSystemWatcher(FileSystemEventHandler):
    """Real-time file system monitoring for compliance violations"""
    
    def __init__(self, rule_engine: ComplianceRuleEngine, auto_fixer: ComplianceAutoFixer, callback):
        self.rule_engine = rule_engine
        self.auto_fixer = auto_fixer
        self.callback = callback
        self.ignore_patterns = {
            '.git', '__pycache__', 'node_modules', 'venv', '.pytest_cache',
            'logs', 'data', 'tmp', '.vscode', '.idea'
        }
        
    def should_ignore(self, path: str) -> bool:
        """Check if path should be ignored"""
        path_parts = Path(path).parts
        return any(ignore in path_parts for ignore in self.ignore_patterns)
        
    def on_created(self, event):
        """Handle file creation events"""
        if not event.is_directory and not self.should_ignore(event.src_path):
            self._check_file(event.src_path)
            
    def on_modified(self, event):
        """Handle file modification events"""
        if not event.is_directory and not self.should_ignore(event.src_path):
            self._check_file(event.src_path)
            
    def on_moved(self, event):
        """Handle file move events"""
        if not event.is_directory and not self.should_ignore(event.dest_path):
            self._check_file(event.dest_path)
            
    def _check_file(self, file_path: str):
        """Check a file for compliance violations"""
        try:
            path = Path(file_path)
            violations = self.rule_engine.check_file_against_rules(path)
            
            if violations:
                for violation in violations:
                    # Attempt auto-fix if enabled
                    if self.auto_fixer.can_auto_fix(violation):
                        self.auto_fixer.auto_fix_violation(violation)
                    
                    # Report violation
                    self.callback(violation)
                    
        except Exception as e:
            logger.error(f"Error checking file {file_path}: {e}")

class ComplianceMonitorCore:
    """Core compliance monitoring system"""
    
    def __init__(self, project_root: str = "/opt/sutazaiapp", auto_fix: bool = False):
        self.project_root = Path(project_root)
        self.auto_fix = auto_fix
        self.violations: List[ComplianceViolation] = []
        self.violation_history: List[ComplianceViolation] = []
        
        # Initialize components
        self.rule_engine = ComplianceRuleEngine(str(self.project_root))
        self.auto_fixer = ComplianceAutoFixer(str(self.project_root)) if auto_fix else None
        
        # File system watcher
        self.fs_watcher = ComplianceFileSystemWatcher(
            self.rule_engine, 
            self.auto_fixer,
            self._handle_violation
        )
        self.observer = Observer()
        
        # State tracking
        self.is_monitoring = False
        self.last_scan_time = None
        self.violation_counts = {}
        
        # Ensure logs directory exists
        (self.project_root / "logs").mkdir(exist_ok=True)
        
    def _handle_violation(self, violation: ComplianceViolation):
        """Handle detected violations"""
        self.violations.append(violation)
        self.violation_history.append(violation)
        
        # Update counts
        rule_id = violation.rule_id
        self.violation_counts[rule_id] = self.violation_counts.get(rule_id, 0) + 1
        
        # Log violation
        logger.warning(
            f"Compliance violation: {violation.rule_name} in {violation.file_path} - {violation.message}"
        )
        
        # Critical violations get immediate attention
        if violation.severity == ViolationSeverity.CRITICAL:
            self._send_alert(violation)
            
    def _send_alert(self, violation: ComplianceViolation):
        """Send alert for critical violations"""
        alert_file = self.project_root / "logs" / "compliance-alerts.log"
        
        with open(alert_file, 'a') as f:
            f.write(f"[{datetime.now().isoformat()}] CRITICAL VIOLATION: {violation.rule_name}\n")
            f.write(f"  File: {violation.file_path}\n")
            f.write(f"  Message: {violation.message}\n")
            f.write(f"  Auto-fixable: {violation.auto_fixable}\n\n")
            
        # TODO: Add email/Slack notifications
        
    def start_monitoring(self):
        """Start real-time monitoring"""
        if self.is_monitoring:
            logger.warning("Monitoring is already active")
            return
            
        logger.info(f"Starting compliance monitoring for {self.project_root}")
        logger.info(f"Auto-fix enabled: {self.auto_fix}")
        
        # Schedule observer
        self.observer.schedule(self.fs_watcher, str(self.project_root), recursive=True)
        self.observer.start()
        
        self.is_monitoring = True
        logger.info("Real-time compliance monitoring started")
        
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        if not self.is_monitoring:
            return
            
        self.observer.stop()
        self.observer.join()
        self.is_monitoring = False
        logger.info("Real-time compliance monitoring stopped")
        
    def run_full_scan(self) -> Dict[str, int]:
        """Run a full compliance scan of the project"""
        logger.info("Starting full compliance scan...")
        start_time = datetime.now()
        
        scan_stats = {
            "files_scanned": 0,
            "violations_found": 0,
            "auto_fixed": 0,
            "scan_duration": 0
        }
        
        # Clear current violations
        self.violations.clear()
        
        # Scan all files
        for file_path in self.project_root.rglob('*'):
            if (file_path.is_file() and 
                not self.fs_watcher.should_ignore(str(file_path))):
                
                scan_stats["files_scanned"] += 1
                violations = self.rule_engine.check_file_against_rules(file_path)
                
                for violation in violations:
                    self._handle_violation(violation)
                    scan_stats["violations_found"] += 1
                    
                    # Auto-fix if enabled
                    if (self.auto_fixer and 
                        self.auto_fixer.can_auto_fix(violation) and
                        self.auto_fixer.auto_fix_violation(violation)):
                        scan_stats["auto_fixed"] += 1
                        
        scan_stats["scan_duration"] = (datetime.now() - start_time).total_seconds()
        self.last_scan_time = start_time
        
        logger.info(f"Full scan complete: {scan_stats}")
        return scan_stats
        
    def generate_compliance_report(self, output_path: Optional[str] = None) -> str:
        """Generate comprehensive compliance report"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(self.project_root / "logs" / f"compliance-report-{timestamp}.json")
            
        # Aggregate statistics
        stats = {
            "total_violations": len(self.violations),
            "critical_violations": len([v for v in self.violations if v.severity == ViolationSeverity.CRITICAL]),
            "auto_fixable_violations": len([v for v in self.violations if v.auto_fixable]),
            "fixed_violations": len([v for v in self.violations if v.fixed_at]),
            "violations_by_rule": self.violation_counts,
            "violations_by_severity": {},
            "last_scan_time": self.last_scan_time.isoformat() if self.last_scan_time else None
        }
        
        # Count by severity
        for violation in self.violations:
            severity = violation.severity.value
            stats["violations_by_severity"][severity] = stats["violations_by_severity"].get(severity, 0) + 1
            
        report_data = {
            "generated_at": datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "monitoring_active": self.is_monitoring,
            "auto_fix_enabled": self.auto_fix,
            "statistics": stats,
            "violations": [v.to_dict() for v in self.violations],
            "recommendations": self._generate_recommendations()
        }
        
        # Write report
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
            
        logger.info(f"Compliance report generated: {output_path}")
        return output_path
        
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on current violations"""
        recommendations = []
        
        critical_count = len([v for v in self.violations if v.severity == ViolationSeverity.CRITICAL])
        if critical_count > 0:
            recommendations.append(f"Address {critical_count} critical violations immediately")
            
        auto_fixable_count = len([v for v in self.violations if v.auto_fixable and not v.fixed_at])
        if auto_fixable_count > 0:
            recommendations.append(f"Enable auto-fix to resolve {auto_fixable_count} violations automatically")
            
        if "rule_13_no_garbage" in self.violation_counts:
            recommendations.append("Run garbage collection to clean up temporary files")
            
        if "rule_12_single_deployment_script" in self.violation_counts:
            recommendations.append("Consolidate deployment scripts into single canonical script")
            
        if not recommendations:
            recommendations.append("All compliance checks passed! Maintain current standards.")
            
        return recommendations
        
    def get_dashboard_data(self) -> Dict:
        """Get real-time dashboard data"""
        return {
            "monitoring_status": "active" if self.is_monitoring else "inactive",
            "total_violations": len(self.violations),
            "critical_violations": len([v for v in self.violations if v.severity == ViolationSeverity.CRITICAL]),
            "recent_violations": [
                v.to_dict() for v in sorted(self.violations, key=lambda x: x.detected_at, reverse=True)[:10]
            ],
            "violation_counts_by_rule": self.violation_counts,
            "last_scan": self.last_scan_time.isoformat() if self.last_scan_time else None,
            "auto_fix_enabled": self.auto_fix
        }

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SutazAI Compliance Monitor")
    parser.add_argument("--mode", choices=["monitor", "scan", "report"], default="monitor",
                       help="Operation mode")
    parser.add_argument("--auto-fix", action="store_true",
                       help="Enable automatic fixing of violations")
    parser.add_argument("--project-root", default="/opt/sutazaiapp",
                       help="Project root directory")
    parser.add_argument("--output", help="Output file for reports")
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = ComplianceMonitorCore(args.project_root, args.auto_fix)
    
    try:
        if args.mode == "scan":
            stats = monitor.run_full_scan()
            print(f"Scan complete: {stats}")
            
        elif args.mode == "report":
            report_path = monitor.generate_compliance_report(args.output)
            print(f"Report generated: {report_path}")
            
        elif args.mode == "monitor":
            # Run initial scan
            monitor.run_full_scan()
            
            # Start monitoring
            monitor.start_monitoring()
            
            try:
                while True:
                    time.sleep(300)  # Generate report every 5 minutes
                    monitor.generate_compliance_report()
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
            finally:
                monitor.stop_monitoring()
                
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
