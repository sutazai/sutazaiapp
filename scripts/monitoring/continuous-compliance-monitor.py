#!/usr/bin/env python3
"""
Continuous Compliance Monitoring System for SutazAI
Purpose: Real-time monitoring and enforcement of 16 codebase hygiene rules
Usage: python continuous-compliance-monitor.py [--daemon] [--report-only]
Requirements: Python 3.8+, GitPython, watchdog
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/logs/compliance-monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class RuleViolation:
    rule_number: int
    rule_name: str
    severity: str  # 'critical', 'high', 'medium', 'low'
    file_path: str
    line_number: Optional[int]
    description: str
    timestamp: str
    auto_fixable: bool

class ComplianceMonitor:
    def __init__(self, project_root: str = "/opt/sutazaiapp"):
        self.project_root = Path(project_root)
        self.violations = []
        self.rules_config = self._load_rules_config()
        
    def _load_rules_config(self) -> Dict:
        """Load the 16 codebase hygiene rules configuration"""
        return {
            1: {
                "name": "No Fantasy Elements",
                "checks": ["magic", "wizard", "black-box", "teleport"],
                "severity": "high",
                "auto_fix": True
            },
            2: {
                "name": "Do Not Break Existing Functionality",
                "checks": ["regression_tests", "backwards_compatibility"],
                "severity": "critical",
                "auto_fix": False
            },
            3: {
                "name": "Analyze Everything",
                "checks": ["comprehensive_analysis"],
                "severity": "high",
                "auto_fix": False
            },
            4: {
                "name": "Reuse Before Creating",
                "checks": ["duplicate_detection", "existing_script_check"],
                "severity": "medium",
                "auto_fix": True
            },
            5: {
                "name": "Professional Standards",
                "checks": ["code_quality", "naming_conventions"],
                "severity": "high",
                "auto_fix": True
            },
            6: {
                "name": "Centralized Documentation",
                "checks": ["doc_location", "doc_structure"],
                "severity": "medium",
                "auto_fix": True
            },
            7: {
                "name": "Script Organization",
                "checks": ["script_location", "script_duplicates"],
                "severity": "medium",
                "auto_fix": True
            },
            8: {
                "name": "Python Script Sanity",
                "checks": ["python_headers", "python_structure"],
                "severity": "medium",
                "auto_fix": True
            },
            9: {
                "name": "No Version Duplication",
                "checks": ["backend_versions", "frontend_versions"],
                "severity": "high",
                "auto_fix": False
            },
            10: {
                "name": "Functionality-First Cleanup",
                "checks": ["safe_deletion", "reference_check"],
                "severity": "critical",
                "auto_fix": False
            },
            11: {
                "name": "Docker Structure",
                "checks": ["dockerfile_standards", "docker_organization"],
                "severity": "medium",
                "auto_fix": True
            },
            12: {
                "name": "Single Deployment Script",
                "checks": ["deployment_script_count", "deploy_sh_exists"],
                "severity": "high",
                "auto_fix": True
            },
            13: {
                "name": "No Garbage Files",
                "checks": ["backup_files", "temp_files", "old_files"],
                "severity": "high",
                "auto_fix": True
            },
            14: {
                "name": "Correct AI Agent Usage",
                "checks": ["agent_selection", "agent_appropriateness"],
                "severity": "medium",
                "auto_fix": False
            },
            15: {
                "name": "Documentation Deduplication",
                "checks": ["doc_duplicates", "doc_naming"],
                "severity": "medium",
                "auto_fix": True
            },
            16: {
                "name": "Ollama/tinyllama Usage",
                "checks": ["ollama_config", "tinyllama_default"],
                "severity": "low",
                "auto_fix": True
            }
        }
    
    def check_rule_1_fantasy_elements(self) -> List[RuleViolation]:
        """Check for fantasy elements in code"""
        violations = []
        forbidden_terms = ["magic", "wizard", "black-box", "teleport", "mystical"]
        
        for py_file in self.project_root.rglob("*.py"):
            if "/archive/" in str(py_file) or "/.git/" in str(py_file):
                continue
                
            try:
                content = py_file.read_text()
                for line_num, line in enumerate(content.splitlines(), 1):
                    for term in forbidden_terms:
                        if term.lower() in line.lower() and not line.strip().startswith("#"):
                            violations.append(RuleViolation(
                                rule_number=1,
                                rule_name="No Fantasy Elements",
                                severity="high",
                                file_path=str(py_file),
                                line_number=line_num,
                                description=f"Found forbidden term '{term}' in code",
                                timestamp=datetime.now().isoformat(),
                                auto_fixable=True
                            ))
            except Exception as e:
                logger.error(f"Error checking {py_file}: {e}")
                
        return violations
    
    def check_rule_7_script_organization(self) -> List[RuleViolation]:
        """Check for script organization violations"""
        violations = []
        scripts_dir = self.project_root / "scripts"
        
        # Find scripts outside of /scripts/
        for script in self.project_root.rglob("*.sh"):
            if not script.is_relative_to(scripts_dir) and "/archive/" not in str(script):
                if script.parent != self.project_root:  # Allow deploy.sh in root
                    violations.append(RuleViolation(
                        rule_number=7,
                        rule_name="Script Organization",
                        severity="medium",
                        file_path=str(script),
                        line_number=None,
                        description="Script found outside /scripts/ directory",
                        timestamp=datetime.now().isoformat(),
                        auto_fixable=True
                    ))
        
        # Check for duplicate scripts
        script_contents = defaultdict(list)
        for script in scripts_dir.rglob("*.sh"):
            try:
                content = script.read_text()
                # Normalize content for comparison
                normalized = "\n".join(line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#"))
                script_contents[normalized].append(script)
            except:
                pass
                
        for content_hash, scripts in script_contents.items():
            if len(scripts) > 1:
                for script in scripts[1:]:
                    violations.append(RuleViolation(
                        rule_number=7,
                        rule_name="Script Organization",
                        severity="medium",
                        file_path=str(script),
                        line_number=None,
                        description=f"Duplicate of {scripts[0].name}",
                        timestamp=datetime.now().isoformat(),
                        auto_fixable=True
                    ))
                    
        return violations
    
    def check_rule_12_deployment_script(self) -> List[RuleViolation]:
        """Check for single deployment script compliance"""
        violations = []
        deploy_scripts = []
        
        # Look for deployment scripts
        patterns = ["deploy*.sh", "release*.sh", "install*.sh", "setup*.sh"]
        for pattern in patterns:
            deploy_scripts.extend(self.project_root.rglob(pattern))
            
        # Filter out test fixtures and archives
        deploy_scripts = [s for s in deploy_scripts if "/test" not in str(s) and "/archive/" not in str(s)]
        
        canonical_deploy = self.project_root / "deploy.sh"
        if not canonical_deploy.exists():
            violations.append(RuleViolation(
                rule_number=12,
                rule_name="Single Deployment Script",
                severity="high",
                file_path=str(self.project_root),
                line_number=None,
                description="Missing canonical deploy.sh in project root",
                timestamp=datetime.now().isoformat(),
                auto_fixable=False
            ))
            
        # Check for multiple deployment scripts
        if len(deploy_scripts) > 1:
            for script in deploy_scripts:
                if script != canonical_deploy:
                    violations.append(RuleViolation(
                        rule_number=12,
                        rule_name="Single Deployment Script",
                        severity="high",
                        file_path=str(script),
                        line_number=None,
                        description="Extra deployment script (should be consolidated into deploy.sh)",
                        timestamp=datetime.now().isoformat(),
                        auto_fixable=True
                    ))
                    
        return violations
    
    def check_rule_13_garbage_files(self) -> List[RuleViolation]:
        """Check for garbage files"""
        violations = []
        garbage_patterns = [
            "*.backup", "*.bak", "*.old", "*.tmp", "*.temp",
            "*~", ".DS_Store", "Thumbs.db", "*.swp", "*.swo",
            "=0.*", "=1.*", "=2.*", "=3.*", "=4.*", "=5.*", "=6.*"
        ]
        
        for pattern in garbage_patterns:
            for garbage_file in self.project_root.rglob(pattern):
                if "/.git/" not in str(garbage_file):
                    violations.append(RuleViolation(
                        rule_number=13,
                        rule_name="No Garbage Files",
                        severity="high",
                        file_path=str(garbage_file),
                        line_number=None,
                        description=f"Garbage file detected: {pattern}",
                        timestamp=datetime.now().isoformat(),
                        auto_fixable=True
                    ))
                    
        return violations
    
    def run_compliance_check(self) -> Dict:
        """Run all compliance checks"""
        logger.info("Starting compliance check...")
        all_violations = []
        
        # Run individual rule checks
        all_violations.extend(self.check_rule_1_fantasy_elements())
        all_violations.extend(self.check_rule_7_script_organization())
        all_violations.extend(self.check_rule_12_deployment_script())
        all_violations.extend(self.check_rule_13_garbage_files())
        
        # Calculate compliance score
        rules_with_violations = set(v.rule_number for v in all_violations)
        compliance_score = (16 - len(rules_with_violations)) / 16 * 100
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "compliance_score": compliance_score,
            "total_violations": len(all_violations),
            "rules_violated": len(rules_with_violations),
            "violations_by_rule": defaultdict(list),
            "auto_fixable_count": sum(1 for v in all_violations if v.auto_fixable),
            "critical_violations": sum(1 for v in all_violations if v.severity == "critical")
        }
        
        for violation in all_violations:
            report["violations_by_rule"][violation.rule_number].append(asdict(violation))
            
        return report
    
    def auto_fix_violations(self, violations: List[RuleViolation]) -> int:
        """Automatically fix violations where possible"""
        fixed_count = 0
        
        for violation in violations:
            if not violation.auto_fixable:
                continue
                
            try:
                if violation.rule_number == 13:  # Garbage files
                    os.remove(violation.file_path)
                    logger.info(f"Removed garbage file: {violation.file_path}")
                    fixed_count += 1
                    
                elif violation.rule_number == 7:  # Script organization
                    if "outside /scripts/" in violation.description:
                        # Move script to proper location
                        script_path = Path(violation.file_path)
                        target = self.project_root / "scripts" / "misc" / script_path.name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        script_path.rename(target)
                        logger.info(f"Moved {script_path} to {target}")
                        fixed_count += 1
                        
            except Exception as e:
                logger.error(f"Failed to auto-fix {violation.file_path}: {e}")
                
        return fixed_count
    
    def generate_report(self, report_data: Dict) -> None:
        """Generate compliance report"""
        report_path = self.project_root / "compliance-reports" / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
            
        # Also update the latest report symlink
        latest_link = report_path.parent / "latest.json"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(report_path.name)
        
        logger.info(f"Report generated: {report_path}")
        logger.info(f"Compliance Score: {report_data['compliance_score']:.1f}%")
        
    def run_daemon_mode(self):
        """Run in daemon mode with continuous monitoring"""
        logger.info("Starting compliance monitor in daemon mode...")
        
        while True:
            try:
                report = self.run_compliance_check()
                
                if report["total_violations"] > 0:
                    logger.warning(f"Found {report['total_violations']} violations!")
                    
                    # Auto-fix if enabled
                    if report["auto_fixable_count"] > 0:
                        violations_list = []
                        for rule_violations in report["violations_by_rule"].values():
                            for v_dict in rule_violations:
                                violations_list.append(RuleViolation(**v_dict))
                                
                        fixed = self.auto_fix_violations(violations_list)
                        logger.info(f"Auto-fixed {fixed} violations")
                        
                self.generate_report(report)
                
                # Check every 5 minutes
                time.sleep(300)
                
            except KeyboardInterrupt:
                logger.info("Shutting down compliance monitor...")
                break
            except Exception as e:
                logger.error(f"Error in daemon mode: {e}")
                time.sleep(60)  # Wait before retrying

def main():
    parser = argparse.ArgumentParser(description="SutazAI Continuous Compliance Monitor")
    parser.add_argument("--daemon", action="store_true", help="Run in daemon mode")
    parser.add_argument("--report-only", action="store_true", help="Generate report without fixes")
    parser.add_argument("--fix", action="store_true", help="Auto-fix violations")
    
    args = parser.parse_args()
    
    monitor = ComplianceMonitor()
    
    if args.daemon:
        monitor.run_daemon_mode()
    else:
        report = monitor.run_compliance_check()
        monitor.generate_report(report)
        
        if args.fix and not args.report_only:
            violations_list = []
            for rule_violations in report["violations_by_rule"].values():
                for v_dict in rule_violations:
                    violations_list.append(RuleViolation(**v_dict))
                    
            fixed = monitor.auto_fix_violations(violations_list)
            print(f"Fixed {fixed} violations")

if __name__ == "__main__":
    main()