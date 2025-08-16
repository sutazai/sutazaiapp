#!/usr/bin/env python3
"""
import logging

logger = logging.getLogger(__name__)
🔧 Professional Codebase Rule Enforcement Validator
Comprehensive validation system for all 20 Fundamental Rules + Core Principles

This script validates the entire SutazAI codebase against the enforcement rules
defined in /opt/sutazaiapp/IMPORTANT/Enforcement_Rules
"""

import os
import sys
import re
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
import argparse


@dataclass
class RuleViolation:
    """Represents a rule violation with context and remediation"""
    rule_number: int
    rule_name: str
    violation_type: str
    file_path: str
    line_number: int
    description: str
    severity: str  # 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'
    remediation: str


class RuleEnforcer:
    """Comprehensive rule enforcement system"""
    
    def __init__(self, codebase_root: str):
        self.root = Path(codebase_root)
        self.violations: List[RuleViolation] = []
        self.rules_doc = self.root / "IMPORTANT" / "Enforcement_Rules"
        
    def validate_all_rules(self) -> Dict[str, Any]:
        """Execute all rule validations and return comprehensive report"""
        logger.info("🔧 Starting Professional Codebase Rule Validation...")
        
        # Rule 1: Real Implementation Only - No Fantasy Code
        self._validate_rule_01_real_implementation()
        
        # Rule 2: Never Break Existing Functionality
        self._validate_rule_02_no_breaking_changes()
        
        # Rule 3: Comprehensive Analysis Required
        self._validate_rule_03_comprehensive_analysis()
        
        # Rule 4: Investigate Existing Files & Consolidate First
        self._validate_rule_04_consolidation()
        
        # Rule 5: Professional Project Standards
        self._validate_rule_05_professional_standards()
        
        # Rule 6: Centralized Documentation
        self._validate_rule_06_documentation()
        
        # Rule 7: Script Organization & Control
        self._validate_rule_07_script_organization()
        
        # Rule 8: Python Script Excellence
        self._validate_rule_08_python_excellence()
        
        # Rule 9: Single Source Frontend/Backend
        self._validate_rule_09_single_source()
        
        # Rule 10: Functionality-First Cleanup
        self._validate_rule_10_cleanup()
        
        # Rule 11: Docker Excellence
        self._validate_rule_11_docker_excellence()
        
        # Rule 12: Universal Deployment Script
        self._validate_rule_12_deployment()
        
        # Rule 13: Zero Tolerance for Waste
        self._validate_rule_13_no_waste()
        
        # Rule 14: Specialized Claude Sub-Agent Usage
        self._validate_rule_14_agent_usage()
        
        # Rule 15: Documentation Quality
        self._validate_rule_15_doc_quality()
        
        # Rule 16: Local LLM Operations
        self._validate_rule_16_local_llm()
        
        # Rule 17: Canonical Documentation Authority
        self._validate_rule_17_canonical_docs()
        
        # Rule 18: Mandatory Documentation Review
        self._validate_rule_18_doc_review()
        
        # Rule 19: Change Tracking Requirements
        self._validate_rule_19_change_tracking()
        
        # Rule 20: MCP Server Protection
        self._validate_rule_20_mcp_protection()
        
        return self._generate_compliance_report()
    
    def _validate_rule_01_real_implementation(self):
        """Rule 1: Real Implementation Only - No Fantasy Code"""
        logger.info("📌 Validating Rule 1: Real Implementation Only...")
        
        # Check for fantasy/placeholder code
        fantasy_patterns = [
            r'\/\/\s*TODO.*magic\s+happens',
            r'\/\/.*future.*implementation',
            r'placeholder.*service',
            r'abstract.*handler',
            r'Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test.*implementation.*production',
            r'imaginary.*system'
        ]
        
        for pattern in fantasy_patterns:
            self._scan_files_for_pattern(
                pattern, 1, "Fantasy Code Detected",
                "Found abstract/placeholder implementation",
                "CRITICAL",
                "Replace with concrete implementation"
            )
    
    def _validate_rule_02_no_breaking_changes(self):
        """Rule 2: Never Break Existing Functionality"""
        logger.info("📌 Validating Rule 2: Never Break Existing Functionality...")
        
        # Check for potential breaking changes
        breaking_patterns = [
            r'\.remove\(',
            r'del\s+\w+\[',
            r'DROP\s+TABLE',
            r'ALTER\s+TABLE.*DROP'
        ]
        
        for pattern in breaking_patterns:
            self._scan_files_for_pattern(
                pattern, 2, "Potential Breaking Change",
                "Found code that may break existing functionality",
                "HIGH",
                "Ensure backward compatibility or proper migration"
            )
    
    def _validate_rule_05_professional_standards(self):
        """Rule 5: Professional Project Standards"""
        logger.info("📌 Validating Rule 5: Professional Project Standards...")
        
        # Count test files
        test_files = list(self.root.rglob("test_*.py")) + list(self.root.rglob("*_test.py"))
        source_files = list(self.root.rglob("*.py"))
        
        if len(source_files) > 0:
            test_ratio = len(test_files) / len(source_files)
            if test_ratio < 0.3:  # Less than 30% test coverage
                self.violations.append(RuleViolation(
                    5, "Professional Standards", "Insufficient Test Coverage",
                    str(self.root), 0,
                    f"Test ratio {test_ratio:.2%} below professional standards",
                    "MEDIUM", "Add comprehensive test coverage"
                ))
    
    def _validate_rule_07_script_organization(self):
        """Rule 7: Script Organization & Control"""
        logger.info("📌 Validating Rule 7: Script Organization...")
        
        # Check for scattered scripts
        scripts_dir = self.root / "scripts"
        if not scripts_dir.exists():
            self.violations.append(RuleViolation(
                7, "Script Organization", "Missing Scripts Directory",
                str(self.root), 0,
                "No centralized /scripts/ directory found",
                "HIGH", "Create organized /scripts/ directory structure"
            ))
    
    def _validate_rule_11_docker_excellence(self):
        """Rule 11: Docker Excellence"""
        logger.info("📌 Validating Rule 11: Docker Excellence...")
        
        # Check for non-root users in Dockerfiles
        dockerfiles = list(self.root.rglob("Dockerfile*"))
        for dockerfile in dockerfiles:
            try:
                content = dockerfile.read_text()
                if "USER " not in content:
                    self.violations.append(RuleViolation(
                        11, "Docker Excellence", "Missing USER directive",
                        str(dockerfile), 0,
                        "Dockerfile missing non-root USER directive",
                        "HIGH", "Add USER directive for security"
                    ))
            except Exception:
                pass
    
    def _validate_rule_13_no_waste(self):
        """Rule 13: Zero Tolerance for Waste"""
        logger.info("📌 Validating Rule 13: Zero Tolerance for Waste...")
        
        # Check for TODO/FIXME markers
        waste_patterns = [
            r'TODO:|FIXME:|XXX:|HACK:'
        ]
        
        for pattern in waste_patterns:
            self._scan_files_for_pattern(
                pattern, 13, "Technical Debt Marker",
                "Found technical debt marker in code",
                "LOW",
                "Resolve or document the issue"
            )
    
    def _validate_rule_20_mcp_protection(self):
        """Rule 20: MCP Server Protection"""
        logger.info("📌 Validating Rule 20: MCP Server Protection...")
        
        mcp_json = self.root / ".mcp.json"
        if mcp_json.exists():
            # Check if .mcp.json has been modified recently
            import time
            mtime = os.path.getmtime(mcp_json)
            if time.time() - mtime < 3600:  # Modified in last hour
                self.violations.append(RuleViolation(
                    20, "MCP Protection", "Recent MCP Modification",
                    str(mcp_json), 0,
                    "MCP configuration modified recently - verify authorization",
                    "CRITICAL", "Ensure MCP changes are authorized"
                ))
    
    def _scan_files_for_pattern(self, pattern: str, rule_num: int, 
                               violation_type: str, description: str,
                               severity: str, remediation: str):
        """Scan all files for a specific pattern and record violations"""
        try:
            result = subprocess.run([
                'grep', '-rn', pattern, str(self.root),
                '--include=*.py', '--include=*.js', '--include=*.ts'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if ':' in line:
                        parts = line.split(':', 2)
                        if len(parts) >= 2:
                            file_path = parts[0]
                            line_num = parts[1] if parts[1].isdigit() else 0
                            
                            self.violations.append(RuleViolation(
                                rule_num, f"Rule {rule_num}", violation_type,
                                file_path, int(line_num), description,
                                severity, remediation
                            ))
        except Exception as e:
            logger.warning(f"Warning: Could not scan for pattern {pattern}: {e}")
    
    def _generate_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        report = {
            "timestamp": subprocess.run(['date', '-u'], capture_output=True, text=True).stdout.strip(),
            "total_violations": len(self.violations),
            "violations_by_severity": {},
            "violations_by_rule": {},
            "violations": [],
            "compliance_score": 0,
            "summary": ""
        }
        
        # Count violations by severity
        for violation in self.violations:
            severity = violation.severity
            report["violations_by_severity"][severity] = report["violations_by_severity"].get(severity, 0) + 1
            
            rule = violation.rule_number
            report["violations_by_rule"][rule] = report["violations_by_rule"].get(rule, 0) + 1
            
            report["violations"].append({
                "rule": violation.rule_number,
                "rule_name": violation.rule_name,
                "type": violation.violation_type,
                "file": violation.file_path,
                "line": violation.line_number,
                "description": violation.description,
                "severity": violation.severity,
                "remediation": violation.remediation
            })
        
        # Calculate compliance score (0-100)
        total_files = len(list(self.root.rglob("*.py"))) + len(list(self.root.rglob("*.js")))
        if total_files > 0:
            violation_ratio = len(self.violations) / total_files
            report["compliance_score"] = max(0, 100 - (violation_ratio * 100))
        
        # Generate summary
        critical = report["violations_by_severity"].get("CRITICAL", 0)
        high = report["violations_by_severity"].get("HIGH", 0)
        
        if critical > 0:
            report["summary"] = f"❌ CRITICAL ISSUES: {critical} violations require immediate attention"
        elif high > 0:
            report["summary"] = f"⚠️  HIGH PRIORITY: {high} violations should be addressed"
        else:
            report["summary"] = "✅ GOOD COMPLIANCE: No critical or high priority violations"
        
        return report


def main():
    parser = argparse.ArgumentParser(description="SutazAI Rule Enforcement Validator")
    parser.add_argument("--root", default="/opt/sutazaiapp", help="Codebase root directory")
    parser.add_argument("--output", help="Output JSON report file")
    parser.add_argument("--summary", action="store_true", help="Show summary only")
    
    args = parser.parse_args()
    
    enforcer = RuleEnforcer(args.root)
    report = enforcer.validate_all_rules()
    
    if args.summary:
        logger.info(f"\n🔧 RULE ENFORCEMENT SUMMARY")
        logger.info(f"{'='*50}")
        logger.info(f"Total Violations: {report['total_violations']}")
        logger.info(f"Compliance Score: {report['compliance_score']:.1f}%")
        logger.info(f"Status: {report['summary']}")
        
        if report['violations_by_severity']:
            logger.info(f"\nViolations by Severity:")
            for severity, count in sorted(report['violations_by_severity'].items()):
                logger.info(f"  {severity}: {count}")
    else:
        logger.info(json.dumps(report, indent=2))
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"📄 Report saved to {args.output}")
    
    # Exit with appropriate code
    critical = report["violations_by_severity"].get("CRITICAL", 0)
    high = report["violations_by_severity"].get("HIGH", 0)
    
    if critical > 0:
        sys.exit(2)  # Critical violations
    elif high > 0:
        sys.exit(1)  # High priority violations
    else:
        sys.exit(0)  # Clean


if __name__ == "__main__":
    main()