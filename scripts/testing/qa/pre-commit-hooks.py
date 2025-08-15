#!/usr/bin/env python3
"""
import logging

logger = logging.getLogger(__name__)
Pre-commit hooks for quality gates enforcement.
Ensures all 20 Fundamental Rules compliance before allowing commits.

Version: SutazAI v93 - Quality Gates Implementation
Author: QA Validation Specialist (Claude Code)
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

class QualityGateEnforcer:
    """Enforces quality gates before commit."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.failed_checks = []
        self.warnings = []
        
    def run_command(self, cmd: List[str], capture_output: bool = True) -> Tuple[bool, str]:
        """Run a command and return success status and output."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=capture_output,
                text=True,
                cwd=self.project_root
            )
            return result.returncode == 0, result.stdout + result.stderr
        except Exception as e:
            return False, str(e)
    
    def check_rule_compliance(self) -> bool:
        """Check compliance with all 20 Fundamental Rules."""
        logger.info("🔧 Checking rule compliance...")
        
        # Check if Enforcement Rules document exists
        enforcement_rules = self.project_root / "IMPORTANT" / "Enforcement_Rules"
        if not enforcement_rules.exists():
            self.failed_checks.append("CRITICAL: Enforcement Rules document missing")
            return False
        
        # Run rule validator if available
        rule_validator = self.project_root / "scripts" / "enforcement" / "rule_validator_simple.py"
        if rule_validator.exists():
            success, output = self.run_command([
                "python3", str(rule_validator), "--quick"
            ])
            if not success:
                self.failed_checks.append(f"Rule validation failed: {output}")
                return False
        
        logger.info("✅ Rule compliance check passed")
        return True
    
    def check_code_formatting(self) -> bool:
        """Check code formatting with black and isort."""
        logger.info("🎨 Checking code formatting...")
        
        # Check with black
        success, output = self.run_command([
            "black", "--check", "--quiet", "backend/", "agents/", "tests/", "scripts/"
        ])
        if not success:
            self.failed_checks.append("Code formatting issues detected (run: make format)")
            return False
        
        # Check with isort
        success, output = self.run_command([
            "isort", "--check-only", "--quiet", "backend/", "agents/", "tests/", "scripts/"
        ])
        if not success:
            self.failed_checks.append("Import sorting issues detected (run: make format)")
            return False
        
        logger.info("✅ Code formatting check passed")
        return True
    
    def check_code_style(self) -> bool:
        """Check code style with flake8."""
        logger.info("🔍 Checking code style...")
        
        success, output = self.run_command([
            "flake8", "backend/", "agents/", "tests/", "scripts/"
        ])
        if not success:
            self.failed_checks.append(f"Code style violations: {output}")
            return False
        
        logger.info("✅ Code style check passed")
        return True
    
    def check_security(self) -> bool:
        """Run basic security checks."""
        logger.info("🛡️ Running security checks...")
        
        # Check for common security issues with bandit
        success, output = self.run_command([
            "bandit", "-r", "backend/", "agents/", "-f", "txt", "--severity-level", "high"
        ])
        if not success and "No issues identified" not in output:
            self.warnings.append(f"Security issues detected: {output}")
        
        # Check for hardcoded secrets
        success, output = self.run_command([
            "grep", "-r", "-i", "--include=*.py", 
            "-E", "(password|secret|key|token)\\s*=\\s*['\"][^'\"]+['\"]",
            "backend/", "agents/"
        ])
        if success:  # grep returns 0 if it finds matches
            self.failed_checks.append("Potential hardcoded secrets detected")
            return False
        
        logger.info("✅ Security checks passed")
        return True
    
    def check_mcp_protection(self) -> bool:
        """Verify MCP servers are protected (Rule 20)."""
        logger.info("🔒 Checking MCP server protection...")
        
        mcp_config = self.project_root / ".mcp.json"
        if not mcp_config.exists():
            self.failed_checks.append("CRITICAL: .mcp.json missing (Rule 20 violation)")
            return False
        
        # Check if .mcp.json is in staged changes
        success, output = self.run_command([
            "git", "diff", "--cached", "--name-only"
        ])
        if success and ".mcp.json" in output:
            self.failed_checks.append("CRITICAL: .mcp.json modification detected (Rule 20 violation)")
            return False
        
        logger.info("✅ MCP protection check passed")
        return True
    
    def check_changelog_compliance(self) -> bool:
        """Check CHANGELOG.md compliance (Rule 18)."""
        logger.info("📋 Checking CHANGELOG compliance...")
        
        # Check root CHANGELOG.md exists
        changelog = self.project_root / "CHANGELOG.md"
        if not changelog.exists():
            self.failed_checks.append("CRITICAL: Root CHANGELOG.md missing (Rule 18 violation)")
            return False
        
        # Check if any source files changed without CHANGELOG update
        success, output = self.run_command([
            "git", "diff", "--cached", "--name-only"
        ])
        if success:
            changed_files = output.strip().split('\n') if output.strip() else []
            source_changes = any(
                f.endswith(('.py', '.js', '.ts', '.yml', '.yaml', '.json'))
                for f in changed_files
            )
            changelog_updated = any(
                'CHANGELOG.md' in f for f in changed_files
            )
            
            if source_changes and not changelog_updated:
                self.warnings.append("Source files changed but CHANGELOG.md not updated")
        
        logger.info("✅ CHANGELOG compliance check passed")
        return True
    
    def check_basic_tests(self) -> bool:
        """Run basic smoke tests if available."""
        logger.info("🧪 Running basic tests...")
        
        # Run smoke tests if available
        smoke_tests = self.project_root / "tests" / "e2e" / "test_smoke.py"
        if smoke_tests.exists():
            success, output = self.run_command([
                "pytest", str(smoke_tests), "-v", "--tb=short"
            ])
            if not success:
                self.warnings.append(f"Smoke tests failed: {output}")
        else:
            self.warnings.append("No smoke tests found")
        
        logger.info("✅ Basic tests check completed")
        return True
    
    def run_all_checks(self) -> bool:
        """Run all quality gate checks."""
        logger.info("🚀 Starting pre-commit quality gate validation...")
        logger.info(f"Project root: {self.project_root}")
        
        checks = [
            ("Rule Compliance", self.check_rule_compliance),
            ("Code Formatting", self.check_code_formatting),
            ("Code Style", self.check_code_style),
            ("Security", self.check_security),
            ("MCP Protection", self.check_mcp_protection),
            ("CHANGELOG Compliance", self.check_changelog_compliance),
            ("Basic Tests", self.check_basic_tests),
        ]
        
        all_passed = True
        for check_name, check_func in checks:
            try:
                if not check_func():
                    all_passed = False
            except Exception as e:
                self.failed_checks.append(f"{check_name} check failed with error: {e}")
                all_passed = False
        
        return all_passed
    
    def report_results(self, passed: bool) -> None:
        """Report quality gate results."""
        logger.info("\n" + "="*60)
        logger.info("🎯 QUALITY GATE RESULTS")
        logger.info("="*60)
        
        if passed:
            logger.info("✅ ALL QUALITY GATES PASSED")
            logger.info("🏆 Commit approved - maintaining code excellence!")
        else:
            logger.info("❌ QUALITY GATE FAILURES DETECTED")
            logger.info("🚫 Commit rejected - fix issues before committing")
            
            if self.failed_checks:
                logger.error("\n💥 CRITICAL ISSUES (must fix):")
                for issue in self.failed_checks:
                    logger.info(f"  • {issue}")
        
        if self.warnings:
            logger.warning("\n⚠️  WARNINGS (recommended to fix):")
            for warning in self.warnings:
                logger.warning(f"  • {warning}")
        
        if not passed:
            logger.info("\n🔧 SUGGESTED FIXES:")
            logger.info("  • Run: make format (fix formatting)")
            logger.info("  • Run: make lint (check style issues)")
            logger.info("  • Run: make test-smoke (verify functionality)")
            logger.error("  • Review and fix critical issues listed above")
            logger.info("  • Update CHANGELOG.md if source files changed")
        
        logger.info("="*60)

def main():
    """Main entry point for pre-commit hook."""
    enforcer = QualityGateEnforcer()
    
    # Check if we're in a git repository
    if not (Path.cwd() / ".git").exists():
        logger.info("⚠️  Not in a git repository, skipping pre-commit checks")
        return 0
    
    start_time = time.time()
    passed = enforcer.run_all_checks()
    duration = time.time() - start_time
    
    enforcer.report_results(passed)
    logger.info(f"\n⏱️  Quality gate validation completed in {duration:.1f}s")
    
    return 0 if passed else 1

if __name__ == "__main__":
    sys.exit(main())