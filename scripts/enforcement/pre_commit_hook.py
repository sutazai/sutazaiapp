#!/usr/bin/env python3
"""
import logging

logger = logging.getLogger(__name__)
🔧 PRE-COMMIT HOOK - Zero-Tolerance Rule Enforcement
Prevents any code from being committed that violates the 20 Fundamental Rules
"""

import sys
import os
import subprocess
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from comprehensive_rule_enforcer import ComprehensiveRuleEnforcer

def get_staged_files():
    """Get list of staged files for commit"""
    result = subprocess.run(
        ['git', 'diff', '--cached', '--name-only'],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        return result.stdout.strip().split('\n')
    return []

def main():
    """Pre-commit hook main function"""
    logger.info("🔧 SUPREME VALIDATOR - Pre-Commit Rule Enforcement")
    logger.info("=" * 60)
    
    # Get repository root
    repo_root = subprocess.run(
        ['git', 'rev-parse', '--show-toplevel'],
        capture_output=True,
        text=True
    ).stdout.strip()
    
    if not repo_root:
        repo_root = "/opt/sutazaiapp"
    
    # Get staged files
    staged_files = get_staged_files()
    if not staged_files or staged_files == ['']:
        logger.info("✅ No files staged for commit")
        return 0
    
    logger.info(f"Validating {len(staged_files)} staged files...")
    
    # Initialize enforcer
    enforcer = ComprehensiveRuleEnforcer(repo_root, auto_fix=False)
    
    # Run validation
    report = enforcer.validate_all_rules()
    
    # Check for critical violations
    critical_count = report["violations_by_severity"].get("CRITICAL", 0)
    high_count = report["violations_by_severity"].get("HIGH", 0)
    
    if critical_count > 0:
        logger.error("\n❌ COMMIT BLOCKED: Critical rule violations detected!")
        logger.error(f"   Found {critical_count} CRITICAL violations")
        logger.error("\nCritical violations must be fixed before committing:")
        
        for violation in report["violations"][:5]:  # Show first 5
            if violation["severity"] == "CRITICAL":
                logger.info(f"  • Rule {violation['rule']}: {violation['description']}")
                logger.info(f"    File: {violation['file']}:{violation['line']}")
                logger.info(f"    Fix: {violation['remediation']}")
        
        logger.info("\nRun 'python scripts/enforcement/comprehensive_rule_enforcer.py' for full report")
        return 1
    
    if high_count > 0:
        logger.warning(f"\n⚠️  WARNING: {high_count} HIGH priority violations detected")
        logger.info("Consider fixing these before committing:")
        
        for violation in report["violations"][:3]:  # Show first 3
            if violation["severity"] == "HIGH":
                logger.info(f"  • Rule {violation['rule']}: {violation['description']}")
        
        # Allow commit with warning for HIGH violations
        response = input("\nProceed with commit anyway? (y/N): ")
        if response.lower() != 'y':
            logger.info("Commit cancelled")
            return 1
    
    logger.info(f"\n✅ Compliance Score: {report['compliance_score']}%")
    logger.info("✅ Pre-commit validation passed")
    return 0

if __name__ == "__main__":
    sys.exit(main())