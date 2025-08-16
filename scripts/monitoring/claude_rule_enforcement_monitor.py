#!/usr/bin/env python3
"""
CLAUDE.md Rule Enforcement Monitor
Automated detection and reporting of rule violations
"""

import os
import json
import subprocess
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ClaudeRuleEnforcer:
    """Automated enforcement of CLAUDE.md rules"""
    
    def __init__(self, root_path: str = "/opt/sutazaiapp"):
        self.root_path = Path(root_path)
        self.violations = {}
        self.compliance_score = 100
        
    def check_rule_1_fantasy_code(self) -> Dict:
        """Rule 1: Detect fantasy/placeholder implementations"""
        violations = []
        fantasy_patterns = ["TODO", "FIXME", "placeholder", "mock", "fake", "dummy", "stub"]
        
        try:
            for pattern in fantasy_patterns:
                result = subprocess.run(
                    ["grep", "-r", pattern, str(self.root_path), "--include=*.py"],
                    capture_output=True, text=True
                )
                if result.stdout:
                    violations.extend(result.stdout.strip().split('\n'))
        except Exception as e:
            logger.error(f"Rule 1 check failed: {e}")
            
        return {
            "rule": "Rule 1: Real Implementation Only",
            "violations": len(violations),
            "severity": "CRITICAL" if violations else "COMPLIANT",
            "details": violations[:10] if violations else []
        }
    
    def check_rule_4_duplicates(self) -> Dict:
        """Rule 4: Detect duplicate files and directories"""
        duplicates = {
            "frontend": [],
            "backend": [],
            "agents": []
        }
        
        # Check for duplicate directories
        for pattern in ["frontend", "backend"]:
            result = subprocess.run(
                ["find", str(self.root_path), "-type", "d", "-name", f"{pattern}*"],
                capture_output=True, text=True
            )
            if result.stdout:
                dirs = result.stdout.strip().split('\n')
                if len(dirs) > 1:
                    duplicates[pattern] = dirs
                    
        # Check for duplicate agent configs
        result = subprocess.run(
            ["find", str(self.root_path), "-name", "*agent*.json"],
            capture_output=True, text=True
        )
        if result.stdout:
            agent_files = result.stdout.strip().split('\n')
            if len(agent_files) > 10:  # Threshold for too many agent files
                duplicates["agents"] = agent_files
                
        violation_count = sum(len(v) for v in duplicates.values())
        
        return {
            "rule": "Rule 4: No Duplication",
            "violations": violation_count,
            "severity": "CRITICAL" if violation_count > 5 else "MAJOR" if violation_count > 0 else "COMPLIANT",
            "details": duplicates
        }
    
    def check_rule_7_script_organization(self) -> Dict:
        """Rule 7: Check script organization"""
        violations = []
        
        # Check for scripts outside proper directories
        result = subprocess.run(
            ["find", str(self.root_path), "-name", "*.py", "-type", "f"],
            capture_output=True, text=True
        )
        
        if result.stdout:
            scripts = result.stdout.strip().split('\n')
            for script in scripts:
                # Check if script is in proper location
                if not any(dir in script for dir in ["/scripts/", "/backend/", "/tests/", "/src/"]):
                    if not any(skip in script for skip in ["/node_modules/", "/.venv/", "__pycache__"]):
                        violations.append(script)
                        
        return {
            "rule": "Rule 7: Script Organization",
            "violations": len(violations),
            "severity": "MAJOR" if violations else "COMPLIANT",
            "details": violations[:10] if violations else []
        }
    
    def check_rule_8_python_standards(self) -> Dict:
        """Rule 8: Check Python script standards"""
        violations = []
        
        # Sample check - look for scripts without proper structure
        scripts_path = self.root_path / "scripts"
        if scripts_path.exists():
            for py_file in scripts_path.rglob("*.py"):
                try:
                    with open(py_file, 'r') as f:
                        content = f.read()
                        # Check for basic standards
                        if "#!/usr/bin/env python3" not in content[:100]:
                            violations.append(f"{py_file}: Missing shebang")
                        if '"""' not in content[:500] and "'''" not in content[:500]:
                            violations.append(f"{py_file}: Missing docstring")
                        if "import logging" not in content and "print(" in content:
                            violations.append(f"{py_file}: Using print instead of logging")
                except Exception:
                    pass
                    
        return {
            "rule": "Rule 8: Python Excellence",
            "violations": len(violations),
            "severity": "MAJOR" if len(violations) > 10 else "MINOR" if violations else "COMPLIANT",
            "details": violations[:10] if violations else []
        }
    
    def check_rule_9_single_source(self) -> Dict:
        """Rule 9: Check for duplicate frontend/backend"""
        duplicates = []
        
        # Check for multiple frontend/backend directories
        for component in ["frontend", "backend"]:
            dirs = list(self.root_path.rglob(f"*/{component}"))
            if len(dirs) > 1:
                duplicates.extend([str(d) for d in dirs])
                
        return {
            "rule": "Rule 9: Single Source",
            "violations": len(duplicates),
            "severity": "CRITICAL" if duplicates else "COMPLIANT",
            "details": duplicates
        }
    
    def check_rule_13_waste(self) -> Dict:
        """Rule 13: Detect waste (archives, backups, etc.)"""
        waste_dirs = []
        
        waste_patterns = ["archive", "backup", "old", "deprecated", "temp", "tmp"]
        for pattern in waste_patterns:
            result = subprocess.run(
                ["find", str(self.root_path), "-type", "d", "-name", f"*{pattern}*"],
                capture_output=True, text=True
            )
            if result.stdout:
                waste_dirs.extend(result.stdout.strip().split('\n'))
                
        return {
            "rule": "Rule 13: Zero Waste",
            "violations": len(waste_dirs),
            "severity": "CRITICAL" if len(waste_dirs) > 10 else "MAJOR" if waste_dirs else "COMPLIANT",
            "details": waste_dirs[:10] if waste_dirs else []
        }
    
    def check_rule_18_changelog(self) -> Dict:
        """Rule 18: Check CHANGELOG.md compliance"""
        missing_changelogs = []
        
        # Check key directories for CHANGELOG.md
        key_dirs = ["backend", "frontend", "scripts", "docker", "tests", "docs"]
        for dir_name in key_dirs:
            dir_path = self.root_path / dir_name
            if dir_path.exists():
                changelog = dir_path / "CHANGELOG.md"
                if not changelog.exists():
                    missing_changelogs.append(str(dir_path))
                    
        return {
            "rule": "Rule 18: Documentation Review",
            "violations": len(missing_changelogs),
            "severity": "MAJOR" if missing_changelogs else "COMPLIANT",
            "details": missing_changelogs
        }
    
    def check_rule_20_mcp_protection(self) -> Dict:
        """Rule 20: Check MCP server protection"""
        violations = []
        
        # Check for unauthorized MCP modifications
        mcp_config = self.root_path / ".mcp.json"
        if mcp_config.exists():
            # Check file permissions and recent modifications
            stat_info = os.stat(mcp_config)
            if stat_info.st_mode & 0o222:  # Check if writable
                violations.append("MCP config is writable - should be protected")
                
        # Check for MCP wrapper modifications
        wrapper_path = self.root_path / "scripts" / "mcp" / "wrappers"
        if wrapper_path.exists():
            # Check for recent modifications without authorization
            pass  # Would need git history analysis
            
        return {
            "rule": "Rule 20: MCP Protection",
            "violations": len(violations),
            "severity": "CRITICAL" if violations else "COMPLIANT",
            "details": violations
        }
    
    def check_file_organization(self) -> Dict:
        """Check CLAUDE.md file organization rules"""
        violations = []
        
        # Check for files in root that shouldn't be there
        root_files = list(self.root_path.glob("*.py"))
        root_files.extend(list(self.root_path.glob("*.txt")))
        root_files.extend(list(self.root_path.glob("*.md")))
        
        allowed_root = ["CHANGELOG.md", "CLAUDE.md", "README.md", "LICENSE"]
        for file in root_files:
            if file.name not in allowed_root:
                violations.append(f"Root folder violation: {file.name}")
                
        return {
            "rule": "File Organization",
            "violations": len(violations),
            "severity": "MAJOR" if violations else "COMPLIANT",
            "details": violations[:10] if violations else []
        }
    
    def run_compliance_check(self) -> Dict:
        """Run all compliance checks"""
        logger.info("Starting CLAUDE.md compliance check...")
        
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "root_path": str(self.root_path),
            "checks": []
        }
        
        # Run all checks
        checks = [
            self.check_rule_1_fantasy_code,
            self.check_rule_4_duplicates,
            self.check_rule_7_script_organization,
            self.check_rule_8_python_standards,
            self.check_rule_9_single_source,
            self.check_rule_13_waste,
            self.check_rule_18_changelog,
            self.check_rule_20_mcp_protection,
            self.check_file_organization
        ]
        
        total_violations = 0
        critical_count = 0
        
        for check in checks:
            try:
                result = check()
                results["checks"].append(result)
                total_violations += result["violations"]
                if result["severity"] == "CRITICAL":
                    critical_count += 1
                logger.info(f"{result['rule']}: {result['severity']} ({result['violations']} violations)")
            except Exception as e:
                logger.error(f"Check failed: {e}")
                
        # Calculate compliance score
        self.compliance_score = max(0, 100 - (total_violations * 2) - (critical_count * 10))
        
        results["summary"] = {
            "total_violations": total_violations,
            "critical_violations": critical_count,
            "compliance_score": self.compliance_score,
            "status": "FAILED" if self.compliance_score < 70 else "WARNING" if self.compliance_score < 90 else "PASSED"
        }
        
        return results
    
    def save_report(self, results: Dict, output_path: str = None):
        """Save compliance report"""
        if not output_path:
            output_path = self.root_path / "logs" / f"claude_compliance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Report saved to {output_path}")
        
    def generate_github_issue(self, results: Dict) -> str:
        """Generate GitHub issue content for violations"""
        issue_content = f"""## CLAUDE.md Compliance Report

**Date:** {results['timestamp']}
**Compliance Score:** {results['summary']['compliance_score']}/100
**Status:** {results['summary']['status']}

### Violations Summary
- Total Violations: {results['summary']['total_violations']}
- Critical Violations: {results['summary']['critical_violations']}

### Critical Issues Requiring Immediate Attention
"""
        
        for check in results['checks']:
            if check['severity'] == 'CRITICAL':
                issue_content += f"\n#### {check['rule']}\n"
                issue_content += f"- Violations: {check['violations']}\n"
                if check.get('details'):
                    issue_content += "- Examples:\n"
                    details_list = check['details']
                    if isinstance(details_list, list):
                        for detail in details_list[:3]:
                            issue_content += f"  - {detail}\n"
                    elif isinstance(details_list, dict):
                        for key, value in list(details_list.items())[:3]:
                            issue_content += f"  - {key}: {len(value) if isinstance(value, list) else value} items\n"
                        
        issue_content += "\n### Action Required\n"
        issue_content += "1. Review the full compliance report\n"
        issue_content += "2. Address critical violations immediately\n"
        issue_content += "3. Implement automated enforcement to prevent future violations\n"
        
        return issue_content


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='CLAUDE.md Rule Enforcement Monitor')
    parser.add_argument('--root', default='/opt/sutazaiapp', help='Root path to check')
    parser.add_argument('--output', help='Output path for report')
    parser.add_argument('--github-issue', action='store_true', help='Generate GitHub issue content')
    
    args = parser.parse_args()
    
    enforcer = ClaudeRuleEnforcer(args.root)
    results = enforcer.run_compliance_check()
    
    if args.output:
        enforcer.save_report(results, args.output)
    else:
        enforcer.save_report(results)
        
    if args.github_issue:
        issue_content = enforcer.generate_github_issue(results)
        print("\n" + "="*50)
        print("GitHub Issue Content:")
        print("="*50)
        print(issue_content)
        
    # Exit with error code if compliance failed
    if results['summary']['status'] == 'FAILED':
        logger.error("Compliance check FAILED")
        exit(1)
    elif results['summary']['status'] == 'WARNING':
        logger.warning("Compliance check passed with warnings")
        exit(0)
    else:
        logger.info("Compliance check PASSED")
        exit(0)


if __name__ == "__main__":
    main()