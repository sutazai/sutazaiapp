#!/usr/bin/env python3

"""
Verification script to check if test issues have been fixed.
This script analyzes the test files and reports any remaining issues.
"""

import os
import re
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("verification.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("verify_fixes")

# Base directory
BASE_DIR = Path("/opt/sutazaiapp")

class FixVerifier:
    """Class to verify that fixes have been applied correctly."""
    
    def __init__(self):
        """Initialize the verifier."""
        self.results = {
            "indentation_and_decorators": {"status": "Not checked", "details": []},
            "coroutine_warnings": {"status": "Not checked", "details": []},
            "sync_exception_test": {"status": "Not checked", "details": []},
            "pytest_config": {"status": "Not checked", "details": []}
        }
    
    def verify_indentation_and_decorators(self):
        """Verify that indentation and decorator issues have been fixed."""
        logger.info("Verifying indentation and decorator fixes...")
        
        tests_dir = BASE_DIR / "tests"
        if not tests_dir.exists():
            logger.error("Tests directory not found")
            self.results["indentation_and_decorators"]["status"] = "Failed"
            self.results["indentation_and_decorators"]["details"].append("Tests directory not found")
            return
        
        issues_found = 0
        files_checked = 0
        
        for root, _, files in os.walk(tests_dir):
            for file in files:
                if file.endswith(".py") and file.startswith("test_"):
                    files_checked += 1
                    file_path = Path(root) / file
                    
                    with open(file_path, "r") as f:
                        content = f.read()
                    
                    # Check for duplicate decorators
                    if re.search(r'@pytest\.mark\.asyncio\s+@pytest\.mark\.asyncio', content):
                        issues_found += 1
                        self.results["indentation_and_decorators"]["details"].append(
                            f"Duplicate decorators found in {file_path.relative_to(BASE_DIR)}"
                        )
                    
                    # Check for indentation issues in decorator lines
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if '@pytest.mark.asyncio' in line:
                            # Check if the next line is a method definition
                            if i + 1 < len(lines) and re.match(r'\s+(async )?def test_', lines[i + 1]):
                                # Get indentation of the method definition
                                method_indent_match = re.match(r'^(\s+)', lines[i + 1])
                                if method_indent_match:
                                    method_indent = method_indent_match.group(1)
                                    # Check if decorator has the same indentation
                                    decorator_indent_match = re.match(r'^(\s+)', line)
                                    if not decorator_indent_match or decorator_indent_match.group(1) != method_indent:
                                        issues_found += 1
                                        self.results["indentation_and_decorators"]["details"].append(
                                            f"Indentation issue at line {i+1} in {file_path.relative_to(BASE_DIR)}"
                                        )
        
        if issues_found == 0 and files_checked > 0:
            self.results["indentation_and_decorators"]["status"] = "Passed"
            self.results["indentation_and_decorators"]["details"].append(f"Checked {files_checked} files, no issues found")
        else:
            self.results["indentation_and_decorators"]["status"] = "Failed"
            self.results["indentation_and_decorators"]["details"].append(f"Found {issues_found} issues in {files_checked} files")
    
    def verify_coroutine_warnings(self):
        """Verify that coroutine warnings have been fixed in agent_manager.py."""
        logger.info("Verifying coroutine warning fixes...")
        
        file_path = BASE_DIR / "core_system" / "orchestrator" / "agent_manager.py"
        if not file_path.exists():
            logger.error("agent_manager.py not found")
            self.results["coroutine_warnings"]["status"] = "Failed"
            self.results["coroutine_warnings"]["details"].append("agent_manager.py not found")
            return
        
        with open(file_path, "r") as f:
            content = f.read()
        
        # Check for non-awaited coroutines
        pattern = r'(?<!await\s)(?<!=\s)self\.(notify_agent_status|update_agent_status|notify_job_complete|notify_job_failed|notify_job_status)\('
        matches = re.findall(pattern, content)
        
        if len(matches) == 0:
            self.results["coroutine_warnings"]["status"] = "Passed"
            self.results["coroutine_warnings"]["details"].append("No unawaited coroutines found")
        else:
            self.results["coroutine_warnings"]["status"] = "Failed"
            self.results["coroutine_warnings"]["details"].append(f"Found {len(matches)} unawaited coroutines")
            for match in matches[:5]:  # Limit to first 5 for brevity
                self.results["coroutine_warnings"]["details"].append(f"Unawaited coroutine: {match}")
    
    def verify_sync_exception_test(self):
        """Verify that the sync_exception test has been fixed."""
        logger.info("Verifying sync exception test fix...")
        
        file_path = BASE_DIR / "tests" / "test_sync_manager_complete_coverage.py"
        if not file_path.exists():
            logger.error("test_sync_manager_complete_coverage.py not found")
            self.results["sync_exception_test"]["status"] = "Failed"
            self.results["sync_exception_test"]["details"].append("test_sync_manager_complete_coverage.py not found")
            return
        
        with open(file_path, "r") as f:
            content = f.read()
        
        # Check if the test_sync_exception method exists and has the correct implementation
        if re.search(r'def test_sync_exception\(self, sync_manager\):', content) and \
           re.search(r'sync_manager\.exception_handler\s*=\s*MagicMock\(\)', content) and \
           re.search(r'sync_manager\.sync_exception\(test_exception\)', content) and \
           re.search(r'sync_manager\.exception_handler\.assert_called_once_with\(test_exception\)', content):
            self.results["sync_exception_test"]["status"] = "Passed"
            self.results["sync_exception_test"]["details"].append("test_sync_exception method has correct implementation")
        else:
            self.results["sync_exception_test"]["status"] = "Failed"
            self.results["sync_exception_test"]["details"].append("test_sync_exception method has incorrect implementation")
    
    def verify_pytest_config(self):
        """Verify that pytest is properly configured."""
        logger.info("Verifying pytest configuration...")
        
        # Check pyproject.toml
        pyproject_path = BASE_DIR / "pyproject.toml"
        if not pyproject_path.exists():
            logger.error("pyproject.toml not found")
            self.results["pytest_config"]["status"] = "Failed"
            self.results["pytest_config"]["details"].append("pyproject.toml not found")
            return
        
        with open(pyproject_path, "r") as f:
            pyproject_content = f.read()
        
        # Check if pytest configuration exists in pyproject.toml
        if "[tool.pytest.ini_options]" in pyproject_content and \
           "asyncio_mode" in pyproject_content:
            pyproject_ok = True
            self.results["pytest_config"]["details"].append("pyproject.toml has pytest configuration")
        else:
            pyproject_ok = False
            self.results["pytest_config"]["details"].append("pyproject.toml missing pytest configuration")
        
        # Check conftest.py
        conftest_path = BASE_DIR / "tests" / "conftest.py"
        if not conftest_path.exists():
            logger.error("conftest.py not found")
            self.results["pytest_config"]["status"] = "Failed"
            self.results["pytest_config"]["details"].append("conftest.py not found")
            return
        
        with open(conftest_path, "r") as f:
            conftest_content = f.read()
        
        # Check if conftest.py has necessary fixtures
        if "def sync_manager" in conftest_content and \
           "def agent_manager" in conftest_content:
            conftest_ok = True
            self.results["pytest_config"]["details"].append("conftest.py has necessary fixtures")
        else:
            conftest_ok = False
            self.results["pytest_config"]["details"].append("conftest.py missing necessary fixtures")
        
        # Overall status
        if pyproject_ok and conftest_ok:
            self.results["pytest_config"]["status"] = "Passed"
        else:
            self.results["pytest_config"]["status"] = "Failed"
    
    def run_all_verifications(self):
        """Run all verification checks."""
        self.verify_indentation_and_decorators()
        self.verify_coroutine_warnings()
        self.verify_sync_exception_test()
        self.verify_pytest_config()
        
        # Generate overall status
        passed = sum(1 for check in self.results.values() if check["status"] == "Passed")
        total = len(self.results)
        
        logger.info(f"Verification complete: {passed}/{total} checks passed")
        
        # Generate verification report
        report = {
            "summary": {
                "checks_passed": passed,
                "total_checks": total,
                "percentage": round(passed / total * 100) if total > 0 else 0
            },
            "details": self.results
        }
        
        # Save report to file
        report_path = BASE_DIR / "verification_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        # Generate markdown report
        markdown_report = self.generate_markdown_report(report)
        markdown_path = BASE_DIR / "VERIFICATION_REPORT.md"
        with open(markdown_path, "w") as f:
            f.write(markdown_report)
        
        logger.info(f"Verification report saved to {report_path} and {markdown_path}")
        
        return passed == total
    
    def generate_markdown_report(self, report):
        """Generate a markdown report from the verification results."""
        md = "# Verification Report\n\n"
        
        # Add summary
        summary = report["summary"]
        md += f"## Summary\n\n"
        md += f"- **Checks Passed**: {summary['checks_passed']}/{summary['total_checks']}\n"
        md += f"- **Percentage**: {summary['percentage']}%\n\n"
        
        # Add status emoji for each check
        status_emoji = {"Passed": "✅", "Failed": "❌", "Not checked": "❓"}
        
        # Add details for each verification
        md += "## Details\n\n"
        
        for check_name, check_result in report["details"].items():
            status = check_result["status"]
            emoji = status_emoji.get(status, "❓")
            
            md += f"### {check_name.replace('_', ' ').title()} {emoji}\n\n"
            md += f"**Status**: {status}\n\n"
            
            if check_result["details"]:
                md += "**Details**:\n\n"
                for detail in check_result["details"]:
                    md += f"- {detail}\n"
            
            md += "\n"
        
        # Add timestamp
        from datetime import datetime
        md += f"\n\n*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
        
        return md

def main():
    """Main function to run the verification process."""
    logger.info("Starting verification process...")
    
    verifier = FixVerifier()
    all_passed = verifier.run_all_verifications()
    
    if all_passed:
        logger.info("All checks passed! The codebase is properly fixed.")
        return 0
    else:
        logger.warning("Some checks failed. See verification report for details.")
        return 1

if __name__ == "__main__":
    main() 