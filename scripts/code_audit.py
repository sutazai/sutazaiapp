#!/usr/bin/env python3.11
"""
Comprehensive Code Audit Script for SutazAI

This script performs a thorough audit of the codebase including:
- Security scanning
- Code quality checks
- Dependency analysis
- Performance profiling
- Documentation validation
"""

import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure logging
# Use project relative paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "code_audit.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("SutazAI.CodeAudit")

class CodeAuditor:
    def __init__(self, project_root: str = None):
        if project_root is None:
            project_root = PROJECT_ROOT
        self.project_root = Path(project_root)
        self.venv_path = self.project_root / "venv"
        self.log_dir = self.project_root / "logs"
        self.audit_results: Dict[str, List[Dict]] = {
            "security": [],
            "quality": [],
            "dependencies": [],
            "performance": [],
            "documentation": [],
        }
        
        # Create necessary directories
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def run_security_scan(self) -> None:
        """Run security scanning tools."""
        logger.info("Running security scan...")
        
        # Run Bandit
        try:
            result = subprocess.run(
                ["bandit", "-r", str(self.project_root), "-f", "json"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                self.audit_results["security"] = json.loads(result.stdout)
            else:
                logger.error(f"Bandit scan failed: {result.stderr}")
        except Exception as e:
            logger.error(f"Error running Bandit: {e}")

    def run_quality_checks(self) -> None:
        """Run code quality checks."""
        logger.info("Running quality checks...")
        
        # Run Pylint
        try:
            result = subprocess.run(
                ["pylint", "--output-format=json", str(self.project_root)],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                self.audit_results["quality"] = json.loads(result.stdout)
            else:
                logger.error(f"Pylint check failed: {result.stderr}")
        except Exception as e:
            logger.error(f"Error running Pylint: {e}")

    def check_dependencies(self) -> None:
        """Analyze project dependencies."""
        logger.info("Checking dependencies...")
        
        # Check requirements.txt
        req_file = self.project_root / "requirements.txt"
        if req_file.exists():
            try:
                with open(req_file) as f:
                    requirements = f.readlines()
                self.audit_results["dependencies"].append({
                    "type": "requirements",
                    "file": str(req_file),
                    "dependencies": requirements,
                })
            except Exception as e:
                logger.error(f"Error reading requirements.txt: {e}")

        # Check for outdated packages
        try:
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format=json"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                self.audit_results["dependencies"].extend(json.loads(result.stdout))
            else:
                logger.error(f"Pip outdated check failed: {result.stderr}")
        except Exception as e:
            logger.error(f"Error checking outdated packages: {e}")

    def check_performance(self) -> None:
        """Run performance profiling."""
        logger.info("Running performance checks...")
        
        # Run cProfile on main modules
        main_modules = [
            "backend/main.py",
            "ai_agents/base_agent.py",
            "model_management/model_manager.py",
        ]
        
        for module in main_modules:
            module_path = self.project_root / module
            if module_path.exists():
                try:
                    profile_name = f"profile_{Path(module).stem}.prof"
                    result = subprocess.run(
                        [
                            "python",
                            "-m",
                            "cProfile",
                            "-o",
                            profile_name,
                            str(module_path),
                        ],
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode == 0:
                        self.audit_results["performance"].append({
                            "module": str(module),
                            "profile": profile_name,
                        })
                    else:
                        logger.error(f"Profiling failed for {module}: {result.stderr}")
                except Exception as e:
                    logger.error(f"Error profiling {module}: {e}")

    def check_documentation(self) -> None:
        """Validate documentation completeness."""
        logger.info("Checking documentation...")
        
        # Check for missing docstrings
        try:
            result = subprocess.run(
                ["pydocstyle", "--json", str(self.project_root)],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                self.audit_results["documentation"] = []
                for issue in json.loads(result.stdout):
                    self.audit_results["documentation"].append({
                        "type": "docstring",
                        "code": issue.get("code"),
                        "message": issue.get("message"),
                        "file": issue.get("file"),
                    })
            else:
                logger.error(f"Pydocstyle check failed: {result.stderr}")
        except Exception as e:
            logger.error(f"Error running pydocstyle: {e}")

        # Check README files
        for readme in self.project_root.rglob("README.md"):
            try:
                with open(readme) as f:
                    content = f.read()
                self.audit_results["documentation"].append({
                    "type": "readme",
                    "file": str(readme),
                    "has_content": bool(content.strip()),
                })
            except Exception as e:
                logger.error(f"Error reading {readme}: {e}")

    def generate_report(self) -> None:
        """Generate a comprehensive audit report."""
        report_path = self.log_dir / f"audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            # Ensure log directory exists
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, "w") as f:
                json.dump(self.audit_results, f, indent=2)
            logger.info(f"Audit report generated: {report_path}")
        except Exception as e:
            logger.error(f"Error generating report: {e}")

    def run_audit(self) -> None:
        """Run all audit checks."""
        logger.info("Starting comprehensive code audit...")
        
        self.run_security_scan()
        self.run_quality_checks()
        self.check_dependencies()
        self.check_performance()
        self.check_documentation()
        self.generate_report()
        
        logger.info("Code audit completed")

def main():
    """Main entry point."""
    auditor = CodeAuditor()
    auditor.run_audit()

if __name__ == "__main__":
    main() 