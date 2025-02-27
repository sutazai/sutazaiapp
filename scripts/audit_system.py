#!/usr/bin/env python3
"""
Comprehensive System Audit Script for SutazAI

This script performs an extremely detailed and comprehensive check of the entire application. It verifies the project structure, expected files, virtual environment, dependencies, sensitive keyword scanning, and performs a basic syntax check on all Python files.

The audit includes:
- Directory structure validation
- Critical file existence
- Virtual environment presence
- Comparison of requirements.txt with parked wheel files
- Scanning for sensitive keywords (e.g., JWT, token, password, secret)
- Basic syntax checking for Python files

Usage:
python3 scripts/audit_system.py

Logs are written to logs/audit.log for a detailed report.
"""

import logging
import os
import re
import sys

# Configure logging
LOG_DIR = os.path.join("logs")
    if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
    LOG_FILE = os.path.join(LOG_DIR, "audit.log")
    
    logging.basicConfig(
    level=logging.INFO,
    filename=LOG_FILE,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Also log to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)


    def check_directory_structure():
    logging.info("Checking directory structure...")
    expected_dirs = [
    "ai_agents",
    "model_management",
    "backend",
    "web_ui",
    "scripts",
    "packages",
    "logs",
    "doc_data",
]
missing_dirs = []
    for d in expected_dirs:
        if not os.path.isdir(d):
        missing_dirs.append(d)
        logging.error(f"Missing directory: {d}")
            else:
            logging.info(f"Found directory: {d}")
                if missing_dirs:
                logging.error("One or more essential directories are missing!")
                    else:
                    logging.info("All expected directories are present.")
                    return missing_dirs
                    
                    
                        def check_expected_files():
                        logging.info("Checking expected critical files...")
                        expected_files = [
                        "backend/main.py",
                        "backend/api_routes.py",
                        "requirements.txt",
                        "scripts/deploy.sh",
                        "scripts/setup_repos.sh",
                        "README.md",
                    ]
                    missing_files = []
                        for f in expected_files:
                            if not os.path.isfile(f):
                            missing_files.append(f)
                            logging.error(f"Missing file: {f}")
                                else:
                                logging.info(f"Found file: {f}")
                                    if missing_files:
                                    logging.error("One or more critical files are missing!")
                                        else:
                                        logging.info("All expected critical files are present.")
                                        return missing_files
                                        
                                        
                                            def check_virtualenv():
                                            logging.info("Checking for virtual environment...")
                                                if not os.path.isdir("venv"):
                                                logging.warning("Virtual environment (venv) directory is missing.")
                                                return False
                                                logging.info("Virtual environment found.")
                                                return True
                                                
                                                
                                                    def check_requirements_vs_wheels():
                                                    logging.info("Checking requirements.txt dependencies against packages/wheels...")
                                                        if not os.path.isfile("requirements.txt"):
                                                        logging.error("No requirements.txt file found!")
                                                        return []
                                                            try:
                                                            with open("requirements.txt") as f:
                                                            requirements = [
                                                            line.strip() for line in f if line.strip() and not line.startswith("#")
                                                        ]
                                                        except Exception as e:
                                                        logging.exception(f"Error reading requirements.txt: {e}")
                                                        return []
                                                        
                                                        wheels_dir = os.path.join("packages", "wheels")
                                                            if not os.path.isdir(wheels_dir):
                                                            logging.error("Wheels directory not found in packages/wheels!")
                                                            return requirements
                                                            wheel_files = os.listdir(wheels_dir)
                                                            missing_wheels = []
                                                                for req in requirements:
                                                                pkg_name = re.split("[<>=]", req)[0].strip().lower()
                                                                matched = any(pkg_name in wheel.lower() for wheel in wheel_files)
                                                                    if not matched:
                                                                    missing_wheels.append(req)
                                                                    logging.error(f"No matching wheel found for requirement: {req}")
                                                                        else:
                                                                        logging.info(f"Found wheel for requirement: {req}")
                                                                            if missing_wheels:
                                                                            logging.warning(
                                                                            "Some requirements are missing corresponding wheels in packages/wheels.",
                                                                        )
                                                                            else:
                                                                            logging.info("All requirements have corresponding wheels.")
                                                                            return missing_wheels
                                                                            
                                                                            
                                                                                def search_for_sensitive_keywords():
                                                                                logging.info(
                                                                                "Scanning code for sensitive keywords (e.g., JWT, token, password, secret)...",
                                                                            )
                                                                            sensitive_keywords = ["JWT", "token", "password", "secret"]
                                                                            issues_found = []
                                                                                for root, dirs, files in os.walk("."):
                                                                                # Skip certain directories
                                                                                    if "venv" in root or "node_modules" in root:
                                                                                continue
                                                                                    for file in files:
                                                                                        if file.endswith((".py", ".js", ".ts", ".txt", ".sh", ".md")):
                                                                                        file_path = os.path.join(root, file)
                                                                                            try:
                                                                                            with open(file_path, encoding="utf-8", errors="ignore") as f:
                                                                                            content = f.read()
                                                                                                for keyword in sensitive_keywords:
                                                                                                    if keyword in content:
                                                                                                    logging.warning(
                                                                                                    f"Sensitive keyword '{keyword}' found in {file_path}",
                                                                                                )
                                                                                                issues_found.append((file_path, keyword))
                                                                                                except Exception as e:
                                                                                                logging.exception(f"Error reading file {file_path}: {e}")
                                                                                                    if not issues_found:
                                                                                                    logging.info("No sensitive keywords found in code.")
                                                                                                    return issues_found
                                                                                                    
                                                                                                    
                                                                                                        def syntax_check():
                                                                                                        logging.info("Performing basic syntax check for Python files...")
                                                                                                        issues_found = []
                                                                                                            for root, dirs, files in os.walk("."):
                                                                                                                if "venv" in root or "node_modules" in root:
                                                                                                            continue
                                                                                                                for file in files:
                                                                                                                    if file.endswith(".py"):
                                                                                                                    file_path = os.path.join(root, file)
                                                                                                                        try:
                                                                                                                        with open(file_path, encoding="utf-8", errors="ignore") as f:
                                                                                                                        source = f.read()
                                                                                                                        compile(source, file_path, "exec")
                                                                                                                        logging.info(f"Syntax OK: {file_path}")
                                                                                                                        except Exception as e:
                                                                                                                        logging.exception(f"Syntax error in {file_path}: {e}")
                                                                                                                        issues_found.append((file_path, str(e)))
                                                                                                                            if not issues_found:
                                                                                                                            logging.info("All Python files pass syntax check.")
                                                                                                                            return issues_found
                                                                                                                            
                                                                                                                            
                                                                                                                                def main():
                                                                                                                                logging.info("Starting comprehensive system audit...")
                                                                                                                                missing_dirs = check_directory_structure()
                                                                                                                                missing_files = check_expected_files()
                                                                                                                                venv_exists = check_virtualenv()
                                                                                                                                missing_wheels = check_requirements_vs_wheels()
                                                                                                                                sensitive_issues = search_for_sensitive_keywords()
                                                                                                                                syntax_issues = syntax_check()
                                                                                                                                
                                                                                                                                logging.info("\nAudit Summary:")
                                                                                                                                    if missing_dirs:
                                                                                                                                    logging.info(f"Missing directories: {missing_dirs}")
                                                                                                                                        if missing_files:
                                                                                                                                        logging.info(f"Missing files: {missing_files}")
                                                                                                                                            if not venv_exists:
                                                                                                                                            logging.info("Virtual environment is missing.")
                                                                                                                                                if missing_wheels:
                                                                                                                                                logging.info(f"Missing wheel files for: {missing_wheels}")
                                                                                                                                                    if sensitive_issues:
                                                                                                                                                    logging.info(f"Sensitive keyword issues found: {sensitive_issues}")
                                                                                                                                                        if syntax_issues:
                                                                                                                                                        logging.info(f"Syntax issues found in files: {syntax_issues}")
                                                                                                                                                        
                                                                                                                                                            if (
                                                                                                                                                            missing_dirs
                                                                                                                                                            or missing_files
                                                                                                                                                            or (not venv_exists)
                                                                                                                                                            or missing_wheels
                                                                                                                                                            or sensitive_issues
                                                                                                                                                            or syntax_issues
                                                                                                                                                            ):
                                                                                                                                                            logging.error("Audit completed with issues. Please address the above errors.")
                                                                                                                                                            sys.exit(1)
                                                                                                                                                                else:
                                                                                                                                                                logging.info("Audit completed successfully. System integrity verified.")
                                                                                                                                                                sys.exit(0)
                                                                                                                                                                
                                                                                                                                                                
                                                                                                                                                                    if __name__ == "__main__":
                                                                                                                                                                    main()
                                                                                                                                                                    