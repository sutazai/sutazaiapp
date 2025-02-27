#!/usr/bin/env python3.11
"""
SutazAI Comprehensive System Review Script

This script performs an in-depth analysis of the entire SutazAI application,
examining code quality, performance, security, and system architecture.
"""

import ast
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List

# Configure logging
logging.basicConfig(
level=logging.INFO,
format="%(asctime)s - %(levelname)s: %(message)s",
handlers=[
logging.FileHandler(
"/opt/sutazaiapp/logs/comprehensive_system_review.log",
),
logging.StreamHandler(sys.stdout),
],
)
logger = logging.getLogger("SutazAI.SystemReview")


    class ComprehensiveSystemReviewer:
    """Comprehensive system review framework"""
    
        def __init__(self, base_path: str = "/opt/sutazaiapp"):
        self.base_path = base_path
        self.log_dir = os.path.join(base_path, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
            def find_python_files(self) -> List[str]:
            """Find all Python files in project"""
            python_files = []
                for root, _, files in os.walk(self.base_path):
                    for file in files:
                        if file.endswith(".py"):
                        python_files.append(os.path.join(root, file))
                        return python_files
                        
                            def analyze_code_structure(self, file_path: str) -> Dict[str, Any]:
                            """
                            Perform a detailed analysis of a Python file's structure.
                            
                            Args:
                            file_path: Path to the Python file
                            
                            Returns:
                            Dictionary containing code structure details
                            """
                                try:
                                with open(file_path) as f:
                                source = f.read()
                                
                                # Parse the source code
                                tree = ast.parse(source)
                                
                                # Analyze code structure
                                structure = {
                                "filename": os.path.basename(file_path),
                                "classes": [],
                                "functions": [],
                                "imports": [],
                                "complexity": {
                                "total_lines": len(source.splitlines()),
                                "class_count": 0,
                                "function_count": 0,
                                "import_count": 0,
                                },
                            }
                            
                            # Traverse the AST
                                for node in ast.walk(tree):
                                    if isinstance(node, ast.ClassDef):
                                    class_info = {
                                    "name": node.name,
                                    "methods": [
                                    method.name
                                        for method in node.body
                                            if isinstance(method, ast.FunctionDef)
                                            ],
                                        }
                                        structure["classes"].append(class_info)
                                        structure["complexity"]["class_count"] += 1
                                        
                                            elif isinstance(node, ast.FunctionDef):
                                            structure["functions"].append(
                                            {
                                            "name": node.name,
                                            "args": [arg.arg for arg in node.args.args],
                                            "line_number": node.lineno,
                                        }
                                    )
                                    structure["complexity"]["function_count"] += 1
                                    
                                        elif isinstance(node, (ast.Import, ast.ImportFrom)):
                                            if isinstance(node, ast.Import):
                                            imports = [alias.name for alias in node.names]
                                                else:
                                                imports = [
                                                f"{node.module}.{alias.name}" for alias in node.names
                                            ]
                                            structure["imports"].extend(imports)
                                            structure["complexity"]["import_count"] += len(imports)
                                            
                                            return structure
                                            
                                            except Exception as e:
                                            logger.error(f"Error analyzing {file_path}: {e}")
                                            return {}
                                            
                                                def detect_code_smells(self, file_path: str) -> List[Dict[str, Any]]:
                                                """
                                                Detect potential code smells and anti-patterns.
                                                
                                                Args:
                                                file_path: Path to the Python file
                                                
                                                Returns:
                                                List of detected code smells
                                                """
                                                code_smells = []
                                                
                                                    try:
                                                    with open(file_path) as f:
                                                    source_lines = f.readlines()
                                                    
                                                    # Detect long methods/functions
                                                        for i, line in enumerate(source_lines):
                                                            if "def " in line:
                                                            method_lines = 0
                                                                for j in range(i + 1, len(source_lines)):
                                                                    if source_lines[j].strip() and not source_lines[j].startswith(
                                                                    " "
                                                                    ):
                                                                break
                                                                method_lines += 1
                                                                
                                                                    if method_lines > 50:  # Arbitrary threshold
                                                                    code_smells.append(
                                                                    {
                                                                    "type": "long_method",
                                                                    "line": i + 1,
                                                                    "description": (
                                                                    f"Method is too long " f"({method_lines} lines)"
                                                                    ),
                                                                }
                                                            )
                                                            
                                                            # Detect duplicate code
                                                            source_code = "".join(source_lines)
                                                            duplicates = self._find_duplicate_code(source_code)
                                                            code_smells.extend(duplicates)
                                                            
                                                            # Detect overly complex conditionals
                                                            conditionals = self._detect_complex_conditionals(source_code)
                                                            code_smells.extend(conditionals)
                                                            
                                                            except Exception as e:
                                                            logger.error(f"Error detecting code smells in {file_path}: {e}")
                                                            
                                                            return code_smells
                                                            
                                                                def _find_duplicate_code(self, source_code: str) -> List[Dict[str, Any]]:
                                                                """
                                                                Find duplicate code segments.
                                                                
                                                                Args:
                                                                source_code: Full source code as a string
                                                                
                                                                Returns:
                                                                List of duplicate code segments
                                                                """
                                                                duplicates = []
                                                                lines = source_code.split("\n")
                                                                line_count = len(lines)
                                                                
                                                                    for i in range(line_count):
                                                                    # Look for duplicates at least 5 lines apart
                                                                        for j in range(i + 5, line_count):
                                                                        # Compare 3-line segments
                                                                        segment_length = 3
                                                                            if j + segment_length > line_count:
                                                                        break
                                                                        
                                                                        segment1 = lines[i : i + segment_length]
                                                                        segment2 = lines[j : j + segment_length]
                                                                        
                                                                        # Remove whitespace and comments for comparison
                                                                        clean_segment1 = [
                                                                        re.sub(r"#.*$", "", line).strip() for line in segment1
                                                                    ]
                                                                    clean_segment2 = [
                                                                    re.sub(r"#.*$", "", line).strip() for line in segment2
                                                                ]
                                                                
                                                                    if clean_segment1 == clean_segment2 and any(clean_segment1):
                                                                    duplicates.append(
                                                                    {
                                                                    "type": "duplicate_code",
                                                                    "first_occurrence": i + 1,
                                                                    "second_occurrence": j + 1,
                                                                    "segment": segment1,
                                                                }
                                                            )
                                                            
                                                            return duplicates
                                                            
                                                                def _detect_complex_conditionals(self, source_code: str) -> List[Dict[str, Any]]:
                                                                """
                                                                Detect overly complex conditional statements.
                                                                
                                                                Args:
                                                                source_code: Full source code as a string
                                                                
                                                                Returns:
                                                                List of complex conditional statements
                                                                """
                                                                complex_conditionals = []
                                                                
                                                                # Regex to find complex if/elif/else statements
                                                                complex_if_pattern = re.compile(
                                                                r"(if|elif)\s*\(.*\band\b.*\bor\b.*\):|"
                                                                r"(if|elif)\s*\(.*\bor\b.*\band\b.*\):",
                                                            )
                                                            
                                                                for match in complex_if_pattern.finditer(source_code):
                                                                complex_conditionals.append(
                                                                {
                                                                "type": "complex_conditional",
                                                                "line": source_code[: match.start()].count("\n") + 1,
                                                                "description": (
                                                                "Complex conditional with multiple " "AND/OR operators"
                                                                ),
                                                            }
                                                        )
                                                        
                                                        return complex_conditionals
                                                        
                                                            def run_linters(self, file_path: str) -> Dict[str, Any]:
                                                            """
                                                            Run comprehensive linting on a Python file.
                                                            
                                                            Args:
                                                            file_path: Path to the Python file
                                                            
                                                            Returns:
                                                            Linting results
                                                            """
                                                                try:
                                                                # Run pylint
                                                                pylint_result = subprocess.run(
                                                                ["pylint", file_path],
                                                                capture_output=True,
                                                                text=True,
                                                                check=False,
                                                            )
                                                            
                                                            # Run mypy
                                                            mypy_result = subprocess.run(
                                                            ["mypy", file_path],
                                                            capture_output=True,
                                                            text=True,
                                                            check=False,
                                                        )
                                                        
                                                        # Run bandit for security checks
                                                        bandit_result = subprocess.run(
                                                        ["bandit", "-r", file_path],
                                                        capture_output=True,
                                                        text=True,
                                                        check=False,
                                                    )
                                                    
                                                    return {
                                                    "pylint": pylint_result.stdout,
                                                    "mypy": mypy_result.stdout,
                                                    "bandit": bandit_result.stdout,
                                                }
                                                
                                                except Exception as e:
                                                logger.error(f"Error running linters on {file_path}: {e}")
                                                return {}
                                                
                                                    def comprehensive_system_review(self) -> Dict[str, Any]:
                                                    """
                                                    Perform a comprehensive review of the entire system.
                                                    
                                                    Returns:
                                                    Comprehensive system review report
                                                    """
                                                    # Find all Python files
                                                    python_files = self.find_python_files()
                                                    
                                                    # Comprehensive review report
                                                    system_review_report = {
                                                    "timestamp": datetime.now().isoformat(),
                                                    "total_files": len(python_files),
                                                    "code_structure": [],
                                                    "code_smells": [],
                                                    "linting_results": {},
                                                }
                                                
                                                # Analyze each Python file
                                                    for file_path in python_files:
                                                    # Analyze code structure
                                                    structure = self.analyze_code_structure(file_path)
                                                    system_review_report["code_structure"].append(structure)
                                                    
                                                    # Detect code smells
                                                    code_smells = self.detect_code_smells(file_path)
                                                    system_review_report["code_smells"].extend(code_smells)
                                                    
                                                    # Run linters
                                                    linting_results = self.run_linters(file_path)
                                                    system_review_report["linting_results"][file_path] = linting_results
                                                    
                                                    # Generate report file
                                                    report_path = os.path.join(
                                                    self.log_dir,
                                                    f'system_review_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
                                                )
                                                with open(report_path, "w") as f:
                                                json.dump(system_review_report, f, indent=2)
                                                
                                                logger.info(f"Comprehensive system review report saved to {report_path}")
                                                return system_review_report
                                                
                                                    def generate_summary_report(self, review_report: Dict[str, Any]) -> None:
                                                    """
                                                    Generate a human-readable summary report.
                                                    
                                                    Args:
                                                    review_report: Comprehensive system review report
                                                    """
                                                    summary_path = os.path.join(
                                                    self.log_dir,
                                                    f'system_review_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md',
                                                )
                                                
                                                with open(summary_path, "w") as f:
                                                f.write("# SutazAI Comprehensive System Review Summary\n\n")
                                                f.write(f"**Timestamp:** {review_report['timestamp']}\n\n")
                                                
                                                # Code Structure Overview
                                                f.write("## Code Structure Overview\n")
                                                f.write(f"- **Total Python Files:** {review_report['total_files']}\n")
                                                
                                                # Code Smells
                                                f.write("\n## Code Smells and Anti-Patterns\n")
                                                code_smells = review_report["code_smells"]
                                                f.write(f"- **Total Code Smells:** {len(code_smells)}\n")
                                                
                                                # Categorize code smells
                                                smell_types = {}
                                                    for smell in code_smells:
                                                    smell_type = smell.get("type", "unknown")
                                                    smell_types[smell_type] = smell_types.get(smell_type, 0) + 1
                                                    
                                                        for smell_type, count in smell_types.items():
                                                        f.write(
                                                        f"  - {smell_type.replace('_', ' ').title()}: {count}\n",
                                                    )
                                                    
                                                    # Linting Results Summary
                                                    f.write("\n## Linting Results Summary\n")
                                                    linting_results = review_report["linting_results"]
                                                    
                                                    pylint_issues = sum(
                                                    len(result.get("pylint", "").splitlines())
                                                        for result in linting_results.values()
                                                    )
                                                    mypy_issues = sum(
                                                    len(result.get("mypy", "").splitlines())
                                                        for result in linting_results.values()
                                                    )
                                                    bandit_issues = sum(
                                                    len(result.get("bandit", "").splitlines())
                                                        for result in linting_results.values()
                                                    )
                                                    
                                                    f.write(f"- **Pylint Issues:** {pylint_issues}\n")
                                                    f.write(f"- **MyPy Type Checking Issues:** {mypy_issues}\n")
                                                    f.write(f"- **Bandit Security Issues:** {bandit_issues}\n")
                                                    
                                                    logger.info(f"System review summary generated at {summary_path}")
                                                    
                                                        def main(self) -> None:
                                                        """
                                                        Execute comprehensive system review
                                                        """
                                                        logger.info("Starting comprehensive system review")
                                                        
                                                        # Perform system review
                                                        review_report = self.comprehensive_system_review()
                                                        
                                                        # Generate summary report
                                                        self.generate_summary_report(review_report)
                                                        
                                                        logger.info("Comprehensive system review completed successfully")
                                                        
                                                        
                                                            def main():
                                                            """
                                                            Main entry point for system review
                                                            """
                                                            reviewer = ComprehensiveSystemReviewer()
                                                            reviewer.main()
                                                            
                                                            
                                                                if __name__ == "__main__":
                                                                main()
                                                                