#!/usr/bin/env python3
import ast
import io
import logging
import os
import tokenize
from typing import Any, Dict, List, Optional


class SyntaxAnalyzer:
    def __init__(
        self,
        base_path: str = "/opt/sutazaiapp",
        log_file: str = "/opt/sutazaiapp/logs/syntax_analysis.log",
        ):
        """
        Initialize the Syntax Analyzer with logging and base path configuration.

        Args:
        base_path: Root directory to search for Python files
        log_file: Path to the log file for recording analysis results
        """
        self.base_path = base_path

        # Configure logging
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

        # Tracking for analysis results
        self.syntax_issues: Dict[str, List[Dict[str, Any]]] = {}

        def find_python_files(self) -> List[str]:
            """
            Recursively find all Python files in the base path.

            Returns:
            List of Python file paths
            """
            python_files = []
            for root, _, files in os.walk(self.base_path):
            for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
                return python_files

            def analyze_file_syntax(self, file_path: str) -> List[Dict[str, Any]]:
                """
                Perform a comprehensive syntax analysis on a Python file.

                Args:
                file_path: Path to the Python file to analyze

                Returns:
                List of syntax issues found
                """
                issues = []

                try:
                    with open(file_path, encoding="utf-8") as f:
                    source_code = f.read()

                    # AST-based syntax parsing
                    try:
                        ast.parse(source_code)
                        except SyntaxError as e:
                            issues.append(
                            {
                            "type": "AST Syntax Error",
                            "line": e.lineno,
                            "offset": e.offset,
                            "text": str(e),
                            "severity": "high",
                            },
                            )

                            # Tokenization-based analysis
                            try:
                                tokens = list(
                                tokenize.generate_tokens(io.StringIO(source_code).readline),
                                )

                                # Check for indentation inconsistencies
                                last_indent = 0
                                for token in tokens:
                                if token[0] == tokenize.INDENT:
                                    current_indent = len(token[1])
                                    if current_indent - last_indent > 4:
                                        issues.append(
                                        {
                                        "type": "Indentation Inconsistency",
                                        "line": token[2][0],
                                        "text": f"Large indentation jump from {last_indent} to {current_indent}",
                                        "severity": "medium",
                                        },
                                        )
                                        last_indent = current_indent
                                        elif token[0] == tokenize.DEDENT:
                                            last_indent = 0

                                            # Check for potential syntax issues in tokens
                                            for i, token in enumerate(tokens):
                                            if token[0] == tokenize.ERRORTOKEN:
                                                issues.append(
                                                {
                                                "type": "Tokenization Error",
                                                "line": token[2][0],
                                                "text": f"Unexpected token: {token[1]}",
                                                "severity": "high",
                                                },
                                                )

                                                except tokenize.TokenError as e:
                                                    issues.append(
                                                    {
                                                    "type": "Tokenization Error",
                                                    "text": str(e),
                                                    "severity": "high",
                                                    },
                                                    )

                                                    except Exception as e:
                                                        issues.append(
                                                        {
                                                        "type": "File Read Error",
                                                        "text": str(e),
                                                        "severity": "critical",
                                                        },
                                                        )

                                                        return issues

                                                    def analyze_project_syntax(self) -> Dict[str, List[Dict[str, Any]]]:
                                                        """
                                                        Analyze syntax for all Python files in the project.

                                                        Returns:
                                                        Dictionary of files and their syntax issues
                                                        """
                                                        python_files = self.find_python_files()

                                                        for file_path in python_files:
                                                        try:
                                                            file_issues = self.analyze_file_syntax(file_path)
                                                            if file_issues:
                                                                self.syntax_issues[file_path] = file_issues
                                                                self.logger.warning(
                                                                f"Syntax issues found in {file_path}: {file_issues}",
                                                                )
                                                                except Exception as e:
                                                                    self.logger.error(f"Error analyzing {file_path}: {e}")

                                                                    return self.syntax_issues

                                                                def generate_report(self) -> str:
                                                                    """
                                                                    Generate a comprehensive syntax analysis report.

                                                                    Returns:
                                                                    Formatted report string
                                                                    """
                                                                    report = ["Comprehensive Syntax Analysis Report", "=" * 40]

                                                                    if not self.syntax_issues:
                                                                        report.append("No syntax issues found in the project.")
                                                                        return "\n".join(report)

                                                                    report.append(f"Total files with issues: {len(self.syntax_issues)}")

                                                                    for file_path, issues in self.syntax_issues.items():
                                                                    report.append(f"\nFile: {file_path}")
                                                                    report.append("-" * (len(file_path) + 6))

                                                                    for issue in issues:
                                                                    report.append(f"  - Type: {issue['type']}")
                                                                    report.append(f"    Severity: {issue['severity']}")
                                                                    report.append(
                                                                    f"    Details: {issue.get('text', 'No additional details')}",
                                                                    )
                                                                    if "line" in issue:
                                                                        report.append(f"    Line: {issue['line']}")

                                                                        return "\n".join(report)


                                                                    def main():
                                                                        analyzer = SyntaxAnalyzer()
                                                                        analyzer.analyze_project_syntax()

                                                                        report = analyzer.generate_report()
                                                                        print(report)

                                                                        # Optionally, write the report to a file
                                                                        report_path = "/opt/sutazaiapp/logs/syntax_analysis_report.txt"
                                                                        with open(report_path, "w") as f:
                                                                        f.write(report)

                                                                        print(f"\nDetailed report saved to {report_path}")


                                                                        if __name__ == "__main__":
                                                                            main()
