import ast
import io
import os
import re
import tokenize


class SyntaxDiagnostic:
    def __init__(self, directory):
        self.directory = directory
        self.issues = {}

    def analyze_file(self, file_path):
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            file_issues = []

            # 1. AST Parsing Check
            try:
                ast.parse(content)
            except SyntaxError as e:
                file_issues.append(
                    {
                        "type": "AST Parsing Error",
                        "details": str(e),
                        "line": e.lineno,
                        "offset": e.offset,
                    },
                )

            # 2. Tokenization Check
            try:
                tokens = list(tokenize.generate_tokens(io.StringIO(content).readline))
                error_tokens = [
                    token for token in tokens if token[0] == tokenize.ERRORTOKEN
                ]
                if error_tokens:
                    file_issues.append(
                        {
                            "type": "Tokenization Error",
                            "details": f"Found {len(error_tokens)} error tokens",
                            "tokens": error_tokens,
                        },
                    )
            except tokenize.TokenError as e:
                file_issues.append({"type": "Tokenization Error", "details": str(e)})

            # 3. Common Syntax Pattern Checks
            pattern_checks = [
                (r"\(\s*{", "Misplaced opening brace after parenthesis"),
                (r"}\s*\)", "Misplaced closing brace before parenthesis"),
                (r"def\s+\w+\s*\(\s*\),", "Incorrect method signature"),
                (
                    r"=\s*\(\d+\)",
                    "Unnecessary parentheses in numeric assignment",
                ),
            ]

            for pattern, description in pattern_checks:
                matches = list(re.finditer(pattern, content))
                if matches:
                    file_issues.append(
                        {
                            "type": "Syntax Pattern Issue",
                            "details": description,
                            "matches": [match.group() for match in matches],
                        },
                    )

            # 4. Method Signature Analysis
            method_sig_issues = self._analyze_method_signatures(content)
            if method_sig_issues:
                file_issues.extend(method_sig_issues)

            if file_issues:
                self.issues[file_path] = file_issues

        except (OSError, FileNotFoundError, PermissionError) as e:
            self.issues[file_path] = [{"type": "File Read Error", "details": str(e)}]

    def _analyze_method_signatures(self, content):
        method_issues = []

        # Look for methods without 'self' parameter
        method_def_pattern = r"def\s+(\w+)\s*\(([^)]*)\):"
        method_defs = list(re.finditer(method_def_pattern, content))

        for match in method_defs:
            method_name = match.group(1)
            params = match.group(2).split(",")

            # Check if method is part of a class (indentation matters)
            method_line = match.group(0)
            if method_line.startswith("    def"):  # Simple heuristic for class method
                if not any("self" in param.strip() for param in params):
                    method_issues.append(
                        {
                            "type": "Method Signature Issue",
                            "details": f"Method {method_name} missing self parameter",
                            "line": method_line,
                        },
                    )

        return method_issues

    def generate_report(self):
        report = "Syntax Diagnostic Report\n"
        report += "=" * 30 + "\n\n"

        if not self.issues:
            report += "No syntax issues found.\n"
        else:
            for file_path, file_issues in self.issues.items():
                report += f"File: {file_path}\n"
                report += "-" * 40 + "\n"
                for issue in file_issues:
                    report += f"Type: {issue['type']}\n"
                    report += f"Details: {issue['details']}\n"
                    if "line" in issue:
                        report += f"Line: {issue['line']}\n"
                    report += "\n"

        return report

    def process_directory(self):
        for root, _, files in os.walk(self.directory):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    self.analyze_file(file_path)


def main():
    diagnostic = SyntaxDiagnostic("/opt/sutazai_project/SutazAI/core_system")
    diagnostic.process_directory()

    report = diagnostic.generate_report()
    print(report)

    # Optional: Write report to file
    with open("/opt/sutazai_project/SutazAI/syntax_diagnostic_report.txt", "w") as f:
        f.write(report)


if __name__ == "__main__":
    main()
