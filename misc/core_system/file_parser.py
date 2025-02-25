#!/usr/bin/env python3
import ast
import logging
import os


class RobustFileParser:
    """
    A robust file parsing utility to handle various encoding and syntax challenges.
    """

    def __init__(
        self, base_path: str = "/media/ai/SutazAI_Storage/SutazAI/v1"
    ):
        """
        Initialize the robust file parser.

        Args:
            base_path (str): Base path to search for Python files
        """
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s: %(message)s",
            handlers=[
                logging.FileHandler("/var/log/sutazai/file_parsing.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

        self.base_path = base_path
        self.encodings = [
            "utf-8",
            "latin-1",
            "iso-8859-1",
            "cp1252",
            "utf-16",
            "ascii",
        ]

        self.parsing_report = {
            "total_files": 0,
            "parsed_files": 0,
            "failed_files": [],
            "encoding_attempts": {},
        }

    def parse_python_files(self) -> dict:
        """
        Parse all Python files in the base path, handling various encoding challenges.

        Returns:
            dict: Comprehensive parsing report
        """
        for root, _, files in os.walk(self.base_path):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    self._parse_file(file_path)

        self._generate_report()
        return self.parsing_report

    def _parse_file(self, file_path: str):
        """
        Attempt to parse a single Python file with multiple encoding strategies.

        Args:
            file_path (str): Path to the Python file
        """
        self.parsing_report["total_files"] += 1

        for encoding in self.encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    content = f.read()

                    # Track encoding attempts
                    self.parsing_report["encoding_attempts"].setdefault(
                        file_path, []
                    ).append(encoding)

                    try:
                        # Attempt to parse the AST
                        ast.parse(content)

                        # If parsing succeeds
                        self.parsing_report["parsed_files"] += 1
                        self.logger.info(
                            f"Successfully parsed {file_path} with {encoding} encoding"
                        )
                        break

                    except SyntaxError as syntax_error:
                        # Log specific syntax errors
                        self.logger.warning(
                            f"Syntax error in {file_path} with {encoding} encoding: "
                            f"Line {syntax_error.lineno}, {syntax_error.text}"
                        )
                        continue

            except UnicodeDecodeError:
                # If decoding fails, continue to next encoding
                continue

        else:
            # If no encoding works
            self.parsing_report["failed_files"].append(file_path)
            self.logger.error(f"Could not parse {file_path} with any encoding")

    def _generate_report(self):
        """Generate a comprehensive parsing report."""
        report_path = "/var/log/sutazai/file_parsing_report.json"

        with open(report_path, "w") as f:
            import json

            json.dump(self.parsing_report, f, indent=2)

        print("\n🔍 File Parsing Report 🔍")
        print(f"Total Files: {self.parsing_report['total_files']}")
        print(f"Parsed Files: {self.parsing_report['parsed_files']}")
        print(f"Failed Files: {len(self.parsing_report['failed_files'])}")

        if self.parsing_report["failed_files"]:
            print("\n❌ Files that could not be parsed:")
            for failed_file in self.parsing_report["failed_files"][
                :10
            ]:  # Limit to first 10
                print(f"  - {failed_file}")


def main():
    parser = RobustFileParser()
    parser.parse_python_files()


if __name__ == "__main__":
    main()
