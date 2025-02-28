#!/usr/bin/env python3.11
import ast
import os


def fix_syntax_errors(directory):
    """Attempt to fix syntax errors in Python files."""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, encoding="utf-8") as f:
                    source = f.read()

                    # Try parsing the AST
                    ast.parse(source)
                    except SyntaxError as e:
                        print(f"Syntax error in {filepath}: {e}")

                        # Basic error correction strategies
                        corrected_source = correct_common_syntax_errors(source)

                        # Write corrected source
                        with open(filepath, "w", encoding="utf-8") as f:
                        f.write(corrected_source)

                        print(f"Attempted to fix {filepath}")


                        def correct_common_syntax_errors(source):
                            """Apply common syntax error fixes."""
                            # Fix indentation
                            lines = source.split("\n")
                            fixed_lines = []
                            current_indent = 0

                            for line in lines:
                                stripped = line.lstrip()
                                if stripped:
                                    # Detect indentation level changes
                                    if stripped.startswith(
                                        ("def ", "class ", "if ", "for ", "while ", "try:", "except")):
                                        fixed_lines.append(
                                            " " * current_indent + stripped)
                                        current_indent += 4
                                        elif stripped.startswith(
                                            ("return", "break", "continue", "pass")):
                                        fixed_lines.append(
                                            " " * current_indent + stripped)
                                        current_indent = max(
                                            0,
                                            current_indent - 4)
                                        else:
                                        fixed_lines.append(
                                            " " * current_indent + stripped)
                                        else:
                                        fixed_lines.append("")

                                    return "\n".join(fixed_lines)


                                    def main():
                                        """Main execution function."""
                                        import sys

                                        if len(sys.argv) != 2:
                                            print(
                                                "Usage: fix_syntax_errors.py <directory>")
                                            sys.exit(1)

                                            directory = sys.argv[1]
                                            if not os.path.isdir(directory):
                                                print(
                                                    f"Error: {directory} is not a directory")
                                                sys.exit(1)

                                                fix_syntax_errors(directory)


                                                if __name__ == "__main__":
                                                    main()
