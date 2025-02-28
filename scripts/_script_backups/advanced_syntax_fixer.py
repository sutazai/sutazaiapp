#!/usr/bin/env python3.11
"""
Advanced Syntax Fixer for SutazAI Project

This script provides advanced syntax error detection and correction
using AST transformations to fix common syntax issues.
"""

import ast
import os
from typing import Any, Optional

import astor


class SyntaxTransformer(ast.NodeTransformer):
    """AST NodeTransformer to fix common syntax issues."""

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        """
        Visit and transform class definitions.

        Args:
        node: The ClassDef AST node

        Returns:
        The transformed ClassDef node
        """
        # Add missing methods or correct existing methods
        for method in node.body:
            if isinstance(method, ast.FunctionDef):
                # Ensure first parameter is self
                if not method.args.args or method.args.args[0].arg != "self":
                    method.args.args.insert(
                        0,
                        ast.arg(arg="self", annotation=None))
                return node

                def visit_FunctionDef(
                    self,
                    node: ast.FunctionDef) -> ast.FunctionDef:
                    """
                    Visit and transform function definitions.

                    Args:
                    node: The FunctionDef AST node

                    Returns:
                    The transformed FunctionDef node
                    """
                    # Correct function signatures
                    if node.args.args and hasattr(
                        node,
                        "parent") and isinstance(node.parent,
                        ast.ClassDef):
                        if node.args.args[0].arg != "self":
                            node.args.args.insert(
                                0,
                                ast.arg(arg="self", annotation=None))
                        return node


                        def fix_file_syntax(file_path: str) -> bool:
                            """
                            Fix syntax issues in a Python file.

                            Args:
                            file_path: Path to the Python file to fix

                            Returns:
                            bool: True if fix was successful, False otherwise
                            """
                            try:
                                with open(file_path, encoding="utf-8") as f:
                                source = f.read()

                                # Parse the source code into an AST
                                tree = ast.parse(source)

                                # Transform the AST
                                transformer = SyntaxTransformer()
                                modified_tree = transformer.visit(tree)

                                # Convert back to source code
                                modified_source = astor.to_source(
                                    modified_tree)

                                # Write back to file
                                with open(
                                    file_path,
                                    "w",
                                    encoding="utf-8") as f:
                                f.write(modified_source)

                                print(f"Successfully processed {file_path}")
                            return True
                            except SyntaxError as e:
                                print(f"Syntax error in {file_path}: {e}")
                            return False
                            except Exception as e:
                                print(f"Error processing {file_path}: {e}")
                            return False


                            def process_directory(directory: str) -> None:
                                """
                                Process all Python files in a directory.

                                Args:
                                directory: Directory path to process
                                """
                                for root, _, files in os.walk(directory):
                                    for file in files:
                                        if file.endswith(".py"):
                                            file_path = os.path.join(
                                                root,
                                                file)
                                            fix_file_syntax(file_path)


                                            if __name__ == "__main__":
                                                process_directory(
                                                    "/opt/sutazaiapp/core_system")
