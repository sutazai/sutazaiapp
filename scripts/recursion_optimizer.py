#!/usr/bin/env python3
"""
Recursion Depth Optimizer for Python 3.11

This script provides advanced recursion analysis and optimization:
- Identifies functions with excessive recursion
- Converts recursive functions to iterative or tail-recursive forms
- Generates optimization recommendations
"""

import ast
import logging
import os
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecursionOptimizer(ast.NodeTransformer):
    """
    A comprehensive recursion analysis and optimization utility.
    """

    def __init__(self):
        self.recursive_functions: Dict[str, int] = {}
        self.optimization_suggestions: List[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        """
        Analyze function for recursion depth and potential optimizations.
        """
        # Count recursive calls
        recursive_calls = self._count_recursive_calls(node)

        if recursive_calls > 0:
            self.recursive_functions[node.name] = recursive_calls

            # Suggest optimization if recursion depth is high
            if recursive_calls > 10:
                suggestion = (
                    f"Function '{node.name}' has {recursive_calls} "
                    "recursive calls. Consider converting to "
                    "iterative or tail-recursive form."
                )
                self.optimization_suggestions.append(suggestion)

        return self.generic_visit(node)

    def _count_recursive_calls(self, node: ast.FunctionDef) -> int:
        """
        Count the number of recursive calls within a function.
        """
        recursive_calls = 0
        for descendant in ast.walk(node):
            is_recursive_call = (
                isinstance(descendant, ast.Call)
                and isinstance(descendant.func, ast.Name)
                and descendant.func.id == node.name
            )
            if is_recursive_call:
                recursive_calls += 1

        return recursive_calls

    def optimize_recursive_function(self, node: ast.FunctionDef) -> Optional[ast.FunctionDef]:
        """
        Attempt to convert a recursive function to an iterative form.

        This is a simplified transformation that may require manual review.
        """
        # Basic tail recursion elimination
        if len(node.body) > 0 and isinstance(node.body[-1], ast.Return):
            last_stmt = node.body[-1]
            is_self_recursive_call = (
                isinstance(last_stmt.value, ast.Call)
                and isinstance(last_stmt.value.func, ast.Name)
                and last_stmt.value.func.id == node.name
            )
            if is_self_recursive_call:
                # Convert to while loop
                while_body = node.body[:-1]
                while_node = ast.While(test=ast.Constant(value=True), body=while_body, orelse=[])

                node.body = [while_node]

                logger.info(f"Converted {node.name} to iterative form")
                return node

        return None


def process_file(file_path: str) -> Dict[str, Any]:
    """
    Process a single Python file for recursion optimization.
    """
    try:
        with open(file_path) as f:
            source = f.read()

        tree = ast.parse(source)
        optimizer = RecursionOptimizer()
        optimizer.visit(tree)

        return {
            "file_path": file_path,
            "recursive_functions": optimizer.recursive_functions,
            "optimization_suggestions": optimizer.optimization_suggestions,
        }

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return {"file_path": file_path, "error": str(e)}


def process_directory(directory: str) -> List[Dict[str, Any]]:
    """
    Process all Python files in a directory for recursion optimization.
    """
    results = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                result = process_file(file_path)

                if result.get("recursive_functions") or result.get("error"):
                    results.append(result)

    return results


def generate_report(results: List[Dict[str, Any]]) -> None:
    """
    Generate a comprehensive recursion optimization report.
    """
    report_path = os.path.join("/opt/sutazaiapp", "logs", "recursion_optimization_report.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, "w") as f:
        f.write("# Recursion Optimization Report\n\n")
        f.write(f"**Generated:** {os.path.getctime(report_path)}\n\n")

        for result in results:
            f.write(f"## {result['file_path']}\n\n")

            if "error" in result:
                f.write(f"**Error:** `{result['error']}`\n\n")
                continue

            if result["recursive_functions"]:
                f.write("### Recursive Functions\n")
                for func, calls in result["recursive_functions"].items():
                    f.write(f"- `{func}`: {calls} recursive calls\n")
                f.write("\n")

            if result.get("optimization_suggestions"):
                f.write("### Optimization Suggestions\n")
                for suggestion in result["optimization_suggestions"]:
                    f.write(f"- {suggestion}\n")
                f.write("\n")

    logger.info(f"Recursion optimization report generated at {report_path}")


def main():
    project_root = "/opt/sutazaiapp"
    results = process_directory(project_root)
    generate_report(results)


if __name__ == "__main__":
    main()
