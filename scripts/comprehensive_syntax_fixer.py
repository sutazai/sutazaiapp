import ast
import io
import os
import re
import tokenize

import astor


def tokenize_and_fix(content):
    """Use tokenize to identify and fix syntax issues"""
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(content).readline))
        fixed_tokens = []

        for token_type, token_string, start, end, line in tokens:
            # Fix common syntax issues during tokenization
            if token_type == tokenize.ERRORTOKEN:
                # Skip or replace error tokens
                continue
            fixed_tokens.append((token_type, token_string, start, end, line))

        # Reconstruct source code
        return tokenize.untokenize(fixed_tokens)
    except tokenize.TokenError:
        return content


def ast_transform(content):
    """Use AST transformation to fix structural issues"""
    try:
        tree = ast.parse(content)

        class SyntaxFixer(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                # Ensure first parameter is self for methods
                if not node.args.args or node.args.args[0].arg != "self":
                    node.args.args.insert(0, ast.arg(arg="self"))
                return node

            def visit_ClassDef(self, node):
                # Add missing __init__ if not present
                has_init = any(
                    isinstance(method, ast.FunctionDef) and method.name == "__init__"
                    for method in node.body
                )
                if not has_init:
                    init_method = ast.FunctionDef(
                        name="__init__",
                        args=ast.arguments(
                            args=[ast.arg(arg="self")],
                            posonlyargs=[],
                            kwonlyargs=[],
                            kw_defaults=[],
                            defaults=[],
                        ),
                        body=[ast.Pass()],
                        decorator_list=[],
                    )
                    node.body.insert(0, init_method)
                return node

        transformer = SyntaxFixer()
        modified_tree = transformer.visit(tree)
        return astor.to_source(modified_tree)
    except SyntaxError:
        return content


def regex_fix(content):
    """Use regex to fix common syntax patterns"""
    # Remove unnecessary parentheses
    content = re.sub(r"=\s*\((\d+)\)", r"= \1", content)

    # Fix method signatures
    content = re.sub(r"def\s+(\w+)\s*\(\s*\),", r"def \1(self, ", content)

    # Add missing self parameter
    content = re.sub(r"def\s+(\w+)\s*\((\w+)\):", r"def \1(self, \2):", content)

    return content


def fix_file_syntax(file_path):
    try:
        with open(file_path, "r") as f:
            content = f.read()

        # Apply multiple fixing strategies
        content = regex_fix(content)
        content = tokenize_and_fix(content)
        content = ast_transform(content)

        # Final syntax validation
        try:
            ast.parse(content)
        except SyntaxError as e:
            print(f"Remaining syntax error in {file_path}: {e}")
            return False

        with open(file_path, "w") as f:
            f.write(content)

        print(f"Successfully processed {file_path}")
        return True

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def process_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                fix_file_syntax(file_path)


if __name__ == "__main__":
    process_directory("/opt/sutazaiapp/core_system")
