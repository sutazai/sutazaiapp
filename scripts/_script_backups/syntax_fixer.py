#!/usr/bin/env python3.11
import ast
import io
import re
import sys
import tokenize
from pathlib import Path

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


        class SyntaxTransformer(ast.NodeTransformer):
            def visit_ClassDef(self, node):
                # Add missing methods or correct existing methods
                has_init = any(
                isinstance(
                    method,
                    ast.FunctionDef) and method.name == "__init__"
                for method in node.body
                    )

                    if not has_init:
                        # Add a basic __init__ method
                        init_method = ast.FunctionDef(
                        name="__init__",
                        args=ast.arguments(
                        args=[ast.arg(arg="self", annotation=None)],
                        posonlyargs=[],
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[],
                        ),
                        body=[ast.Pass()],
                        decorator_list=[],
                    returns=None,
                    )
                    node.body.insert(0, init_method)

                    # Fix method signatures in class methods
                    for method in node.body:
                        if isinstance(method, ast.FunctionDef):
                            # Ensure first parameter is self for instance methods
                            if method.args.args and len(method.args.args) > 0:
                                if method.args.args[0].arg != "self":
                                    method.args.args.insert(
                                        0,
                                        ast.arg(arg="self", annotation=None))
                                    else:
                                    method.args.args.insert(
                                        0,
                                        ast.arg(arg="self", annotation=None))

                                return self.generic_visit(node)

                                def visit_FunctionDef(self, node):
                                    # Skip this transformation for methods within classes
                                    # as they're handled in visit_ClassDef
                                return self.generic_visit(node)


                                def ast_transform(content):
                                    """Use AST transformation to fix structural issues"""
                                    try:
                                        tree = ast.parse(content)
                                        transformer = SyntaxTransformer()
                                        modified_tree = transformer.visit(tree)
                                    return astor.to_source(modified_tree)
                                    except SyntaxError:
                                    return content


                                    def regex_fix(content):
                                        """Use regex to fix common syntax patterns"""
                                        # Fix missing colons
                                        content = re.sub(
                                        r"(
                                            if|elif|else|for|while|def|class|try|except|finally|with)\s+([^:]+)(?<!\n|:)$",
                                        r"\1 \2:",
                                        content,
                                        flags=re.MULTILINE,
                                        )

                                        # Fix indentation (convert tabs to spaces)
                                        content = re.sub(
                                        r"^\t+",
                                        lambda match: "    " * len(
                                            match.group(0)),
                                        content,
                                        flags=re.MULTILINE,
                                        )

                                        # Fix parentheses in function calls
                                        content = re.sub(
                                            r"(\w+)\s+\(", r"\1(", content)

                                        # Fix missing parentheses in print statements (Python 3)
                                        content = re.sub(
                                            r"print\s+([^(].*?)$", r"print(\1)", content, flags=re.MULTILINE)

                                        # Fix common typos
                                        content = re.sub(
                                            r"\bimpotr\b",
                                            "import",
                                            content)
                                        content = re.sub(
                                            r"\bfrom\s+(\w+)\s+imports\b",
                                            r"from \1 import",
                                            content)
                                        content = re.sub(
                                            r"\belse\s+if\b",
                                            "elif",
                                            content)

                                    return content


                                    def fix_file_syntax(file_path):
                                                                                """Fix syntax errors in \
                                            a single Python file"""
                                        try:
                                            with open(
                                                file_path,
                                                encoding="utf-8") as f:
                                            content = f.read()

                                            # Apply multiple fixing strategies
                                            original_content = content

                                            # Step 1: Apply regex fixes for common patterns
                                            content = regex_fix(content)

                                            # Step 2: Apply tokenization fixes
                                            content = tokenize_and_fix(content)

                                            # Step 3: Apply AST transformation if possible
                                            try:
                                                content = ast_transform(
                                                    content)
                                                except Exception as e:
                                                    print(
                                                        f"Warning: AST transformation failed for {file_path}: {e}")

                                                    # Final syntax validation
                                                    try:
                                                        ast.parse(content)
                                                        if original_content != content:
                                                            with open(
                                                                file_path,
                                                                "w",
                                                                encoding="utf-8") as f:
                                                            f.write(content)
                                                            print(
                                                                f"✓ Successfully fixed syntax in {file_path}")
                                                        return True
                                                        print(
                                                            f"✓ No syntax issues found in {file_path}")
                                                    return True
                                                    except SyntaxError as e:
                                                        print(
                                                            f"✗ Remaining syntax error in {file_path}: {e}")
                                                    return False

                                                    except Exception as e:
                                                        print(
                                                            f"✗ Error processing {file_path}: {e}")
                                                    return False


                                                    def process_directory(
                                                        directory,
                                                        extensions=None):
                                                                                                                """Process all Python files in a directory and \
                                                            its subdirectories"""
                                                        if extensions is None:
                                                            extensions = [".py"]

                                                            fixed_files = 0
                                                            error_files = 0

                                                            for path in Path(
                                                                directory).rglob("*"):
                                                                if path.is_file() and path.suffix in extensions:
                                                                    if fix_file_syntax(
                                                                        path):
                                                                        fixed_files += 1
                                                                        else:
                                                                        error_files += 1

                                                                    return fixed_files, error_files


                                                                    def main():
                                                                        """Main entry point for syntax error fixing"""
                                                                        if len(
                                                                            sys.argv) < 2:
                                                                            print(
                                                                                "Usage: python syntax_fixer.py <directory> [extensions]")
                                                                            print(
                                                                                "Example: python syntax_fixer.py . .py,
                                                                                .pyw")
                                                                        return

                                                                        directory = sys.argv[1]
                                                                        extensions = (
                                                                        [f".{ext}" for ext in sys.argv[2].split(
                                                                            ",
                                                                            ")] if len(sys.argv) > 2 else [".py"]
                                                                        )

                                                                        print(
                                                                                                                                                f"Starting syntax fixing in \
                                                                            {directory} for files with extensions: {extensions}",
                                                                        )
                                                                        fixed_files, error_files = process_directory(
                                                                            directory,
                                                                            extensions)

                                                                        print(
                                                                            "\nSummary:")
                                                                        print(
                                                                            f"- Directory processed: {directory}")
                                                                        print(
                                                                            f"- Files processed successfully: {fixed_files}")
                                                                        print(
                                                                            f"- Files with remaining errors: {error_files}")


                                                                        if __name__ == "__main__":
                                                                            main()
