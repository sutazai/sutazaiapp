import ast
import os

import astor


class SyntaxTransformer(ast.NodeTransformer):
    def visit_ClassDef(self, node):
        # Add missing methods or correct existing methods
        for method in node.body:
            if isinstance(method, ast.FunctionDef):
                # Ensure first parameter is self
                if not method.args.args or method.args.args[0].arg != "self":
                    method.args.args.insert(0, ast.arg(arg="self"))
        return node

    def visit_FunctionDef(self, node):
        # Correct function signatures
        if not node.args.args or node.args.args[0].arg != "self":
            node.args.args.insert(0, ast.arg(arg="self"))
        return node


def fix_file_syntax(file_path):
    try:
        with open(file_path) as f:
            source = f.read()

        # Parse the source code into an AST
        tree = ast.parse(source)

        # Transform the AST
        transformer = SyntaxTransformer()
        modified_tree = transformer.visit(tree)

        # Convert back to source code
        modified_source = astor.to_source(modified_tree)

        # Write back to file
        with open(file_path, "w") as f:
            f.write(modified_source)

        print(f"Successfully processed {file_path}")
    except SyntaxError as e:
        print(f"Syntax error in {file_path}: {e}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")


def process_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                fix_file_syntax(file_path)


if __name__ == "__main__":
    process_directory("/opt/sutazaiapp/core_system")
