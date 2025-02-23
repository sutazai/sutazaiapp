import ast
import os


def fix_syntax_errors(directory):
    """Attempt to fix syntax errors in Python files."""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r') as f:
                        source = f.read()
                    
                    # Try parsing the AST
                    ast.parse(source)
                except SyntaxError as e:
                    print(f"Syntax error in {filepath}: {e}")
                    
                    # Basic error correction strategies
                    corrected_source = correct_common_syntax_errors(source)
                    
                    # Write corrected source
                    with open(filepath, 'w') as f:
                        f.write(corrected_source)
                    
                    print(f"Attempted to fix {filepath}")


def correct_common_syntax_errors(source):
    """Apply common syntax error corrections."""
    # Remove trailing whitespaces
    lines = source.split('\n')
    lines = [line.rstrip() for line in lines]
    
    # Ensure final newline
    if lines and lines[-1].strip():
        lines.append('')
    
    # Basic indentation correction
    corrected_lines = []
    indent_level = 0
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith(('def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'except', 'else:', 'elif ')):
            indent_level += 1
        elif stripped.startswith(('return', 'break', 'continue', 'pass')):
            indent_level = max(0, indent_level - 1)
        
        corrected_line = ' ' * (4 * indent_level) + stripped
        corrected_lines.append(corrected_line)
    
    return '\n'.join(corrected_lines)


def main():
    """Main entry point for syntax error fixing."""
    fix_syntax_errors('core_system')


if __name__ == '__main__':
    main() 