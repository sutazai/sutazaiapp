#!/bin/bash

# Script to fix indentation of @pytest.mark.asyncio decorators in all test files

echo "Fixing indentation of @pytest.mark.asyncio decorators in all test files..."

# Process each test file
for test_file in $(find tests -name "test_*.py"); do
    echo "Processing $test_file..."
    
    # Create a temporary file
    tempfile=$(mktemp)
    
    # Fix the indentation in the file
    python3 -c "
import re

with open('$test_file', 'r') as f:
    content = f.read()

# Remove duplicate decorators (just in case)
content = re.sub(r'@pytest\.mark\.asyncio\s+@pytest\.mark\.asyncio', '@pytest.mark.asyncio', content)

# Fix indentation - ensure decorators are properly indented with class method
lines = content.split('\\n')
fixed_lines = []
in_class = False
class_indent = ''

for i, line in enumerate(lines):
    # Track when we enter a class
    if re.match(r'^class\s+', line):
        in_class = True
        class_indent = ''
    
    # Check for method definitions in a class
    if in_class and re.match(r'^\s+def\s+', line):
        method_indent = re.match(r'^(\s+)', line).group(1)
        
        # If previous line has a pytest.mark.asyncio without proper indentation
        if i > 0 and '@pytest.mark.asyncio' in lines[i-1] and not lines[i-1].startswith(method_indent):
            # Replace previous line with properly indented version
            fixed_lines[-1] = method_indent + '@pytest.mark.asyncio'
        
    fixed_lines.append(line)

with open('$test_file', 'w') as f:
    f.write('\\n'.join(fixed_lines))

print(f'Fixed indentation in {test_file}')
" 2>/dev/null || echo "Failed to process $test_file"
done

echo "All test files processed!" 