#!/bin/bash

# Script to fix test_sync_manager_complete_coverage.py issues

echo "Fixing issues in test_sync_manager_complete_coverage.py..."

# Create a backup
cp tests/test_sync_manager_complete_coverage.py tests/test_sync_manager_complete_coverage.py.bak

# Rewrite the file with proper indentation and remove duplicate decorators
python3 -c "
import re

with open('tests/test_sync_manager_complete_coverage.py', 'r') as f:
    content = f.read()

# Remove all duplicate decorators
while '@pytest.mark.asyncio\n@pytest.mark.asyncio' in content:
    content = content.replace('@pytest.mark.asyncio\n@pytest.mark.asyncio', '@pytest.mark.asyncio')

# Fix indentation of all decorators
lines = content.split('\n')
fixed_lines = []
in_class = False
class_indent = ''
method_indent = '    '  # Default method indentation (4 spaces)

for i, line in enumerate(lines):
    # Check if this is a class definition
    if line.strip().startswith('class '):
        in_class = True
        fixed_lines.append(line)
        continue
    
    # If in a class and looking at a method definition
    if in_class and re.match(r'^\s+def\s+', line):
        # Get the indentation of this method
        method_indent = re.match(r'^(\s+)', line).group(1)
        fixed_lines.append(line)
        continue
    
    # If this line is a decorator and we're in a class
    if in_class and '@pytest.mark.asyncio' in line and not line.startswith(method_indent):
        # Fix the indentation by adding proper spaces
        fixed_lines.append(method_indent + '@pytest.mark.asyncio')
    else:
        fixed_lines.append(line)

# Write the fixed content back
with open('tests/test_sync_manager_complete_coverage.py', 'w') as f:
    f.write('\n'.join(fixed_lines))

print('Fixed indentation and removed duplicate decorators in test_sync_manager_complete_coverage.py')
"

echo "Fix completed!" 