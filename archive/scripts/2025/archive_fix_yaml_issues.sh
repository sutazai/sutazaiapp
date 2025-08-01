#!/bin/bash
#
# Fix common YAML issues in docker-compose files
#

echo "🔧 Fixing YAML issues in docker-compose files..."

# Fix multiple merge keys in docker-compose-complete-agi.yml
if [ -f "docker-compose-complete-agi.yml" ]; then
    echo "Checking docker-compose-complete-agi.yml..."
    
    # Check for the specific pattern of multiple merge keys
    if grep -A2 "environment:" docker-compose-complete-agi.yml | grep -q "<<: \*.*\n.*<<: \*"; then
        echo "Found multiple merge keys issue, fixing..."
        
        # Create a temporary file
        cp docker-compose-complete-agi.yml docker-compose-complete-agi.yml.bak
        
        # Use Python to fix the issue properly
        python3 << 'EOF'
import yaml
import re

# Read the file
with open('docker-compose-complete-agi.yml', 'r') as f:
    content = f.read()

# Fix multiple consecutive merge keys
# This pattern finds environment sections with multiple <<: keys
pattern = r'(environment:\s*\n)(\s*<<: \*\w+\s*\n)(\s*<<: \*\w+\s*\n)(\s*<<: \*\w+\s*\n)?'

def fix_merge_keys(match):
    indent = match.group(2).split('<<:')[0]
    keys = []
    for group in [match.group(2), match.group(3), match.group(4)]:
        if group:
            key = group.strip().replace('<<: *', '').strip()
            keys.append(f'*{key}')
    
    if len(keys) > 1:
        return f"{match.group(1)}{indent}<<: [{', '.join(keys)}]\n"
    else:
        return match.group(0)

# Apply the fix
content = re.sub(pattern, fix_merge_keys, content)

# Write back
with open('docker-compose-complete-agi.yml', 'w') as f:
    f.write(content)

print("Fixed multiple merge keys issue")
EOF
        
        # Validate the fixed file
        if python3 -c "import yaml; yaml.safe_load(open('docker-compose-complete-agi.yml'))" 2>/dev/null; then
            echo "✅ YAML file is now valid!"
            rm -f docker-compose-complete-agi.yml.bak
        else
            echo "❌ Fix failed, restoring backup..."
            mv docker-compose-complete-agi.yml.bak docker-compose-complete-agi.yml
        fi
    else
        echo "No multiple merge keys issue found"
    fi
fi

# Check for other common YAML issues
echo ""
echo "Checking for other common YAML issues..."

# Check for tabs
if grep -P '\t' docker-compose*.yml 2>/dev/null; then
    echo "⚠️  WARNING: Found tabs in YAML files. YAML requires spaces, not tabs!"
    echo "Files with tabs:"
    grep -l -P '\t' docker-compose*.yml
fi

# Check for trailing spaces
if grep -E ' +$' docker-compose*.yml 2>/dev/null; then
    echo "⚠️  WARNING: Found trailing spaces in YAML files"
    # Fix trailing spaces
    for file in docker-compose*.yml; do
        if [ -f "$file" ]; then
            sed -i 's/[[:space:]]*$//' "$file"
            echo "Fixed trailing spaces in $file"
        fi
    done
fi

echo ""
echo "✅ YAML fixes completed!" 