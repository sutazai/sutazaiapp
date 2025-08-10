#!/usr/bin/env python3
"""
Fix common issues in agent app.py files
"""
import re
from pathlib import Path

def fix_indentation_errors(content: str) -> str:
    """Fix common indentation errors"""
    # Fix duplicate try statements
    content = re.sub(r'try:\s*try:', 'try:', content)
    
    # Fix missing indentation after try/except
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        fixed_lines.append(line)
        
        # If we find a try: or except: without proper indentation on next line
        if line.strip().endswith(':') and i + 1 < len(lines):
            next_line = lines[i + 1]
            # Check if next non-empty line has proper indentation
            if next_line.strip() and not next_line.startswith((' ', '\t')):
                # Add proper indentation
                lines[i + 1] = '    ' + next_line
        
        i += 1
    
    return '\n'.join(fixed_lines)

def fix_agent_app_file(agent_dir: Path):
    """Fix app.py file in an agent directory"""
    app_file = agent_dir / 'app.py'
    
    if not app_file.exists():
        return False
        
    try:
        with open(app_file, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Apply fixes
        content = fix_indentation_errors(content)
        
        # Only write if changed
        if content != original_content:
            with open(app_file, 'w') as f:
                f.write(content)
            print(f"✓ Fixed app.py for {agent_dir.name}")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"✗ Error fixing {agent_dir.name}: {e}")
        return False

def main():
    """Fix all agent app.py files"""
    agents_dir = Path('/opt/sutazaiapp/agents')
    
    print("Scanning for agent app.py files with issues...\n")
    
    fixed_count = 0
    checked_count = 0
    
    for agent_dir in agents_dir.iterdir():
        if agent_dir.is_dir() and (agent_dir / 'app.py').exists():
            checked_count += 1
            if fix_agent_app_file(agent_dir):
                fixed_count += 1
    
    print(f"\n✅ Checked {checked_count} agents")
    print(f"✅ Fixed {fixed_count} app.py files")
    
    if fixed_count > 0:
        print("\nAgents with fixed app.py files need to be restarted.")

if __name__ == "__main__":
    main()