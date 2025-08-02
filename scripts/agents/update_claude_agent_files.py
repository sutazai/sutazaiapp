#!/usr/bin/env python3
"""
Purpose: Add CLAUDE.md review directive to Claude agent definition files
Usage: python update_claude_agent_files.py
Requirements: None (uses built-in modules)
"""

import os
import glob
import re

def update_agent_file(filepath):
    """Add CLAUDE.md directive to agent definition file"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Skip if already contains CLAUDE.md reference
        if "CLAUDE.md" in content:
            return True, "Already contains CLAUDE.md reference"
        
        # Skip summary/status files
        if any(keyword in os.path.basename(filepath).upper() for keyword in ['SUMMARY', 'STATUS', 'PROTOCOL']):
            return True, "Skipped summary/status file"
        
        # Add directive after the agent name/title
        lines = content.split('\n')
        
        # Find the first line that isn't empty or a comment
        insert_index = 0
        for i, line in enumerate(lines):
            if line.strip() and not line.startswith('#'):
                insert_index = i + 1
                break
        
        # Create the CLAUDE.md directive
        claude_directive = [
            "",
            "## Important: Codebase Standards",
            "",
            "**MANDATORY**: Before performing any task, you MUST first review `/opt/sutazaiapp/CLAUDE.md` to understand:",
            "- Codebase standards and conventions",
            "- Implementation requirements and best practices",
            "- Rules for avoiding fantasy elements",
            "- System stability and performance guidelines",
            "- Clean code principles and organization rules",
            "",
            "This file contains critical rules that must be followed to maintain code quality and system integrity.",
            ""
        ]
        
        # Insert the directive
        for i, line in enumerate(claude_directive):
            lines.insert(insert_index + i, line)
        
        # Write updated content
        updated_content = '\n'.join(lines)
        with open(filepath, 'w') as f:
            f.write(updated_content)
        
        return True, "Successfully updated"
        
    except Exception as e:
        return False, str(e)

def main():
    """Update all Claude agent definition files"""
    print("🔧 Updating Claude agent definition files with CLAUDE.md directive")
    print("=" * 60)
    
    # Find all agent definition files
    agent_files = glob.glob("/opt/sutazaiapp/.claude/agents/*.md")
    
    if not agent_files:
        print("❌ No Claude agent files found!")
        return 1
    
    print(f"Found {len(agent_files)} agent definition files")
    print()
    
    success_count = 0
    skip_count = 0
    error_count = 0
    errors = []
    
    for agent_file in sorted(agent_files):
        filename = os.path.basename(agent_file)
        success, result = update_agent_file(agent_file)
        
        if success:
            if "Skipped" in result or "Already contains" in result:
                print(f"⏭️  Skipped: {filename} ({result})")
                skip_count += 1
            else:
                print(f"✅ Updated: {filename}")
                success_count += 1
        else:
            print(f"❌ Failed: {filename}")
            errors.append(f"{filename}: {result}")
            error_count += 1
    
    print()
    print("📊 Summary:")
    print(f"✅ Successfully updated: {success_count}")
    print(f"⏭️  Skipped: {skip_count}")
    print(f"❌ Failed: {error_count}")
    
    if errors:
        print("\n❌ Errors:")
        for error in errors:
            print(f"  - {error}")
    
    print()
    print("✅ Claude agent files have been updated!")
    print("All agents will now review CLAUDE.md before performing tasks.")
    
    return 0 if error_count == 0 else 1

if __name__ == "__main__":
    exit(main())