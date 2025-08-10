#!/usr/bin/env python3
"""
Purpose: Add CLAUDE.md review directive to all agent configurations
Usage: python add_claude_md_directive.py
Requirements: json module (built-in)
"""

import json
import os
import glob

def add_claude_directive(config_file):
    """Add CLAUDE.md review directive to agent configuration"""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Add to system prompt if it exists
        if 'system_prompt' in config:
            original_prompt = config['system_prompt']
            if "CLAUDE.md" not in original_prompt:
                claude_directive = (
                    "IMPORTANT: Before performing any task, you MUST first review "
                    "/opt/sutazaiapp/CLAUDE.md to understand the codebase standards "
                    "and implementation requirements. This file contains critical rules "
                    "for maintaining code quality, avoiding fantasy elements, and ensuring "
                    "system stability.\n\n"
                )
                config['system_prompt'] = claude_directive + original_prompt
        
        # Add to instructions if it exists
        if 'instructions' in config:
            original_instructions = config['instructions']
            if "CLAUDE.md" not in original_instructions:
                claude_instruction = (
                    "1. ALWAYS review /opt/sutazaiapp/CLAUDE.md before starting any task\n"
                    "2. Follow all rules and standards defined in CLAUDE.md\n"
                )
                config['instructions'] = claude_instruction + original_instructions
        
        # Add to goals if it exists
        if 'goals' in config and isinstance(config['goals'], list):
            claude_goal = "Review and follow all standards in /opt/sutazaiapp/CLAUDE.md"
            if claude_goal not in config['goals']:
                config['goals'].insert(0, claude_goal)
        
        # Write updated configuration
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        return True, config_file
        
    except Exception as e:
        return False, f"{config_file}: {str(e)}"

def main():
    """Update all agent configurations"""
    print("🔧 Adding CLAUDE.md directive to all agent configurations")
    print("=" * 60)
    
    # Find all agent config files
    config_files = glob.glob("/opt/sutazaiapp/agents/configs/*.json")
    
    if not config_files:
        print("❌ No agent configuration files found!")
        return 1
    
    print(f"Found {len(config_files)} configuration files")
    print()
    
    success_count = 0
    error_count = 0
    errors = []
    
    for config_file in sorted(config_files):
        filename = os.path.basename(config_file)
        success, result = add_claude_directive(config_file)
        
        if success:
            print(f"✅ Updated: {filename}")
            success_count += 1
        else:
            print(f"❌ Failed: {filename}")
            errors.append(result)
            error_count += 1
    
    print()
    print("📊 Summary:")
    print(f"✅ Successfully updated: {success_count}")
    print(f"❌ Failed: {error_count}")
    
    if errors:
        print("\n❌ Errors:")
        for error in errors:
            print(f"  - {error}")
    
    print()
    print("✅ All agent configurations have been updated with CLAUDE.md directive!")
    print("Agents will now review codebase standards before performing any tasks.")
    
    return 0 if error_count == 0 else 1

if __name__ == "__main__":
    exit(main())