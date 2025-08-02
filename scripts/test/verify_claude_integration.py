#!/usr/bin/env python3
"""
Purpose: Verify CLAUDE.md integration in all agent configurations
Usage: python verify_claude_integration.py
Requirements: json module (built-in)
"""

import json
import os
import glob

def check_json_config(filepath):
    """Check if JSON config contains CLAUDE.md reference"""
    try:
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        claude_found = False
        locations = []
        
        # Check modelfile field (primary location for agent configs)
        if 'modelfile' in config and isinstance(config['modelfile'], str):
            if 'CLAUDE.md' in config['modelfile']:
                claude_found = True
                locations.append('modelfile')
        
        # Check various other fields
        for field in ['system_prompt', 'instructions', 'prompt', 'description']:
            if field in config and isinstance(config[field], str):
                if 'CLAUDE.md' in config[field]:
                    claude_found = True
                    locations.append(field)
        
        # Check goals list
        if 'goals' in config and isinstance(config['goals'], list):
            for goal in config['goals']:
                if 'CLAUDE.md' in str(goal):
                    claude_found = True
                    locations.append('goals')
                    break
        
        return claude_found, locations
        
    except Exception as e:
        return None, [str(e)]

def check_md_file(filepath):
    """Check if MD file contains CLAUDE.md reference"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        if 'CLAUDE.md' in content:
            # Find the context
            lines = content.split('\n')
            claude_lines = []
            for i, line in enumerate(lines):
                if 'CLAUDE.md' in line:
                    # Get surrounding context
                    start = max(0, i - 2)
                    end = min(len(lines), i + 3)
                    claude_lines.extend(lines[start:end])
            return True, claude_lines[:5]  # Return first 5 relevant lines
        
        return False, []
        
    except Exception as e:
        return None, [str(e)]

def main():
    """Verify CLAUDE.md integration across all agent files"""
    print("🔍 Verifying CLAUDE.md Integration")
    print("=" * 60)
    
    # Check JSON configurations
    print("\n📋 Checking JSON Configurations:")
    print("-" * 40)
    
    json_files = glob.glob("/opt/sutazaiapp/agents/configs/*.json")
    json_with_claude = 0
    json_without_claude = 0
    json_errors = 0
    
    for json_file in sorted(json_files):
        filename = os.path.basename(json_file)
        has_claude, info = check_json_config(json_file)
        
        if has_claude is None:
            print(f"❌ {filename}: Error - {info[0]}")
            json_errors += 1
        elif has_claude:
            print(f"✅ {filename}: Found in {', '.join(info)}")
            json_with_claude += 1
        else:
            print(f"⚠️  {filename}: No CLAUDE.md reference found")
            json_without_claude += 1
    
    # Check MD files
    print("\n📋 Checking Claude Agent Definitions:")
    print("-" * 40)
    
    md_files = glob.glob("/opt/sutazaiapp/.claude/agents/*.md")
    md_with_claude = 0
    md_without_claude = 0
    md_errors = 0
    
    # Skip summary/status files
    skip_keywords = ['SUMMARY', 'STATUS', 'PROTOCOL', 'REPORT']
    
    for md_file in sorted(md_files):
        filename = os.path.basename(md_file)
        
        # Skip non-agent files
        if any(keyword in filename.upper() for keyword in skip_keywords):
            continue
        
        has_claude, info = check_md_file(md_file)
        
        if has_claude is None:
            print(f"❌ {filename}: Error - {info[0]}")
            md_errors += 1
        elif has_claude:
            md_with_claude += 1
        else:
            print(f"⚠️  {filename}: No CLAUDE.md reference found")
            md_without_claude += 1
    
    # Summary
    print("\n📊 Integration Summary:")
    print("=" * 60)
    print(f"\nJSON Configurations:")
    print(f"  ✅ With CLAUDE.md: {json_with_claude}")
    print(f"  ⚠️  Without CLAUDE.md: {json_without_claude}")
    print(f"  ❌ Errors: {json_errors}")
    print(f"  📁 Total: {len(json_files)}")
    
    print(f"\nClaude Agent Definitions:")
    print(f"  ✅ With CLAUDE.md: {md_with_claude}")
    print(f"  ⚠️  Without CLAUDE.md: {md_without_claude}")
    print(f"  ❌ Errors: {md_errors}")
    
    total_integrated = json_with_claude + md_with_claude
    total_files = len(json_files) + md_with_claude + md_without_claude
    percentage = (total_integrated / total_files * 100) if total_files > 0 else 0
    
    print(f"\n🎯 Overall Integration: {percentage:.1f}%")
    
    if json_without_claude == 0 and md_without_claude == 0:
        print("\n✅ All agent configurations now include CLAUDE.md directive!")
        print("Agents will review codebase standards before performing any tasks.")
    else:
        print(f"\n⚠️  {json_without_claude + md_without_claude} files still need CLAUDE.md integration.")
    
    return 0

if __name__ == "__main__":
    exit(main())