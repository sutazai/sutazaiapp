#!/usr/bin/env python3
"""
Update all agent definition files to use tinyllama:latest as the default model
"""

import os
import re
from pathlib import Path

def update_agent_model(file_path):
    """Update a single agent file to use tinyllama"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Pattern to match the model line in YAML frontmatter
    pattern = r'(model:\s*)(.*?)(\n)'
    
    # Replace with tinyllama:latest
    updated_content = re.sub(pattern, r'\1tinyllama:latest\3', content)
    
    # Only write if changed
    if content != updated_content:
        with open(file_path, 'w') as f:
            f.write(updated_content)
        return True
    return False

def main():
    agents_dir = Path("/opt/sutazaiapp/.claude/agents")
    updated_count = 0
    
    print("Updating all agents to use tinyllama:latest...")
    
    # Process all .md files in agents directory
    for agent_file in agents_dir.glob("*.md"):
        # Skip non-agent files
        if agent_file.name in ["AGENT_CLEANUP_SUMMARY.md", "COMPREHENSIVE_INVESTIGATION_PROTOCOL.md"]:
            continue
            
        if update_agent_model(agent_file):
            print(f"✅ Updated: {agent_file.name}")
            updated_count += 1
        else:
            print(f"⏭️  Skipped: {agent_file.name} (already using tinyllama or no model field)")
    
    print(f"\n✨ Complete! Updated {updated_count} agent files to use tinyllama:latest")
    print("\n💡 TinyLlama benefits:")
    print("   - Size: Only 637MB (vs 3.8GB for llama3.2:3b)")
    print("   - RAM: ~1-2GB usage (vs 4-6GB for larger models)")
    print("   - Speed: Faster inference on CPU")
    print("   - Context: 2048 tokens (sufficient for most agent tasks)")

if __name__ == "__main__":
    main()