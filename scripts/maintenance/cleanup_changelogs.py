#!/usr/bin/env python3
"""
Smart CHANGELOG cleanup - removes auto-generated templates while preserving important ones
"""

import os
from pathlib import Path

# Key CHANGELOGs to preserve
PRESERVE_PATHS = {
    "/opt/sutazaiapp/docs/CHANGELOG.md",           # Main canonical
    "/opt/sutazaiapp/backend/CHANGELOG.md",        # Backend changes
    "/opt/sutazaiapp/frontend/CHANGELOG.md",       # Frontend changes
    "/opt/sutazaiapp/agents/CHANGELOG.md",         # Agent changes
    "/opt/sutazaiapp/IMPORTANT/CHANGELOG.md",      # Critical docs
    "/opt/sutazaiapp/scripts/CHANGELOG.md",        # Scripts changes
    "/opt/sutazaiapp/configs/CHANGELOG.md",        # Config changes
}

def is_auto_generated(filepath):
    """Check if file is an auto-generated template"""
    try:
        with open(filepath, 'r') as f:
            content = f.read(500)  # Read first 500 chars
            return "generated_by: scripts/utils/ensure_changelogs.py" in content
    except:
        return False

def has_real_content(filepath):
    """Check if file has content beyond template"""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            # Count non-template lines
            real_lines = 0
            for line in lines:
                line = line.strip()
                if line and not any(marker in line for marker in [
                    "title:", "generated_by:", "purpose:", "Conventions:",
                    "Template entry:", "Follow Conventional", "Keep entries",
                    "Include date", "This folder maintains", "authoritative",
                    "---", "#", ">"
                ]):
                    real_lines += 1
            return real_lines > 2
    except:
        return False

def main():
    print("🔍 Smart CHANGELOG Cleanup")
    print("=" * 50)
    
    removed = 0
    preserved = 0
    
    # Find all CHANGELOG.md files
    for root, dirs, files in os.walk("/opt/sutazaiapp"):
        # Skip hidden and vendor directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'node_modules']
        
        if "CHANGELOG.md" in files:
            filepath = os.path.join(root, "CHANGELOG.md")
            
            # Check if should preserve
            if filepath in PRESERVE_PATHS:
                print(f"  ✅ Preserving key: {filepath}")
                preserved += 1
                continue
            
            # Check if auto-generated without content
            if is_auto_generated(filepath) and not has_real_content(filepath):
                print(f"  ❌ Removing template: {filepath}")
                os.remove(filepath)
                removed += 1
            else:
                # Check if it has actual content
                try:
                    size = os.path.getsize(filepath)
                    if size < 100:  # Very small, likely empty
                        print(f"  ❌ Removing empty: {filepath}")
                        os.remove(filepath)
                        removed += 1
                    else:
                        print(f"  ⚠️  Keeping (has content): {filepath}")
                        preserved += 1
                except:
                    pass
    
    print("\n" + "=" * 50)
    print(f"✅ CLEANUP COMPLETE")
    print(f"  Removed: {removed} auto-generated/empty CHANGELOGs")
    print(f"  Preserved: {preserved} CHANGELOGs with content")

if __name__ == "__main__":
    main()