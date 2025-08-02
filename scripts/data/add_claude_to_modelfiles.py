#!/usr/bin/env python3
"""
Purpose: Add CLAUDE.md review directive to agent modelfiles
Usage: python add_claude_to_modelfiles.py
Requirements: json module (built-in)
"""

import json
import os
import glob

def update_modelfile(config_file):
    """Add CLAUDE.md directive to agent modelfile"""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Check if modelfile exists
        if 'modelfile' not in config:
            return False, "No modelfile field found"
        
        modelfile = config['modelfile']
        
        # Skip if already contains CLAUDE.md
        if "CLAUDE.md" in modelfile:
            return True, "Already contains CLAUDE.md reference"
        
        # Find the SYSTEM section
        if "SYSTEM" in modelfile:
            # Add CLAUDE.md directive after SYSTEM
            claude_directive = (
                "\n\n**CRITICAL REQUIREMENT**: Before performing any task, you MUST first review "
                "/opt/sutazaiapp/CLAUDE.md to understand the codebase standards, implementation "
                "requirements, and rules for maintaining code quality and system stability. "
                "This file contains mandatory guidelines that must be followed.\n"
            )
            
            # Insert after the first paragraph of SYSTEM
            lines = modelfile.split('\n')
            system_index = -1
            
            for i, line in enumerate(lines):
                if line.startswith("SYSTEM"):
                    # Find the end of the first paragraph after SYSTEM
                    for j in range(i + 1, len(lines)):
                        if lines[j].strip() == "" or lines[j].startswith("#"):
                            system_index = j
                            break
                    break
            
            if system_index > 0:
                lines.insert(system_index, claude_directive)
                config['modelfile'] = '\n'.join(lines)
            else:
                # Add at the beginning of SYSTEM content
                config['modelfile'] = modelfile.replace(
                    "SYSTEM", 
                    "SYSTEM" + claude_directive
                )
        
        # Write updated configuration
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        return True, "Successfully updated"
        
    except Exception as e:
        return False, str(e)

def main():
    """Update all agent configurations with CLAUDE.md directive"""
    print("🔧 Adding CLAUDE.md directive to agent modelfiles")
    print("=" * 60)
    
    # Find all agent config files
    config_files = glob.glob("/opt/sutazaiapp/agents/configs/*.json")
    
    if not config_files:
        print("❌ No agent configuration files found!")
        return 1
    
    print(f"Found {len(config_files)} configuration files")
    print()
    
    success_count = 0
    skip_count = 0
    error_count = 0
    no_modelfile_count = 0
    errors = []
    
    for config_file in sorted(config_files):
        filename = os.path.basename(config_file)
        success, result = update_modelfile(config_file)
        
        if success:
            if "Already contains" in result:
                print(f"⏭️  {filename}: {result}")
                skip_count += 1
            else:
                print(f"✅ {filename}: {result}")
                success_count += 1
        else:
            if "No modelfile" in result:
                print(f"⚠️  {filename}: {result}")
                no_modelfile_count += 1
            else:
                print(f"❌ {filename}: {result}")
                errors.append(f"{filename}: {result}")
                error_count += 1
    
    print()
    print("📊 Summary:")
    print(f"✅ Successfully updated: {success_count}")
    print(f"⏭️  Already had CLAUDE.md: {skip_count}")
    print(f"⚠️  No modelfile field: {no_modelfile_count}")
    print(f"❌ Errors: {error_count}")
    
    if errors:
        print("\n❌ Errors:")
        for error in errors:
            print(f"  - {error}")
    
    print()
    if success_count > 0:
        print("✅ Agent modelfiles have been updated with CLAUDE.md directive!")
        print("Agents will now review codebase standards before performing any tasks.")
    
    return 0 if error_count == 0 else 1

if __name__ == "__main__":
    exit(main())