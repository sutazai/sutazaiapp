#!/usr/bin/env python3
"""
Fix YAML Structure Issues

This script fixes agents where the environment section was added incorrectly
before the YAML front matter instead of within it.
"""

import re
from pathlib import Path

class YAMLStructureFixer:
    """Fixes YAML front matter structure issues"""
    
    def __init__(self, agents_dir: str = "/opt/sutazaiapp/.claude/agents"):
        self.agents_dir = Path(agents_dir)
        
    def fix_yaml_structure(self, agent_file: Path) -> bool:
        """Fix YAML front matter structure for an agent file"""
        
        try:
            with open(agent_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if file has environment section before YAML front matter
            if content.startswith('environment:'):
                print(f"  🔧 Fixing YAML structure: {agent_file.name}")
                
                # Create backup
                backup_file = agent_file.with_suffix('.md.backup2')
                if not backup_file.exists():
                    with open(backup_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                
                # Extract environment section and rest of content
                lines = content.split('\n')
                env_lines = []
                rest_lines = []
                
                in_env_section = True
                for line in lines:
                    if in_env_section and (line.startswith('  - ') or line.startswith('environment:')):
                        env_lines.append(line)
                    else:
                        in_env_section = False
                        rest_lines.append(line)
                
                # Find where YAML front matter starts in the rest
                yaml_start_idx = -1
                for i, line in enumerate(rest_lines):
                    if line.strip() == '---':
                        yaml_start_idx = i
                        break
                
                if yaml_start_idx >= 0:
                    # Reconstruct with proper YAML structure
                    new_content = []
                    new_content.append('---')
                    new_content.append('')
                    new_content.append('## Important: Codebase Standards')
                    
                    # Find the end of the compliance header
                    compliance_end_idx = -1
                    for i, line in enumerate(rest_lines):
                        if 'This file contains critical rules' in line:
                            # Find the next double newline or next YAML section
                            for j in range(i, len(rest_lines)):
                                if rest_lines[j].strip() and not rest_lines[j].startswith('**') and not rest_lines[j].startswith('-') and 'file contains' not in rest_lines[j]:
                                    compliance_end_idx = j
                                    break
                            break
                    
                    if compliance_end_idx > 0:
                        # Add compliance header
                        for i in range(yaml_start_idx + 1, compliance_end_idx):
                            new_content.append(rest_lines[i])
                        
                        new_content.append('')
                        
                        # Add environment section
                        for env_line in env_lines:
                            new_content.append(env_line)
                        
                        # Add the rest starting from YAML content
                        for i in range(compliance_end_idx, len(rest_lines)):
                            new_content.append(rest_lines[i])
                    else:
                        # Fallback: just insert environment after compliance header
                        for line in rest_lines[yaml_start_idx + 1:]:
                            new_content.append(line)
                            if 'This file contains critical rules' in line:
                                new_content.append('')
                                for env_line in env_lines:
                                    new_content.append(env_line)
                                new_content.append('')
                    
                    # Write fixed content
                    with open(agent_file, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(new_content))
                    
                    print(f"    ✅ Fixed: {agent_file.name}")
                    return True
                else:
                    print(f"    ❌ Could not find YAML front matter in: {agent_file.name}")
                    return False
            else:
                print(f"  ℹ️  Already correct structure: {agent_file.name}")
                return True
                
        except Exception as e:
            print(f"  ❌ Error fixing {agent_file.name}: {str(e)}")
            return False
    
    def fix_all_yaml_issues(self) -> dict:
        """Fix YAML structure issues for all agents"""
        
        # Get agent files that need fixing (those that start with environment:)
        files_to_fix = []
        
        exclude_patterns = [
            'AGENT_CLEANUP_SUMMARY.md',
            'AGENT_COMPLIANCE_REPORT.md', 
            'COMPREHENSIVE_INVESTIGATION_PROTOCOL.md',
            'COMPLETE_CLEANUP_STATUS.md',
            'DUPLICATE_AGENTS_REPORT.md',
            'FINAL_COMPLIANCE_SUMMARY.md',
            'team_collaboration_standards.md'
        ]
        
        for md_file in self.agents_dir.glob("*.md"):
            if md_file.name not in exclude_patterns:
                try:
                    with open(md_file, 'r', encoding='utf-8') as f:
                        first_line = f.readline().strip()
                    
                    if first_line.startswith('environment:'):
                        files_to_fix.append(md_file)
                except:
                    continue
        
        print(f"Found {len(files_to_fix)} files with YAML structure issues")
        
        results = {"fixed": 0, "already_correct": 0, "failed": 0}
        
        for agent_file in files_to_fix:
            if self.fix_yaml_structure(agent_file):
                results["fixed"] += 1
            else:
                results["failed"] += 1
        
        return results

def main():
    """Main function"""
    fixer = YAMLStructureFixer()
    
    print("🚀 Fixing YAML structure issues...")
    results = fixer.fix_all_yaml_issues()
    
    print(f"\n📊 YAML Structure Fix Summary:")
    print(f"  Fixed: {results['fixed']}")
    print(f"  Failed: {results['failed']}")
    
    if results["failed"] > 0:
        print("Some files could not be fixed. Manual intervention may be required.")

if __name__ == "__main__":
    main()