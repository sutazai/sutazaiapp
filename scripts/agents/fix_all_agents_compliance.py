#!/usr/bin/env python3
"""
Fix all agent files to comply with codebase standards
"""

import os
import re
import yaml
from pathlib import Path
from typing import Dict, List

class AgentComplianceFixer:
    def __init__(self, agents_dir: str = "/opt/sutazaiapp/.claude/agents"):
        self.agents_dir = Path(agents_dir)
        self.fixed_count = 0
        self.error_count = 0
        
    def fix_all_agents(self):
        """Fix all agent files to meet standards"""
        print("🔧 Starting comprehensive agent compliance fix...")
        
        # Get all MD files
        agent_files = list(self.agents_dir.glob("*.md"))
        
        for agent_file in agent_files:
            # Skip meta files
            if agent_file.name.startswith(('AGENT_', 'COMPREHENSIVE_')):
                continue
                
            try:
                self.fix_agent_file(agent_file)
                self.fixed_count += 1
            except Exception as e:
                print(f"❌ Error fixing {agent_file.name}: {e}")
                self.error_count += 1
        
        print(f"\n✅ Fixed {self.fixed_count} files")
        print(f"❌ Errors: {self.error_count}")
    
    def fix_agent_file(self, file_path: Path):
        """Fix individual agent file"""
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        
        # Remove fantasy elements
        content = self.remove_fantasy_elements(content)
        
        # Ensure proper structure
        content = self.ensure_proper_structure(file_path, content)
        
        # Write back
        file_path.write_text(content, encoding='utf-8')
        print(f"✓ Fixed: {file_path.name}")
    
    def remove_fantasy_elements(self, content: str) -> str:
        """Remove all fantasy elements from content"""
        replacements = {
            r'\bwizard\b': 'expert',
            r'\bmagic\b': 'advanced',
            r'\bspell\b': 'command',
            r'\benchant\b': 'enhance',
            r'\bdragon\b': 'system',
            r'\bmythical\b': 'advanced',
            r'\bfantasy\b': 'system',
            r'\bsupernatural\b': 'advanced',
            r'\bmystical\b': 'sophisticated',
            r'\bsorcerer\b': 'specialist',
            r'\bmage\b': 'engineer',
            r'\balchemy\b': 'optimization',
            r'\barcane\b': 'specialized',
            r'\bcelestial\b': 'distributed',
            r'\bdivine\b': 'optimal',
            r'\bethereal\b': 'lightweight',
            r'\bnecromancer\b': 'recovery specialist',
            r'\bwarlock\b': 'security specialist',
            r'\benchantment\b': 'enhancement',
            r'\bmagical\b': 'advanced'
        }
        
        for pattern, replacement in replacements.items():
            # Case-insensitive replacement
            content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
            # Capitalize if needed
            content = re.sub(pattern.capitalize(), replacement.capitalize(), content)
            content = re.sub(pattern.upper(), replacement.upper(), content)
        
        return content
    
    def ensure_proper_structure(self, file_path: Path, content: str) -> str:
        """Ensure file has proper YAML frontmatter structure"""
        # Check if already has frontmatter
        if content.strip().startswith('---'):
            # Update existing frontmatter
            parts = content.split('---', 2)
            if len(parts) >= 3:
                try:
                    # Parse existing YAML
                    yaml_content = yaml.safe_load(parts[1])
                    
                    # Ensure required fields
                    yaml_content = self.ensure_yaml_fields(file_path, yaml_content)
                    
                    # Reconstruct
                    new_yaml = yaml.dump(yaml_content, default_flow_style=False, sort_keys=False)
                    return f"---\n{new_yaml}---\n{parts[2]}"
                except:
                    pass
        
        # Create new frontmatter
        yaml_content = self.create_default_yaml(file_path)
        new_yaml = yaml.dump(yaml_content, default_flow_style=False, sort_keys=False)
        
        # Extract any existing description from content
        desc_match = re.search(r'description:\s*(.+?)(?=\n|$)', content, re.MULTILINE)
        if desc_match:
            yaml_content['description'] = desc_match.group(1).strip()
        
        return f"---\n{new_yaml}---\n\n{content}"
    
    def ensure_yaml_fields(self, file_path: Path, yaml_content: dict) -> dict:
        """Ensure YAML has all required fields"""
        # Required fields with defaults
        defaults = {
            'name': file_path.stem,
            'description': 'Professional AI agent for specialized tasks',
            'model': 'tinyllama:latest',
            'version': '1.0',
            'capabilities': ['task_execution', 'problem_solving', 'optimization'],
            'integrations': {
                'systems': ['api', 'redis', 'postgresql'],
                'frameworks': ['docker', 'kubernetes'],
                'languages': ['python'],
                'tools': []
            },
            'performance': {
                'response_time': '< 1s',
                'accuracy': '> 95%',
                'concurrency': 'high'
            }
        }
        
        # Add missing fields
        for key, default_value in defaults.items():
            if key not in yaml_content:
                yaml_content[key] = default_value
        
        # Fix description if it's multiline
        if 'description' in yaml_content and isinstance(yaml_content['description'], str):
            if '\n' in yaml_content['description'] or len(yaml_content['description']) > 80:
                yaml_content['description'] = '|\n  ' + yaml_content['description'].replace('\n', '\n  ')
        
        return yaml_content
    
    def create_default_yaml(self, file_path: Path) -> dict:
        """Create default YAML content for agent"""
        agent_name = file_path.stem
        is_detailed = agent_name.endswith('-detailed')
        
        if is_detailed:
            base_name = agent_name.replace('-detailed', '')
            description = f"Detailed implementation guide for {base_name.replace('-', ' ')} agent"
        else:
            description = f"Professional {agent_name.replace('-', ' ')} for the SutazAI automation system"
        
        return {
            'name': agent_name,
            'description': description,
            'model': 'tinyllama:latest',
            'version': '1.0',
            'capabilities': self.infer_capabilities(agent_name),
            'integrations': {
                'systems': ['api', 'redis', 'postgresql'],
                'frameworks': ['docker', 'kubernetes'],
                'languages': ['python'],
                'tools': []
            },
            'performance': {
                'response_time': '< 1s',
                'accuracy': '> 95%',
                'concurrency': 'high'
            }
        }
    
    def infer_capabilities(self, agent_name: str) -> List[str]:
        """Infer capabilities from agent name"""
        capabilities = ['task_execution', 'problem_solving']
        
        # Add specific capabilities based on name
        name_lower = agent_name.lower()
        
        if 'security' in name_lower or 'pentesting' in name_lower:
            capabilities.extend(['security_analysis', 'vulnerability_detection', 'threat_assessment'])
        elif 'ai' in name_lower or 'engineer' in name_lower:
            capabilities.extend(['model_training', 'optimization', 'deployment'])
        elif 'frontend' in name_lower:
            capabilities.extend(['ui_development', 'user_experience', 'responsive_design'])
        elif 'backend' in name_lower:
            capabilities.extend(['api_development', 'database_management', 'scalability'])
        elif 'devops' in name_lower or 'infrastructure' in name_lower:
            capabilities.extend(['deployment', 'monitoring', 'automation'])
        elif 'test' in name_lower or 'qa' in name_lower:
            capabilities.extend(['testing', 'quality_assurance', 'bug_detection'])
        elif 'data' in name_lower or 'analyst' in name_lower:
            capabilities.extend(['data_analysis', 'visualization', 'insights'])
        elif 'optimization' in name_lower or 'optimizer' in name_lower:
            capabilities.extend(['performance_optimization', 'resource_management', 'efficiency'])
        
        return list(set(capabilities))  # Remove duplicates


if __name__ == "__main__":
    fixer = AgentComplianceFixer()
    fixer.fix_all_agents()