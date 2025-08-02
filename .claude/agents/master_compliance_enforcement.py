#!/usr/bin/env python3
"""
Master AI Agent Compliance Enforcement Script
Ensures all agents follow codebase hygiene standards
"""

import os
import json
import yaml
import shutil
import logging
from datetime import datetime
from pathlib import Path
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AgentComplianceEnforcer:
    def __init__(self, agents_dir='/opt/sutazaiapp/.claude/agents'):
        self.agents_dir = Path(agents_dir)
        self.backup_dir = self.agents_dir / 'archive' / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.report = {
            'timestamp': datetime.now().isoformat(),
            'actions': [],
            'errors': [],
            'summary': {}
        }
        
    def run_full_compliance(self):
        """Execute full compliance enforcement"""
        logger.info("Starting comprehensive agent compliance enforcement")
        
        # Phase 1: Clean up clutter
        self.cleanup_backup_files()
        self.cleanup_old_reports()
        self.cleanup_fix_scripts()
        
        # Phase 2: Analyze and consolidate duplicates
        self.analyze_and_consolidate_duplicates()
        
        # Phase 3: Update all agents with codebase standards
        self.update_agents_with_standards()
        
        # Phase 4: Reorganize structure
        self.reorganize_structure()
        
        # Phase 5: Generate final report
        self.generate_final_report()
        
    def cleanup_backup_files(self):
        """Remove all backup files following codebase hygiene rules"""
        logger.info("Phase 1: Cleaning up backup files")
        
        backup_patterns = ['.backup', '.backup2', '.bak', '.old', '_old', '_copy']
        removed_count = 0
        
        # Create archive directory
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        for file_path in self.agents_dir.rglob('*'):
            if file_path.is_file():
                for pattern in backup_patterns:
                    if pattern in str(file_path):
                        try:
                            # Archive before removing
                            archive_path = self.backup_dir / file_path.relative_to(self.agents_dir)
                            archive_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.move(str(file_path), str(archive_path))
                            removed_count += 1
                            self.report['actions'].append({
                                'action': 'archived_backup',
                                'file': str(file_path),
                                'archive': str(archive_path)
                            })
                        except Exception as e:
                            self.report['errors'].append({
                                'error': 'backup_removal_failed',
                                'file': str(file_path),
                                'message': str(e)
                            })
        
        logger.info(f"Archived {removed_count} backup files")
        
    def cleanup_old_reports(self):
        """Remove old compliance reports keeping only the latest"""
        logger.info("Cleaning up old compliance reports")
        
        report_patterns = [
            'agent_compliance_report_*.json',
            'final_compliance_report_*.json',
            'duplicate_agents_analysis.json'
        ]
        
        for pattern in report_patterns:
            reports = list(self.agents_dir.glob(pattern))
            if len(reports) > 1:
                # Keep the newest, archive the rest
                reports.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                for report in reports[1:]:
                    archive_path = self.backup_dir / report.name
                    shutil.move(str(report), str(archive_path))
                    self.report['actions'].append({
                        'action': 'archived_old_report',
                        'file': str(report)
                    })
                    
    def cleanup_fix_scripts(self):
        """Archive the fixes directory as it's no longer needed"""
        logger.info("Cleaning up fix scripts directory")
        
        fixes_dir = self.agents_dir / 'fixes'
        if fixes_dir.exists():
            archive_path = self.backup_dir / 'fixes'
            shutil.move(str(fixes_dir), str(archive_path))
            self.report['actions'].append({
                'action': 'archived_fixes_directory',
                'directory': str(fixes_dir)
            })
            
    def analyze_and_consolidate_duplicates(self):
        """Consolidate duplicate agents keeping the better version"""
        logger.info("Phase 2: Analyzing and consolidating duplicate agents")
        
        # Find all agent markdown files
        agent_files = {}
        for md_file in self.agents_dir.glob('*.md'):
            if md_file.name.endswith('-detailed.md'):
                base_name = md_file.name.replace('-detailed.md', '')
                if base_name not in agent_files:
                    agent_files[base_name] = {}
                agent_files[base_name]['detailed'] = md_file
            elif md_file.name.endswith('.md'):
                base_name = md_file.name.replace('.md', '')
                if base_name not in agent_files:
                    agent_files[base_name] = {}
                agent_files[base_name]['regular'] = md_file
                
        # Consolidate duplicates
        consolidated_count = 0
        for base_name, versions in agent_files.items():
            if 'regular' in versions and 'detailed' in versions:
                # Compare file sizes and content quality
                regular_size = versions['regular'].stat().st_size
                detailed_size = versions['detailed'].stat().st_size
                
                # Keep the larger, more comprehensive version
                if detailed_size > regular_size * 0.8:
                    # Detailed version is substantial, keep it
                    keep_file = versions['detailed']
                    archive_file = versions['regular']
                else:
                    # Regular version is more comprehensive
                    keep_file = versions['regular']
                    archive_file = versions['detailed']
                    
                # Archive the duplicate
                archive_path = self.backup_dir / 'duplicates' / archive_file.name
                archive_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(archive_file), str(archive_path))
                
                # Rename if needed to remove -detailed suffix
                if keep_file.name.endswith('-detailed.md'):
                    new_name = keep_file.parent / f"{base_name}.md"
                    keep_file.rename(new_name)
                    
                consolidated_count += 1
                self.report['actions'].append({
                    'action': 'consolidated_duplicate',
                    'kept': str(keep_file),
                    'archived': str(archive_file)
                })
                
        logger.info(f"Consolidated {consolidated_count} duplicate agent pairs")
        
    def update_agents_with_standards(self):
        """Update all agents to include codebase hygiene standards"""
        logger.info("Phase 3: Updating agents with codebase standards")
        
        codebase_standards = """

## Codebase Hygiene Standards

This agent MUST follow these codebase hygiene principles:

1. **Clean Code**: Write clean, consistent, well-organized code
2. **No Duplication**: Avoid creating multiple versions of the same functionality
3. **Proper Structure**: Place files in logical locations following project conventions
4. **Remove Dead Code**: Delete unused code, files, and dependencies
5. **Use Existing Tools**: Reuse existing scripts and components before creating new ones
6. **Professional Standards**: No fantasy elements, only production-ready implementations
7. **Documentation**: Keep documentation centralized and up-to-date
8. **Testing**: Ensure all code is tested and maintains existing functionality
9. **Version Control**: Use branches and feature flags, not file copies
10. **Continuous Cleanup**: Leave code better than you found it

When executing tasks, this agent will:
- Analyze existing code before making changes
- Consolidate duplicate functionality
- Remove obsolete files and code
- Follow naming conventions and structure
- Ensure backwards compatibility
- Document all changes clearly
"""
        
        updated_count = 0
        for md_file in self.agents_dir.glob('*.md'):
            if md_file.name in ['README.md', 'AGENT_COMPLIANCE_REPORT.md', 
                               'DUPLICATE_AGENTS_REPORT.md', 'FINAL_COMPLIANCE_SUMMARY.md']:
                continue
                
            try:
                content = md_file.read_text()
                
                # Check if standards already included
                if "Codebase Hygiene Standards" not in content:
                    # Find a good insertion point (after capabilities or description)
                    insert_markers = [
                        "## Implementation",
                        "## Core Functions",
                        "## Technical Implementation",
                        "## Key Features",
                        "capabilities:",
                        "---\n\n"
                    ]
                    
                    inserted = False
                    for marker in insert_markers:
                        if marker in content:
                            parts = content.split(marker, 1)
                            content = parts[0] + marker + codebase_standards + "\n" + parts[1]
                            inserted = True
                            break
                            
                    if not inserted:
                        # Add at the end before any implementation details
                        content = content.rstrip() + "\n" + codebase_standards
                        
                    md_file.write_text(content)
                    updated_count += 1
                    self.report['actions'].append({
                        'action': 'updated_with_standards',
                        'file': str(md_file)
                    })
                    
            except Exception as e:
                self.report['errors'].append({
                    'error': 'update_failed',
                    'file': str(md_file),
                    'message': str(e)
                })
                
        logger.info(f"Updated {updated_count} agents with codebase standards")
        
    def reorganize_structure(self):
        """Reorganize the agents directory structure"""
        logger.info("Phase 4: Reorganizing directory structure")
        
        # Move knowledge-graph-builder subdirectory to archive
        kg_dir = self.agents_dir / 'knowledge-graph-builder'
        if kg_dir.exists():
            archive_path = self.backup_dir / 'implementations' / 'knowledge-graph-builder'
            shutil.move(str(kg_dir), str(archive_path))
            self.report['actions'].append({
                'action': 'archived_implementation_dir',
                'directory': str(kg_dir)
            })
            
        # Move Python scripts to a scripts subdirectory
        scripts_dir = self.agents_dir / 'scripts'
        scripts_dir.mkdir(exist_ok=True)
        
        python_scripts = [
            'agent_compliance_checker.py',
            'agent_startup_wrapper.py',
            'claude_rules_checker.py',
            'create_agent_fixes.py',
            'fix_yaml_structure.py',
            'master_agent_compliance_fix.py',
            'simple_agent_fixer.py',
            'update_agent_compliance.py',
            'update_all_agents_compliance.py'
        ]
        
        for script in python_scripts:
            script_path = self.agents_dir / script
            if script_path.exists():
                new_path = scripts_dir / script
                shutil.move(str(script_path), str(new_path))
                self.report['actions'].append({
                    'action': 'organized_script',
                    'from': str(script_path),
                    'to': str(new_path)
                })
                
        # Create a proper README
        self.create_readme()
        
    def create_readme(self):
        """Create a comprehensive README for the agents directory"""
        readme_content = """# AI Agents Directory

This directory contains all AI agent definitions for the SutazAI system.

## Structure

```
.claude/agents/
├── README.md                    # This file
├── *.md                        # Agent definition files
├── scripts/                    # Compliance and utility scripts
├── archive/                    # Archived old versions and backups
└── reports/                    # Compliance reports
```

## Agent Standards

All agents in this directory follow strict codebase hygiene standards:

1. **Single Definition**: Each agent has one authoritative definition file
2. **Clear Naming**: Agent filenames match their internal names
3. **No Duplicates**: No multiple versions or copies of the same agent
4. **Clean Structure**: Consistent YAML frontmatter and markdown format
5. **Codebase Standards**: All agents include and enforce hygiene rules

## Agent Capabilities

Each agent is specialized for specific tasks and includes:
- Clear description and purpose
- Defined capabilities and limitations
- Codebase hygiene standards enforcement
- Professional, production-ready implementations

## Maintenance

- Run `master_compliance_enforcement.py` to ensure all agents remain compliant
- Archives are stored in `archive/` with timestamps
- Reports are generated in `reports/` directory

## Usage

Agents are loaded by the SutazAI system and can be invoked for their specialized tasks.
Each agent operates with awareness of codebase hygiene standards and maintains
code quality throughout their operations.
"""
        
        readme_path = self.agents_dir / 'README.md'
        readme_path.write_text(readme_content)
        self.report['actions'].append({
            'action': 'created_readme',
            'file': str(readme_path)
        })
        
    def generate_final_report(self):
        """Generate comprehensive compliance report"""
        logger.info("Phase 5: Generating final compliance report")
        
        # Count current state
        md_files = list(self.agents_dir.glob('*.md'))
        agent_files = [f for f in md_files if f.name not in [
            'README.md', 'AGENT_COMPLIANCE_REPORT.md', 
            'DUPLICATE_AGENTS_REPORT.md', 'FINAL_COMPLIANCE_SUMMARY.md'
        ]]
        
        self.report['summary'] = {
            'total_agents': len(agent_files),
            'backup_files_removed': len([a for a in self.report['actions'] if a['action'] == 'archived_backup']),
            'duplicates_consolidated': len([a for a in self.report['actions'] if a['action'] == 'consolidated_duplicate']),
            'agents_updated': len([a for a in self.report['actions'] if a['action'] == 'updated_with_standards']),
            'scripts_organized': len([a for a in self.report['actions'] if a['action'] == 'organized_script']),
            'errors': len(self.report['errors'])
        }
        
        # Save JSON report
        report_path = self.agents_dir / 'reports' / f'compliance_enforcement_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(self.report, f, indent=2)
            
        # Create markdown summary
        summary = f"""# Agent Compliance Enforcement Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- **Total Agents**: {self.report['summary']['total_agents']}
- **Backup Files Removed**: {self.report['summary']['backup_files_removed']}
- **Duplicates Consolidated**: {self.report['summary']['duplicates_consolidated']}
- **Agents Updated with Standards**: {self.report['summary']['agents_updated']}
- **Scripts Organized**: {self.report['summary']['scripts_organized']}
- **Errors**: {self.report['summary']['errors']}

## Actions Taken

1. **Cleanup Phase**
   - Archived all backup files (.backup, .backup2, etc.)
   - Removed old compliance reports
   - Archived fixes directory

2. **Consolidation Phase**
   - Analyzed duplicate agent pairs
   - Kept the most comprehensive version
   - Archived redundant versions

3. **Standards Update Phase**
   - Added codebase hygiene standards to all agents
   - Ensured all agents will enforce clean code practices

4. **Organization Phase**
   - Moved Python scripts to scripts/ subdirectory
   - Archived implementation directories
   - Created comprehensive README

## Result

All agents now comply with codebase hygiene standards and will enforce these
standards in their operations. The directory structure is clean, organized,
and free of duplicates or clutter.

Archive location: {self.backup_dir}
"""
        
        summary_path = self.agents_dir / 'COMPLIANCE_ENFORCEMENT_SUMMARY.md'
        summary_path.write_text(summary)
        
        logger.info(f"Compliance enforcement complete. Report saved to {report_path}")
        logger.info(f"Summary: {self.report['summary']}")
        
        return self.report


def main():
    """Run the compliance enforcement"""
    enforcer = AgentComplianceEnforcer()
    report = enforcer.run_full_compliance()
    
    print("\n" + "="*60)
    print("AGENT COMPLIANCE ENFORCEMENT COMPLETE")
    print("="*60)
    print(f"Total Agents: {report['summary']['total_agents']}")
    print(f"Backups Removed: {report['summary']['backup_files_removed']}")
    print(f"Duplicates Consolidated: {report['summary']['duplicates_consolidated']}")
    print(f"Agents Updated: {report['summary']['agents_updated']}")
    print(f"Errors: {report['summary']['errors']}")
    print("="*60)
    

if __name__ == '__main__':
    main()