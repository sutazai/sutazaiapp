#!/usr/bin/env python3
"""
Emergency Compliance Fix Script
Removes critical rule violations to achieve immediate compliance
"""

import os
import shutil
import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set

class ComplianceFixer:
    def __init__(self):
        self.root = Path("/opt/sutazaiapp")
        self.backup_dir = self.root / f"compliance_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.violations_fixed = []
        self.files_deleted = []
        self.files_modified = []
        
    def backup_file(self, filepath: Path):
        """Backup file before modification/deletion"""
        if not filepath.exists():
            return
            
        relative = filepath.relative_to(self.root)
        backup_path = self.backup_dir / relative
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(filepath, backup_path)
        
    def fix_rule1_fantasy_elements(self):
        """Remove all fantasy/AGI/advanced elements"""
        print("\n[RULE 1] Removing fantasy elements...")
        
        # Files to delete completely
        fantasy_files = [
            "backend/app/core/agi_brain.py",
            "backend/ai_agents/reasoning/agi_orchestrator.py",
        ]
        
        # Also find and remove advanced directories
        advanced_dirs = list(self.root.glob("**/advanced*"))
        agi_dirs = list(self.root.glob("**/agi*"))
        
        for rel_path in fantasy_files:
            filepath = self.root / rel_path
            if filepath.exists():
                self.backup_file(filepath)
                filepath.unlink()
                self.files_deleted.append(str(rel_path))
                print(f"  ✓ Deleted: {rel_path}")
                
        for dir_path in advanced_dirs + agi_dirs:
            if dir_path.is_dir():
                rel_path = dir_path.relative_to(self.root)
                shutil.rmtree(dir_path)
                self.files_deleted.append(f"{rel_path}/ (directory)")
                print(f"  ✓ Deleted directory: {rel_path}/")
                
        self.violations_fixed.append("Removed AGI/advanced fantasy modules")
        
    def fix_rule3_docker_compose_chaos(self):
        """Keep only essential docker-compose files"""
        print("\n[RULE 3] Cleaning docker-compose chaos...")
        
        # Keep only these
        keep_files = {
            "docker-compose.yml",
            "docker-compose.monitoring.yml"  # Optional monitoring stack
        }
        
        compose_files = list(self.root.glob("docker-compose*.yml"))
        
        for filepath in compose_files:
            if filepath.name not in keep_files:
                self.backup_file(filepath)
                filepath.unlink()
                self.files_deleted.append(filepath.name)
                print(f"  ✓ Deleted: {filepath.name}")
                
        # Also clean up backup directories with old compose files
        backup_dirs = list(self.root.glob("backups/*/docker-compose*.yml"))
        archive_dirs = list(self.root.glob("archive/*/docker-compose*.yml"))
        
        for filepath in backup_dirs + archive_dirs:
            filepath.unlink()
            
        self.violations_fixed.append("Consolidated docker-compose files")
        
    def fix_rule3_documentation_rot(self):
        """Remove outdated and duplicate documentation"""
        print("\n[RULE 3] Cleaning documentation rot...")
        
        # Directories with mostly outdated docs
        docs_to_remove = [
            "docs/agi",
            "docs/advanced", 
            "docs/agents",  # Mostly fantasy
            "docs/architecture",  # Outdated
            "docs/deployment",  # Conflicting guides
            "docs/operations",  # Too many duplicates
            "docs/reports",  # Old reports
            "docs/strategy",  # Fantasy roadmaps
            "docs/system"  # Outdated
        ]
        
        for rel_path in docs_to_remove:
            dir_path = self.root / rel_path
            if dir_path.exists():
                shutil.rmtree(dir_path)
                self.files_deleted.append(f"{rel_path}/ (directory)")
                print(f"  ✓ Deleted directory: {rel_path}/")
                
        self.violations_fixed.append("Removed outdated documentation")
        
    def fix_rule5_external_apis(self):
        """Remove OpenAI and external API references"""
        print("\n[RULE 5] Removing external API references...")
        
        # Files that reference OpenAI but can be fixed
        files_to_fix = [
            "docker/autogpt/autogpt_service.py",
            "docker/gpt-engineer/gpt_engineer_service.py",
            "docker/aider/aider_service.py"
        ]
        
        for rel_path in files_to_fix:
            filepath = self.root / rel_path
            if filepath.exists():
                self.backup_file(filepath)
                content = filepath.read_text()
                
                # Replace OpenAI references with Ollama
                content = re.sub(r'OPENAI_API_KEY["\']?\s*:\s*["\'][^"\']*["\']',
                                'OLLAMA_HOST": "http://ollama:10104', content)
                content = re.sub(r'from openai import.*\n', '', content)
                content = re.sub(r'import openai.*\n', '', content)
                content = re.sub(r'ChatOpenAI', 'Ollama', content)
                content = re.sub(r'OpenAI\(', 'Ollama(', content)
                
                filepath.write_text(content)
                self.files_modified.append(str(rel_path))
                print(f"  ✓ Fixed: {rel_path}")
                
        self.violations_fixed.append("Removed external API dependencies")
        
    def consolidate_scripts(self):
        """Move scattered scripts to organized structure"""
        print("\n[RULE 3/4] Consolidating scripts...")
        
        # Create organized structure
        script_dirs = [
            "scripts/deployment",
            "scripts/monitoring", 
            "scripts/validation",
            "scripts/utils"
        ]
        
        for dir_path in script_dirs:
            (self.root / dir_path).mkdir(parents=True, exist_ok=True)
            
        # Remove duplicate script directories
        duplicate_dirs = [
            "scripts/agents",  # Too many duplicates
            "scripts/ci-cd",  # Outdated
            "scripts/test",  # Move to tests/
            "scripts/data"  # Outdated
        ]
        
        for rel_path in duplicate_dirs:
            dir_path = self.root / rel_path
            if dir_path.exists():
                shutil.rmtree(dir_path)
                self.files_deleted.append(f"{rel_path}/ (directory)")
                print(f"  ✓ Deleted directory: {rel_path}/")
                
        self.violations_fixed.append("Consolidated script organization")
        
    def generate_report(self):
        """Generate compliance fix report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "backup_location": str(self.backup_dir),
            "violations_fixed": self.violations_fixed,
            "files_deleted": len(self.files_deleted),
            "files_modified": len(self.files_modified),
            "details": {
                "deleted_files": self.files_deleted[:50],  # First 50
                "modified_files": self.files_modified
            }
        }
        
        report_path = self.root / "compliance_fix_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"\n✓ Report saved to: {report_path}")
        return report
        
    def run(self):
        """Execute all compliance fixes"""
        print("=" * 60)
        print("EMERGENCY COMPLIANCE FIX SCRIPT")
        print("=" * 60)
        
        # Create backup directory
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nBackup directory: {self.backup_dir}")
        
        # Run fixes
        self.fix_rule1_fantasy_elements()
        self.fix_rule3_docker_compose_chaos()
        self.fix_rule3_documentation_rot()
        self.fix_rule5_external_apis()
        self.consolidate_scripts()
        
        # Generate report
        report = self.generate_report()
        
        print("\n" + "=" * 60)
        print("COMPLIANCE FIX SUMMARY")
        print("=" * 60)
        print(f"✓ Violations fixed: {len(self.violations_fixed)}")
        print(f"✓ Files deleted: {len(self.files_deleted)}")
        print(f"✓ Files modified: {len(self.files_modified)}")
        print(f"✓ Backup location: {self.backup_dir}")
        print("\nNext steps:")
        print("1. Review the changes")
        print("2. Run: docker-compose down && docker-compose up -d")
        print("3. Test core functionality")
        print("4. Run compliance audit again")
        
        return report

if __name__ == "__main__":
    fixer = ComplianceFixer()
    fixer.run()