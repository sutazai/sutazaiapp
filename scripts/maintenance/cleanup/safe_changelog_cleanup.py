"""
Safe CHANGELOG.md Cleanup Script
=================================

Safely removes auto-generated CHANGELOG.md waste files while preserving legitimate changelog files.
Created in response to critical file proliferation analysis (556 CHANGELOG.md files found).

Author: Garbage Collection System
Date: 2025-08-18
Purpose: Remove 382+ auto-generated template files, preserve legitimate changelogs
"""

import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
import re
import argparse
import json

class ChangelogCleanupManager:
    def __init__(self, root_path: str = "/opt/sutazaiapp"):
        self.root_path = Path(root_path)
        self.backup_path = self.root_path / "cleanup_backup" / "changelogs"
        self.report_path = self.root_path / "docs" / "reports"
        self.dry_run = False
        
        self.waste_signatures = [
            "rule-enforcement-system",
            "2025-08-18 15:05:54 UTC",
            "Establishing CHANGELOG.md for Rule 18 compliance",
            "every directory must have change tracking"
        ]
        
        self.preserve_patterns = [
            r"SutazAI.*platform",
            r"Database.*Investigation",
            r"Documentation Change Log",
            r"Version \d+\.\d+\.\d+",
            r"BREAKING CHANGE",
            r"feature:",
            r"fix:",
            r"Added:",
            r"Changed:",
            r"Deprecated:",
            r"Removed:",
            r"Fixed:",
            r"Security:"
        ]
        
    def create_backup_directory(self):
        """Create backup directory for safety."""
        self.backup_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created backup directory: {self.backup_path}")
        
    def find_all_changelogs(self) -> List[Path]:
        """Find all CHANGELOG.md files excluding node_modules."""
        changelogs = []
        for changelog in self.root_path.rglob("CHANGELOG.md"):
            if "node_modules" not in str(changelog):
                changelogs.append(changelog)
        return sorted(changelogs)
    
    def analyze_changelog_file(self, filepath: Path) -> Dict:
        """Analyze a single changelog file to determine if it's waste or legitimate."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            analysis = {
                'filepath': str(filepath),
                'size': filepath.stat().st_size,
                'line_count': len(content.splitlines()),
                'is_empty': len(content.strip()) == 0,
                'has_waste_signatures': False,
                'has_preserve_patterns': False,
                'classification': 'UNKNOWN',
                'confidence': 0.0,
                'created_time': datetime.fromtimestamp(filepath.stat().st_ctime).isoformat(),
                'modified_time': datetime.fromtimestamp(filepath.stat().st_mtime).isoformat()
            }
            
            waste_score = 0
            for signature in self.waste_signatures:
                if signature in content:
                    analysis['has_waste_signatures'] = True
                    waste_score += 1
                    
            preserve_score = 0
            for pattern in self.preserve_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    analysis['has_preserve_patterns'] = True
                    preserve_score += 1
                    
            if analysis['is_empty']:
                analysis['classification'] = 'SAFE_REMOVE'
                analysis['confidence'] = 1.0
            elif waste_score >= 2 and preserve_score == 0:
                analysis['classification'] = 'SAFE_REMOVE'
                analysis['confidence'] = 0.95
            elif waste_score >= 1 and preserve_score == 0 and analysis['line_count'] == 44:
                analysis['classification'] = 'SAFE_REMOVE'
                analysis['confidence'] = 0.9
            elif preserve_score > 0:
                analysis['classification'] = 'PRESERVE'
                analysis['confidence'] = 0.8 + (preserve_score * 0.05)
            elif analysis['size'] < 100:
                analysis['classification'] = 'SAFE_REMOVE'
                analysis['confidence'] = 0.7
            else:
                analysis['classification'] = 'INVESTIGATE'
                analysis['confidence'] = 0.5
                
            return analysis
            
        except Exception as e:
            return {
                'filepath': str(filepath),
                'error': str(e),
                'classification': 'INVESTIGATE',
                'confidence': 0.0
            }
    
    def backup_file(self, filepath: Path) -> bool:
        """Backup a file before removal."""
        try:
            relative_path = filepath.relative_to(self.root_path)
            backup_file = self.backup_path / f"{str(relative_path).replace('/', '_')}"
            shutil.copy2(filepath, backup_file)
            return True
        except Exception as e:
            print(f"⚠️ Failed to backup {filepath}: {e}")
            return False
    
    def safe_remove_file(self, filepath: Path, analysis: Dict) -> bool:
        """Safely remove a file with backup."""
        if self.dry_run:
            print(f"🔍 [DRY RUN] Would remove: {filepath}")
            return True
            
        try:
            if self.backup_file(filepath):
                filepath.unlink()
                print(f"🗑️ Removed: {filepath} (confidence: {analysis['confidence']:.1%})")
                return True
            else:
                print(f"⚠️ Skipped removal (backup failed): {filepath}")
                return False
        except Exception as e:
            print(f"❌ Failed to remove {filepath}: {e}")
            return False
    
    def execute_cleanup(self, dry_run: bool = False) -> Dict:
        """Execute the complete cleanup process."""
        self.dry_run = dry_run
        
        if not dry_run:
            self.create_backup_directory()
            
        print(f"🔍 Starting CHANGELOG.md cleanup analysis...")
        print(f"📁 Root directory: {self.root_path}")
        print(f"🔧 Mode: {'DRY RUN' if dry_run else 'LIVE EXECUTION'}")
        
        all_changelogs = self.find_all_changelogs()
        print(f"📊 Found {len(all_changelogs)} CHANGELOG.md files")
        
        analyses = []
        for changelog in all_changelogs:
            analysis = self.analyze_changelog_file(changelog)
            analyses.append(analysis)
            
        safe_remove = [a for a in analyses if a['classification'] == 'SAFE_REMOVE']
        preserve = [a for a in analyses if a['classification'] == 'PRESERVE']
        investigate = [a for a in analyses if a['classification'] == 'INVESTIGATE']
        
        print(f"\n📈 ANALYSIS RESULTS:")
        print(f"🗑️ Safe to remove: {len(safe_remove)} files")
        print(f"🛡️ Preserve: {len(preserve)} files")
        print(f"🔍 Investigate: {len(investigate)} files")
        
        removed_count = 0
        for analysis in safe_remove:
            if analysis['confidence'] >= 0.8:
                filepath = Path(analysis['filepath'])
                if self.safe_remove_file(filepath, analysis):
                    removed_count += 1
                    
        summary = {
            'timestamp': datetime.utcnow().isoformat(),
            'total_files_found': len(all_changelogs),
            'safe_to_remove': len(safe_remove),
            'preserved': len(preserve),
            'investigate': len(investigate),
            'actually_removed': removed_count,
            'dry_run': dry_run,
            'analyses': analyses
        }
        
        report_file = self.report_path / f"changelog_cleanup_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"\n✅ CLEANUP COMPLETE:")
        print(f"🗑️ Files removed: {removed_count}")
        print(f"🛡️ Files preserved: {len(preserve)}")
        print(f"🔍 Files requiring investigation: {len(investigate)}")
        print(f"📄 Detailed report: {report_file}")
        
        if not dry_run and removed_count > 0:
            print(f"💾 Backup location: {self.backup_path}")
            
        return summary
    
    def list_preserved_files(self):
        """Show which files will be preserved."""
        changelogs = self.find_all_changelogs()
        print("\n🛡️ FILES TO PRESERVE (High Value Changelogs):")
        
        for changelog in changelogs:
            analysis = self.analyze_changelog_file(changelog)
            if analysis['classification'] == 'PRESERVE':
                print(f"  ✅ {changelog}")
                print(f"     Size: {analysis['size']} bytes, Lines: {analysis['line_count']}")
                
    def list_removal_candidates(self):
        """Show which files are candidates for removal."""
        changelogs = self.find_all_changelogs()
        print("\n🗑️ FILES TO REMOVE (Auto-Generated Waste):")
        
        for changelog in changelogs:
            analysis = self.analyze_changelog_file(changelog)
            if analysis['classification'] == 'SAFE_REMOVE' and analysis['confidence'] >= 0.8:
                print(f"  ❌ {changelog}")
                print(f"     Confidence: {analysis['confidence']:.1%}, Size: {analysis['size']} bytes")

def main():
    parser = argparse.ArgumentParser(description="Safe CHANGELOG.md cleanup tool")
    parser.add_argument('--dry-run', action='store_true', help='Show what would be removed without actually removing')
    parser.add_argument('--list-preserve', action='store_true', help='List files that will be preserved')
    parser.add_argument('--list-remove', action='store_true', help='List files that will be removed')
    parser.add_argument('--root', default='/opt/sutazaiapp', help='Root directory to clean')
    
    args = parser.parse_args()
    
    cleaner = ChangelogCleanupManager(args.root)
    
    if args.list_preserve:
        cleaner.list_preserved_files()
    elif args.list_remove:
        cleaner.list_removal_candidates()
    else:
        cleaner.execute_cleanup(dry_run=args.dry_run)

if __name__ == "__main__":
    main()