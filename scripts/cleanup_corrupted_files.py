#!/usr/bin/env python3
"""
Cleanup Script for Corrupted Files with "" Pattern
Author: Claude Code
Date: 2025-08-17
Purpose: Fix files corrupted with placeholder text pattern
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple
import shutil
from datetime import datetime

# Define the corruption patterns to fix
CORRUPTION_PATTERNS = [
    # The main repeated pattern
    (r'Mock', 'Mock'),
    # Variations that might exist
    (r'Mocks', 'Mock'),
    (r'Mock', 'Mock'),
    # Clean up any standalone patterns
    (r'', ''),
]

# Files to skip (system files, binaries, etc.)
SKIP_PATTERNS = [
    '.git/',
    '__pycache__/',
    'node_modules/',
    '.pyc',
    '.pyo',
    '.so',
    '.dll',
    '.exe',
    '.bin',
    '.jpg',
    '.png',
    '.gif',
    '.svg',
    '.ico',
    '.pdf',
    '.zip',
    '.tar',
    '.gz',
]

# File extensions to process
PROCESS_EXTENSIONS = [
    '.py', '.js', '.ts', '.jsx', '.tsx',
    '.md', '.txt', '.yml', '.yaml', '.json',
    '.sh', '.bash', '.zsh',
    '.html', '.css', '.scss', '.sass',
    '.xml', '.toml', '.ini', '.cfg', '.conf',
    '.java', '.cpp', '.c', '.h', '.hpp',
    '.go', '.rs', '.rb', '.php',
]

class FileCleanup:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.fixed_files = []
        self.error_files = []
        self.backup_dir = Path(f"/tmp/cleanup_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
    def should_process_file(self, file_path: Path) -> bool:
        """Check if file should be processed."""
        str_path = str(file_path)
        
        # Skip if in skip patterns
        for pattern in SKIP_PATTERNS:
            if pattern in str_path:
                return False
        
        # Check if has valid extension
        return any(str(file_path).endswith(ext) for ext in PROCESS_EXTENSIONS)
    
    def backup_file(self, file_path: Path) -> None:
        """Create a backup of the file before modification."""
        relative_path = file_path.relative_to(self.root_dir)
        backup_path = self.backup_dir / relative_path
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, backup_path)
    
    def fix_file(self, file_path: Path) -> Tuple[bool, int]:
        """Fix corruption in a single file."""
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            original_content = content
            replacements = 0
            
            # Apply all corruption patterns
            for pattern, replacement in CORRUPTION_PATTERNS:
                new_content = re.sub(pattern, replacement, content)
                if new_content != content:
                    replacements += len(re.findall(pattern, content))
                    content = new_content
            
            # Only write if changes were made
            if content != original_content:
                # Backup first
                self.backup_file(file_path)
                
                # Write fixed content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                return True, replacements
            
            return False, 0
            
        except Exception as e:
            self.error_files.append((file_path, str(e)))
            return False, 0
    
    def scan_and_fix(self) -> dict:
        """Scan and fix all corrupted files."""
        print(f"Starting cleanup in: {self.root_dir}")
        print(f"Backup directory: {self.backup_dir}")
        print("-" * 60)
        
        total_files = 0
        total_replacements = 0
        
        # Walk through all files
        for root, dirs, files in os.walk(self.root_dir):
            # Skip .git directory
            dirs[:] = [d for d in dirs if d != '.git']
            
            for file_name in files:
                file_path = Path(root) / file_name
                
                if not self.should_process_file(file_path):
                    continue
                
                total_files += 1
                fixed, replacements = self.fix_file(file_path)
                
                if fixed:
                    self.fixed_files.append((file_path, replacements))
                    total_replacements += replacements
                    print(f"✓ Fixed: {file_path.relative_to(self.root_dir)} ({replacements} replacements)")
        
        return {
            'total_files_scanned': total_files,
            'files_fixed': len(self.fixed_files),
            'total_replacements': total_replacements,
            'errors': len(self.error_files),
            'backup_dir': str(self.backup_dir)
        }
    
    def generate_report(self, stats: dict) -> str:
        """Generate a cleanup report."""
        report = []
        report.append("# File Cleanup Report")
        report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        report.append("")
        
        report.append("## Summary")
        report.append(f"- Total files scanned: {stats['total_files_scanned']}")
        report.append(f"- Files fixed: {stats['files_fixed']}")
        report.append(f"- Total replacements: {stats['total_replacements']}")
        report.append(f"- Errors encountered: {stats['errors']}")
        report.append(f"- Backup location: {stats['backup_dir']}")
        report.append("")
        
        if self.fixed_files:
            report.append("## Fixed Files")
            for file_path, replacements in sorted(self.fixed_files, key=lambda x: x[1], reverse=True)[:50]:
                rel_path = file_path.relative_to(self.root_dir)
                report.append(f"- `{rel_path}` - {replacements} replacements")
            
            if len(self.fixed_files) > 50:
                report.append(f"- ... and {len(self.fixed_files) - 50} more files")
            report.append("")
        
        if self.error_files:
            report.append("## Errors")
            for file_path, error in self.error_files[:20]:
                rel_path = file_path.relative_to(self.root_dir)
                report.append(f"- `{rel_path}`: {error}")
            
            if len(self.error_files) > 20:
                report.append(f"- ... and {len(self.error_files) - 20} more errors")
            report.append("")
        
        report.append("## Corruption Pattern Fixed")
        report.append("The following pattern was removed from all files:")
        report.append("```")
        report.append("Mock")
        report.append("```")
        report.append("This pattern was replaced with 'Mock' in test files where appropriate.")
        report.append("")
        
        report.append("## Next Steps")
        report.append("1. Review the changes with `git diff`")
        report.append("2. Run tests to ensure functionality is preserved")
        report.append("3. If issues arise, backups are available at: " + stats['backup_dir'])
        report.append("4. Commit the fixes once validated")
        
        return "\n".join(report)


def main():
    """Main execution function."""
    root_dir = "/opt/sutazaiapp"
    
    if not os.path.exists(root_dir):
        print(f"Error: Directory {root_dir} does not exist")
        sys.exit(1)
    
    cleanup = FileCleanup(root_dir)
    stats = cleanup.scan_and_fix()
    
    print("\n" + "=" * 60)
    print("CLEANUP COMPLETE")
    print("=" * 60)
    print(f"Files fixed: {stats['files_fixed']}")
    print(f"Total replacements: {stats['total_replacements']}")
    print(f"Errors: {stats['errors']}")
    print(f"Backup saved to: {stats['backup_dir']}")
    
    # Generate and save report
    report = cleanup.generate_report(stats)
    report_path = Path(root_dir) / "docs" / "reports" / "CORRUPTION_CLEANUP_REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\nDetailed report saved to: {report_path}")
    
    return 0 if stats['errors'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())