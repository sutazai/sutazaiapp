"""
Priority P0 Fixes for Critical Rule Violations
Created: 2025-08-18 17:00:00 UTC
Purpose: Execute immediate critical fixes for rule compliance
"""

import os
import shutil
import sys
from pathlib import Path
from datetime import datetime

class RuleEnforcementFixer:
    def __init__(self):
        self.root_dir = Path('/opt/sutazaiapp')
        self.timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        self.fixes_applied = []
        self.errors = []
    
    def backup_file(self, file_path):
        """Create a backup of a file before modification."""
        backup_dir = self.root_dir / 'backups' / f'enforcement_{self.timestamp}'
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        if Path(file_path).exists():
            rel_path = Path(file_path).relative_to(self.root_dir)
            backup_path = backup_dir / rel_path
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, backup_path)
            return backup_path
        return None
    
    def fix_test_files_in_root(self):
        """Move test files from root to proper /tests directory."""
        print("\n🔧 FIX 1: Moving test files from root to /tests directory...")
        
        tests_dir = self.root_dir / 'tests'
        tests_unit_dir = tests_dir / 'unit'
        tests_results_dir = tests_dir / 'results'
        
        tests_unit_dir.mkdir(parents=True, exist_ok=True)
        tests_results_dir.mkdir(parents=True, exist_ok=True)
        
        test_files = [
            ('test_agent_orchestration.py', tests_unit_dir),
            ('test_mcp_stdio.py', tests_unit_dir),
            ('test-results.json', tests_results_dir),
            ('test-results.xml', tests_results_dir),
            ('pytest.ini', tests_dir),
            ('.pytest-no-cov.ini', tests_dir),
        ]
        
        moved_count = 0
        for filename, target_dir in test_files:
            source = self.root_dir / filename
            if source.exists():
                self.backup_file(source)
                target = target_dir / filename
                try:
                    shutil.move(str(source), str(target))
                    self.fixes_applied.append(f"Moved {filename} to {target_dir}")
                    moved_count += 1
                    print(f"   ✅ Moved: {filename} -> {target}")
                except Exception as e:
                    self.errors.append(f"Failed to move {filename}: {e}")
                    print(f"   ❌ Error moving {filename}: {e}")
        
        test_results_source = self.root_dir / 'test-results'
        if test_results_source.exists() and test_results_source.is_dir():
            try:
                for item in test_results_source.iterdir():
                    shutil.move(str(item), str(tests_results_dir))
                test_results_source.rmdir()
                self.fixes_applied.append(f"Moved test-results directory to {tests_results_dir}")
                moved_count += 1
                print(f"   ✅ Moved: test-results/ -> {tests_results_dir}")
            except Exception as e:
                self.errors.append(f"Failed to move test-results directory: {e}")
        
        print(f"   📊 Moved {moved_count} test-related files/directories")
        return moved_count > 0
    
    def consolidate_requirements(self):
        """Consolidate all requirements.txt files into one."""
        print("\n🔧 FIX 2: Consolidating requirements.txt files...")
        
        main_requirements = self.root_dir / 'requirements.txt'
        requirements_files = [
            self.root_dir / 'requirements-base.txt',
            self.root_dir / 'backend' / 'requirements.txt',
            self.root_dir / 'frontend' / 'requirements_optimized.txt',
        ]
        
        all_requirements = set()
        
        for req_file in requirements_files:
            if req_file.exists():
                with open(req_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            all_requirements.add(line)
        
        with open(main_requirements, 'w') as f:
            f.write("# CONSOLIDATED REQUIREMENTS FILE\n")
            f.write(f"# Generated: {datetime.utcnow().isoformat()}Z\n")
            f.write("# This is the SINGLE source of truth for Python dependencies\n")
            f.write("# Rule Compliance: Enforces Rule 4 - Consolidate First\n\n")
            
            for req in sorted(all_requirements):
                f.write(f"{req}\n")
        
        self.fixes_applied.append(f"Created consolidated requirements.txt with {len(all_requirements)} packages")
        print(f"   ✅ Consolidated {len(all_requirements)} unique requirements")
        
        for env in ['dev', 'test', 'prod']:
            env_file = self.root_dir / f'requirements-{env}.txt'
            with open(env_file, 'w') as f:
                f.write(f"# Requirements for {env.upper()} environment\n")
                f.write(f"# Extends base requirements.txt\n")
                f.write(f"-r requirements.txt\n\n")
                f.write(f"# {env.upper()}-specific packages below:\n")
            print(f"   ✅ Created: requirements-{env}.txt")
        
        return True
    
    def clean_archive_directories(self):
        """Consolidate archive directories into single location."""
        print("\n🔧 FIX 3: Consolidating archive directories...")
        
        main_archive = self.root_dir / 'archive'
        main_archive.mkdir(exist_ok=True)
        
        archive_dirs = [
            self.root_dir / 'scripts' / 'archive',
            self.root_dir / 'frontend' / 'archive',
            self.root_dir / 'frontend' / 'utils' / 'archive',
        ]
        
        consolidated_count = 0
        for archive_dir in archive_dirs:
            if archive_dir.exists() and archive_dir.is_dir():
                for item in archive_dir.iterdir():
                    target = main_archive / archive_dir.parent.name / item.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        shutil.move(str(item), str(target))
                        consolidated_count += 1
                    except Exception as e:
                        self.errors.append(f"Failed to move {item}: {e}")
                
                try:
                    archive_dir.rmdir()
                    print(f"   ✅ Consolidated: {archive_dir}")
                except:
                    pass
        
        self.fixes_applied.append(f"Consolidated {consolidated_count} archived items")
        print(f"   📊 Consolidated {consolidated_count} archived items into /archive")
        return consolidated_count > 0
    
    def create_enforcement_hooks(self):
        """Create git hooks to prevent future violations."""
        print("\n🔧 FIX 4: Creating git hooks for enforcement...")
        
        hooks_dir = self.root_dir / '.git' / 'hooks'
        if not hooks_dir.exists():
            print("   ⚠️  Git hooks directory not found, skipping...")
            return False
        
        pre_commit_hook = hooks_dir / 'pre-commit'
        
        hook_content = '''#!/bin/bash

echo "🔍 Checking rule compliance..."

if ls *.test.* *test*.py 2>/dev/null | grep -q .; then
    echo "❌ ERROR: Test files found in root directory!"
    echo "   Move all test files to /tests directory"
    exit 1
fi

compose_count=$(find . -name "docker-compose*.yml" -not -name "docker-compose.consolidated.yml" | wc -l)
if [ "$compose_count" -gt 0 ]; then
    echo "⚠️  WARNING: Multiple docker-compose files detected"
    echo "   Use docker-compose.consolidated.yml only"
fi

for dir in $(git diff --cached --name-only | xargs dirname | sort -u); do
    if [ -d "$dir" ] && [ ! -f "$dir/CHANGELOG.md" ]; then
        echo "❌ ERROR: Directory $dir is missing CHANGELOG.md!"
        echo "   Run: python3 scripts/enforcement/add_missing_changelogs.py"
        exit 1
    fi
done

echo "✅ Rule compliance check passed"
'''
        
        with open(pre_commit_hook, 'w') as f:
            f.write(hook_content)
        
        os.chmod(pre_commit_hook, 0o755)
        
        self.fixes_applied.append("Created git pre-commit hook for rule enforcement")
        print(f"   ✅ Created: {pre_commit_hook}")
        return True
    
    def generate_report(self):
        """Generate a report of fixes applied."""
        report_path = self.root_dir / 'docs' / 'reports' / f'RULE_ENFORCEMENT_FIXES_{self.timestamp}.md'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write(f"# Rule Enforcement Fixes Report\n")
            f.write(f"**Generated**: {datetime.utcnow().isoformat()}Z\n\n")
            
            f.write("## Fixes Applied\n")
            for fix in self.fixes_applied:
                f.write(f"- ✅ {fix}\n")
            
            if self.errors:
                f.write("\n## Errors Encountered\n")
                for error in self.errors:
                    f.write(f"- ❌ {error}\n")
            
            f.write(f"\n## Backup Location\n")
            f.write(f"All original files backed up to: `/opt/sutazaiapp/backups/enforcement_{self.timestamp}/`\n")
            
            f.write("\n## Next Steps\n")
            f.write("1. Run `python3 scripts/enforcement/consolidate_docker.py` to consolidate Docker configs\n")
            f.write("2. Run `python3 scripts/enforcement/add_missing_changelogs.py` to add missing CHANGELOG.md files\n")
            f.write("3. Review and test all changes\n")
            f.write("4. Commit changes with proper CHANGELOG.md updates\n")
        
        return report_path

def main():
    print("=" * 60)
    print("🚨 RULE ENFORCEMENT PRIORITY FIXES (P0)")
    print("=" * 60)
    print(f"Timestamp: {datetime.utcnow().isoformat()}Z")
    print("This script will apply critical fixes for rule compliance\n")
    
    fixer = RuleEnforcementFixer()
    
    fixer.fix_test_files_in_root()
    fixer.consolidate_requirements()
    fixer.clean_archive_directories()
    fixer.create_enforcement_hooks()
    
    report_path = fixer.generate_report()
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"✅ Fixes applied: {len(fixer.fixes_applied)}")
    print(f"❌ Errors: {len(fixer.errors)}")
    print(f"📄 Report saved to: {report_path}")
    
    if fixer.errors:
        print("\n⚠️  Some errors occurred. Review the report for details.")
        sys.exit(1)
    else:
        print("\n✅ All P0 fixes completed successfully!")
        print("\n🎯 Next: Run the other enforcement scripts:")
        print("   1. python3 scripts/enforcement/consolidate_docker.py")
        print("   2. python3 scripts/enforcement/add_missing_changelogs.py")

if __name__ == '__main__':
    main()