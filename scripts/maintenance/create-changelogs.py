#!/usr/bin/env python3
"""
Create CHANGELOG.md files in all directories that don't have them.
"""

import os
from pathlib import Path
from datetime import datetime

BASE_DIR = Path("/opt/sutazaiapp")
SKIP_DIRS = {'.git', '.venv', 'venv', 'node_modules', '__pycache__', 'dist', 'build', '.pytest_cache', '.mypy_cache'}

def create_changelog(directory: Path) -> bool:
    """Create a CHANGELOG.md file in the given directory."""
    changelog_path = directory / "CHANGELOG.md"
    
    if changelog_path.exists():
        return False
    
    dirname = directory.name
    if dirname == "":
        dirname = "Root"
    
    content = f"""# Changelog - {dirname}

All notable changes to this directory will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - {datetime.now().strftime("%Y-%m-%d")}

### Added
- Initial directory structure
- Basic configuration files

### Changed
- N/A

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- N/A
"""
    
    try:
        changelog_path.write_text(content)
        return True
    except Exception as e:
        print(f"❌ Error creating {changelog_path}: {e}")
        return False

def main():
    print("📝 Creating CHANGELOG.md files in all directories...")
    print("")
    
    created = 0
    skipped = 0
    errors = 0
    
    for root, dirs, files in os.walk(BASE_DIR):
        # Remove directories we should skip from the dirs list
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in SKIP_DIRS]
        
        root_path = Path(root)
        
        # Skip if it's a hidden directory
        if any(part.startswith('.') for part in root_path.parts):
            continue
        
        # Skip if it's in our skip list
        if any(skip_dir in root_path.parts for skip_dir in SKIP_DIRS):
            continue
        
        if create_changelog(root_path):
            created += 1
            # Only show first 20 to avoid spam
            if created <= 20:
                print(f"✅ Created: {root_path}/CHANGELOG.md")
            elif created == 21:
                print("... (showing first 20 only)")
        else:
            if (root_path / "CHANGELOG.md").exists():
                skipped += 1
            else:
                errors += 1
    
    print("")
    print("📊 Summary:")
    print(f"  ✅ Created: {created} CHANGELOG.md files")
    print(f"  ⏭️  Skipped: {skipped} (already exist)")
    if errors > 0:
        print(f"  ❌ Errors: {errors}")
    print("")
    print("✅ CHANGELOG.md creation complete!")
    
    # Calculate coverage
    total_dirs = created + skipped + errors
    if total_dirs > 0:
        coverage = ((created + skipped) / total_dirs) * 100
        print(f"📈 Coverage: {coverage:.1f}% of directories have CHANGELOG.md")
        
        if coverage >= 90:
            print("🎉 EXCELLENT! Achieved 90%+ CHANGELOG.md coverage!")
        elif coverage >= 70:
            print("⚠️  Good progress, but more directories need CHANGELOG.md")
        else:
            print("❌ Low coverage, many directories still missing CHANGELOG.md")

if __name__ == "__main__":
    main()