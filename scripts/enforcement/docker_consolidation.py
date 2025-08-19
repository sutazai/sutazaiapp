"""
Docker Configuration Consolidation Script
Safely consolidates Docker compose files while preserving functionality
Date: 2025-08-18
"""

import os
import sys
import shutil
import subprocess
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

class DockerConsolidator:
    def __init__(self, project_root: str = "/opt/sutazaiapp"):
        self.project_root = Path(project_root)
        self.backup_dir = self.project_root / f"backups/consolidation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.docker_files = []
        self.running_containers = []
        self.changes_made = []
        
    def setup_backup(self) -> None:
        """Create backup directory and save current state"""
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created backup directory: {self.backup_dir}")
        
        compose_files = [
            self.project_root / "docker-compose.yml",
            self.project_root / "docker" / "docker-compose.yml",
            self.project_root / "docker" / "docker-compose.consolidated.yml"
        ]
        
        for file in compose_files:
            if file.exists():
                shutil.copy2(file, self.backup_dir / file.name)
                print(f"✓ Backed up: {file.name}")
        
        try:
            result = subprocess.run(
                ["docker", "ps", "--format", "json"],
                capture_output=True, text=True, check=True
            )
            containers_file = self.backup_dir / "running_containers.json"
            containers_file.write_text(result.stdout)
            print(f"✓ Saved container state to: running_containers.json")
        except subprocess.CalledProcessError as e:
            print(f"⚠ Warning: Could not save container state: {e}")
    
    def find_docker_files(self) -> List[Path]:
        """Find all docker-compose files in the project"""
        patterns = ["docker-compose*.yml", "docker-compose*.yaml"]
        found_files = []
        
        for pattern in patterns:
            found_files.extend(self.project_root.rglob(pattern))
        
        found_files = sorted(set(found_files))
        self.docker_files = found_files
        return found_files
    
    def check_file_duplicates(self) -> List[Tuple[Path, Path]]:
        """Find duplicate docker-compose files by content hash"""
        file_hashes = {}
        duplicates = []
        
        for file in self.docker_files:
            if file.exists():
                content = file.read_bytes()
                file_hash = hashlib.md5(content).hexdigest()
                
                if file_hash in file_hashes:
                    duplicates.append((file_hashes[file_hash], file))
                else:
                    file_hashes[file_hash] = file
        
        return duplicates
    
    def check_file_references(self, file_path: Path) -> List[str]:
        """Check which scripts reference a specific docker-compose file"""
        references = []
        scripts_dir = self.project_root / "scripts"
        
        if scripts_dir.exists():
            try:
                result = subprocess.run(
                    ["grep", "-r", str(file_path.name), str(scripts_dir)],
                    capture_output=True, text=True
                )
                if result.stdout:
                    references = result.stdout.strip().split('\n')
            except subprocess.CalledProcessError:
                pass
        
        return references
    
    def validate_docker_compose(self, file_path: Path) -> bool:
        """Validate a docker-compose file"""
        try:
            result = subprocess.run(
                ["docker-compose", "-f", str(file_path), "config"],
                capture_output=True, text=True, check=True,
                cwd=str(self.project_root)
            )
            return True
        except subprocess.CalledProcessError:
            return False
    
    def remove_duplicate(self, duplicate: Path, original: Path) -> bool:
        """Safely remove a duplicate docker-compose file"""
        print(f"\n🔍 Processing duplicate: {duplicate}")
        print(f"   Original: {original}")
        
        references = self.check_file_references(duplicate)
        if references:
            print(f"⚠ Warning: Found {len(references)} references to {duplicate.name}")
            for ref in references[:5]:  # Show first 5 references
                print(f"   - {ref.split(':')[0]}")
        
        if not self.validate_docker_compose(original):
            print(f"✗ Error: Original file {original} is not valid. Skipping removal.")
            return False
        
        try:
            duplicate.unlink()
            self.changes_made.append(f"Removed duplicate: {duplicate}")
            print(f"✓ Removed duplicate: {duplicate}")
            
            if references:
                self.update_references(duplicate, original)
            
            return True
        except Exception as e:
            print(f"✗ Error removing {duplicate}: {e}")
            return False
    
    def update_references(self, old_path: Path, new_path: Path) -> None:
        """Update script references from old path to new path"""
        scripts_dir = self.project_root / "scripts"
        
        if scripts_dir.exists():
            for script_file in scripts_dir.rglob("*.sh"):
                try:
                    content = script_file.read_text()
                    if str(old_path.name) in content:
                        backup_path = self.backup_dir / "scripts" / script_file.relative_to(self.project_root)
                        backup_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(script_file, backup_path)
                        
                        updated_content = content.replace(
                            str(old_path.relative_to(self.project_root)),
                            str(new_path.relative_to(self.project_root))
                        )
                        script_file.write_text(updated_content)
                        self.changes_made.append(f"Updated references in: {script_file}")
                        print(f"   ✓ Updated: {script_file.name}")
                except Exception as e:
                    print(f"   ⚠ Could not update {script_file}: {e}")
    
    def consolidate(self, dry_run: bool = True) -> None:
        """Main consolidation process"""
        print("=" * 60)
        print("Docker Configuration Consolidation")
        print("=" * 60)
        
        if dry_run:
            print("🔍 DRY RUN MODE - No changes will be made")
        else:
            print("⚠️  LIVE MODE - Changes will be applied")
            self.setup_backup()
        
        print("\n📁 Finding Docker compose files...")
        files = self.find_docker_files()
        print(f"Found {len(files)} docker-compose files:")
        for file in files:
            print(f"  - {file.relative_to(self.project_root)}")
        
        print("\n🔍 Checking for duplicates...")
        duplicates = self.check_file_duplicates()
        
        if duplicates:
            print(f"Found {len(duplicates)} duplicate pairs:")
            for original, duplicate in duplicates:
                print(f"  - {original.relative_to(self.project_root)}")
                print(f"    = {duplicate.relative_to(self.project_root)}")
                
                if not dry_run:
                    if "docker/" in str(duplicate) and "docker/" not in str(original):
                        self.remove_duplicate(duplicate, original)
                    elif "docker/" in str(original) and "docker/" not in str(duplicate):
                        self.remove_duplicate(original, duplicate)
        else:
            print("✓ No exact duplicates found")
        
        print("\n📊 Analyzing consolidation opportunities...")
        primary_compose = self.project_root / "docker-compose.yml"
        consolidated = self.project_root / "docker" / "docker-compose.consolidated.yml"
        
        if primary_compose.exists() and consolidated.exists():
            primary_size = primary_compose.stat().st_size
            consolidated_size = consolidated.stat().st_size
            
            print(f"Primary compose: {primary_size:,} bytes")
            print(f"Consolidated compose: {consolidated_size:,} bytes")
            
            if consolidated_size > primary_size:
                print(f"ℹ️  Consolidated file is {consolidated_size - primary_size:,} bytes larger")
                print("   Consider reviewing and merging necessary services")
        
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        
        if dry_run:
            print("DRY RUN COMPLETE - No changes made")
            print("\nTo apply changes, run with --apply flag")
        else:
            if self.changes_made:
                print(f"✓ Made {len(self.changes_made)} changes:")
                for change in self.changes_made:
                    print(f"  - {change}")
                
                changelog = self.backup_dir / "CHANGES.md"
                with open(changelog, 'w') as f:
                    f.write(f"# Docker Consolidation Changes\n")
                    f.write(f"Date: {datetime.now().isoformat()}\n\n")
                    for change in self.changes_made:
                        f.write(f"- {change}\n")
                print(f"\n✓ Change log saved to: {changelog}")
            else:
                print("ℹ️  No changes were necessary")
        
        print("\n📋 Recommendations:")
        print("1. Test docker-compose config after changes")
        print("2. Verify all containers still run properly")
        print("3. Run Playwright tests to ensure functionality")
        print("4. Update CHANGELOG.md with consolidation details")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Safely consolidate Docker compose configurations"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes (default is dry-run)"
    )
    parser.add_argument(
        "--project-root",
        default="/opt/sutazaiapp",
        help="Project root directory"
    )
    
    args = parser.parse_args()
    
    consolidator = DockerConsolidator(args.project_root)
    consolidator.consolidate(dry_run=not args.apply)

if __name__ == "__main__":
    main()