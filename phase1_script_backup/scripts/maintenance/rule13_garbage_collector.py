#!/usr/bin/env python3
"""
Purpose: Rule 13 enforcement - No garbage, no rot
Usage: python rule13_garbage_collector.py --task-file=<path> [--dry-run]
Requirements: asyncio, pathlib, gitignore parser
"""

import asyncio
import json
import logging
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
import fnmatch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GarbageCollector:
    """Enforces Rule 13: No garbage, no rot"""
    
    # Garbage patterns to detect
    GARBAGE_PATTERNS = [
        # Backup files
        "*.backup", "*.backup*", "*.bak", "*.back", "*~",
        "*.old", "*.orig", "*.save", "*.sav",
        
        # Temporary files
        "*.tmp", "*.temp", "*.swp", "*.cache", 
        ".*.swp", ".*.swo", "*.$$$", "*.~*",
        
        # OS-specific junk
        ".DS_Store", "Thumbs.db", "desktop.ini", 
        "*.pyc", "__pycache__", ".pytest_cache",
        
        # Editor artifacts
        ".idea/", ".vscode/", "*.sublime-*",
        ".*.kate-swp", ".#*", "#*#",
        
        # Build artifacts that shouldn't be committed
        "*.log", "*.pid", "*.seed", "*.pid.lock",
        "npm-debug.log*", "yarn-debug.log*", "yarn-error.log*",
        
        # Common test/debug files
        "test.py", "test2.py", "temp.py", "debug.py",
        "test_*.py.bak", "old_*.py", "*_old.py",
        "copy_of_*", "*_copy.*", "*_copy[0-9].*",
        
        # Archive indicators
        "*_archive/", "archive_*/", "old/", "deprecated/",
        "backup/", "bkp/", "save/", "OLD*/",
    ]
    
    # Patterns that might be legitimate (require verification)
    SUSPICIOUS_PATTERNS = [
        "*_v[0-9].*", "*_version[0-9].*", "*_final.*",
        "*_new.*", "*_latest.*", "*_updated.*",
        "draft_*", "wip_*", "todo_*",
    ]
    
    # Safe directories to skip
    SAFE_DIRS = {
        ".git", "node_modules", "venv", ".venv", 
        "env", ".env", "migrations", "__pycache__",
        ".pytest_cache", ".tox", "htmlcov", ".coverage"
    }
    
    def __init__(self, project_root: Path, dry_run: bool = False):
        self.project_root = project_root
        self.dry_run = dry_run
        self.archive_root = project_root / "archive"
        self.violations = []
        self.fixed = []
        self.skipped = []
        self.stats = {
            "files_scanned": 0,
            "violations_found": 0,
            "violations_fixed": 0,
            "space_recovered": 0,
            "suspicious_files": 0
        }
        
    async def scan_for_garbage(self) -> List[Path]:
        """Scan project for garbage files"""
        logger.info(f"Scanning for garbage files in {self.project_root}")
        garbage_files = []
        
        # Use asyncio for parallel directory scanning
        tasks = []
        for root, dirs, files in os.walk(self.project_root):
            # Skip safe directories
            dirs[:] = [d for d in dirs if d not in self.SAFE_DIRS]
            
            # Create async tasks for file checking
            for file in files:
                file_path = Path(root) / file
                tasks.append(self._check_file_async(file_path))
        
        # Process all files in parallel
        results = await asyncio.gather(*tasks)
        garbage_files = [f for f in results if f is not None]
        
        self.stats["files_scanned"] = len(tasks)
        self.stats["violations_found"] = len(garbage_files)
        
        return garbage_files
    
    async def _check_file_async(self, file_path: Path) -> Path:
        """Async check if file is garbage"""
        try:
            # Check against garbage patterns
            for pattern in self.GARBAGE_PATTERNS:
                if fnmatch.fnmatch(str(file_path), f"*{pattern}*") or \
                   fnmatch.fnmatch(file_path.name, pattern):
                    return file_path
            
            # Check file age for temporary files
            if await self._is_old_temp_file(file_path):
                return file_path
                
            # Check for empty files that might be garbage
            if file_path.stat().st_size == 0 and file_path.suffix in ['.tmp', '.temp', '.log']:
                return file_path
                
        except Exception as e:
            logger.debug(f"Error checking {file_path}: {e}")
            
        return None
    
    async def _is_old_temp_file(self, file_path: Path) -> bool:
        """Check if file is an old temporary file"""
        try:
            # Check if it matches temp patterns and is older than 7 days
            temp_patterns = ['tmp', 'temp', 'test', 'debug']
            if any(p in file_path.name.lower() for p in temp_patterns):
                mtime = file_path.stat().st_mtime
                age_days = (datetime.now().timestamp() - mtime) / 86400
                return age_days > 7
        except Exception as e:
            # Suppressed exception (was bare except)
            logger.debug(f"Suppressed exception: {e}")
            pass
        return False
    
    async def verify_safe_to_remove(self, file_path: Path) -> Tuple[bool, str]:
        """Verify file can be safely removed (Rule 10 compliance)"""
        # Check if file is referenced in code
        references = await self._find_references(file_path)
        if references:
            return False, f"Referenced in {len(references)} locations"
        
        # Check if it's in .gitignore (already ignored = safe to remove)
        if await self._is_gitignored(file_path):
            return True, "File is gitignored"
        
        # Check if it's a build artifact
        if self._is_build_artifact(file_path):
            return True, "Build artifact"
        
        # Check against suspicious patterns (might be legitimate)
        for pattern in self.SUSPICIOUS_PATTERNS:
            if fnmatch.fnmatch(file_path.name, pattern):
                self.stats["suspicious_files"] += 1
                return False, "Suspicious pattern - requires manual review"
        
        return True, "No references found"
    
    async def _find_references(self, file_path: Path) -> List[str]:
        """Find references to file in codebase"""
        references = []
        search_name = file_path.name
        
        # Use ripgrep for fast searching
        cmd = [
            "rg", "-l", "--no-ignore", 
            f"\\b{re.escape(search_name)}\\b",
            str(self.project_root)
        ]
        
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL
            )
            stdout, _ = await proc.communicate()
            
            if proc.returncode == 0 and stdout:
                refs = stdout.decode().strip().split('\n')
                # Filter out self-reference
                references = [r for r in refs if r and not r.endswith(search_name)]
                
        except Exception as e:
            logger.debug(f"Error searching for references: {e}")
            
        return references
    
    async def _is_gitignored(self, file_path: Path) -> bool:
        """Check if file is in .gitignore"""
        try:
            relative_path = file_path.relative_to(self.project_root)
            cmd = ["git", "check-ignore", str(relative_path)]
            
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self.project_root),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            
            return proc.returncode == 0
        except Exception as e:
            logger.warning(f"Exception caught, returning: {e}")
            return False
    
    def _is_build_artifact(self, file_path: Path) -> bool:
        """Check if file is a build artifact"""
        build_indicators = [
            'dist/', 'build/', 'target/', '.next/', 
            'out/', 'coverage/', '.coverage'
        ]
        
        path_str = str(file_path)
        return any(indicator in path_str for indicator in build_indicators)
    
    async def archive_file(self, file_path: Path) -> bool:
        """Archive file before removal"""
        try:
            # Create archive directory with date
            date_str = datetime.now().strftime("%Y-%m-%d")
            archive_dir = self.archive_root / f"rule13-cleanup-{date_str}"
            
            # Preserve directory structure
            relative_path = file_path.relative_to(self.project_root)
            archive_path = archive_dir / relative_path
            
            if not self.dry_run:
                archive_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, archive_path)
                
            logger.info(f"Archived: {file_path} -> {archive_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to archive {file_path}: {e}")
            return False
    
    async def remove_garbage(self, garbage_files: List[Path]) -> Dict:
        """Remove garbage files safely"""
        logger.info(f"Processing {len(garbage_files)} garbage files")
        
        for file_path in garbage_files:
            try:
                # Verify safe to remove
                safe, reason = await self.verify_safe_to_remove(file_path)
                
                if not safe:
                    logger.warning(f"Skipping {file_path}: {reason}")
                    self.skipped.append({
                        "file": str(file_path),
                        "reason": reason
                    })
                    continue
                
                # Get file size for stats
                file_size = file_path.stat().st_size
                
                # Archive before removal
                if await self.archive_file(file_path):
                    if not self.dry_run:
                        file_path.unlink()
                        
                    self.fixed.append(str(file_path))
                    self.stats["violations_fixed"] += 1
                    self.stats["space_recovered"] += file_size
                    
                    logger.info(f"Removed: {file_path} ({file_size} bytes)")
                    
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                self.violations.append({
                    "file": str(file_path),
                    "error": str(e)
                })
        
        return self.stats
    
    async def clean_empty_directories(self):
        """Remove empty directories left after cleanup"""
        empty_dirs = []
        
        for root, dirs, files in os.walk(self.project_root, topdown=False):
            # Skip safe directories
            if any(safe in root for safe in self.SAFE_DIRS):
                continue
                
            # Check if directory is empty
            if not dirs and not files:
                empty_dirs.append(Path(root))
        
        for empty_dir in empty_dirs:
            try:
                if not self.dry_run:
                    empty_dir.rmdir()
                logger.info(f"Removed empty directory: {empty_dir}")
            except Exception as e:
                logger.debug(f"Could not remove {empty_dir}: {e}")
    
    async def generate_report(self) -> Dict:
        """Generate enforcement report"""
        report = {
            "rule": 13,
            "description": "No garbage, no rot",
            "timestamp": datetime.now().isoformat(),
            "dry_run": self.dry_run,
            "statistics": self.stats,
            "violations": {
                "found": self.stats["violations_found"],
                "fixed": self.stats["violations_fixed"],
                "skipped": len(self.skipped),
                "errors": len(self.violations)
            },
            "space_recovered": f"{self.stats['space_recovered'] / 1024 / 1024:.2f} MB",
            "details": {
                "fixed_files": self.fixed[:20],  # First 20
                "skipped_files": self.skipped[:10],  # First 10
                "errors": self.violations
            }
        }
        
        if self.stats["suspicious_files"] > 0:
            report["warnings"] = {
                "suspicious_files": self.stats["suspicious_files"],
                "message": "Files with suspicious patterns require manual review"
            }
        
        return report
    
    async def enforce(self) -> Dict:
        """Main enforcement method"""
        logger.info(f"Starting Rule 13 enforcement (dry_run={self.dry_run})")
        
        # Scan for garbage
        garbage_files = await self.scan_for_garbage()
        
        # Remove garbage
        if garbage_files:
            await self.remove_garbage(garbage_files)
            
        # Clean empty directories
        await self.clean_empty_directories()
        
        # Generate report
        report = await self.generate_report()
        
        logger.info(f"Rule 13 enforcement completed: {self.stats}")
        
        return report


async def main():
    """Main entry point for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Rule 13: Garbage Collector")
    parser.add_argument("--task-file", help="Task configuration file")
    parser.add_argument("--project-root", default="/opt/sutazaiapp", help="Project root")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    
    args = parser.parse_args()
    
    # Load task configuration if provided
    if args.task_file and Path(args.task_file).exists():
        with open(args.task_file) as f:
            task_config = json.load(f)
            project_root = Path(task_config.get("project_root", args.project_root))
            dry_run = task_config.get("dry_run", args.dry_run)
    else:
        project_root = Path(args.project_root)
        dry_run = args.dry_run
    
    # Create and run garbage collector
    collector = GarbageCollector(project_root, dry_run)
    report = await collector.enforce()
    
    # Output report
    print(json.dumps(report, indent=2))
    
    return 0 if report["violations"]["errors"] == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))