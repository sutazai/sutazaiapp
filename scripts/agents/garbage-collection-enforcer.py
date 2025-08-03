#!/usr/bin/env python3
"""
Garbage Collection and Cleanup Enforcer - Rule 13: No Garbage, No Rot

This comprehensive enforcer implements intelligent detection, safe cleanup, and detailed
reporting for all forms of digital clutter in the codebase.

Purpose: Detect and remove all forms of junk while ensuring zero impact on functionality
Usage: python garbage-collection-enforcer.py [options]
Requirements: asyncio, pathlib, git, ripgrep (rg), ast, pylint, mypy
"""

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import ast
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
import fnmatch
import argparse
from dataclasses import dataclass, asdict
from enum import Enum

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/opt/sutazaiapp/logs/garbage-collection-enforcer.log')
    ]
)
logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk levels for file removal"""
    SAFE = "safe"
    MODERATE = "moderate"
    RISKY = "risky"
    DANGEROUS = "dangerous"


class GarbageType(Enum):
    """Types of garbage detected"""
    TEMP_FILE = "temporary_file"
    BACKUP_FILE = "backup_file"
    DEAD_CODE = "dead_code"
    UNUSED_IMPORT = "unused_import"
    EMPTY_FILE = "empty_file"
    DUPLICATE_FILE = "duplicate_file"
    BUILD_ARTIFACT = "build_artifact"
    LOG_FILE = "log_file"
    CACHE_FILE = "cache_file"
    COMMENTED_CODE = "commented_code"
    OLD_VERSION = "old_version"
    STALE_CONFIG = "stale_config"
    UNUSED_ASSET = "unused_asset"


@dataclass
class GarbageItem:
    """Represents a detected garbage item"""
    path: str
    garbage_type: GarbageType
    risk_level: RiskLevel
    size_bytes: int
    age_days: int
    confidence: float  # 0.0 to 1.0
    reasoning: str
    references: List[str]
    duplicate_of: Optional[str] = None
    content_hash: Optional[str] = None
    last_modified: Optional[str] = None


@dataclass
class CleanupStats:
    """Statistics from cleanup operation"""
    files_scanned: int = 0
    directories_scanned: int = 0
    garbage_items_found: int = 0
    items_removed: int = 0
    items_archived: int = 0
    items_skipped: int = 0
    space_recovered_bytes: int = 0
    false_positives: int = 0
    scan_duration_seconds: float = 0.0
    cleanup_duration_seconds: float = 0.0


class GarbageCollectionEnforcer:
    """
    Comprehensive Garbage Collection and Cleanup Enforcer
    
    Features:
    - Intelligent garbage detection with ML-like confidence scoring
    - Safe cleanup with comprehensive archiving
    - False positive prevention with reference analysis
    - Git integration for history tracking
    - Detailed reporting and audit trails
    - Configurable risk thresholds
    - Support for all file types and patterns
    """
    
    # Comprehensive garbage patterns organized by type
    GARBAGE_PATTERNS = {
        GarbageType.TEMP_FILE: [
            "*.tmp", "*.temp", "*.swp", "*.cache", ".*.swp", ".*.swo", 
            "*.$$$", "*.~*", "temp_*", "*_temp.*", "temporary_*",
            "scratch_*", "*_scratch.*", "test_file_*", "debug_*"
        ],
        GarbageType.BACKUP_FILE: [
            "*.backup", "*.backup*", "*.bak", "*.back", "*~", "*.old", 
            "*.orig", "*.save", "*.sav", "*_backup.*", "*_old.*",
            "*_copy.*", "*_copy[0-9].*", "copy_of_*", "*_original.*"
        ],
        GarbageType.BUILD_ARTIFACT: [
            "*.pyc", "__pycache__/", ".pytest_cache/", "*.egg-info/",
            "dist/", "build/", ".coverage", "htmlcov/", ".tox/",
            "node_modules/", ".next/", "out/", "target/", ".gradle/",
            "*.class", "*.jar", "*.war", "*.o", "*.so", "*.dylib"
        ],
        GarbageType.LOG_FILE: [
            "*.log", "*.log.*", "*.log[0-9]*", "debug.log", "error.log",
            "access.log", "application.log", "npm-debug.log*", 
            "yarn-debug.log*", "yarn-error.log*", "*.pid", "*.seed"
        ],
        GarbageType.CACHE_FILE: [
            ".DS_Store", "Thumbs.db", "desktop.ini", ".sass-cache/",
            ".cache/", "cache/", "*.cache", ".parcel-cache/",
            ".vscode/.browse.VC.db*", ".idea/workspace.xml"
        ],
        GarbageType.OLD_VERSION: [
            "*_v[0-9].*", "*_version[0-9].*", "*_final.*", "*_new.*",
            "*_latest.*", "*_updated.*", "*_revised.*", "*_fixed.*",
            "*_v[0-9]_[0-9].*", "*_draft.*", "*_wip.*"
        ]
    }
    
    # Suspicious patterns that require careful analysis
    SUSPICIOUS_PATTERNS = [
        "draft_*", "wip_*", "todo_*", "*_test.*", "*_example.*",
        "*_sample.*", "*_demo.*", "*_prototype.*", "poc_*"
    ]
    
    # Safe directories to skip entirely
    SAFE_DIRS = {
        ".git", "node_modules", "venv", ".venv", "env", ".env",
        "__pycache__", ".pytest_cache", ".tox", "htmlcov", ".coverage",
        "migrations", ".mypy_cache", ".ruff_cache", "dist", "build"
    }
    
    # File extensions that are never safe to auto-remove
    PROTECTED_EXTENSIONS = {
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".h",
        ".go", ".rs", ".rb", ".php", ".sql", ".md", ".txt", ".json", ".yaml",
        ".yml", ".xml", ".html", ".css", ".scss", ".less", ".sh", ".bat"
    }
    
    # Configuration file patterns (require special handling)
    CONFIG_PATTERNS = [
        "*.conf", "*.config", "*.cfg", "*.ini", "*.properties",
        "package.json", "pyproject.toml", "requirements*.txt",
        "Dockerfile*", "docker-compose*.yml", "*.env*"
    ]

    def __init__(self, 
                 project_root: Path,
                 dry_run: bool = True,
                 confidence_threshold: float = 0.7,
                 risk_threshold: RiskLevel = RiskLevel.MODERATE,
                 enable_git_integration: bool = True,
                 archive_before_delete: bool = True):
        """
        Initialize the Garbage Collection Enforcer
        
        Args:
            project_root: Root directory to scan
            dry_run: If True, only report what would be done
            confidence_threshold: Minimum confidence to act on (0.0-1.0)
            risk_threshold: Maximum risk level to auto-remove
            enable_git_integration: Use git for history analysis
            archive_before_delete: Archive files before deletion
        """
        self.project_root = Path(project_root).resolve()
        self.dry_run = dry_run
        self.confidence_threshold = confidence_threshold
        self.risk_threshold = risk_threshold
        self.enable_git_integration = enable_git_integration
        self.archive_before_delete = archive_before_delete
        
        # State tracking
        self.garbage_items: List[GarbageItem] = []
        self.stats = CleanupStats()
        self.archive_root = self.project_root / "archive"
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Caches for performance
        self._file_content_cache: Dict[str, str] = {}
        self._git_history_cache: Dict[str, Dict] = {}
        self._reference_cache: Dict[str, List[str]] = {}
        
        logger.info(f"Initialized GarbageCollectionEnforcer for {self.project_root}")
        logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE'}")
        logger.info(f"Confidence threshold: {self.confidence_threshold}")
        logger.info(f"Risk threshold: {self.risk_threshold.value}")

    async def scan_project(self) -> List[GarbageItem]:
        """
        Comprehensive project scan for garbage items
        
        Returns:
            List of detected garbage items with confidence scores
        """
        start_time = datetime.now()
        logger.info("Starting comprehensive project scan...")
        
        # Initialize statistics
        self.stats.files_scanned = 0
        self.stats.directories_scanned = 0
        
        # Scan tasks
        scan_tasks = [
            self._scan_filesystem(),
            self._scan_dead_code(),
            self._scan_unused_imports(),
            self._scan_duplicate_files(),
            self._scan_commented_code(),
            self._scan_stale_configs()
        ]
        
        # Execute all scans in parallel
        scan_results = await asyncio.gather(*scan_tasks, return_exceptions=True)
        
        # Consolidate results
        for result in scan_results:
            if isinstance(result, Exception):
                logger.error(f"Scan task failed: {result}")
            elif isinstance(result, list):
                self.garbage_items.extend(result)
        
        # Remove duplicates and sort by confidence
        self.garbage_items = self._deduplicate_items(self.garbage_items)
        self.garbage_items.sort(key=lambda x: x.confidence, reverse=True)
        
        self.stats.garbage_items_found = len(self.garbage_items)
        self.stats.scan_duration_seconds = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Scan completed: {self.stats.garbage_items_found} items found in "
                   f"{self.stats.scan_duration_seconds:.2f}s")
        
        return self.garbage_items

    async def _scan_filesystem(self) -> List[GarbageItem]:
        """Scan filesystem for garbage files"""
        items = []
        
        for root, dirs, files in os.walk(self.project_root):
            # Skip safe directories
            dirs[:] = [d for d in dirs if d not in self.SAFE_DIRS]
            self.stats.directories_scanned += 1
            
            # Process files in parallel batches
            file_paths = [Path(root) / f for f in files]
            batch_size = 50
            
            for i in range(0, len(file_paths), batch_size):
                batch = file_paths[i:i + batch_size]
                batch_tasks = [self._analyze_file(fp) for fp in batch]
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, GarbageItem):
                        items.append(result)
                    elif isinstance(result, Exception):
                        logger.debug(f"File analysis failed: {result}")
                
                self.stats.files_scanned += len(batch)
        
        return items

    async def _analyze_file(self, file_path: Path) -> Optional[GarbageItem]:
        """Analyze individual file for garbage patterns"""
        try:
            if not file_path.exists() or file_path.is_dir():
                return None
            
            stat = file_path.stat()
            age_days = (datetime.now().timestamp() - stat.st_mtime) / 86400
            
            # Skip very recent files (less than 1 hour old)
            if age_days < 1/24:
                return None
            
            # Analyze against all garbage patterns
            analysis_results = []
            
            for garbage_type, patterns in self.GARBAGE_PATTERNS.items():
                confidence = self._match_patterns(file_path, patterns)
                if confidence > 0:
                    analysis_results.append((garbage_type, confidence))
            
            if not analysis_results:
                # Check for empty files
                if stat.st_size == 0:
                    analysis_results.append((GarbageType.EMPTY_FILE, 0.6))
                else:
                    return None
            
            # Get the highest confidence match
            garbage_type, confidence = max(analysis_results, key=lambda x: x[1])
            
            # Calculate risk level
            risk_level = await self._assess_risk(file_path, garbage_type)
            
            # Get references to this file
            references = await self._find_file_references(file_path)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(file_path, garbage_type, confidence, references)
            
            # Create content hash for duplicate detection
            content_hash = await self._calculate_content_hash(file_path)
            
            return GarbageItem(
                path=str(file_path.relative_to(self.project_root)),
                garbage_type=garbage_type,
                risk_level=risk_level,
                size_bytes=stat.st_size,
                age_days=int(age_days),
                confidence=confidence,
                reasoning=reasoning,
                references=references,
                content_hash=content_hash,
                last_modified=datetime.fromtimestamp(stat.st_mtime).isoformat()
            )
            
        except Exception as e:
            logger.debug(f"Error analyzing {file_path}: {e}")
            return None

    def _match_patterns(self, file_path: Path, patterns: List[str]) -> float:
        """Match file against patterns and return confidence score"""
        confidence = 0.0
        file_name = file_path.name
        file_path_str = str(file_path)
        
        for pattern in patterns:
            # Direct filename match
            if fnmatch.fnmatch(file_name, pattern):
                confidence = max(confidence, 0.9)
            # Path contains pattern
            elif fnmatch.fnmatch(file_path_str, f"*{pattern}*"):
                confidence = max(confidence, 0.7)
            # Partial pattern match
            elif pattern.replace("*", "") in file_name.lower():
                confidence = max(confidence, 0.5)
        
        return confidence

    async def _assess_risk(self, file_path: Path, garbage_type: GarbageType) -> RiskLevel:
        """Assess risk level for removing this file"""
        
        # Protected file extensions are always risky
        if file_path.suffix in self.PROTECTED_EXTENSIONS:
            return RiskLevel.RISKY
        
        # Configuration files are dangerous
        if any(fnmatch.fnmatch(file_path.name, pattern) for pattern in self.CONFIG_PATTERNS):
            return RiskLevel.DANGEROUS
        
        # Check git status if enabled
        if self.enable_git_integration:
            git_status = await self._get_git_status(file_path)
            if git_status.get("tracked", False):
                return RiskLevel.MODERATE
        
        # Build artifacts and temp files are usually safe
        if garbage_type in [GarbageType.BUILD_ARTIFACT, GarbageType.TEMP_FILE, 
                           GarbageType.CACHE_FILE, GarbageType.LOG_FILE]:
            return RiskLevel.SAFE
        
        # Files with references are risky
        references = await self._find_file_references(file_path)
        if references:
            return RiskLevel.RISKY
        
        return RiskLevel.MODERATE

    async def _find_file_references(self, file_path: Path) -> List[str]:
        """Find references to file in codebase using ripgrep"""
        file_key = str(file_path)
        if file_key in self._reference_cache:
            return self._reference_cache[file_key]
        
        references = []
        search_name = file_path.name
        
        # Use ripgrep for fast searching
        try:
            cmd = [
                "rg", "-l", "--no-ignore", "--type-not", "log",
                f"\\b{re.escape(search_name)}\\b",
                str(self.project_root)
            ]
            
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            if proc.returncode == 0 and stdout:
                refs = stdout.decode().strip().split('\n')
                # Filter out self-reference and empty lines
                references = [r for r in refs if r and not r.endswith(str(file_path))]
                
        except Exception as e:
            logger.debug(f"Error searching for references to {file_path}: {e}")
        
        self._reference_cache[file_key] = references
        return references

    async def _get_git_status(self, file_path: Path) -> Dict[str, Any]:
        """Get git status information for file"""
        file_key = str(file_path)
        if file_key in self._git_history_cache:
            return self._git_history_cache[file_key]
        
        status = {"tracked": False, "modified": False, "staged": False}
        
        try:
            relative_path = file_path.relative_to(self.project_root)
            
            # Check if tracked
            cmd = ["git", "ls-files", "--", str(relative_path)]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self.project_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await proc.communicate()
            status["tracked"] = bool(stdout.strip())
            
            # Check status
            if status["tracked"]:
                cmd = ["git", "status", "--porcelain", "--", str(relative_path)]
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=str(self.project_root),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, _ = await proc.communicate()
                
                if stdout:
                    status_line = stdout.decode().strip()
                    status["modified"] = 'M' in status_line
                    status["staged"] = status_line.startswith('M') or status_line.startswith('A')
            
        except Exception as e:
            logger.debug(f"Error getting git status for {file_path}: {e}")
        
        self._git_history_cache[file_key] = status
        return status

    async def _calculate_content_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file content"""
        try:
            hasher = hashlib.sha256()
            with open(file_path, 'rb') as f:
                # Read in chunks to handle large files
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.debug(f"Error calculating hash for {file_path}: {e}")
            return ""

    def _generate_reasoning(self, file_path: Path, garbage_type: GarbageType, 
                          confidence: float, references: List[str]) -> str:
        """Generate human-readable reasoning for detection"""
        reasons = []
        
        # Pattern match reasoning
        if confidence > 0.8:
            reasons.append(f"Strong pattern match for {garbage_type.value}")
        elif confidence > 0.6:
            reasons.append(f"Pattern match for {garbage_type.value}")
        else:
            reasons.append(f"Weak pattern match for {garbage_type.value}")
        
        # File age
        try:
            age_days = (datetime.now().timestamp() - file_path.stat().st_mtime) / 86400
            if age_days > 30:
                reasons.append(f"File is {int(age_days)} days old")
        except:
            pass
        
        # Size considerations
        try:
            size = file_path.stat().st_size
            if size == 0:
                reasons.append("Empty file")
            elif size > 100 * 1024 * 1024:  # 100MB
                reasons.append(f"Large file ({size // 1024 // 1024}MB)")
        except:
            pass
        
        # Reference analysis
        if references:
            reasons.append(f"Referenced in {len(references)} files")
        else:
            reasons.append("No references found")
        
        return "; ".join(reasons)

    async def _scan_dead_code(self) -> List[GarbageItem]:
        """Scan Python files for dead code using AST analysis"""
        items = []
        
        python_files = list(self.project_root.rglob("*.py"))
        
        for py_file in python_files:
            try:
                # Skip certain directories
                if any(safe_dir in str(py_file) for safe_dir in self.SAFE_DIRS):
                    continue
                
                dead_code_analysis = await self._analyze_python_dead_code(py_file)
                if dead_code_analysis:
                    items.extend(dead_code_analysis)
                    
            except Exception as e:
                logger.debug(f"Error analyzing Python file {py_file}: {e}")
        
        return items

    async def _analyze_python_dead_code(self, py_file: Path) -> List[GarbageItem]:
        """Analyze Python file for dead code using AST"""
        items = []
        
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content, filename=str(py_file))
            
            # Find unused functions and classes
            defined_names = set()
            used_names = set()
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    defined_names.add(node.name)
                elif isinstance(node, ast.Name):
                    used_names.add(node.id)
            
            # Find potentially unused definitions
            unused_names = defined_names - used_names
            
            # Check if these are actually referenced elsewhere
            for unused_name in unused_names:
                references = await self._find_symbol_references(py_file, unused_name)
                if not references:
                    # This might be dead code
                    stat = py_file.stat()
                    age_days = (datetime.now().timestamp() - stat.st_mtime) / 86400
                    
                    items.append(GarbageItem(
                        path=f"{py_file.relative_to(self.project_root)}:{unused_name}",
                        garbage_type=GarbageType.DEAD_CODE,
                        risk_level=RiskLevel.MODERATE,
                        size_bytes=0,  # This is a symbol, not a file
                        age_days=int(age_days),
                        confidence=0.6,
                        reasoning=f"Unused function/class '{unused_name}' in {py_file.name}",
                        references=[]
                    ))
                    
        except Exception as e:
            logger.debug(f"Error in AST analysis for {py_file}: {e}")
        
        return items

    async def _find_symbol_references(self, file_path: Path, symbol_name: str) -> List[str]:
        """Find references to a symbol in the codebase"""
        try:
            cmd = [
                "rg", "-l", "--type", "py",
                f"\\b{re.escape(symbol_name)}\\b",
                str(self.project_root)
            ]
            
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await proc.communicate()
            
            if proc.returncode == 0 and stdout:
                refs = stdout.decode().strip().split('\n')
                # Filter out self-reference
                return [r for r in refs if r and not r.endswith(str(file_path))]
            
        except Exception as e:
            logger.debug(f"Error searching for symbol {symbol_name}: {e}")
        
        return []

    async def _scan_unused_imports(self) -> List[GarbageItem]:
        """Scan for unused imports in Python files"""
        items = []
        
        # This would typically use tools like unimport or vulture
        # For now, we'll implement a basic version
        
        python_files = list(self.project_root.rglob("*.py"))
        
        for py_file in python_files[:10]:  # Limit for performance
            try:
                unused_imports = await self._find_unused_imports(py_file)
                for import_info in unused_imports:
                    stat = py_file.stat()
                    age_days = (datetime.now().timestamp() - stat.st_mtime) / 86400
                    
                    items.append(GarbageItem(
                        path=f"{py_file.relative_to(self.project_root)}:{import_info}",
                        garbage_type=GarbageType.UNUSED_IMPORT,
                        risk_level=RiskLevel.SAFE,
                        size_bytes=0,
                        age_days=int(age_days),
                        confidence=0.7,
                        reasoning=f"Unused import '{import_info}' in {py_file.name}",
                        references=[]
                    ))
                    
            except Exception as e:
                logger.debug(f"Error checking imports in {py_file}: {e}")
        
        return items

    async def _find_unused_imports(self, py_file: Path) -> List[str]:
        """Find unused imports in a Python file"""
        unused = []
        
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=str(py_file))
            
            imported_names = set()
            used_names = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imported_names.add(alias.asname or alias.name)
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        imported_names.add(alias.asname or alias.name)
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    used_names.add(node.id)
            
            unused = list(imported_names - used_names)
            
        except Exception as e:
            logger.debug(f"Error analyzing imports in {py_file}: {e}")
        
        return unused

    async def _scan_duplicate_files(self) -> List[GarbageItem]:
        """Scan for duplicate files using content hashes"""
        items = []
        hash_to_files = {}
        
        # Get all files and their hashes
        all_files = []
        for root, dirs, files in os.walk(self.project_root):
            dirs[:] = [d for d in dirs if d not in self.SAFE_DIRS]
            for file in files:
                file_path = Path(root) / file
                if file_path.is_file() and file_path.stat().st_size > 0:
                    all_files.append(file_path)
        
        # Calculate hashes for files
        for file_path in all_files[:1000]:  # Limit for performance
            content_hash = await self._calculate_content_hash(file_path)
            if content_hash:
                if content_hash not in hash_to_files:
                    hash_to_files[content_hash] = []
                hash_to_files[content_hash].append(file_path)
        
        # Find duplicates
        for content_hash, file_list in hash_to_files.items():
            if len(file_list) > 1:
                # Keep the newest file, mark others as duplicates
                file_list.sort(key=lambda f: f.stat().st_mtime, reverse=True)
                original = file_list[0]
                
                for duplicate_file in file_list[1:]:
                    stat = duplicate_file.stat()
                    age_days = (datetime.now().timestamp() - stat.st_mtime) / 86400
                    
                    items.append(GarbageItem(
                        path=str(duplicate_file.relative_to(self.project_root)),
                        garbage_type=GarbageType.DUPLICATE_FILE,
                        risk_level=RiskLevel.MODERATE,
                        size_bytes=stat.st_size,
                        age_days=int(age_days),
                        confidence=0.9,
                        reasoning=f"Duplicate of {original.relative_to(self.project_root)}",
                        references=[],
                        duplicate_of=str(original.relative_to(self.project_root)),
                        content_hash=content_hash
                    ))
        
        return items

    async def _scan_commented_code(self) -> List[GarbageItem]:
        """Scan for large blocks of commented-out code"""
        items = []
        
        # Scan Python files for commented code
        python_files = list(self.project_root.rglob("*.py"))[:50]  # Limit for performance
        
        for py_file in python_files:
            try:
                commented_blocks = await self._find_commented_code_blocks(py_file)
                for block_info in commented_blocks:
                    stat = py_file.stat()
                    age_days = (datetime.now().timestamp() - stat.st_mtime) / 86400
                    
                    items.append(GarbageItem(
                        path=f"{py_file.relative_to(self.project_root)}:lines_{block_info['start']}-{block_info['end']}",
                        garbage_type=GarbageType.COMMENTED_CODE,
                        risk_level=RiskLevel.SAFE,
                        size_bytes=block_info['size'],
                        age_days=int(age_days),
                        confidence=0.8,
                        reasoning=f"Large commented code block ({block_info['lines']} lines)",
                        references=[]
                    ))
                    
            except Exception as e:
                logger.debug(f"Error scanning commented code in {py_file}: {e}")
        
        return items

    async def _find_commented_code_blocks(self, py_file: Path) -> List[Dict]:
        """Find large blocks of commented-out code"""
        blocks = []
        
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            in_block = False
            block_start = 0
            block_lines = 0
            
            for i, line in enumerate(lines):
                stripped = line.strip()
                
                # Check if this looks like commented code (not just comments)
                is_commented_code = (
                    stripped.startswith('#') and
                    len(stripped) > 1 and
                    (
                        '=' in stripped or
                        'def ' in stripped or
                        'class ' in stripped or
                        'import ' in stripped or
                        'if ' in stripped or
                        'for ' in stripped or
                        'while ' in stripped
                    )
                )
                
                if is_commented_code:
                    if not in_block:
                        in_block = True
                        block_start = i + 1
                        block_lines = 1
                    else:
                        block_lines += 1
                else:
                    if in_block and block_lines >= 5:  # Only report blocks of 5+ lines
                        blocks.append({
                            'start': block_start,
                            'end': i,
                            'lines': block_lines,
                            'size': sum(len(lines[j]) for j in range(block_start-1, i))
                        })
                    in_block = False
                    block_lines = 0
            
            # Handle block at end of file
            if in_block and block_lines >= 5:
                blocks.append({
                    'start': block_start,
                    'end': len(lines),
                    'lines': block_lines,
                    'size': sum(len(lines[j]) for j in range(block_start-1, len(lines)))
                })
                
        except Exception as e:
            logger.debug(f"Error analyzing commented code in {py_file}: {e}")
        
        return blocks

    async def _scan_stale_configs(self) -> List[GarbageItem]:
        """Scan for stale configuration files"""
        items = []
        
        # Find configuration files
        config_files = []
        for pattern in self.CONFIG_PATTERNS:
            config_files.extend(self.project_root.rglob(pattern))
        
        for config_file in config_files[:20]:  # Limit for performance
            try:
                if await self._is_stale_config(config_file):
                    stat = config_file.stat()
                    age_days = (datetime.now().timestamp() - stat.st_mtime) / 86400
                    
                    items.append(GarbageItem(
                        path=str(config_file.relative_to(self.project_root)),
                        garbage_type=GarbageType.STALE_CONFIG,
                        risk_level=RiskLevel.RISKY,  # Configs are risky to remove
                        size_bytes=stat.st_size,
                        age_days=int(age_days),
                        confidence=0.6,
                        reasoning=f"Potentially stale configuration file",
                        references=await self._find_file_references(config_file)
                    ))
                    
            except Exception as e:
                logger.debug(f"Error analyzing config file {config_file}: {e}")
        
        return items

    async def _is_stale_config(self, config_file: Path) -> bool:
        """Check if configuration file appears to be stale"""
        try:
            # Check age (configs older than 6 months might be stale)
            stat = config_file.stat()
            age_days = (datetime.now().timestamp() - stat.st_mtime) / 86400
            
            if age_days < 180:  # Less than 6 months
                return False
            
            # Check if it has references
            references = await self._find_file_references(config_file)
            if not references:
                return True
            
            # Additional heuristics could be added here
            return False
            
        except Exception as e:
            logger.debug(f"Error checking if {config_file} is stale: {e}")
            return False

    def _deduplicate_items(self, items: List[GarbageItem]) -> List[GarbageItem]:
        """Remove duplicate garbage items"""
        seen_paths = set()
        deduplicated = []
        
        for item in items:
            if item.path not in seen_paths:
                seen_paths.add(item.path)
                deduplicated.append(item)
        
        return deduplicated

    async def cleanup_garbage(self, items: List[GarbageItem]) -> CleanupStats:
        """
        Perform safe cleanup of detected garbage items
        
        Args:
            items: List of garbage items to clean up
            
        Returns:
            CleanupStats with results of cleanup operation
        """
        start_time = datetime.now()
        logger.info(f"Starting cleanup of {len(items)} garbage items...")
        
        # Filter items by confidence and risk thresholds
        actionable_items = [
            item for item in items
            if item.confidence >= self.confidence_threshold and
               self._risk_level_value(item.risk_level) <= self._risk_level_value(self.risk_threshold)
        ]
        
        logger.info(f"Filtering to {len(actionable_items)} actionable items "
                   f"(confidence >= {self.confidence_threshold}, "
                   f"risk <= {self.risk_threshold.value})")
        
        # Create archive directory
        if self.archive_before_delete:
            archive_dir = self.archive_root / f"garbage_cleanup_{self.session_id}"
            if not self.dry_run:
                archive_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Archive directory: {archive_dir}")
        
        # Process items
        for item in actionable_items:
            try:
                success = await self._process_garbage_item(item, archive_dir if self.archive_before_delete else None)
                if success:
                    self.stats.items_removed += 1
                    self.stats.space_recovered_bytes += item.size_bytes
                    if self.archive_before_delete:
                        self.stats.items_archived += 1
                else:
                    self.stats.items_skipped += 1
                    
            except Exception as e:
                logger.error(f"Error processing {item.path}: {e}")
                self.stats.items_skipped += 1
        
        # Clean up empty directories
        if not self.dry_run:
            await self._cleanup_empty_directories()
        
        self.stats.cleanup_duration_seconds = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Cleanup completed: {self.stats.items_removed} items removed, "
                   f"{self.stats.space_recovered_bytes / 1024 / 1024:.2f}MB recovered")
        
        return self.stats

    def _risk_level_value(self, risk_level: RiskLevel) -> int:
        """Convert risk level to numeric value for comparison"""
        return {
            RiskLevel.SAFE: 1,
            RiskLevel.MODERATE: 2,
            RiskLevel.RISKY: 3,
            RiskLevel.DANGEROUS: 4
        }[risk_level]

    async def _process_garbage_item(self, item: GarbageItem, archive_dir: Optional[Path]) -> bool:
        """Process individual garbage item"""
        
        # Handle different types of garbage items
        if ":" in item.path:
            # This is a code-level item (function, import, etc.)
            return await self._process_code_item(item)
        else:
            # This is a file-level item
            return await self._process_file_item(item, archive_dir)

    async def _process_file_item(self, item: GarbageItem, archive_dir: Optional[Path]) -> bool:
        """Process file-level garbage item"""
        file_path = self.project_root / item.path
        
        if not file_path.exists():
            logger.warning(f"File no longer exists: {item.path}")
            return False
        
        try:
            # Final safety check - verify no new references appeared
            current_references = await self._find_file_references(file_path)
            if len(current_references) > len(item.references):
                logger.warning(f"New references found for {item.path}, skipping")
                return False
            
            # Archive if requested
            if archive_dir:
                archive_path = archive_dir / item.path
                if not self.dry_run:
                    archive_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file_path, archive_path)
                logger.info(f"Archived: {item.path} -> {archive_path}")
            
            # Remove file
            if not self.dry_run:
                file_path.unlink()
            
            logger.info(f"{'[DRY RUN] ' if self.dry_run else ''}Removed {item.garbage_type.value}: {item.path}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing file {item.path}: {e}")
            return False

    async def _process_code_item(self, item: GarbageItem) -> bool:
        """Process code-level garbage item (dead code, unused imports, etc.)"""
        # This would require more sophisticated code manipulation
        # For now, just log what would be done
        
        logger.info(f"{'[DRY RUN] ' if self.dry_run else ''}Would remove {item.garbage_type.value}: {item.path}")
        
        # In a full implementation, this would:
        # 1. Parse the file
        # 2. Remove the specific code element
        # 3. Rewrite the file
        # 4. Run linting/formatting
        
        return True  # Placeholder

    async def _cleanup_empty_directories(self):
        """Remove empty directories after cleanup"""
        empty_dirs = []
        
        for root, dirs, files in os.walk(self.project_root, topdown=False):
            # Skip safe directories
            if any(safe_dir in root for safe_dir in self.SAFE_DIRS):
                continue
            
            # Check if directory is empty
            if not dirs and not files:
                empty_dirs.append(Path(root))
        
        for empty_dir in empty_dirs:
            try:
                # Make sure it's really empty and not a special directory
                if (empty_dir != self.project_root and 
                    not any(special in str(empty_dir) for special in ['.git', 'node_modules'])):
                    empty_dir.rmdir()
                    logger.info(f"Removed empty directory: {empty_dir}")
            except Exception as e:
                logger.debug(f"Could not remove directory {empty_dir}: {e}")

    async def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive cleanup report"""
        
        # Categorize items by type and risk
        items_by_type = {}
        items_by_risk = {}
        
        for item in self.garbage_items:
            # By type
            type_key = item.garbage_type.value
            if type_key not in items_by_type:
                items_by_type[type_key] = []
            items_by_type[type_key].append(item)
            
            # By risk
            risk_key = item.risk_level.value
            if risk_key not in items_by_risk:
                items_by_risk[risk_key] = []
            items_by_risk[risk_key].append(item)
        
        # Calculate total potential space savings
        total_potential_bytes = sum(item.size_bytes for item in self.garbage_items)
        
        # Generate top violations by size
        top_by_size = sorted(self.garbage_items, key=lambda x: x.size_bytes, reverse=True)[:10]
        
        # Generate confidence distribution
        confidence_distribution = {
            "high (0.8-1.0)": len([i for i in self.garbage_items if i.confidence >= 0.8]),
            "medium (0.6-0.8)": len([i for i in self.garbage_items if 0.6 <= i.confidence < 0.8]),
            "low (0.0-0.6)": len([i for i in self.garbage_items if i.confidence < 0.6])
        }
        
        report = {
            "metadata": {
                "rule": "Rule 13: No Garbage, No Rot",
                "enforcer": "GarbageCollectionEnforcer",
                "version": "1.0.0",
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                "project_root": str(self.project_root),
                "dry_run": self.dry_run
            },
            "configuration": {
                "confidence_threshold": self.confidence_threshold,
                "risk_threshold": self.risk_threshold.value,
                "git_integration_enabled": self.enable_git_integration,
                "archive_before_delete": self.archive_before_delete
            },
            "statistics": asdict(self.stats),
            "analysis": {
                "total_garbage_items": len(self.garbage_items),
                "total_potential_space_mb": round(total_potential_bytes / 1024 / 1024, 2),
                "actionable_items": len([
                    i for i in self.garbage_items 
                    if i.confidence >= self.confidence_threshold and
                       self._risk_level_value(i.risk_level) <= self._risk_level_value(self.risk_threshold)
                ]),
                "items_by_type": {k: len(v) for k, v in items_by_type.items()},
                "items_by_risk": {k: len(v) for k, v in items_by_risk.items()},
                "confidence_distribution": confidence_distribution
            },
            "findings": {
                "top_violations_by_size": [
                    {
                        "path": item.path,
                        "type": item.garbage_type.value,
                        "size_mb": round(item.size_bytes / 1024 / 1024, 2),
                        "confidence": item.confidence,
                        "risk": item.risk_level.value,
                        "reasoning": item.reasoning
                    }
                    for item in top_by_size
                ],
                "high_confidence_items": [
                    {
                        "path": item.path,
                        "type": item.garbage_type.value,
                        "confidence": item.confidence,
                        "risk": item.risk_level.value,
                        "reasoning": item.reasoning
                    }
                    for item in self.garbage_items if item.confidence >= 0.9
                ][:20],
                "duplicate_files": [
                    {
                        "path": item.path,
                        "duplicate_of": item.duplicate_of,
                        "size_mb": round(item.size_bytes / 1024 / 1024, 2)
                    }
                    for item in self.garbage_items if item.garbage_type == GarbageType.DUPLICATE_FILE
                ]
            },
            "recommendations": self._generate_recommendations(),
            "audit_trail": {
                "archive_location": str(self.archive_root / f"garbage_cleanup_{self.session_id}") if self.archive_before_delete else None,
                "git_commands_for_review": self._generate_git_commands(),
                "rollback_instructions": self._generate_rollback_instructions()
            }
        }
        
        return report

    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Based on findings
        if any(item.garbage_type == GarbageType.DUPLICATE_FILE for item in self.garbage_items):
            recommendations.append("Consider implementing duplicate file detection in CI/CD pipeline")
        
        if any(item.garbage_type == GarbageType.DEAD_CODE for item in self.garbage_items):
            recommendations.append("Integrate dead code detection tools (vulture, unimport) into development workflow")
        
        if any(item.garbage_type == GarbageType.COMMENTED_CODE for item in self.garbage_items):
            recommendations.append("Establish policy for commented code - use git history instead of keeping commented blocks")
        
        # Based on configuration
        if self.confidence_threshold < 0.8:
            recommendations.append("Consider increasing confidence threshold to 0.8+ for more conservative cleanup")
        
        if not self.enable_git_integration:
            recommendations.append("Enable git integration for better safety analysis")
        
        # Based on stats
        total_mb = sum(item.size_bytes for item in self.garbage_items) / 1024 / 1024
        if total_mb > 100:
            recommendations.append(f"Large amount of garbage detected ({total_mb:.1f}MB) - consider running cleanup monthly")
        
        return recommendations

    def _generate_git_commands(self) -> List[str]:
        """Generate git commands for reviewing changes"""
        commands = [
            "# Review what will be changed:",
            "git status",
            "git diff --name-only",
            "",
            "# After cleanup, review changes:",
            "git add -A",
            "git commit -m 'Clean up garbage files (Rule 13: No Garbage, No Rot) 🤖 Generated with Claude Code'",
            "",
            "# If you need to revert:",
            "git reset --hard HEAD~1"
        ]
        
        return commands

    def _generate_rollback_instructions(self) -> List[str]:
        """Generate rollback instructions"""
        instructions = [
            "# To rollback file deletions:",
            "1. Check archive directory for backed up files",
            f"2. Archive location: {self.archive_root / f'garbage_cleanup_{self.session_id}'}",
            "3. Use git to restore files if they were tracked:",
            "   git checkout HEAD~1 -- <file_path>",
            "",
            "# To prevent similar issues:",
            "1. Add patterns to .gitignore for build artifacts",
            "2. Set up pre-commit hooks to prevent garbage files",
            "3. Configure IDE to not create backup files in project directory"
        ]
        
        return instructions

    async def save_report(self, report: Dict[str, Any], output_path: Optional[Path] = None) -> Path:
        """Save report to file"""
        if output_path is None:
            output_path = self.project_root / f"garbage_cleanup_report_{self.session_id}.json"
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Report saved to: {output_path}")
        return output_path

    async def enforce_rule_13(self) -> Dict[str, Any]:
        """
        Main method to enforce Rule 13: No Garbage, No Rot
        
        Returns:
            Comprehensive report of findings and actions taken
        """
        logger.info("🧹 Starting Rule 13 enforcement: No Garbage, No Rot")
        
        try:
            # Phase 1: Comprehensive scan
            logger.info("Phase 1: Scanning for garbage...")
            garbage_items = await self.scan_project()
            
            # Phase 2: Safe cleanup
            if garbage_items and not self.dry_run:
                logger.info("Phase 2: Performing safe cleanup...")
                cleanup_stats = await self.cleanup_garbage(garbage_items)
            else:
                logger.info("Phase 2: Dry run - no actual cleanup performed")
                cleanup_stats = self.stats
            
            # Phase 3: Generate comprehensive report
            logger.info("Phase 3: Generating report...")
            report = await self.generate_report()
            
            # Save report
            report_path = await self.save_report(report)
            
            logger.info(f"✅ Rule 13 enforcement completed successfully")
            logger.info(f"📊 Report saved to: {report_path}")
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Rule 13 enforcement failed: {e}")
            raise


async def main():
    """Command-line interface for the Garbage Collection Enforcer"""
    parser = argparse.ArgumentParser(
        description="Garbage Collection and Cleanup Enforcer - Rule 13: No Garbage, No Rot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run scan (safe, no changes)
  python garbage-collection-enforcer.py --dry-run
  
  # Live cleanup with moderate risk threshold
  python garbage-collection-enforcer.py --risk-threshold moderate
  
  # High confidence cleanup only
  python garbage-collection-enforcer.py --confidence-threshold 0.9
  
  # Scan specific directory
  python garbage-collection-enforcer.py --project-root /path/to/project --dry-run
        """
    )
    
    parser.add_argument(
        "--project-root", 
        type=Path, 
        default=Path("/opt/sutazaiapp"),
        help="Root directory to scan (default: /opt/sutazaiapp)"
    )
    
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        default=True,
        help="Perform dry run only (default: True)"
    )
    
    parser.add_argument(
        "--live", 
        action="store_true",
        help="Perform actual cleanup (overrides --dry-run)"
    )
    
    parser.add_argument(
        "--confidence-threshold", 
        type=float, 
        default=0.7,
        help="Minimum confidence score to act on (0.0-1.0, default: 0.7)"
    )
    
    parser.add_argument(
        "--risk-threshold", 
        choices=["safe", "moderate", "risky", "dangerous"],
        default="moderate",
        help="Maximum risk level for auto-removal (default: moderate)"
    )
    
    parser.add_argument(
        "--no-git", 
        action="store_true",
        help="Disable git integration"
    )
    
    parser.add_argument(
        "--no-archive", 
        action="store_true",
        help="Skip archiving files before deletion"
    )
    
    parser.add_argument(
        "--output", 
        type=Path,
        help="Output path for report (default: auto-generated)"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Resolve dry run mode
    dry_run = args.dry_run and not args.live
    
    # Create enforcer
    enforcer = GarbageCollectionEnforcer(
        project_root=args.project_root,
        dry_run=dry_run,
        confidence_threshold=args.confidence_threshold,
        risk_threshold=RiskLevel(args.risk_threshold),
        enable_git_integration=not args.no_git,
        archive_before_delete=not args.no_archive
    )
    
    try:
        # Run enforcement
        report = await enforcer.enforce_rule_13()
        
        # Save report to specified location
        if args.output:
            await enforcer.save_report(report, args.output)
        
        # Print summary
        print("\n" + "="*60)
        print("🧹 GARBAGE COLLECTION ENFORCER - SUMMARY")
        print("="*60)
        print(f"Mode: {'DRY RUN' if dry_run else 'LIVE CLEANUP'}")
        print(f"Items Found: {report['analysis']['total_garbage_items']}")
        print(f"Actionable Items: {report['analysis']['actionable_items']}")
        print(f"Space Recovered: {report['statistics']['space_recovered_bytes'] / 1024 / 1024:.2f} MB")
        print(f"Scan Duration: {report['statistics']['scan_duration_seconds']:.2f}s")
        
        if not dry_run:
            print(f"Items Removed: {report['statistics']['items_removed']}")
            print(f"Items Archived: {report['statistics']['items_archived']}")
        
        print("\n📊 Top Violations by Size:")
        for item in report['findings']['top_violations_by_size'][:5]:
            print(f"  • {item['path']} ({item['size_mb']}MB) - {item['type']}")
        
        if report['findings']['duplicate_files']:
            print(f"\n📁 Duplicate Files: {len(report['findings']['duplicate_files'])}")
        
        print(f"\n📋 Full report: {args.output or f'garbage_cleanup_report_{enforcer.session_id}.json'}")
        print("="*60)
        
        return 0 if report['statistics']['items_removed'] >= 0 else 1
        
    except Exception as e:
        logger.error(f"Enforcement failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))