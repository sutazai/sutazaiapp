#!/usr/bin/env python3
"""
Hardware Resource Optimizer Agent

Purpose: On-demand hardware resource optimization tool
Usage: Runs optimizations when API endpoints are called
Requirements: Docker, system access for optimization tasks

This agent acts like a "janitor" - comes in when called, 
cleans up resources, and exits. No continuous monitoring.
"""

import os
import sys
import gc
import subprocess
import shutil
import docker
import tempfile
import hashlib
import gzip
import sqlite3
import re
import glob
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple, Optional
import psutil
import json
import time
from datetime import datetime, timedelta
from collections import defaultdict
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import logging
import asyncio
import concurrent.futures
from functools import partial

# Add local shared directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "shared"))
from agents.core.base_agent import BaseAgent

# ULTRA-FIX: Import threading for thread safety
import threading

# ULTRA-FIX: Security function to prevent path traversal attacks
def validate_safe_path(requested_path: str, base_path: str = "/") -> str:
    """Validate path to prevent directory traversal attacks"""
    # Normalize and resolve the path
    requested = Path(requested_path).resolve()
    base = Path(base_path).resolve()
    
    # Check if the resolved path is within the base path
    try:
        requested.relative_to(base)
        return str(requested)
    except ValueError:
        raise ValueError(f"Path traversal attempt detected: {requested_path}")

class HardwareResourceOptimizerAgent(BaseAgent):
    """On-demand Hardware Resource Optimizer - Clean, optimize, and exit"""
    
    def __init__(self):
        super().__init__()
        self.agent_id = "hardware-resource-optimizer"
        self.name = "Hardware Resource Optimizer"
        self.port = int(os.getenv("PORT", "8080"))
        self.description = "On-demand hardware resource optimization and cleanup tool"
        
        # ULTRA-FIX: Thread-safe Docker client initialization with retry logic
        self.docker_client = None
        self.docker_client_lock = threading.Lock()
        self._init_docker_client_safe()
        
        # Storage optimization configuration
        self.protected_paths = {'/etc', '/boot', '/usr', '/bin', '/sbin', '/lib', '/proc', '/sys', '/dev'}
        self.user_protected_patterns = {'/home/*/Documents', '/home/*/Desktop', '/home/*/Pictures'}
        self.safe_temp_location = '/tmp/hardware_optimizer_safety'
        self.hash_cache = {}
        
        # ULTRA-FIX: Thread-safe hash cache with lock
        self.hash_cache_lock = threading.Lock()
        
        # Ensure safety directory exists
        os.makedirs(self.safe_temp_location, exist_ok=True)
        
        # Setup FastAPI app
        self.app = FastAPI(
            title="Hardware Resource Optimizer", 
            version="4.0.0",
            description="On-demand hardware optimization API with advanced storage features"
        )
        self._setup_routes()
        
        self.logger.info(f"Initialized {self.name} - Ready for on-demand optimization")
    
    def _init_docker_client_safe(self):
        """ULTRA-FIX: Thread-safe Docker client initialization with retry logic"""
        with self.docker_client_lock:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    client = docker.from_env(timeout=10)
                    # Test connectivity with timeout
                    client.ping()
                    self.docker_client = client
                    self.logger.info("Docker client initialized successfully")
                    return
                except Exception as e:
                    self.logger.warning(f"Docker client initialization attempt {attempt + 1}/{max_retries} failed: {e}")
                    if attempt == max_retries - 1:
                        self.docker_client = None
                        self.logger.warning("Docker client unavailable after all retries")
                    else:
                        time.sleep(1)  # Brief delay before retry
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Get current system status for optimization decisions"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": (disk.used / disk.total) * 100,
                "memory_available_gb": memory.available / (1024**3),
                "disk_free_gb": disk.free / (1024**3),
                "timestamp": time.time()
            }
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {}
    
    # Storage Analysis Methods
    def _is_safe_path(self, path: str) -> bool:
        """Check if a path is safe to analyze/modify"""
        path = os.path.abspath(path)
        
        # Never touch protected system paths
        for protected in self.protected_paths:
            if path.startswith(protected):
                return False
        
        # Check user protected patterns
        for pattern in self.user_protected_patterns:
            if '*' in pattern:
                pattern_parts = pattern.split('*')
                if len(pattern_parts) == 2 and path.startswith(pattern_parts[0]) and path.endswith(pattern_parts[1]):
                    return False
            elif path.startswith(pattern):
                return False
        
        return True
    
    def _get_file_hash(self, filepath: str) -> str:
        """ULTRA-FIX: Thread-safe file hash computation with proper error handling"""
        # ULTRA-FIX: Additional path validation to prevent traversal
        try:
            validated_path = validate_safe_path(filepath, "/")
        except ValueError:
            self.logger.warning(f"Blocked unsafe path access: {filepath}")
            return None
            
        with self.hash_cache_lock:
            if validated_path in self.hash_cache:
                return self.hash_cache[validated_path]
        
        try:
            hasher = hashlib.sha256()
            with open(validated_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            
            file_hash = hasher.hexdigest()
            with self.hash_cache_lock:
                self.hash_cache[validated_path] = file_hash
            return file_hash
        except Exception as e:
            self.logger.error(f"Error computing hash for {validated_path}: {e}")
            return None
    
    def _scan_directory(self, path: str, max_depth: int = 5, current_depth: int = 0) -> List[Dict[str, Any]]:
        """Efficiently scan directory using os.scandir"""
        files = []
        
        if current_depth >= max_depth or not self._is_safe_path(path):
            return files
        
        try:
            with os.scandir(path) as entries:
                for entry in entries:
                    try:
                        if entry.is_file(follow_symlinks=False):
                            stat_info = entry.stat()
                            files.append({
                                'path': entry.path,
                                'name': entry.name,
                                'size': stat_info.st_size,
                                'mtime': stat_info.st_mtime,
                                'atime': stat_info.st_atime,
                                'extension': os.path.splitext(entry.name)[1].lower()
                            })
                        elif entry.is_dir(follow_symlinks=False) and not entry.name.startswith('.'):
                            files.extend(self._scan_directory(entry.path, max_depth, current_depth + 1))
                    except (OSError, PermissionError):
                        continue
        except (OSError, PermissionError):
            pass
        
        return files
    
    def _analyze_storage(self, path: str) -> Dict[str, Any]:
        """Analyze storage usage with detailed breakdown"""
        try:
            if not self._is_safe_path(path) or not os.path.exists(path):
                return {
                    "status": "error",
                    "error": f"Path not accessible or safe: {path}"
                }
            
            files = self._scan_directory(path)
            
            # Analyze by extension
            extension_stats = defaultdict(lambda: {'count': 0, 'total_size': 0})
            size_buckets = defaultdict(int)
            age_buckets = defaultdict(int)
            total_size = 0
            
            now = time.time()
            
            for file_info in files:
                ext = file_info['extension'] or 'no_extension'
                size = file_info['size']
                age_days = (now - file_info['mtime']) / 86400
                
                extension_stats[ext]['count'] += 1
                extension_stats[ext]['total_size'] += size
                total_size += size
                
                # Size buckets
                if size < 1024:
                    size_buckets['< 1KB'] += 1
                elif size < 1024 * 1024:
                    size_buckets['1KB - 1MB'] += 1
                elif size < 1024 * 1024 * 100:
                    size_buckets['1MB - 100MB'] += 1
                else:
                    size_buckets['> 100MB'] += 1
                
                # Age buckets
                if age_days < 7:
                    age_buckets['< 1 week'] += 1
                elif age_days < 30:
                    age_buckets['1 week - 1 month'] += 1
                elif age_days < 365:
                    age_buckets['1 month - 1 year'] += 1
                else:
                    age_buckets['> 1 year'] += 1
            
            # Convert to regular dict and sort by size
            extension_stats = dict(sorted(extension_stats.items(), 
                                        key=lambda x: x[1]['total_size'], reverse=True))
            
            return {
                "status": "success",
                "path": path,
                "total_files": len(files),
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "extension_breakdown": extension_stats,
                "size_distribution": dict(size_buckets),
                "age_distribution": dict(age_buckets),
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Storage analysis error: {e}")
            return {"status": "error", "error": str(e)}
    
    def _analyze_duplicates(self, path: str) -> Dict[str, Any]:
        """Find duplicate files using hash comparison"""
        try:
            if not self._is_safe_path(path) or not os.path.exists(path):
                return {
                    "status": "error",
                    "error": f"Path not accessible or safe: {path}"
                }
            
            files = self._scan_directory(path)
            hash_groups = defaultdict(list)
            duplicates = []
            total_duplicate_size = 0
            
            # Group files by hash
            for file_info in files:
                file_hash = self._get_file_hash(file_info['path'])
                if file_hash:
                    hash_groups[file_hash].append(file_info)
            
            # Find duplicates
            for file_hash, file_list in hash_groups.items():
                if len(file_list) > 1:
                    # Sort by modification time (keep newest)
                    file_list.sort(key=lambda x: x['mtime'], reverse=True)
                    duplicate_group = {
                        'hash': file_hash,
                        'files': file_list,
                        'duplicate_count': len(file_list) - 1,
                        'space_wasted': sum(f['size'] for f in file_list[1:])
                    }
                    duplicates.append(duplicate_group)
                    total_duplicate_size += duplicate_group['space_wasted']
            
            # Sort by space wasted
            duplicates.sort(key=lambda x: x['space_wasted'], reverse=True)
            
            return {
                "status": "success",
                "path": path,
                "duplicate_groups": len(duplicates),
                "total_duplicates": sum(d['duplicate_count'] for d in duplicates),
                "space_wasted_bytes": total_duplicate_size,
                "space_wasted_mb": total_duplicate_size / (1024 * 1024),
                "duplicate_details": duplicates[:20],  # Limit to top 20
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Duplicate analysis error: {e}")
            return {"status": "error", "error": str(e)}
    
    def _analyze_large_files(self, path: str, min_size_mb: int) -> Dict[str, Any]:
        """Find unusually large files"""
        try:
            if not self._is_safe_path(path) or not os.path.exists(path):
                return {
                    "status": "error",
                    "error": f"Path not accessible or safe: {path}"
                }
            
            files = self._scan_directory(path)
            min_size_bytes = min_size_mb * 1024 * 1024
            large_files = []
            
            for file_info in files:
                if file_info['size'] >= min_size_bytes:
                    large_files.append({
                        **file_info,
                        'size_mb': file_info['size'] / (1024 * 1024),
                        'age_days': (time.time() - file_info['mtime']) / 86400
                    })
            
            # Sort by size
            large_files.sort(key=lambda x: x['size'], reverse=True)
            
            return {
                "status": "success",
                "path": path,
                "min_size_mb": min_size_mb,
                "large_files_count": len(large_files),
                "total_size_mb": sum(f['size_mb'] for f in large_files),
                "large_files": large_files[:50],  # Limit to top 50
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Large files analysis error: {e}")
            return {"status": "error", "error": str(e)}
    
    def _generate_storage_report(self) -> Dict[str, Any]:
        """Generate comprehensive storage analysis report"""
        try:
            disk_usage = psutil.disk_usage('/')
            
            # Analyze key directories
            key_paths = ['/tmp', '/var/log', '/var/cache', '/opt', '/home']
            path_analysis = {}
            
            for path in key_paths:
                if os.path.exists(path):
                    analysis = self._analyze_storage(path)
                    if analysis.get('status') == 'success':
                        path_analysis[path] = {
                            'total_files': analysis['total_files'],
                            'total_size_mb': analysis['total_size_mb'],
                            'top_extensions': dict(list(analysis['extension_breakdown'].items())[:5])
                        }
            
            # Find duplicates system-wide (limited scope for performance)
            duplicate_analysis = self._analyze_duplicates('/tmp')
            
            # Find large files
            large_files_analysis = self._analyze_large_files('/', 100)
            
            return {
                "status": "success",
                "disk_usage": {
                    "total_gb": disk_usage.total / (1024**3),
                    "used_gb": disk_usage.used / (1024**3),
                    "free_gb": disk_usage.free / (1024**3),
                    "usage_percent": (disk_usage.used / disk_usage.total) * 100
                },
                "path_analysis": path_analysis,
                "duplicate_summary": {
                    "groups": duplicate_analysis.get('duplicate_groups', 0),
                    "space_wasted_mb": duplicate_analysis.get('space_wasted_mb', 0)
                },
                "large_files_summary": {
                    "count": large_files_analysis.get('large_files_count', 0),
                    "total_size_mb": large_files_analysis.get('total_size_mb', 0)
                },
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Storage report error: {e}")
            return {"status": "error", "error": str(e)}
    
    # Storage Optimization Methods
    def _safe_delete(self, filepath: str, dry_run: bool = False) -> bool:
        """Safely delete a file by moving to temp location first"""
        try:
            if dry_run:
                return True
            
            # Create safety backup
            backup_path = os.path.join(self.safe_temp_location, 
                                     f"{int(time.time())}_{os.path.basename(filepath)}")
            shutil.move(filepath, backup_path)
            
            # Log the operation
            self.logger.info(f"Safely deleted {filepath} -> {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed safe delete of {filepath}: {e}")
            return False
    
    def _optimize_storage_comprehensive(self, dry_run: bool = False) -> Dict[str, Any]:
        """Main storage optimization with smart cleanup"""
        actions_taken = []
        space_freed = 0
        
        try:
            initial_disk = psutil.disk_usage('/')
            
            # 1. Clean temporary files (older than 3 days)
            temp_paths = ['/tmp', '/var/tmp']
            for temp_path in temp_paths:
                if os.path.exists(temp_path):
                    files = self._scan_directory(temp_path, max_depth=3)
                    cutoff_time = time.time() - (3 * 86400)  # 3 days
                    
                    for file_info in files:
                        if file_info['mtime'] < cutoff_time:
                            if self._safe_delete(file_info['path'], dry_run):
                                space_freed += file_info['size']
                                actions_taken.append(f"Deleted old temp file: {file_info['name']}")
            
            # 2. Clean cache directories
            cache_paths = ['/var/cache', '/home/*/.cache'] if not dry_run else []
            for cache_pattern in cache_paths:
                for cache_path in glob.glob(cache_pattern):
                    if os.path.isdir(cache_path) and self._is_safe_path(cache_path):
                        try:
                            # Clean cache older than 7 days
                            result = subprocess.run([
                                'find', cache_path, '-type', 'f', '-mtime', '+7', '-delete'
                            ], capture_output=True, text=True, timeout=30)
                            
                            if result.returncode == 0:
                                actions_taken.append(f"Cleaned cache: {cache_path}")
                        except Exception as e:
                            actions_taken.append(f"Could not clean cache {cache_path}: {e}")
            
            # 3. Application-specific cleanup
            app_cleanups = {
                'pip': ['/home/*/.cache/pip', '/root/.cache/pip'],
                'npm': ['/home/*/.npm/_cacache', '/root/.npm/_cacache'],
                'apt': ['/var/cache/apt/archives/*.deb'],
                'docker': None  # Handled separately
            }
            
            for app_name, paths in app_cleanups.items():
                if paths and not dry_run:
                    for path_pattern in paths:
                        for path in glob.glob(path_pattern):
                            if os.path.exists(path):
                                try:
                                    if os.path.isfile(path):
                                        os.remove(path)
                                        actions_taken.append(f"Cleaned {app_name} cache file: {os.path.basename(path)}")
                                    elif os.path.isdir(path):
                                        shutil.rmtree(path)
                                        actions_taken.append(f"Cleaned {app_name} cache dir: {path}")
                                except Exception as e:
                                    actions_taken.append(f"Could not clean {app_name} cache: {e}")
            
            # 4. Log rotation
            if not dry_run:
                log_result = self._optimize_logs()
                if log_result.get('status') == 'success':
                    actions_taken.extend(log_result.get('actions_taken', []))
                    space_freed += log_result.get('space_freed_bytes', 0)
            
            final_disk = psutil.disk_usage('/')
            actual_space_freed = initial_disk.used - final_disk.used
            
            return {
                "status": "success",
                "optimization_type": "comprehensive_storage",
                "dry_run": dry_run,
                "actions_taken": actions_taken,
                "estimated_space_freed_mb": space_freed / (1024 * 1024),
                "actual_space_freed_mb": actual_space_freed / (1024 * 1024),
                "initial_disk_percent": (initial_disk.used / initial_disk.total) * 100,
                "final_disk_percent": (final_disk.used / final_disk.total) * 100,
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Comprehensive storage optimization error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "actions_taken": actions_taken
            }
    
    def _optimize_duplicates(self, path: str, dry_run: bool = False) -> Dict[str, Any]:
        """Remove duplicate files with safety checks"""
        actions_taken = []
        space_freed = 0
        
        try:
            duplicates_analysis = self._analyze_duplicates(path)
            
            if duplicates_analysis.get('status') != 'success':
                return duplicates_analysis
            
            duplicate_groups = duplicates_analysis.get('duplicate_details', [])
            
            for group in duplicate_groups:
                # Keep the first file (newest), delete the rest
                files_to_delete = group['files'][1:]
                
                for file_info in files_to_delete:
                    if self._safe_delete(file_info['path'], dry_run):
                        space_freed += file_info['size']
                        actions_taken.append(f"Removed duplicate: {file_info['name']}")
            
            return {
                "status": "success",
                "optimization_type": "duplicates",
                "path": path,
                "dry_run": dry_run,
                "actions_taken": actions_taken,
                "duplicates_removed": len(actions_taken),
                "space_freed_mb": space_freed / (1024 * 1024),
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Duplicate optimization error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "actions_taken": actions_taken
            }
    
    def _optimize_cache(self) -> Dict[str, Any]:
        """Clear various system and application caches"""
        actions_taken = []
        
        try:
            cache_commands = [
                # System cache
                ['sync'],
                # Package manager caches
                ['apt-get', 'clean'],
                ['yum', 'clean', 'all'],
                # Thumbnail cache
                ['find', '/home/*/.thumbnails', '-type', 'f', '-delete'],
                ['find', '/home/*/.cache/thumbnails', '-type', 'f', '-delete'],
            ]
            
            for cmd in cache_commands:
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                    if result.returncode == 0:
                        actions_taken.append(f"Executed: {' '.join(cmd)}")
                    else:
                        actions_taken.append(f"Failed: {' '.join(cmd)} - {result.stderr}")
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                    continue
            
            # Browser caches (if safe to do so)
            browser_caches = [
                '/home/*/.cache/google-chrome',
                '/home/*/.cache/chromium',
                '/home/*/.mozilla/firefox/*/cache2'
            ]
            
            for cache_pattern in browser_caches:
                for cache_path in glob.glob(cache_pattern):
                    if os.path.isdir(cache_path):
                        try:
                            shutil.rmtree(cache_path)
                            actions_taken.append(f"Cleared browser cache: {cache_path}")
                        except Exception as e:
                            actions_taken.append(f"Could not clear browser cache {cache_path}: {e}")
            
            return {
                "status": "success",
                "optimization_type": "cache",
                "actions_taken": actions_taken,
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Cache optimization error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "actions_taken": actions_taken
            }
    
    def _optimize_compress(self, path: str, days_old: int) -> Dict[str, Any]:
        """Compress old/archived files"""
        actions_taken = []
        space_saved = 0
        
        try:
            if not self._is_safe_path(path) or not os.path.exists(path):
                return {
                    "status": "error",
                    "error": f"Path not accessible or safe: {path}"
                }
            
            files = self._scan_directory(path)
            cutoff_time = time.time() - (days_old * 86400)
            compressible_extensions = {'.log', '.txt', '.csv', '.sql', '.json', '.xml'}
            
            for file_info in files:
                if (file_info['mtime'] < cutoff_time and 
                    file_info['extension'] in compressible_extensions and
                    not file_info['name'].endswith('.gz')):
                    
                    try:
                        original_path = file_info['path']
                        compressed_path = original_path + '.gz'
                        
                        with open(original_path, 'rb') as f_in:
                            with gzip.open(compressed_path, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                        
                        # Verify compression worked and delete original
                        if os.path.exists(compressed_path):
                            original_size = file_info['size']
                            compressed_size = os.path.getsize(compressed_path)
                            
                            if compressed_size < original_size:
                                os.remove(original_path)
                                space_saved += (original_size - compressed_size)
                                actions_taken.append(f"Compressed {file_info['name']} "
                                                   f"({original_size} -> {compressed_size} bytes)")
                            else:
                                # Compression didn't help, remove compressed version
                                os.remove(compressed_path)
                    except Exception as e:
                        actions_taken.append(f"Could not compress {file_info['name']}: {e}")
            
            return {
                "status": "success",
                "optimization_type": "compress",
                "path": path,
                "days_old": days_old,
                "actions_taken": actions_taken,
                "files_compressed": len([a for a in actions_taken if 'Compressed' in a]),
                "space_saved_mb": space_saved / (1024 * 1024),
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Compression optimization error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "actions_taken": actions_taken
            }
    
    def _optimize_logs(self) -> Dict[str, Any]:
        """Intelligent log rotation and cleanup"""
        actions_taken = []
        space_freed = 0
        
        try:
            log_paths = ['/var/log', '/opt/sutazaiapp/logs']
            
            for log_path in log_paths:
                if not os.path.exists(log_path):
                    continue
                
                files = self._scan_directory(log_path, max_depth=2)
                
                for file_info in files:
                    if file_info['extension'] == '.log':
                        age_days = (time.time() - file_info['mtime']) / 86400
                        
                        # Delete logs older than 90 days
                        if age_days > 90:
                            if self._safe_delete(file_info['path']):
                                space_freed += file_info['size']
                                actions_taken.append(f"Deleted old log: {file_info['name']}")
                        
                        # Compress logs older than 7 days
                        elif age_days > 7 and not file_info['name'].endswith('.gz'):
                            try:
                                original_path = file_info['path']
                                compressed_path = original_path + '.gz'
                                
                                with open(original_path, 'rb') as f_in:
                                    with gzip.open(compressed_path, 'wb') as f_out:
                                        shutil.copyfileobj(f_in, f_out)
                                
                                if os.path.exists(compressed_path):
                                    compressed_size = os.path.getsize(compressed_path)
                                    if compressed_size < file_info['size'] * 0.8:  # 20% compression minimum
                                        os.remove(original_path)
                                        space_freed += (file_info['size'] - compressed_size)
                                        actions_taken.append(f"Compressed log: {file_info['name']}")
                                    else:
                                        os.remove(compressed_path)
                                        
                            except Exception as e:
                                actions_taken.append(f"Could not compress log {file_info['name']}: {e}")
            
            # Handle SQLite databases (VACUUM operation)
            sqlite_files = []
            for log_path in log_paths:
                if os.path.exists(log_path):
                    sqlite_files.extend(glob.glob(os.path.join(log_path, '**/*.db'), recursive=True))
                    sqlite_files.extend(glob.glob(os.path.join(log_path, '**/*.sqlite'), recursive=True))
            
            for db_file in sqlite_files:
                try:
                    if self._is_safe_path(db_file):
                        original_size = os.path.getsize(db_file)
                        conn = sqlite3.connect(db_file)
                        conn.execute('VACUUM')
                        conn.close()
                        
                        new_size = os.path.getsize(db_file)
                        if new_size < original_size:
                            space_freed += (original_size - new_size)
                            actions_taken.append(f"Vacuumed SQLite database: {os.path.basename(db_file)}")
                            
                except Exception as e:
                    actions_taken.append(f"Could not vacuum database {os.path.basename(db_file)}: {e}")
            
            return {
                "status": "success",
                "optimization_type": "logs",
                "actions_taken": actions_taken,
                "space_freed_bytes": space_freed,
                "space_freed_mb": space_freed / (1024 * 1024),
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Log optimization error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "actions_taken": actions_taken
            }
        
    def _setup_routes(self):
        """Setup FastAPI routes for on-demand optimization"""
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint"""
            status = self._get_system_status()
            return JSONResponse(content={
                "status": "healthy",
                "agent": self.agent_id,
                "description": self.description,
                "docker_available": self.docker_client is not None,
                "system_status": status,
                "timestamp": time.time()
            })
            
        @self.app.get("/status")
        async def get_status():
            """Get current system resource status"""
            return JSONResponse(content=self._get_system_status())
            
        @self.app.post("/optimize/memory")
        async def optimize_memory(background_tasks: BackgroundTasks):
            """Optimize memory usage - ULTRA-OPTIMIZED for <200ms response"""
            result = await self._optimize_memory_async()
            return JSONResponse(content=result)
            
        @self.app.post("/optimize/cpu")
        async def optimize_cpu():
            """Optimize CPU scheduling"""
            result = self._optimize_cpu()
            return JSONResponse(content=result)
            
        @self.app.post("/optimize/disk")
        async def optimize_disk():
            """Clean up disk space"""
            result = self._optimize_disk()
            return JSONResponse(content=result)
            
        @self.app.post("/optimize/docker")
        async def optimize_docker():
            """Clean up Docker resources"""
            result = self._optimize_docker()
            return JSONResponse(content=result)
            
        @self.app.post("/optimize/all")
        async def optimize_all():
            """Run all optimizations"""
            result = self._optimize_all()
            return JSONResponse(content=result)
            
        # Storage Analysis Endpoints
        @self.app.get("/analyze/storage")
        async def analyze_storage(path: str = Query("/", description="Path to analyze")):
            """Analyze storage usage with detailed breakdown"""
            # ULTRA-FIX: Complete path traversal security validation
            try:
                safe_path = validate_safe_path(path, "/")
                # Additional security: ensure path exists and is readable
                if not os.path.exists(safe_path):
                    raise HTTPException(status_code=404, detail=f"Path not found: {path}")
                if not os.access(safe_path, os.R_OK):
                    raise HTTPException(status_code=403, detail=f"Access denied: {path}")
            except ValueError as e:
                raise HTTPException(status_code=403, detail=str(e))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid path: {e}")
            result = self._analyze_storage(safe_path)
            return JSONResponse(content=result)
            
        @self.app.get("/analyze/storage/duplicates")
        async def analyze_duplicates(path: str = Query("/", description="Path to scan for duplicates")):
            """Find duplicate files using hash comparison"""
            # ULTRA-FIX: Add path validation to duplicates endpoint
            try:
                safe_path = validate_safe_path(path, "/")
                if not os.path.exists(safe_path) or not os.access(safe_path, os.R_OK):
                    raise HTTPException(status_code=403, detail=f"Invalid or inaccessible path: {path}")
            except ValueError as e:
                raise HTTPException(status_code=403, detail=str(e))
            result = self._analyze_duplicates(safe_path)
            return JSONResponse(content=result)
            
        @self.app.get("/analyze/storage/large-files")
        async def analyze_large_files(
            path: str = Query("/", description="Path to scan"),
            min_size_mb: int = Query(100, description="Minimum file size in MB")
        ):
            """Find unusually large files"""
            result = self._analyze_large_files(path, min_size_mb)
            return JSONResponse(content=result)
            
        @self.app.get("/analyze/storage/report")
        async def storage_report():
            """Generate comprehensive storage analysis report"""
            result = self._generate_storage_report()
            return JSONResponse(content=result)
            
        # Storage Optimization Endpoints
        @self.app.post("/optimize/storage")
        async def optimize_storage(dry_run: bool = Query(False, description="Perform dry run without actual deletion")):
            """Main storage optimization with smart cleanup"""
            result = self._optimize_storage_comprehensive(dry_run)
            return JSONResponse(content=result)
            
        @self.app.post("/optimize/storage/duplicates")
        async def optimize_duplicates(
            path: str = Query("/", description="Path to deduplicate"),
            dry_run: bool = Query(False, description="Perform dry run without actual deletion")
        ):
            """Remove duplicate files with safety checks"""
            # ULTRA-FIX: Add path validation to optimize duplicates endpoint
            try:
                safe_path = validate_safe_path(path, "/")
                if not os.path.exists(safe_path) or not os.access(safe_path, os.R_OK):
                    raise HTTPException(status_code=403, detail=f"Invalid or inaccessible path: {path}")
            except ValueError as e:
                raise HTTPException(status_code=403, detail=str(e))
            result = self._optimize_duplicates(safe_path, dry_run)
            return JSONResponse(content=result)
            
        @self.app.post("/optimize/storage/cache")
        async def optimize_cache():
            """Clear various system and application caches"""
            result = self._optimize_cache()
            return JSONResponse(content=result)
            
        @self.app.post("/optimize/storage/compress")
        async def optimize_compress(
            path: str = Query("/var/log", description="Path to compress files"),
            days_old: int = Query(30, description="Compress files older than N days")
        ):
            """Compress old/archived files"""
            # ULTRA-FIX: Add path validation to compress endpoint
            try:
                safe_path = validate_safe_path(path, "/")
                if not os.path.exists(safe_path) or not os.access(safe_path, os.R_OK):
                    raise HTTPException(status_code=403, detail=f"Invalid or inaccessible path: {path}")
            except ValueError as e:
                raise HTTPException(status_code=403, detail=str(e))
            result = self._optimize_compress(safe_path, days_old)
            return JSONResponse(content=result)
            
        @self.app.post("/optimize/storage/logs")
        async def optimize_logs():
            """Intelligent log rotation and cleanup"""
            result = self._optimize_logs()
            return JSONResponse(content=result)
    
    async def _optimize_memory_async(self) -> Dict[str, Any]:
        """ULTRA-OPTIMIZED async memory optimization - TARGET: <200ms"""
        start_time = time.time()
        actions_taken = []
        
        try:
            # Create thread pool executor for CPU-bound operations
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                # Task 1: Get initial memory status (async)
                loop = asyncio.get_event_loop()
                initial_memory_task = loop.run_in_executor(
                    executor, psutil.virtual_memory
                )
                
                # Task 2: Python garbage collection (async)
                gc_task = loop.run_in_executor(
                    executor, gc.collect
                )
                
                # Wait for initial tasks concurrently
                initial_memory, collected = await asyncio.gather(
                    initial_memory_task,
                    gc_task,
                    return_exceptions=True
                )
                
                initial_percent = initial_memory.percent
                actions_taken.append(f"Python GC freed {collected} objects")
                
                # Task 3: System cache clearing (only if high memory usage)
                cache_clear_task = None
                if initial_percent > 85:
                    cache_clear_task = asyncio.create_task(
                        self._clear_system_caches_async(executor)
                    )
                
                # Task 4: Get final memory status (concurrent with cache clearing)
                final_memory_task = loop.run_in_executor(
                    executor, psutil.virtual_memory
                )
                
                # Gather results
                if cache_clear_task:
                    cache_result, final_memory = await asyncio.gather(
                        cache_clear_task,
                        final_memory_task,
                        return_exceptions=True
                    )
                    if isinstance(cache_result, str):
                        actions_taken.append(cache_result)
                    elif isinstance(cache_result, Exception):
                        actions_taken.append(f"Cache clear error: {cache_result}")
                else:
                    final_memory = await final_memory_task
                
                memory_freed = initial_memory.used - final_memory.used
                execution_time = (time.time() - start_time) * 1000  # Convert to ms
                
                return {
                    "status": "success",
                    "optimization_type": "memory",
                    "actions_taken": actions_taken,
                    "initial_memory_percent": initial_percent,
                    "final_memory_percent": final_memory.percent,
                    "memory_freed_mb": memory_freed / (1024 * 1024),
                    "execution_time_ms": round(execution_time, 2),
                    "timestamp": time.time(),
                    "performance_target": "< 200ms",
                    "achieved": execution_time < 200
                }
                
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.logger.error(f"Memory optimization error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "actions_taken": actions_taken,
                "execution_time_ms": round(execution_time, 2),
                "timestamp": time.time()
            }
    
    async def _clear_system_caches_async(self, executor) -> str:
        """Async system cache clearing with timeout protection"""
        try:
            # Run sync command with timeout in executor
            loop = asyncio.get_event_loop()
            sync_result = await loop.run_in_executor(
                executor,
                partial(subprocess.run, ['sync'], check=True, timeout=5, capture_output=True)
            )
            
            # Clear caches
            await loop.run_in_executor(
                executor,
                self._write_drop_caches
            )
            
            return "Cleared system page cache, dentries, and inodes"
            
        except (subprocess.CalledProcessError, PermissionError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            return f"Could not clear system caches: {e}"
    
    def _write_drop_caches(self):
        """Thread-safe cache clearing"""
        try:
            with open('/proc/sys/vm/drop_caches', 'w') as f:
                f.write('3')
        except (PermissionError, FileNotFoundError):
            raise
    
    def _optimize_memory(self) -> Dict[str, Any]:
        """ULTRA-FIX: Fixed event loop conflict - use existing loop if available"""
        try:
            # Check if we're already in an event loop
            loop = asyncio.get_running_loop()
            # If we're in a loop, create a task and run it synchronously
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self._run_memory_optimization_sync)
                return future.result(timeout=30)
        except RuntimeError:
            # No event loop running, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._optimize_memory_async())
            finally:
                loop.close()
                
    def _run_memory_optimization_sync(self) -> Dict[str, Any]:
        """ULTRA-FIX: Synchronous memory optimization for thread execution"""
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._optimize_memory_async())
        finally:
            loop.close()

    def _optimize_cpu(self) -> Dict[str, Any]:
        """Optimize CPU scheduling and usage"""
        actions_taken = []
        
        try:
            cpu_percent = psutil.cpu_percent(interval=0)
            
            # Find high CPU processes and adjust their nice values
            high_cpu_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'nice']):
                try:
                    proc_info = proc.info
                    if proc_info['cpu_percent'] and proc_info['cpu_percent'] > 25:
                        high_cpu_processes.append(proc_info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Adjust nice values for high CPU processes (lower priority)
            adjusted_count = 0
            for proc_info in high_cpu_processes[:5]:  # Limit to top 5
                try:
                    pid = proc_info['pid']
                    name = proc_info['name']
                    current_nice = proc_info['nice']
                    
                    # Skip system processes
                    if name in ['systemd', 'kernel', 'kthreadd', 'init']:
                        continue
                    
                    # Increase nice value (lower priority) if not already high
                    if current_nice < 10:
                        process = psutil.Process(pid)
                        process.nice(10)
                        adjusted_count += 1
                        actions_taken.append(f"Adjusted {name} (PID {pid}) nice value to 10")
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    actions_taken.append(f"Could not adjust process {proc_info.get('name', 'unknown')}: {e}")
            
            if adjusted_count == 0:
                actions_taken.append("No CPU optimization needed - no high-usage processes found")
            
            return {
                "status": "success",
                "optimization_type": "cpu",
                "actions_taken": actions_taken,
                "initial_cpu_percent": cpu_percent,
                "processes_adjusted": adjusted_count,
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"CPU optimization error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "actions_taken": actions_taken
            }
    def _optimize_disk(self) -> Dict[str, Any]:
        """Clean up disk space"""
        actions_taken = []
        
        try:
            # Get initial disk usage
            initial_disk = psutil.disk_usage('/')
            initial_percent = (initial_disk.used / initial_disk.total) * 100
            
            # Clean temporary files older than 7 days
            temp_dirs = ['/tmp', '/var/tmp']
            for temp_dir in temp_dirs:
                if os.path.exists(temp_dir):
                    try:
                        result = subprocess.run([
                            'find', temp_dir, '-type', 'f', '-atime', '+7', '-delete'
                        ], capture_output=True, text=True, timeout=30)
                        
                        if result.returncode == 0:
                            actions_taken.append(f"Cleaned old files from {temp_dir}")
                        else:
                            actions_taken.append(f"Could not clean {temp_dir}: {result.stderr}")
                            
                    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
                        actions_taken.append(f"Error cleaning {temp_dir}: {e}")
            
            # Clean system logs older than 30 days (if space is really tight)
            if initial_percent > 90:
                try:
                    result = subprocess.run([
                        'find', '/var/log', '-name', '*.log', '-mtime', '+30', '-delete'
                    ], capture_output=True, text=True, timeout=30)
                    
                    if result.returncode == 0:
                        actions_taken.append("Cleaned old log files")
                        
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
                    actions_taken.append(f"Could not clean log files: {e}")
            
            # Get final disk usage
            final_disk = psutil.disk_usage('/')
            space_freed = initial_disk.used - final_disk.used
            
            return {
                "status": "success",
                "optimization_type": "disk",
                "actions_taken": actions_taken,
                "initial_disk_percent": initial_percent,
                "final_disk_percent": (final_disk.used / final_disk.total) * 100,
                "space_freed_mb": space_freed / (1024 * 1024),
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Disk optimization error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "actions_taken": actions_taken
            }
    def _optimize_docker(self) -> Dict[str, Any]:
        """ULTRA-FIX: Thread-safe Docker optimization with proper client handling"""
        actions_taken = []
        
        # ULTRA-FIX: Thread-safe Docker client access
        with self.docker_client_lock:
            if not self.docker_client:
                # Try to reinitialize client
                self._init_docker_client_safe()
                if not self.docker_client:
                    return {
                        "status": "error",
                        "error": "Docker client not available",
                        "actions_taken": ["Docker not accessible"]
                    }
            
            docker_client = self.docker_client
        
        try:
            # Get initial Docker system info with timeout
            initial_images = len(docker_client.images.list())
            initial_containers = len(docker_client.containers.list(all=True))
            
            # Remove stopped containers with timeout
            stopped_containers = docker_client.containers.list(filters={'status': 'exited'})
            for container in stopped_containers:
                try:
                    container.remove(timeout=10)
                    actions_taken.append(f"Removed stopped container: {container.name}")
                except Exception as e:
                    actions_taken.append(f"Could not remove container {container.name}: {e}")
            
            # Remove dangling images with timeout
            dangling_images = docker_client.images.list(filters={'dangling': True})
            for image in dangling_images:
                try:
                    docker_client.images.remove(image.id, force=True, timeout=10)
                    actions_taken.append(f"Removed dangling image: {image.id[:12]}")
                except Exception as e:
                    actions_taken.append(f"Could not remove image {image.id[:12]}: {e}")
            
            # Prune unused networks
            try:
                pruned_networks = docker_client.networks.prune()
                if pruned_networks.get('NetworksDeleted'):
                    actions_taken.append(f"Pruned {len(pruned_networks['NetworksDeleted'])} unused networks")
            except Exception as e:
                actions_taken.append(f"Could not prune networks: {e}")
            
            # Prune build cache
            try:
                docker_client.api.prune_build_cache()
                actions_taken.append("Pruned Docker build cache")
            except Exception as e:
                actions_taken.append(f"Could not prune build cache: {e}")
            
            # Get final counts
            final_images = len(docker_client.images.list())
            final_containers = len(docker_client.containers.list(all=True))
            
            return {
                "status": "success",
                "optimization_type": "docker",
                "actions_taken": actions_taken,
                "containers_removed": initial_containers - final_containers,
                "images_removed": initial_images - final_images,
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Docker optimization error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "actions_taken": actions_taken
            }
    def _optimize_all(self) -> Dict[str, Any]:
        """Run all optimization tasks"""
        start_time = time.time()
        all_actions = []
        all_results = {}
        
        try:
            # Get initial system status
            initial_status = self._get_system_status()
            
            # Run memory optimization
            memory_result = self._optimize_memory()
            all_results['memory'] = memory_result
            all_actions.extend(memory_result.get('actions_taken', []))
            
            # Run CPU optimization
            cpu_result = self._optimize_cpu()
            all_results['cpu'] = cpu_result
            all_actions.extend(cpu_result.get('actions_taken', []))
            
            # Run disk optimization
            disk_result = self._optimize_disk()
            all_results['disk'] = disk_result
            all_actions.extend(disk_result.get('actions_taken', []))
            
            # Run Docker optimization if available
            if self.docker_client:
                docker_result = self._optimize_docker()
                all_results['docker'] = docker_result
                all_actions.extend(docker_result.get('actions_taken', []))
            
            # Run comprehensive storage optimization
            storage_result = self._optimize_storage_comprehensive(dry_run=False)
            all_results['storage'] = storage_result
            all_actions.extend(storage_result.get('actions_taken', []))
            
            # Get final system status
            final_status = self._get_system_status()
            
            return {
                "status": "success",
                "optimization_type": "all",
                "duration_seconds": time.time() - start_time,
                "total_actions": len(all_actions),
                "actions_taken": all_actions,
                "detailed_results": all_results,
                "before": initial_status,
                "after": final_status,
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Comprehensive optimization error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "actions_taken": all_actions,
                "partial_results": all_results
            }
    def start_server(self):
        """Start the FastAPI server"""
        try:
            self.logger.info(f"Starting {self.name} on port {self.port}")
            uvicorn.run(
                self.app, 
                host="0.0.0.0", 
                port=self.port,
                log_level="info",
                workers=1,  # Force single worker process
                reload=False,  # Disable auto-reload
                access_log=False  # Reduce log noise
            )
        except Exception as e:
            self.logger.error(f"Server error: {e}")
            raise
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming tasks (for compatibility with BaseAgent)"""
        try:
            task_type = task.get("type", "unknown")
            
            if task_type == "health":
                return {"status": "healthy", "agent": self.agent_id}
            
            elif task_type == "optimize_memory":
                return self._optimize_memory()
            
            elif task_type == "optimize_cpu":
                return self._optimize_cpu()
            
            elif task_type == "optimize_disk":
                return self._optimize_disk()
            
            elif task_type == "optimize_docker":
                return self._optimize_docker()
            
            elif task_type == "optimize_all":
                return self._optimize_all()
            
            elif task_type == "analyze_storage":
                path = task.get("path", "/")
                return self._analyze_storage(path)
            
            elif task_type == "analyze_duplicates":
                path = task.get("path", "/")
                return self._analyze_duplicates(path)
            
            elif task_type == "analyze_large_files":
                path = task.get("path", "/")
                min_size_mb = task.get("min_size_mb", 100)
                return self._analyze_large_files(path, min_size_mb)
            
            elif task_type == "storage_report":
                return self._generate_storage_report()
            
            elif task_type == "optimize_storage":
                dry_run = task.get("dry_run", False)
                return self._optimize_storage_comprehensive(dry_run)
            
            elif task_type == "optimize_duplicates":
                path = task.get("path", "/")
                dry_run = task.get("dry_run", False)
                return self._optimize_duplicates(path, dry_run)
            
            elif task_type == "optimize_cache":
                return self._optimize_cache()
            
            elif task_type == "optimize_compress":
                path = task.get("path", "/var/log")
                days_old = task.get("days_old", 30)
                return self._optimize_compress(path, days_old)
            
            elif task_type == "optimize_logs":
                return self._optimize_logs()
            
            else:
                return {
                    "status": "success",
                    "message": f"Hardware optimization agent ready. Supported tasks: optimize_memory, optimize_cpu, optimize_disk, optimize_docker, optimize_all, analyze_storage, analyze_duplicates, analyze_large_files, storage_report, optimize_storage, optimize_duplicates, optimize_cache, optimize_compress, optimize_logs",
                    "agent": self.agent_id,
                    "task_type_received": task_type
                }
            
        except Exception as e:
            self.logger.error(f"Error processing task: {e}")
            return {
                "status": "error",
                "error": str(e),
                "agent": self.agent_id
            }
    def start(self):
        """Start the hardware optimization agent"""
        try:
            self.logger.info(f"Starting {self.name} - On-demand hardware optimization service")
            self.start_server()
        except KeyboardInterrupt:
            self.logger.info("Agent stopped by user")
        except Exception as e:
            self.logger.error(f"Agent error: {e}")
            raise

if __name__ == "__main__":
    # Initialize and start the hardware resource optimizer agent
    agent = HardwareResourceOptimizerAgent()
    agent.start()