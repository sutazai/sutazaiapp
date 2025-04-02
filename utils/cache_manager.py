#!/usr/bin/env python3
"""
Cache Manager utility for SutazAI application.
Provides functions to manage and clean cache across the application.
"""

import os
import shutil
import logging
import json
from pathlib import Path
import time

logger = logging.getLogger(__name__)

# Default cache paths
DEFAULT_PATHS = [
    '/opt/sutazaiapp/data/models/.cache',
    '/opt/sutazaiapp/.pytest_cache',
    '/opt/sutazaiapp/.ruff_cache'
]

class CacheManager:
    """Cache management utility for SutazAI application."""
    
    def __init__(self, cache_paths=None, max_cache_size_mb=5000, cache_ttl_days=30):
        """
        Initialize the cache manager.
        
        Args:
            cache_paths: List of cache paths to manage (in addition to default paths)
            max_cache_size_mb: Maximum cache size in MB
            cache_ttl_days: Cache time-to-live in days
        """
        self.cache_paths = DEFAULT_PATHS + (cache_paths or [])
        self.max_cache_size_mb = max_cache_size_mb
        self.cache_ttl_days = cache_ttl_days
    
    def clean_pycache(self, directory=None):
        """
        Recursively remove all __pycache__ directories and .pyc/.pyo files
        
        Args:
            directory: Directory to clean (default: /opt/sutazaiapp)
        """
        if directory is None:
            directory = "/opt/sutazaiapp"
        
        logger.info(f"Cleaning Python cache in {directory}")
        
        for root, dirs, files in os.walk(directory):
            # Remove __pycache__ directories
            if '__pycache__' in dirs:
                pycache_path = os.path.join(root, '__pycache__')
                logger.debug(f"Removing: {pycache_path}")
                try:
                    shutil.rmtree(pycache_path)
                    dirs.remove('__pycache__')  # Don't recurse into deleted directory
                except Exception as e:
                    logger.error(f"Failed to remove {pycache_path}: {e}")
            
            # Remove .pyc and .pyo files
            for file in files:
                if file.endswith(('.pyc', '.pyo')):
                    pyc_file = os.path.join(root, file)
                    logger.debug(f"Removing: {pyc_file}")
                    try:
                        os.unlink(pyc_file)
                    except Exception as e:
                        logger.error(f"Failed to remove {pyc_file}: {e}")
        
        logger.info("Python cache cleanup complete!")
    
    def clean_model_cache(self, preserve_index=True):
        """
        Clean the model cache directory.
        
        Args:
            preserve_index: Whether to preserve the cache index file
        """
        cache_dir = '/opt/sutazaiapp/data/models/.cache'
        if not os.path.exists(cache_dir):
            logger.info(f"Model cache directory {cache_dir} does not exist.")
            return
        
        logger.info(f"Cleaning model cache at {cache_dir}")
        
        # Backup index file if it exists and we want to preserve it
        index_path = os.path.join(cache_dir, 'cache_index.json')
        index_backup = None
        
        if preserve_index and os.path.exists(index_path):
            try:
                with open(index_path, 'r') as f:
                    index_data = json.load(f)
                index_backup = index_data
                logger.info("Cache index file backed up.")
            except Exception as e:
                logger.error(f"Failed to backup cache index: {e}")
        
        # Clean cache directory
        try:
            for item in os.listdir(cache_dir):
                item_path = os.path.join(cache_dir, item)
                if os.path.isfile(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            logger.info("Model cache cleaned.")
        except Exception as e:
            logger.error(f"Failed to clean model cache: {e}")
        
        # Restore index file if needed
        if preserve_index and index_backup is not None:
            try:
                with open(index_path, 'w') as f:
                    json.dump(index_backup, f)
                logger.info("Cache index file restored.")
            except Exception as e:
                logger.error(f"Failed to restore cache index: {e}")
    
    def clean_all_caches(self):
        """Clean all cache directories."""
        logger.info("Starting full cache cleanup...")
        
        # Clean Python bytecode cache
        self.clean_pycache()
        
        # Clean model cache
        self.clean_model_cache()
        
        # Clean other specific caches
        for cache_path in self.cache_paths:
            if cache_path != '/opt/sutazaiapp/data/models/.cache' and os.path.exists(cache_path):
                logger.info(f"Cleaning cache at {cache_path}")
                try:
                    if os.path.isdir(cache_path):
                        shutil.rmtree(cache_path)
                        os.makedirs(cache_path, exist_ok=True)
                    else:
                        os.unlink(cache_path)
                except Exception as e:
                    logger.error(f"Failed to clean {cache_path}: {e}")
        
        logger.info("Full cache cleanup completed!")
    
    def get_cache_size(self):
        """Get the total size of all cache directories in MB."""
        total_size = 0
        
        # Add size of __pycache__ directories
        for root, dirs, files in os.walk("/opt/sutazaiapp"):
            if '__pycache__' in dirs:
                pycache_path = os.path.join(root, '__pycache__')
                total_size += self._get_dir_size(pycache_path)
        
        # Add size of other cache directories
        for cache_path in self.cache_paths:
            if os.path.exists(cache_path):
                total_size += self._get_dir_size(cache_path)
        
        return total_size / (1024 * 1024)  # Convert to MB
    
    def _get_dir_size(self, path):
        """Get the total size of a directory in bytes."""
        total_size = 0
        if os.path.isdir(path):
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    if os.path.isfile(file_path) and not os.path.islink(file_path):
                        total_size += os.path.getsize(file_path)
        elif os.path.isfile(path):
            total_size = os.path.getsize(path)
        return total_size
    
    def enforce_cache_limits(self):
        """Enforce cache size and TTL limits."""
        logger.info("Enforcing cache limits...")
        
        # Check cache size limit
        current_size_mb = self.get_cache_size()
        if current_size_mb > self.max_cache_size_mb:
            logger.info(f"Cache size ({current_size_mb:.2f} MB) exceeds limit ({self.max_cache_size_mb} MB). Cleaning...")
            self.clean_all_caches()
        else:
            logger.info(f"Cache size ({current_size_mb:.2f} MB) within limit ({self.max_cache_size_mb} MB).")
        
        # Check TTL for model cache files
        model_cache_dir = '/opt/sutazaiapp/data/models/.cache'
        if os.path.exists(model_cache_dir):
            now = time.time()
            ttl_seconds = self.cache_ttl_days * 24 * 3600
            
            for item in os.listdir(model_cache_dir):
                if item == 'cache_index.json':
                    continue  # Skip index file
                
                item_path = os.path.join(model_cache_dir, item)
                if os.path.isfile(item_path):
                    mtime = os.path.getmtime(item_path)
                    if now - mtime > ttl_seconds:
                        logger.info(f"Removing expired cache file: {item_path}")
                        try:
                            os.unlink(item_path)
                        except Exception as e:
                            logger.error(f"Failed to remove expired cache file: {e}")


# Convenience functions
def clean_pycache(directory=None):
    """Clean Python cache files."""
    CacheManager().clean_pycache(directory)

def clean_all_caches():
    """Clean all cache directories."""
    CacheManager().clean_all_caches()

def get_cache_size():
    """Get the total size of all cache directories in MB."""
    return CacheManager().get_cache_size()

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    # Run cache cleanup
    manager = CacheManager()
    manager.enforce_cache_limits() 