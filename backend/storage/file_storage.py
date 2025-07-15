"""
Optimized File Storage System for SutazAI
High-performance file operations with compression and deduplication
"""

import asyncio
import hashlib
import json
import logging
import os
import shutil
import time
import zlib
from pathlib import Path
from typing import Dict, List, Any, Optional, BinaryIO
from dataclasses import dataclass
import threading
import mimetypes

logger = logging.getLogger(__name__)

@dataclass
class StorageConfig:
    """File storage configuration"""
    storage_root: str = "/opt/sutazaiapp/storage"
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    compression_enabled: bool = True
    deduplication_enabled: bool = True
    backup_enabled: bool = True
    retention_days: int = 365

class OptimizedFileStorage:
    """Optimized file storage with compression and deduplication"""
    
    def __init__(self, config: StorageConfig = None):
        self.config = config or StorageConfig()
        self.storage_root = Path(self.config.storage_root)
        self.storage_root.mkdir(parents=True, exist_ok=True)
        
        # Storage directories
        self.data_dir = self.storage_root / "data"
        self.index_dir = self.storage_root / "index"
        self.temp_dir = self.storage_root / "temp"
        self.backup_dir = self.storage_root / "backup"
        
        for directory in [self.data_dir, self.index_dir, self.temp_dir, self.backup_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # File index for deduplication
        self.file_index = {}
        self.storage_lock = threading.RLock()
        
        # Statistics
        self.stats = {
            "files_stored": 0,
            "bytes_stored": 0,
            "compression_ratio": 0.0,
            "deduplication_savings": 0,
            "operations": 0
        }
    
    async def initialize(self):
        """Initialize file storage system"""
        logger.info("🔄 Initializing Optimized File Storage")
        
        # Load file index
        await self._load_file_index()
        
        # Start cleanup task
        asyncio.create_task(self._cleanup_task())
        
        logger.info("✅ File storage initialized")
    
    async def _load_file_index(self):
        """Load file index from disk"""
        try:
            index_file = self.index_dir / "file_index.json"
            if index_file.exists():
                with open(index_file, 'r') as f:
                    self.file_index = json.load(f)
                
                logger.info(f"Loaded file index with {len(self.file_index)} entries")
        except Exception as e:
            logger.warning(f"Failed to load file index: {e}")
            self.file_index = {}
    
    async def store_file(self, file_data: bytes, filename: str, metadata: Dict[str, Any] = None) -> str:
        """Store file with optimization"""
        self.stats["operations"] += 1
        
        if len(file_data) > self.config.max_file_size:
            raise ValueError(f"File too large: {len(file_data)} bytes")
        
        # Calculate file hash for deduplication
        file_hash = hashlib.sha256(file_data).hexdigest()
        
        # Check for existing file (deduplication)
        if self.config.deduplication_enabled and file_hash in self.file_index:
            existing_entry = self.file_index[file_hash]
            existing_entry["reference_count"] += 1
            existing_entry["last_accessed"] = time.time()
            
            # Add new filename reference
            if "filenames" not in existing_entry:
                existing_entry["filenames"] = []
            existing_entry["filenames"].append(filename)
            
            await self._save_file_index()
            
            self.stats["deduplication_savings"] += len(file_data)
            return file_hash
        
        # Compress file if enabled
        compressed_data = file_data
        compression_ratio = 1.0
        
        if self.config.compression_enabled:
            compressed_data = zlib.compress(file_data, level=6)
            compression_ratio = len(compressed_data) / len(file_data)
        
        # Store file
        storage_path = self.data_dir / f"{file_hash[:2]}" / f"{file_hash[2:4]}"
        storage_path.mkdir(parents=True, exist_ok=True)
        
        file_path = storage_path / file_hash
        
        with open(file_path, 'wb') as f:
            f.write(compressed_data)
        
        # Update index
        with self.storage_lock:
            self.file_index[file_hash] = {
                "filename": filename,
                "filenames": [filename],
                "original_size": len(file_data),
                "compressed_size": len(compressed_data),
                "compression_ratio": compression_ratio,
                "mime_type": mimetypes.guess_type(filename)[0],
                "stored_at": time.time(),
                "last_accessed": time.time(),
                "reference_count": 1,
                "metadata": metadata or {},
                "compressed": self.config.compression_enabled
            }
        
        await self._save_file_index()
        
        # Update statistics
        self.stats["files_stored"] += 1
        self.stats["bytes_stored"] += len(file_data)
        self.stats["compression_ratio"] = (
            self.stats["compression_ratio"] * (self.stats["files_stored"] - 1) + compression_ratio
        ) / self.stats["files_stored"]
        
        logger.info(f"Stored file {filename} with hash {file_hash}")
        return file_hash
    
    async def retrieve_file(self, file_hash: str) -> Optional[bytes]:
        """Retrieve file by hash"""
        self.stats["operations"] += 1
        
        if file_hash not in self.file_index:
            return None
        
        entry = self.file_index[file_hash]
        
        # Update access time
        entry["last_accessed"] = time.time()
        
        # Get file path
        storage_path = self.data_dir / f"{file_hash[:2]}" / f"{file_hash[2:4]}" / file_hash
        
        if not storage_path.exists():
            logger.error(f"File not found: {file_hash}")
            return None
        
        # Read file
        with open(storage_path, 'rb') as f:
            file_data = f.read()
        
        # Decompress if needed
        if entry.get("compressed", False):
            file_data = zlib.decompress(file_data)
        
        return file_data
    
    async def delete_file(self, file_hash: str, filename: str = None) -> bool:
        """Delete file or reduce reference count"""
        if file_hash not in self.file_index:
            return False
        
        entry = self.file_index[file_hash]
        
        # If filename specified, remove only that reference
        if filename and "filenames" in entry:
            if filename in entry["filenames"]:
                entry["filenames"].remove(filename)
                entry["reference_count"] -= 1
        else:
            # Remove all references
            entry["reference_count"] = 0
        
        # Delete file if no more references
        if entry["reference_count"] <= 0:
            storage_path = self.data_dir / f"{file_hash[:2]}" / f"{file_hash[2:4]}" / file_hash
            
            try:
                if storage_path.exists():
                    storage_path.unlink()
                
                del self.file_index[file_hash]
                await self._save_file_index()
                
                logger.info(f"Deleted file {file_hash}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete file {file_hash}: {e}")
                return False
        else:
            await self._save_file_index()
            return True
    
    async def list_files(self, metadata_filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """List stored files with optional metadata filtering"""
        files = []
        
        for file_hash, entry in self.file_index.items():
            # Apply metadata filter if specified
            if metadata_filter:
                match = True
                for key, value in metadata_filter.items():
                    if key not in entry.get("metadata", {}) or entry["metadata"][key] != value:
                        match = False
                        break
                
                if not match:
                    continue
            
            files.append({
                "hash": file_hash,
                "filename": entry["filename"],
                "filenames": entry.get("filenames", [entry["filename"]]),
                "size": entry["original_size"],
                "compressed_size": entry["compressed_size"],
                "mime_type": entry.get("mime_type"),
                "stored_at": entry["stored_at"],
                "last_accessed": entry["last_accessed"],
                "reference_count": entry["reference_count"],
                "metadata": entry.get("metadata", {})
            })
        
        return files
    
    async def get_file_info(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """Get file information"""
        if file_hash not in self.file_index:
            return None
        
        entry = self.file_index[file_hash]
        return {
            "hash": file_hash,
            "filename": entry["filename"],
            "filenames": entry.get("filenames", [entry["filename"]]),
            "size": entry["original_size"],
            "compressed_size": entry["compressed_size"],
            "compression_ratio": entry["compression_ratio"],
            "mime_type": entry.get("mime_type"),
            "stored_at": entry["stored_at"],
            "last_accessed": entry["last_accessed"],
            "reference_count": entry["reference_count"],
            "metadata": entry.get("metadata", {}),
            "compressed": entry.get("compressed", False)
        }
    
    async def _save_file_index(self):
        """Save file index to disk"""
        try:
            index_file = self.index_dir / "file_index.json"
            temp_file = self.index_dir / "file_index.json.tmp"
            
            with open(temp_file, 'w') as f:
                json.dump(self.file_index, f, indent=2)
            
            # Atomic replace
            temp_file.replace(index_file)
            
        except Exception as e:
            logger.error(f"Failed to save file index: {e}")
    
    async def _cleanup_task(self):
        """Periodic cleanup task"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Clean up old temporary files
                cutoff_time = time.time() - 3600  # 1 hour old
                for temp_file in self.temp_dir.glob("*"):
                    try:
                        if temp_file.stat().st_mtime < cutoff_time:
                            temp_file.unlink()
                    except Exception:
                        pass
                
                # Clean up old files if retention policy enabled
                if self.config.retention_days > 0:
                    cutoff_time = time.time() - (self.config.retention_days * 24 * 3600)
                    
                    expired_files = [
                        file_hash for file_hash, entry in self.file_index.items()
                        if entry["last_accessed"] < cutoff_time
                    ]
                    
                    for file_hash in expired_files:
                        await self.delete_file(file_hash)
                    
                    if expired_files:
                        logger.info(f"Cleaned up {len(expired_files)} expired files")
                
            except Exception as e:
                logger.error(f"Storage cleanup error: {e}")
    
    def get_storage_statistics(self) -> Dict[str, Any]:
        """Get storage statistics"""
        total_compressed_size = sum(
            entry["compressed_size"] for entry in self.file_index.values()
        )
        
        total_original_size = sum(
            entry["original_size"] for entry in self.file_index.values()
        )
        
        return {
            **self.stats,
            "total_files": len(self.file_index),
            "total_original_size_mb": total_original_size / (1024 * 1024),
            "total_compressed_size_mb": total_compressed_size / (1024 * 1024),
            "overall_compression_ratio": total_compressed_size / max(1, total_original_size),
            "deduplication_savings_mb": self.stats["deduplication_savings"] / (1024 * 1024),
            "storage_efficiency": (1 - total_compressed_size / max(1, total_original_size)) * 100
        }

# Global file storage instance
file_storage = OptimizedFileStorage()
