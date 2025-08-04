#!/usr/bin/env python3
"""
Advanced Memory Pool Manager for SutazAI
=======================================

Purpose: Manage shared memory pools for 131 agents with intelligent allocation and optimization
Usage: python scripts/memory-pool-manager.py [--pool-size 4096] [--monitor]
Requirements: Python 3.8+, psutil, mmap

Features:
- Shared memory pools for model weights and data
- Intelligent memory allocation and deallocation
- Memory compression and deduplication
- Pool fragmentation prevention
- Cross-agent memory sharing optimization
"""

import os
import sys
import mmap
import json
import time
import hashlib
import logging
import argparse
import threading
import multiprocessing
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import psutil
import pickle
import gzip
import zlib
import weakref
from enum import Enum
import struct
import uuid

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/logs/memory_pool_manager.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MemoryPoolManager')

class PoolType(Enum):
    """Memory pool types"""
    MODEL_WEIGHTS = "model_weights"
    DATA_BUFFER = "data_buffer" 
    COMMUNICATION = "communication"
    CACHE = "cache"
    TEMPORARY = "temporary"

class CompressionType(Enum):
    """Compression algorithms"""
    NONE = "none"
    ZLIB = "zlib"
    GZIP = "gzip"
    LZ4 = "lz4"  # If available

@dataclass
class MemoryBlock:
    """Memory block metadata"""
    block_id: str
    offset: int
    size: int
    allocated: bool
    owner_agent: Optional[str]
    access_count: int
    last_access: float
    compression: CompressionType
    checksum: str
    data_hash: Optional[str] = None
    ref_count: int = 0

@dataclass
class PoolStats:
    """Pool statistics"""
    total_size: int
    allocated_size: int
    free_size: int
    fragmentation_ratio: float
    allocation_count: int
    deallocation_count: int
    compression_ratio: float
    cache_hit_ratio: float
    average_block_size: float

class MemoryDeduplicator:
    """Memory deduplication system"""
    
    def __init__(self):
        self.hash_to_blocks = defaultdict(list)
        self.block_hashes = {}
        self.dedup_savings = 0
        self.lock = threading.RLock()
    
    def add_block(self, block_id: str, data: bytes) -> Optional[str]:
        """Add block for deduplication, returns existing block ID if duplicate found"""
        data_hash = hashlib.sha256(data).hexdigest()
        
        with self.lock:
            # Check if we already have this data
            if data_hash in self.hash_to_blocks:
                existing_blocks = self.hash_to_blocks[data_hash]
                if existing_blocks:
                    # Found duplicate
                    existing_block_id = existing_blocks[0]
                    self.dedup_savings += len(data)
                    logger.debug(f"Deduplication: {block_id} -> {existing_block_id}")
                    return existing_block_id
            
            # New unique data
            self.hash_to_blocks[data_hash].append(block_id)
            self.block_hashes[block_id] = data_hash
            return None
    
    def remove_block(self, block_id: str):
        """Remove block from deduplication tracking"""
        with self.lock:
            if block_id in self.block_hashes:
                data_hash = self.block_hashes[block_id]
                if data_hash in self.hash_to_blocks:
                    self.hash_to_blocks[data_hash].remove(block_id)
                    if not self.hash_to_blocks[data_hash]:
                        del self.hash_to_blocks[data_hash]
                del self.block_hashes[block_id]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get deduplication statistics"""
        with self.lock:
            unique_blocks = len(self.hash_to_blocks)
            total_blocks = len(self.block_hashes)
            
            return {
                "unique_blocks": unique_blocks,
                "total_blocks": total_blocks,
                "deduplication_ratio": (total_blocks - unique_blocks) / max(total_blocks, 1),
                "savings_bytes": self.dedup_savings
            }

class MemoryCompressor:
    """Memory compression utilities"""
    
    @staticmethod
    def compress(data: bytes, compression_type: CompressionType) -> bytes:
        """Compress data using specified algorithm"""
        if compression_type == CompressionType.NONE:
            return data
        elif compression_type == CompressionType.ZLIB:
            return zlib.compress(data, level=6)
        elif compression_type == CompressionType.GZIP:
            return gzip.compress(data, compresslevel=6)
        else:
            # Fallback to zlib
            return zlib.compress(data, level=6)
    
    @staticmethod
    def decompress(data: bytes, compression_type: CompressionType) -> bytes:
        """Decompress data using specified algorithm"""
        if compression_type == CompressionType.NONE:
            return data
        elif compression_type == CompressionType.ZLIB:
            return zlib.decompress(data)
        elif compression_type == CompressionType.GZIP:
            return gzip.decompress(data)
        else:
            # Fallback to zlib
            return zlib.decompress(data)
    
    @staticmethod
    def get_compression_ratio(original_size: int, compressed_size: int) -> float:
        """Calculate compression ratio"""
        if original_size == 0:
            return 1.0
        return compressed_size / original_size

class MemoryPool:
    """Advanced memory pool with fragmentation prevention"""
    
    def __init__(self, pool_id: str, size: int, pool_type: PoolType):
        self.pool_id = pool_id
        self.size = size
        self.pool_type = pool_type
        self.mmap_file = None
        self.memory_map = None
        self.blocks = {}  # block_id -> MemoryBlock
        self.free_blocks = []  # List of (offset, size) tuples
        self.allocation_lock = threading.RLock()
        self.access_count = 0
        self.hit_count = 0
        self.miss_count = 0
        
        # Initialize memory pool
        self._initialize_pool()
        
        # Add entire pool as one free block initially
        self.free_blocks.append((0, size))
        
        logger.info(f"Created memory pool '{pool_id}' of type {pool_type.value}, size {size} bytes")
    
    def _initialize_pool(self):
        """Initialize the memory-mapped file"""
        try:
            # Create temporary file for memory mapping
            self.mmap_file = f"/tmp/sutazai_pool_{self.pool_id}_{os.getpid()}"
            
            # Create and initialize file
            with open(self.mmap_file, "wb") as f:
                f.write(b'\x00' * self.size)
            
            # Memory map the file
            fd = os.open(self.mmap_file, os.O_RDWR)
            self.memory_map = mmap.mmap(fd, self.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
            os.close(fd)  # Close fd as mmap keeps it open
            
        except Exception as e:
            logger.error(f"Failed to initialize memory pool {self.pool_id}: {e}")
            raise
    
    def allocate(self, size: int, agent_id: str, compression: CompressionType = CompressionType.NONE) -> Optional[str]:
        """Allocate memory block"""
        with self.allocation_lock:
            # Find suitable free block using best-fit algorithm
            best_fit_idx = -1
            best_fit_size = float('inf')
            
            for i, (offset, block_size) in enumerate(self.free_blocks):
                if block_size >= size and block_size < best_fit_size:
                    best_fit_idx = i
                    best_fit_size = block_size
            
            if best_fit_idx == -1:
                # No suitable block found, try garbage collection
                self._garbage_collect()
                
                # Try again after GC
                for i, (offset, block_size) in enumerate(self.free_blocks):
                    if block_size >= size and block_size < best_fit_size:
                        best_fit_idx = i
                        best_fit_size = block_size
                
                if best_fit_idx == -1:
                    logger.warning(f"Pool {self.pool_id}: Out of memory, requested {size} bytes")
                    return None
            
            # Allocate from the best-fit block
            offset, block_size = self.free_blocks.pop(best_fit_idx)
            
            # If block is larger than needed, split it
            if block_size > size:
                # Add remainder back to free blocks
                self.free_blocks.append((offset + size, block_size - size))
                # Keep free blocks sorted by offset for better coalescing
                self.free_blocks.sort(key=lambda x: x[0])
            
            # Create block metadata
            block_id = f"{self.pool_id}_{uuid.uuid4().hex[:8]}"
            block = MemoryBlock(
                block_id=block_id,
                offset=offset,
                size=size,
                allocated=True,
                owner_agent=agent_id,
                access_count=0,
                last_access=time.time(),
                compression=compression,
                checksum=""
            )
            
            self.blocks[block_id] = block
            logger.debug(f"Allocated {size} bytes at offset {offset} for agent {agent_id}")
            
            return block_id
    
    def deallocate(self, block_id: str) -> bool:
        """Deallocate memory block"""
        with self.allocation_lock:
            if block_id not in self.blocks:
                logger.warning(f"Block {block_id} not found for deallocation")
                return False
            
            block = self.blocks[block_id]
            if not block.allocated:
                logger.warning(f"Block {block_id} already deallocated")
                return False
            
            # Mark as deallocated
            block.allocated = False
            
            # Add to free blocks
            self.free_blocks.append((block.offset, block.size))
            
            # Coalesce adjacent free blocks
            self._coalesce_free_blocks()
            
            # Remove from blocks
            del self.blocks[block_id]
            
            logger.debug(f"Deallocated block {block_id}, size {block.size}")
            return True
    
    def write(self, block_id: str, data: bytes, offset: int = 0) -> bool:
        """Write data to memory block"""
        with self.allocation_lock:
            if block_id not in self.blocks:
                logger.error(f"Block {block_id} not found for write")
                self.miss_count += 1
                return False
            
            block = self.blocks[block_id]
            if not block.allocated:
                logger.error(f"Block {block_id} not allocated")
                return False
            
            if offset + len(data) > block.size:
                logger.error(f"Write would exceed block {block_id} size")
                return False
            
            try:
                # Compress data if needed
                if block.compression != CompressionType.NONE:
                    data = MemoryCompressor.compress(data, block.compression)
                
                # Write to memory map
                self.memory_map.seek(block.offset + offset)
                self.memory_map.write(data)
                
                # Update block metadata
                block.last_access = time.time()
                block.access_count += 1
                block.checksum = hashlib.md5(data).hexdigest()
                
                self.access_count += 1
                self.hit_count += 1
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to write to block {block_id}: {e}")
                return False
    
    def read(self, block_id: str, size: Optional[int] = None, offset: int = 0) -> Optional[bytes]:
        """Read data from memory block"""
        with self.allocation_lock:
            if block_id not in self.blocks:
                logger.error(f"Block {block_id} not found for read")
                self.miss_count += 1
                return None
            
            block = self.blocks[block_id]
            if not block.allocated:
                logger.error(f"Block {block_id} not allocated")
                return None
            
            read_size = size if size is not None else (block.size - offset)
            
            if offset + read_size > block.size:
                logger.error(f"Read would exceed block {block_id} size")
                return None
            
            try:
                # Read from memory map
                self.memory_map.seek(block.offset + offset)
                data = self.memory_map.read(read_size)
                
                # Decompress if needed
                if block.compression != CompressionType.NONE:
                    data = MemoryCompressor.decompress(data, block.compression)
                
                # Update block metadata
                block.last_access = time.time()
                block.access_count += 1
                
                self.access_count += 1
                self.hit_count += 1
                
                return data
                
            except Exception as e:
                logger.error(f"Failed to read from block {block_id}: {e}")
                return None
    
    def _coalesce_free_blocks(self):
        """Coalesce adjacent free blocks to prevent fragmentation"""
        if len(self.free_blocks) <= 1:
            return
        
        # Sort by offset
        self.free_blocks.sort(key=lambda x: x[0])
        
        coalesced = []
        current_offset, current_size = self.free_blocks[0]
        
        for offset, size in self.free_blocks[1:]:
            if current_offset + current_size == offset:
                # Adjacent blocks, coalesce
                current_size += size
            else:
                # Non-adjacent, add current and start new
                coalesced.append((current_offset, current_size))
                current_offset, current_size = offset, size
        
        # Add the last block
        coalesced.append((current_offset, current_size))
        
        if len(coalesced) < len(self.free_blocks):
            logger.debug(f"Coalesced {len(self.free_blocks)} free blocks into {len(coalesced)}")
        
        self.free_blocks = coalesced
    
    def _garbage_collect(self):
        """Garbage collect unused blocks"""
        current_time = time.time()
        gc_threshold = 300  # 5 minutes
        
        blocks_to_remove = []
        for block_id, block in self.blocks.items():
            if (block.allocated and 
                block.access_count == 0 and 
                current_time - block.last_access > gc_threshold):
                blocks_to_remove.append(block_id)
        
        for block_id in blocks_to_remove:
            logger.debug(f"Garbage collecting unused block {block_id}")
            self.deallocate(block_id)
    
    def get_stats(self) -> PoolStats:
        """Get pool statistics"""
        with self.allocation_lock:
            allocated_size = sum(block.size for block in self.blocks.values() if block.allocated)
            free_size = sum(size for _, size in self.free_blocks)
            
            # Calculate fragmentation ratio
            largest_free = max((size for _, size in self.free_blocks), default=0)
            fragmentation_ratio = 1.0 - (largest_free / max(free_size, 1))
            
            # Calculate cache hit ratio
            total_accesses = self.hit_count + self.miss_count
            cache_hit_ratio = self.hit_count / max(total_accesses, 1)
            
            # Average block size
            allocated_blocks = [b for b in self.blocks.values() if b.allocated]
            avg_block_size = sum(b.size for b in allocated_blocks) / max(len(allocated_blocks), 1)
            
            return PoolStats(
                total_size=self.size,
                allocated_size=allocated_size,
                free_size=free_size,
                fragmentation_ratio=fragmentation_ratio,
                allocation_count=len(allocated_blocks),
                deallocation_count=0,  # Would need to track this separately
                compression_ratio=0.0,  # Would need to calculate based on compressed blocks
                cache_hit_ratio=cache_hit_ratio,
                average_block_size=avg_block_size
            )
    
    def cleanup(self):
        """Clean up resources"""
        if self.memory_map:
            self.memory_map.close()
        
        if self.mmap_file and os.path.exists(self.mmap_file):
            try:
                os.unlink(self.mmap_file)
            except OSError:
                pass

class MemoryPoolManager:
    """Main memory pool manager"""
    
    def __init__(self, total_memory_mb: int = 4096):
        self.total_memory = total_memory_mb * 1024 * 1024
        self.pools = {}  # pool_id -> MemoryPool
        self.agent_allocations = defaultdict(list)  # agent_id -> [block_ids]
        self.deduplicator = MemoryDeduplicator()
        self.pool_lock = threading.RLock()
        self.stats_history = deque(maxlen=1000)
        self.monitoring_thread = None
        self.running = False
        
        # Create default pools
        self._create_default_pools()
        
        logger.info(f"Memory Pool Manager initialized with {total_memory_mb}MB total memory")
    
    def _create_default_pools(self):
        """Create default memory pools"""
        # Model weights pool (largest, for sharing model parameters)
        model_pool_size = int(self.total_memory * 0.4)  # 40%
        self.create_pool("model_weights", model_pool_size, PoolType.MODEL_WEIGHTS)
        
        # Data buffer pool (for input/output data)
        data_pool_size = int(self.total_memory * 0.3)  # 30%
        self.create_pool("data_buffer", data_pool_size, PoolType.DATA_BUFFER)
        
        # Communication pool (for inter-agent communication)
        comm_pool_size = int(self.total_memory * 0.15)  # 15%
        self.create_pool("communication", comm_pool_size, PoolType.COMMUNICATION)
        
        # Cache pool (for frequently accessed data)
        cache_pool_size = int(self.total_memory * 0.1)  # 10%
        self.create_pool("cache", cache_pool_size, PoolType.CACHE)
        
        # Temporary pool (for temporary allocations)
        temp_pool_size = int(self.total_memory * 0.05)  # 5%
        self.create_pool("temporary", temp_pool_size, PoolType.TEMPORARY)
    
    def create_pool(self, pool_id: str, size: int, pool_type: PoolType) -> bool:
        """Create a new memory pool"""
        with self.pool_lock:
            if pool_id in self.pools:
                logger.warning(f"Pool {pool_id} already exists")
                return False
            
            try:
                pool = MemoryPool(pool_id, size, pool_type)
                self.pools[pool_id] = pool
                logger.info(f"Created pool {pool_id} of type {pool_type.value}, size {size} bytes")
                return True
                
            except Exception as e:
                logger.error(f"Failed to create pool {pool_id}: {e}")
                return False
    
    def allocate_memory(self, agent_id: str, size: int, pool_id: str = "data_buffer", 
                       compression: CompressionType = CompressionType.NONE) -> Optional[str]:
        """Allocate memory for an agent"""
        with self.pool_lock:
            if pool_id not in self.pools:
                logger.error(f"Pool {pool_id} not found")
                return None
            
            pool = self.pools[pool_id]
            block_id = pool.allocate(size, agent_id, compression)
            
            if block_id:
                self.agent_allocations[agent_id].append(block_id)
                logger.debug(f"Allocated {size} bytes for agent {agent_id} in pool {pool_id}")
            
            return block_id
    
    def deallocate_memory(self, block_id: str) -> bool:
        """Deallocate memory block"""
        with self.pool_lock:
            # Find which pool contains this block
            for pool in self.pools.values():
                if block_id in pool.blocks:
                    # Remove from agent allocations
                    for agent_id, blocks in self.agent_allocations.items():
                        if block_id in blocks:
                            blocks.remove(block_id)
                            break
                    
                    return pool.deallocate(block_id)
            
            logger.warning(f"Block {block_id} not found in any pool")
            return False
    
    def write_data(self, block_id: str, data: bytes, offset: int = 0, 
                   enable_dedup: bool = True) -> bool:
        """Write data to memory block with optional deduplication"""
        # Check for deduplication if enabled
        if enable_dedup:
            existing_block = self.deduplicator.add_block(block_id, data)
            if existing_block:
                # Data already exists, create reference instead
                logger.debug(f"Deduplicated write: {block_id} -> {existing_block}")
                return True
        
        # Find which pool contains this block
        for pool in self.pools.values():
            if block_id in pool.blocks:
                return pool.write(block_id, data, offset)
        
        logger.error(f"Block {block_id} not found for write")
        return False
    
    def read_data(self, block_id: str, size: Optional[int] = None, offset: int = 0) -> Optional[bytes]:
        """Read data from memory block"""
        # Find which pool contains this block
        for pool in self.pools.values():
            if block_id in pool.blocks:
                return pool.read(block_id, size, offset)
        
        logger.error(f"Block {block_id} not found for read")
        return None
    
    def share_memory(self, source_agent: str, target_agent: str, block_id: str) -> bool:
        """Share memory block between agents"""
        with self.pool_lock:
            # Find the block
            for pool in self.pools.values():
                if block_id in pool.blocks:
                    block = pool.blocks[block_id]
                    
                    # Add to target agent's allocations
                    if block_id not in self.agent_allocations[target_agent]:
                        self.agent_allocations[target_agent].append(block_id)
                        block.ref_count += 1
                        logger.info(f"Shared block {block_id} from {source_agent} to {target_agent}")
                        return True
                    else:
                        logger.warning(f"Block {block_id} already shared with {target_agent}")
                        return True
            
            logger.error(f"Block {block_id} not found for sharing")
            return False
    
    def cleanup_agent_memory(self, agent_id: str) -> int:
        """Clean up all memory allocated to an agent"""
        with self.pool_lock:
            blocks = self.agent_allocations.get(agent_id, [])
            cleaned_count = 0
            
            for block_id in blocks[:]:  # Copy list to avoid modification during iteration
                if self.deallocate_memory(block_id):
                    cleaned_count += 1
            
            # Clear agent allocations
            if agent_id in self.agent_allocations:
                del self.agent_allocations[agent_id]
            
            logger.info(f"Cleaned up {cleaned_count} memory blocks for agent {agent_id}")
            return cleaned_count
    
    def start_monitoring(self):
        """Start monitoring thread"""
        if self.monitoring_thread is not None:
            return
        
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Memory pool monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring thread"""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
            self.monitoring_thread = None
        logger.info("Memory pool monitoring stopped")
    
    def _monitoring_loop(self):
        """Monitoring loop"""
        while self.running:
            try:
                # Collect stats from all pools
                stats = {}
                for pool_id, pool in self.pools.items():
                    stats[pool_id] = asdict(pool.get_stats())
                
                # Add deduplication stats
                stats["deduplication"] = self.deduplicator.get_stats()
                
                # Add agent allocation stats
                stats["agents"] = {
                    "total_agents": len(self.agent_allocations),
                    "total_allocations": sum(len(blocks) for blocks in self.agent_allocations.values())
                }
                
                # Add timestamp
                stats["timestamp"] = time.time()
                
                # Store in history
                self.stats_history.append(stats)
                
                # Log summary every 5 minutes
                if len(self.stats_history) % 300 == 0:
                    self._log_summary_stats(stats)
                
                time.sleep(1)  # Update every second
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(5)
    
    def _log_summary_stats(self, stats: Dict[str, Any]):
        """Log summary statistics"""
        total_allocated = sum(s["allocated_size"] for s in stats.values() if isinstance(s, dict) and "allocated_size" in s)
        total_free = sum(s["free_size"] for s in stats.values() if isinstance(s, dict) and "free_size" in s)
        
        logger.info(f"Memory Summary: Allocated={total_allocated/1024/1024:.1f}MB, "
                   f"Free={total_free/1024/1024:.1f}MB, "
                   f"Agents={stats['agents']['total_agents']}, "
                   f"Dedup_Ratio={stats['deduplication']['deduplication_ratio']:.3f}")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        with self.pool_lock:
            stats = {}
            
            # Pool stats
            for pool_id, pool in self.pools.items():
                stats[pool_id] = asdict(pool.get_stats())
            
            # Deduplication stats
            stats["deduplication"] = self.deduplicator.get_stats()
            
            # Agent stats
            agent_stats = {}
            for agent_id, blocks in self.agent_allocations.items():
                agent_stats[agent_id] = {
                    "block_count": len(blocks),
                    "total_size": sum(
                        pool.blocks[block_id].size 
                        for pool in self.pools.values() 
                        for block_id in blocks 
                        if block_id in pool.blocks
                    )
                }
            
            stats["agents"] = agent_stats
            
            # System memory stats
            memory = psutil.virtual_memory()
            stats["system"] = {
                "total_memory": memory.total,
                "available_memory": memory.available,
                "used_memory": memory.used,
                "memory_percent": memory.percent
            }
            
            return stats
    
    def export_stats(self, filepath: str):
        """Export statistics to file"""
        stats = self.get_comprehensive_stats()
        stats["export_timestamp"] = time.time()
        stats["history"] = list(self.stats_history)
        
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info(f"Exported memory pool statistics to {filepath}")
    
    def cleanup(self):
        """Clean up all resources"""
        logger.info("Cleaning up memory pool manager...")
        
        # Stop monitoring
        self.stop_monitoring()
        
        # Clean up all pools
        with self.pool_lock:
            for pool in self.pools.values():
                pool.cleanup()
            
            self.pools.clear()
            self.agent_allocations.clear()
        
        logger.info("Memory pool manager cleanup complete")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="SutazAI Memory Pool Manager")
    parser.add_argument("--pool-size", type=int, default=4096,
                       help="Total memory pool size in MB")
    parser.add_argument("--monitor", action="store_true",
                       help="Start monitoring mode")
    parser.add_argument("--stats-export", type=str,
                       help="Export statistics to file")
    parser.add_argument("--test", action="store_true",
                       help="Run test allocations")
    
    args = parser.parse_args()
    
    # Create memory pool manager
    manager = MemoryPoolManager(args.pool_size)
    
    try:
        if args.monitor:
            # Start monitoring
            manager.start_monitoring()
            logger.info("Monitoring memory pools. Press Ctrl+C to stop.")
            
            try:
                while True:
                    time.sleep(60)
                    stats = manager.get_comprehensive_stats()
                    
                    # Print summary
                    total_allocated = sum(
                        s["allocated_size"] for s in stats.values() 
                        if isinstance(s, dict) and "allocated_size" in s
                    )
                    print(f"Total Allocated: {total_allocated/1024/1024:.1f}MB")
                    
            except KeyboardInterrupt:
                logger.info("Stopping monitoring...")
        
        elif args.test:
            # Run test allocations
            logger.info("Running test allocations...")
            
            # Test allocations
            test_agents = [f"agent_{i:03d}" for i in range(10)]
            allocations = []
            
            for agent_id in test_agents:
                # Allocate some memory
                block_id = manager.allocate_memory(agent_id, 1024 * 1024, "data_buffer")  # 1MB
                if block_id:
                    allocations.append((agent_id, block_id))
                    
                    # Write some test data
                    test_data = f"Test data for {agent_id}".encode() * 1000
                    manager.write_data(block_id, test_data)
            
            # Test sharing
            if len(allocations) >= 2:
                source_agent, source_block = allocations[0]
                target_agent = allocations[1][0]
                manager.share_memory(source_agent, target_agent, source_block)
            
            # Print stats
            stats = manager.get_comprehensive_stats()
            print(json.dumps(stats, indent=2, default=str))
            
            # Clean up test allocations
            for agent_id, _ in allocations:
                manager.cleanup_agent_memory(agent_id)
        
        if args.stats_export:
            manager.export_stats(args.stats_export)
            print(f"Statistics exported to {args.stats_export}")
    
    finally:
        manager.cleanup()

if __name__ == "__main__":
    main()