"""
Data Compression Utilities for SutazAI
Advanced compression algorithms for optimal storage
"""

import gzip
import lzma
import zlib
import bz2
import logging
import json
import pickle
from typing import Any, Union, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

class CompressionAlgorithm(str, Enum):
    GZIP = "gzip"
    LZMA = "lzma"
    ZLIB = "zlib"
    BZ2 = "bz2"

class DataCompressor:
    """Advanced data compression utility"""
    
    @staticmethod
    def compress_data(data: Union[str, bytes, dict, list], algorithm: CompressionAlgorithm = CompressionAlgorithm.GZIP, level: int = 6) -> Tuple[bytes, float]:
        """Compress data with specified algorithm"""
        try:
            # Convert data to bytes if needed
            if isinstance(data, (dict, list)):
                data_bytes = json.dumps(data).encode()
            elif isinstance(data, str):
                data_bytes = data.encode()
            else:
                data_bytes = data
            
            original_size = len(data_bytes)
            
            # Apply compression
            if algorithm == CompressionAlgorithm.GZIP:
                compressed = gzip.compress(data_bytes, compresslevel=level)
            elif algorithm == CompressionAlgorithm.LZMA:
                compressed = lzma.compress(data_bytes, preset=level)
            elif algorithm == CompressionAlgorithm.ZLIB:
                compressed = zlib.compress(data_bytes, level=level)
            elif algorithm == CompressionAlgorithm.BZ2:
                compressed = bz2.compress(data_bytes, compresslevel=level)
            else:
                raise ValueError(f"Unsupported compression algorithm: {algorithm}")
            
            compression_ratio = len(compressed) / original_size
            
            return compressed, compression_ratio
            
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            raise
    
    @staticmethod
    def decompress_data(compressed_data: bytes, algorithm: CompressionAlgorithm, data_type: str = "bytes") -> Any:
        """Decompress data"""
        try:
            # Decompress based on algorithm
            if algorithm == CompressionAlgorithm.GZIP:
                decompressed = gzip.decompress(compressed_data)
            elif algorithm == CompressionAlgorithm.LZMA:
                decompressed = lzma.decompress(compressed_data)
            elif algorithm == CompressionAlgorithm.ZLIB:
                decompressed = zlib.decompress(compressed_data)
            elif algorithm == CompressionAlgorithm.BZ2:
                decompressed = bz2.decompress(compressed_data)
            else:
                raise ValueError(f"Unsupported compression algorithm: {algorithm}")
            
            # Convert back to original data type
            if data_type == "json":
                return json.loads(decompressed.decode())
            elif data_type == "str":
                return decompressed.decode()
            else:
                return decompressed
                
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            raise
    
    @staticmethod
    def find_best_compression(data: Union[str, bytes], algorithms: list = None) -> Tuple[CompressionAlgorithm, bytes, float]:
        """Find best compression algorithm for given data"""
        if algorithms is None:
            algorithms = list(CompressionAlgorithm)
        
        best_algorithm = None
        best_compressed = None
        best_ratio = float('inf')
        
        for algorithm in algorithms:
            try:
                compressed, ratio = DataCompressor.compress_data(data, algorithm)
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_compressed = compressed
                    best_algorithm = algorithm
            except Exception as e:
                logger.warning(f"Algorithm {algorithm} failed: {e}")
        
        return best_algorithm, best_compressed, best_ratio

# Global compressor instance
data_compressor = DataCompressor()
