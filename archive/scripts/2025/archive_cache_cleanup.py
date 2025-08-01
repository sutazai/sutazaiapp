#!/usr/bin/env python3
"""
Standalone script to clean application caches.
"""

import sys
import os
import logging
import argparse

# Ensure the project root is in the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the cache manager
from utils.cache_manager import CacheManager

def main():
    """Parse arguments and run cache cleanup."""
    parser = argparse.ArgumentParser(description="SutazAI Cache Cleanup Tool")
    parser.add_argument(
        "--max-size", 
        type=int, 
        default=5000, 
        help="Maximum cache size in MB (default: 5000)"
    )
    parser.add_argument(
        "--ttl", 
        type=int, 
        default=30, 
        help="Cache time-to-live in days (default: 30)"
    )
    parser.add_argument(
        "--pycache-only", 
        action="store_true", 
        help="Clean only Python bytecode cache"
    )
    parser.add_argument(
        "--model-cache-only", 
        action="store_true", 
        help="Clean only model cache"
    )
    parser.add_argument(
        "--enforce-limits", 
        action="store_true", 
        help="Enforce cache size and TTL limits instead of cleaning"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    # Initialize cache manager
    manager = CacheManager(
        max_cache_size_mb=args.max_size,
        cache_ttl_days=args.ttl
    )
    
    # Run requested operation
    if args.enforce_limits:
        logging.info("Enforcing cache limits...")
        manager.enforce_cache_limits()
    elif args.pycache_only:
        logging.info("Cleaning Python bytecode cache only...")
        manager.clean_pycache()
    elif args.model_cache_only:
        logging.info("Cleaning model cache only...")
        manager.clean_model_cache()
    else:
        logging.info("Running full cache cleanup...")
        manager.clean_all_caches()
    
    # Report final cache size
    final_size = manager.get_cache_size()
    logging.info(f"Current cache size: {final_size:.2f} MB")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 