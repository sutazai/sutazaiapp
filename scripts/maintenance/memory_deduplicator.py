#!/usr/bin/env python3
"""
Memory Bank Deduplication Script
Removes duplicate entries from activeContext.md
"""

import hashlib
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set

# Configuration
MEMORY_BANK_DIR = Path("/opt/sutazaiapp/memory-bank")
ACTIVE_CONTEXT_FILE = MEMORY_BANK_DIR / "activeContext.md"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)8s | %(message)s'
)
logger = logging.getLogger(__name__)


def deduplicate_file():
    """Remove duplicate entries from activeContext.md"""
    
    if not ACTIVE_CONTEXT_FILE.exists():
        logger.error(f"File not found: {ACTIVE_CONTEXT_FILE}")
        return
    
    logger.info(f"Starting deduplication of {ACTIVE_CONTEXT_FILE}")
    
    # Read the file
    with open(ACTIVE_CONTEXT_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_size = len(content)
    logger.info(f"Original size: {original_size / 1024 / 1024:.2f}MB")
    
    # Split into sections
    sections = re.split(r'(## Code Changes \(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\))', content)
    
    # Process sections and remove duplicates
    seen_hashes = set()
    unique_sections = []
    duplicates_removed = 0
    
    # Keep header
    if sections and not sections[0].startswith('## Code Changes'):
        unique_sections.append(sections[0])
        sections = sections[1:]
    
    # Process pairs of header + content
    for i in range(0, len(sections), 2):
        if i + 1 < len(sections):
            header = sections[i]
            body = sections[i + 1] if i + 1 < len(sections) else ""
            
            # Create hash of the content (not including timestamp)
            content_hash = hashlib.sha256(body.encode('utf-8')).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_sections.append(header)
                unique_sections.append(body)
            else:
                duplicates_removed += 1
                logger.debug(f"Removing duplicate entry: {header[:50]}")
    
    # Rebuild content
    new_content = ''.join(unique_sections)
    
    # If still too large, keep only the most recent entry
    if len(new_content) > 1024 * 1024:  # > 1MB
        logger.info("File still too large after deduplication, keeping only most recent entry")
        
        # Find the last complete entry
        last_entry_match = list(re.finditer(r'## Code Changes \(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\)', new_content))
        
        if last_entry_match:
            # Keep header and last entry only
            last_entry_start = last_entry_match[-1].start()
            new_content = "# Active Context\n\n" + new_content[last_entry_start:]
    
    # Write the deduplicated content
    with open(ACTIVE_CONTEXT_FILE, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    new_size = len(new_content)
    logger.info(f"Deduplication complete!")
    logger.info(f"Original size: {original_size / 1024 / 1024:.2f}MB")
    logger.info(f"New size: {new_size / 1024 / 1024:.2f}MB")
    logger.info(f"Space saved: {(original_size - new_size) / 1024 / 1024:.2f}MB")
    logger.info(f"Duplicates removed: {duplicates_removed}")
    

if __name__ == "__main__":
    deduplicate_file()