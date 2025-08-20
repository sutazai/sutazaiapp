#!/usr/bin/env python3
"""
Memory Bank Cleanup and Rotation System
Veteran-grade cleanup for activeContext.md bloat prevention

This script handles:
1. Archiving old context entries with compression
2. Rotating activeContext.md to prevent bloat
3. Extracting and preserving valuable data
4. Monitoring file size and auto-rotating
5. Creating forensic backups before cleanup
"""

import os
import sys
import gzip
import json
import shutil
import hashlib
import argparse
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import re

# Configuration
MEMORY_BANK_DIR = Path("/opt/sutazaiapp/memory-bank")
ARCHIVE_DIR = MEMORY_BANK_DIR / "archives"
ACTIVE_CONTEXT_FILE = MEMORY_BANK_DIR / "activeContext.md"
MAX_FILE_SIZE_MB = 1  # Maximum size before rotation
MAX_AGE_DAYS = 7  # Maximum age of entries to keep in active file
COMPRESSION_LEVEL = 9  # Maximum compression for archives

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)8s | %(funcName)s:%(lineno)d | %(message)s',
    handlers=[
        logging.FileHandler(f"/var/log/memory_cleanup_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ContextEntry:
    """Represents a single context entry in activeContext.md"""
    timestamp: datetime
    title: str
    content: str
    line_start: int
    line_end: int
    size_bytes: int


class MemoryBankCleaner:
    """
    Production-grade memory bank cleanup system
    Built with 20 years of experience handling large-scale data cleanup
    """
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.stats = {
            "original_size": 0,
            "final_size": 0,
            "entries_archived": 0,
            "entries_kept": 0,
            "compression_ratio": 0.0
        }
        
    def analyze_file(self) -> List[ContextEntry]:
        """Analyze activeContext.md and extract all entries"""
        if not ACTIVE_CONTEXT_FILE.exists():
            logger.error(f"File not found: {ACTIVE_CONTEXT_FILE}")
            return []
            
        entries = []
        current_entry = None
        current_content = []
        line_num = 0
        
        logger.info(f"Analyzing {ACTIVE_CONTEXT_FILE}")
        
        # Pattern to match section headers like "## Code Changes (2025-08-20 17:50:09)"
        section_pattern = re.compile(r'^## (.+?) \((\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\)$')
        
        with open(ACTIVE_CONTEXT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line_num += 1
                
                match = section_pattern.match(line.strip())
                if match:
                    # Save previous entry if exists
                    if current_entry:
                        current_entry.content = '\n'.join(current_content)
                        current_entry.line_end = line_num - 1
                        current_entry.size_bytes = len(current_entry.content.encode('utf-8'))
                        entries.append(current_entry)
                    
                    # Start new entry
                    title = match.group(1)
                    timestamp_str = match.group(2)
                    try:
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                        timestamp = timestamp.replace(tzinfo=timezone.utc)
                    except ValueError:
                        timestamp = datetime.now(timezone.utc)
                    
                    current_entry = ContextEntry(
                        timestamp=timestamp,
                        title=title,
                        content='',
                        line_start=line_num,
                        line_end=line_num,
                        size_bytes=0
                    )
                    current_content = [line.rstrip()]
                elif current_entry:
                    current_content.append(line.rstrip())
            
            # Don't forget the last entry
            if current_entry:
                current_entry.content = '\n'.join(current_content)
                current_entry.line_end = line_num
                current_entry.size_bytes = len(current_entry.content.encode('utf-8'))
                entries.append(current_entry)
        
        logger.info(f"Found {len(entries)} entries in file")
        return entries
    
    def create_backup(self) -> Optional[Path]:
        """Create a compressed backup of the current file"""
        if not ACTIVE_CONTEXT_FILE.exists():
            return None
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = ARCHIVE_DIR / f"activeContext_backup_{timestamp}.md.gz"
        
        logger.info(f"Creating backup: {backup_path}")
        
        if not self.dry_run:
            ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
            
            # Calculate checksum before compression
            with open(ACTIVE_CONTEXT_FILE, 'rb') as f:
                checksum = hashlib.sha256(f.read()).hexdigest()
            
            # Compress the file
            with open(ACTIVE_CONTEXT_FILE, 'rb') as f_in:
                with gzip.open(backup_path, 'wb', compresslevel=COMPRESSION_LEVEL) as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Save checksum for verification
            checksum_file = backup_path.with_suffix('.md.gz.sha256')
            checksum_file.write_text(f"{checksum}  {backup_path.name}\n")
            
            # Verify backup
            backup_size = backup_path.stat().st_size
            original_size = ACTIVE_CONTEXT_FILE.stat().st_size
            compression_ratio = (1 - backup_size / original_size) * 100
            
            logger.info(f"Backup created: {backup_size / 1024 / 1024:.2f}MB "
                       f"(compression: {compression_ratio:.1f}%)")
            
            self.stats["compression_ratio"] = compression_ratio
            
        return backup_path
    
    def archive_old_entries(self, entries: List[ContextEntry], cutoff_date: datetime) -> Tuple[List[ContextEntry], List[ContextEntry]]:
        """Separate entries into keep and archive based on age"""
        keep = []
        archive = []
        
        for entry in entries:
            if entry.timestamp < cutoff_date:
                archive.append(entry)
            else:
                keep.append(entry)
        
        logger.info(f"Keeping {len(keep)} recent entries, archiving {len(archive)} old entries")
        
        self.stats["entries_kept"] = len(keep)
        self.stats["entries_archived"] = len(archive)
        
        return keep, archive
    
    def write_archive(self, entries: List[ContextEntry], archive_name: str) -> Optional[Path]:
        """Write archived entries to a compressed file"""
        if not entries:
            return None
            
        archive_path = ARCHIVE_DIR / f"{archive_name}.md.gz"
        
        logger.info(f"Writing {len(entries)} entries to {archive_path}")
        
        if not self.dry_run:
            ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
            
            # Combine entries into markdown
            content = "# Archived Context\n\n"
            for entry in entries:
                content += entry.content + "\n\n"
            
            # Write compressed archive
            with gzip.open(archive_path, 'wt', encoding='utf-8', compresslevel=COMPRESSION_LEVEL) as f:
                f.write(content)
            
            # Create index file for quick searching
            index_data = {
                "archive_date": datetime.now(timezone.utc).isoformat(),
                "entry_count": len(entries),
                "date_range": {
                    "start": min(e.timestamp for e in entries).isoformat(),
                    "end": max(e.timestamp for e in entries).isoformat()
                },
                "titles": [e.title for e in entries],
                "total_size": sum(e.size_bytes for e in entries)
            }
            
            index_path = archive_path.with_suffix('.md.gz.index.json')
            with open(index_path, 'w') as f:
                json.dump(index_data, f, indent=2)
            
            logger.info(f"Archive created: {archive_path}")
        
        return archive_path
    
    def write_cleaned_file(self, entries: List[ContextEntry]) -> None:
        """Write the cleaned entries back to activeContext.md"""
        if not entries:
            content = "# Active Context\n\n*No active entries*\n"
        else:
            content = "# Active Context\n\n"
            for entry in sorted(entries, key=lambda e: e.timestamp, reverse=True):
                content += entry.content + "\n\n"
        
        if not self.dry_run:
            # Write atomically using temp file
            temp_file = ACTIVE_CONTEXT_FILE.with_suffix('.tmp')
            temp_file.write_text(content, encoding='utf-8')
            temp_file.replace(ACTIVE_CONTEXT_FILE)
            
            new_size = ACTIVE_CONTEXT_FILE.stat().st_size
            logger.info(f"New file size: {new_size / 1024 / 1024:.2f}MB")
            self.stats["final_size"] = new_size
        else:
            logger.info(f"[DRY RUN] Would write {len(content)} bytes")
    
    def cleanup(self, max_age_days: int = MAX_AGE_DAYS) -> Dict:
        """Main cleanup process"""
        logger.info("=" * 60)
        logger.info("Starting Memory Bank Cleanup")
        logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE'}")
        logger.info("=" * 60)
        
        # Record original size
        if ACTIVE_CONTEXT_FILE.exists():
            self.stats["original_size"] = ACTIVE_CONTEXT_FILE.stat().st_size
            logger.info(f"Original file size: {self.stats['original_size'] / 1024 / 1024:.2f}MB")
        
        # Step 1: Create backup
        backup_path = self.create_backup()
        if backup_path:
            logger.info(f"Backup created: {backup_path}")
        
        # Step 2: Analyze file
        entries = self.analyze_file()
        if not entries:
            logger.warning("No entries found to process")
            return self.stats
        
        # Step 3: Determine cutoff date
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        logger.info(f"Cutoff date: {cutoff_date.isoformat()}")
        
        # Step 4: Separate entries
        keep_entries, archive_entries = self.archive_old_entries(entries, cutoff_date)
        
        # Step 5: Archive old entries
        if archive_entries:
            archive_name = f"archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.write_archive(archive_entries, archive_name)
        
        # Step 6: Write cleaned file
        self.write_cleaned_file(keep_entries)
        
        # Step 7: Report results
        logger.info("=" * 60)
        logger.info("Cleanup Complete!")
        logger.info(f"Original size: {self.stats['original_size'] / 1024 / 1024:.2f}MB")
        logger.info(f"Final size: {self.stats['final_size'] / 1024 / 1024:.2f}MB")
        logger.info(f"Space saved: {(self.stats['original_size'] - self.stats['final_size']) / 1024 / 1024:.2f}MB")
        logger.info(f"Entries kept: {self.stats['entries_kept']}")
        logger.info(f"Entries archived: {self.stats['entries_archived']}")
        if self.stats['compression_ratio'] > 0:
            logger.info(f"Backup compression: {self.stats['compression_ratio']:.1f}%")
        logger.info("=" * 60)
        
        return self.stats


class MemoryMonitor:
    """Monitor memory bank file size and trigger cleanup when needed"""
    
    @staticmethod
    def check_size() -> bool:
        """Check if cleanup is needed"""
        if not ACTIVE_CONTEXT_FILE.exists():
            return False
            
        size_mb = ACTIVE_CONTEXT_FILE.stat().st_size / 1024 / 1024
        logger.info(f"Current file size: {size_mb:.2f}MB (threshold: {MAX_FILE_SIZE_MB}MB)")
        
        return size_mb > MAX_FILE_SIZE_MB
    
    @staticmethod
    def auto_cleanup():
        """Automatically cleanup if needed"""
        if MemoryMonitor.check_size():
            logger.info("File size exceeded threshold, initiating cleanup...")
            cleaner = MemoryBankCleaner(dry_run=False)
            return cleaner.cleanup()
        else:
            logger.info("File size within limits, no cleanup needed")
            return None


def search_archives(search_term: str) -> List[Dict]:
    """Search through archived files for specific content"""
    results = []
    
    if not ARCHIVE_DIR.exists():
        return results
    
    for index_file in ARCHIVE_DIR.glob("*.index.json"):
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        
        # Search in titles
        for title in index_data.get('titles', []):
            if search_term.lower() in title.lower():
                results.append({
                    'archive': index_file.stem.replace('.md.gz.index', ''),
                    'type': 'title',
                    'match': title,
                    'date_range': index_data['date_range']
                })
    
    return results


def main():
    """Main entry point with CLI interface"""
    parser = argparse.ArgumentParser(
        description='Memory Bank Cleanup and Rotation System',
        epilog='Example: python memory_cleanup.py --cleanup --max-age 7'
    )
    
    parser.add_argument('--cleanup', action='store_true',
                       help='Perform cleanup operation')
    parser.add_argument('--monitor', action='store_true',
                       help='Check size and cleanup if needed')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze file without making changes')
    parser.add_argument('--search', type=str,
                       help='Search archives for specific content')
    parser.add_argument('--dry-run', action='store_true',
                       help='Perform dry run without making changes')
    parser.add_argument('--max-age', type=int, default=MAX_AGE_DAYS,
                       help=f'Maximum age of entries to keep (default: {MAX_AGE_DAYS} days)')
    parser.add_argument('--force', action='store_true',
                       help='Force cleanup even if size is within limits')
    
    args = parser.parse_args()
    
    if args.search:
        results = search_archives(args.search)
        if results:
            print(f"\nFound {len(results)} matches:")
            for result in results:
                print(f"  - Archive: {result['archive']}")
                print(f"    Match: {result['match']}")
                print(f"    Date range: {result['date_range']['start']} to {result['date_range']['end']}")
        else:
            print(f"No matches found for '{args.search}'")
    
    elif args.analyze:
        cleaner = MemoryBankCleaner(dry_run=True)
        entries = cleaner.analyze_file()
        if entries:
            print(f"\nFile Analysis:")
            print(f"  Total entries: {len(entries)}")
            print(f"  Date range: {min(e.timestamp for e in entries)} to {max(e.timestamp for e in entries)}")
            print(f"  Total size: {sum(e.size_bytes for e in entries) / 1024 / 1024:.2f}MB")
            
            # Show entry distribution
            from collections import Counter
            titles = Counter(e.title for e in entries)
            print(f"\n  Entry types:")
            for title, count in titles.most_common(10):
                print(f"    - {title}: {count}")
    
    elif args.monitor:
        result = MemoryMonitor.auto_cleanup()
        if result:
            print(f"\nCleanup performed:")
            print(f"  Space saved: {(result['original_size'] - result['final_size']) / 1024 / 1024:.2f}MB")
    
    elif args.cleanup or args.force:
        if args.force or MemoryMonitor.check_size():
            cleaner = MemoryBankCleaner(dry_run=args.dry_run)
            result = cleaner.cleanup(max_age_days=args.max_age)
            
            if not args.dry_run:
                print(f"\nCleanup complete!")
                print(f"  Space saved: {(result['original_size'] - result['final_size']) / 1024 / 1024:.2f}MB")
        else:
            print("File size within limits. Use --force to cleanup anyway.")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()