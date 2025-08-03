#!/usr/bin/env python3
"""
Test and Verification Script for Garbage Collection Enforcer

Purpose: Test the garbage collection enforcer with synthetic test data
Usage: python test-garbage-enforcer.py [options]
Requirements: Python 3.8+
"""

import asyncio
import json
import logging
import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add the agents directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import our enforcer
try:
    from garbage_collection_enforcer import GarbageCollectionEnforcer, GarbageType, RiskLevel
except ImportError as e:
    print(f"Error importing garbage collection enforcer: {e}")
    print("Make sure the enforcer script is in the same directory")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GarbageEnforcerTester:
    """Test suite for the Garbage Collection Enforcer"""
    
    def __init__(self):
        self.test_dir = None
        self.enforcer = None
        
    async def create_test_environment(self) -> Path:
        """Create a test environment with various types of garbage files"""
        self.test_dir = Path(tempfile.mkdtemp(prefix="garbage_test_"))
        logger.info(f"Created test environment: {self.test_dir}")
        
        # Create directory structure
        dirs = [
            "src", "tests", "build", "dist", "cache", ".git", "node_modules",
            "old_versions", "backups", "logs", "temp"
        ]
        
        for dir_name in dirs:
            (self.test_dir / dir_name).mkdir(exist_ok=True)
        
        # Create test files
        test_files = {
            # Temporary files
            "temp/temp_file.tmp": "temporary data",
            "temp/debug.temp": "debug information",
            "cache/app.cache": "cached data",
            "src/.file.swp": "vim swap file",
            
            # Backup files
            "src/main.py.bak": "# backup of main file\nprint('hello')",
            "config.json.backup": '{"test": true}',
            "script.sh~": "#!/bin/bash\necho test",
            "data.old": "old data format",
            
            # Build artifacts
            "build/app.o": "object file content",
            "dist/app.js": "compiled javascript",
            "__pycache__/module.pyc": "compiled python",
            "node_modules/package/index.js": "npm package",
            
            # Log files
            "logs/app.log": "log entry 1\nlog entry 2",
            "logs/debug.log": "debug info",
            "error.log": "error occurred",
            
            # Cache files
            ".DS_Store": "mac metadata",
            "Thumbs.db": "windows thumbnail cache",
            
            # Old versions
            "script_v1.py": "# old version\nprint('version 1')",
            "config_final.json": '{"final": "version"}',
            "app_new.js": "// new version\nconsole.log('new');",
            
            # Duplicate files (same content)
            "file1.txt": "duplicate content here",
            "file1_copy.txt": "duplicate content here",
            "copy_of_file1.txt": "duplicate content here",
            
            # Empty files
            "empty1.txt": "",
            "empty2.log": "",
            "empty.tmp": "",
            
            # Valid files (should not be removed)
            "src/main.py": "#!/usr/bin/env python3\nprint('Hello World')",
            "README.md": "# Test Project\nThis is a test",
            "package.json": '{"name": "test", "version": "1.0.0"}',
            "requirements.txt": "requests==2.25.1\nflask==2.0.1",
            
            # Files with references
            "src/utils.py": "def helper(): pass",
            "src/app.py": "from utils import helper\nhelper()",
        }
        
        # Create all test files
        for file_path, content in test_files.items():
            full_path = self.test_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            
            # Set different modification times for age testing
            if "old" in file_path or "temp" in file_path:
                # Make these files appear older
                old_time = datetime.now() - timedelta(days=30)
                os.utime(full_path, (old_time.timestamp(), old_time.timestamp()))
        
        logger.info(f"Created {len(test_files)} test files")
        return self.test_dir
    
    async def test_scanning(self):
        """Test the scanning functionality"""
        logger.info("Testing scanning functionality...")
        
        # Create enforcer instance
        self.enforcer = GarbageCollectionEnforcer(
            project_root=self.test_dir,
            dry_run=True,
            confidence_threshold=0.5,  # Lower threshold for testing
            risk_threshold=RiskLevel.MODERATE
        )
        
        # Run scan
        garbage_items = await self.enforcer.scan_project()
        
        # Analyze results
        logger.info(f"Found {len(garbage_items)} garbage items")
        
        # Check for expected garbage types
        found_types = {item.garbage_type for item in garbage_items}
        expected_types = {
            GarbageType.TEMP_FILE,
            GarbageType.BACKUP_FILE,
            GarbageType.BUILD_ARTIFACT,
            GarbageType.LOG_FILE,
            GarbageType.CACHE_FILE,
            GarbageType.OLD_VERSION,
            GarbageType.DUPLICATE_FILE,
            GarbageType.EMPTY_FILE
        }
        
        logger.info(f"Found garbage types: {sorted([t.value for t in found_types])}")
        
        # Verify we found some key items
        temp_files = [item for item in garbage_items if item.garbage_type == GarbageType.TEMP_FILE]
        backup_files = [item for item in garbage_items if item.garbage_type == GarbageType.BACKUP_FILE]
        duplicates = [item for item in garbage_items if item.garbage_type == GarbageType.DUPLICATE_FILE]
        
        assert len(temp_files) > 0, "Should find temporary files"
        assert len(backup_files) > 0, "Should find backup files"
        assert len(duplicates) > 0, "Should find duplicate files"
        
        logger.info("✅ Scanning test passed")
        return garbage_items
    
    async def test_risk_assessment(self, garbage_items):
        """Test risk assessment functionality"""
        logger.info("Testing risk assessment...")
        
        # Check that different risk levels are assigned
        risk_levels = {item.risk_level for item in garbage_items}
        logger.info(f"Found risk levels: {sorted([r.value for r in risk_levels])}")
        
        # Temporary files should generally be safe
        temp_files = [item for item in garbage_items if item.garbage_type == GarbageType.TEMP_FILE]
        safe_temp_files = [item for item in temp_files if item.risk_level == RiskLevel.SAFE]
        
        assert len(safe_temp_files) > 0, "Some temporary files should be marked as safe"
        
        logger.info("✅ Risk assessment test passed")
    
    async def test_reference_detection(self, garbage_items):
        """Test reference detection functionality"""
        logger.info("Testing reference detection...")
        
        # Find items with references
        items_with_refs = [item for item in garbage_items if item.references]
        items_without_refs = [item for item in garbage_items if not item.references]
        
        logger.info(f"Items with references: {len(items_with_refs)}")
        logger.info(f"Items without references: {len(items_without_refs)}")
        
        # Should have both types
        assert len(items_without_refs) > 0, "Should find items without references"
        
        logger.info("✅ Reference detection test passed")
    
    async def test_confidence_scoring(self, garbage_items):
        """Test confidence scoring"""
        logger.info("Testing confidence scoring...")
        
        # Check confidence ranges
        high_confidence = [item for item in garbage_items if item.confidence >= 0.8]
        medium_confidence = [item for item in garbage_items if 0.5 <= item.confidence < 0.8]
        low_confidence = [item for item in garbage_items if item.confidence < 0.5]
        
        logger.info(f"High confidence items: {len(high_confidence)}")
        logger.info(f"Medium confidence items: {len(medium_confidence)}")
        logger.info(f"Low confidence items: {len(low_confidence)}")
        
        # Should have items in different confidence ranges
        assert len(high_confidence) > 0, "Should have high confidence items"
        
        logger.info("✅ Confidence scoring test passed")
    
    async def test_dry_run_safety(self, garbage_items):
        """Test that dry run doesn't actually remove files"""
        logger.info("Testing dry run safety...")
        
        # Count files before
        files_before = list(self.test_dir.rglob("*"))
        files_before = [f for f in files_before if f.is_file()]
        count_before = len(files_before)
        
        # Run cleanup in dry run mode
        stats = await self.enforcer.cleanup_garbage(garbage_items)
        
        # Count files after
        files_after = list(self.test_dir.rglob("*"))
        files_after = [f for f in files_after if f.is_file()]
        count_after = len(files_after)
        
        # Should be the same
        assert count_before == count_after, f"File count changed: {count_before} -> {count_after}"
        
        logger.info(f"Files preserved in dry run: {count_after}")
        logger.info("✅ Dry run safety test passed")
    
    async def test_report_generation(self):
        """Test report generation"""
        logger.info("Testing report generation...")
        
        # Generate report
        report = await self.enforcer.generate_report()
        
        # Verify report structure
        required_sections = [
            "metadata", "configuration", "statistics", 
            "analysis", "findings", "recommendations", "audit_trail"
        ]
        
        for section in required_sections:
            assert section in report, f"Report missing section: {section}"
        
        # Verify some content
        assert report["metadata"]["rule"] == "Rule 13: No Garbage, No Rot"
        assert "total_garbage_items" in report["analysis"]
        assert "items_by_type" in report["analysis"]
        
        logger.info("✅ Report generation test passed")
        return report
    
    async def test_live_cleanup(self):
        """Test actual file removal (with a separate test directory)"""
        logger.info("Testing live cleanup...")
        
        # Create a separate test environment for live testing
        live_test_dir = Path(tempfile.mkdtemp(prefix="garbage_live_test_"))
        
        try:
            # Create some test garbage files
            garbage_files = {
                "temp.tmp": "temporary",
                "old.bak": "backup",
                "empty.txt": "",
                ".DS_Store": "cache"
            }
            
            for file_path, content in garbage_files.items():
                (live_test_dir / file_path).write_text(content)
            
            # Create live enforcer
            live_enforcer = GarbageCollectionEnforcer(
                project_root=live_test_dir,
                dry_run=False,  # Live mode
                confidence_threshold=0.5,
                risk_threshold=RiskLevel.SAFE  # Only safe items
            )
            
            # Scan and cleanup
            items = await live_enforcer.scan_project()
            stats = await live_enforcer.cleanup_garbage(items)
            
            # Verify some files were removed
            remaining_files = list(live_test_dir.rglob("*"))
            remaining_files = [f for f in remaining_files if f.is_file()]
            
            logger.info(f"Items removed: {stats.items_removed}")
            logger.info(f"Remaining files: {len(remaining_files)}")
            
            assert stats.items_removed > 0, "Should have removed some items"
            
            logger.info("✅ Live cleanup test passed")
            
        finally:
            # Cleanup live test directory
            shutil.rmtree(live_test_dir, ignore_errors=True)
    
    def cleanup_test_environment(self):
        """Clean up the test environment"""
        if self.test_dir and self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)
            logger.info(f"Cleaned up test environment: {self.test_dir}")
    
    async def run_all_tests(self):
        """Run all tests"""
        logger.info("🧹 Starting Garbage Collection Enforcer Tests")
        logger.info("=" * 60)
        
        try:
            # Create test environment
            await self.create_test_environment()
            
            # Run scanning tests
            garbage_items = await self.test_scanning()
            
            # Run analysis tests
            await self.test_risk_assessment(garbage_items)
            await self.test_reference_detection(garbage_items)
            await self.test_confidence_scoring(garbage_items)
            
            # Run safety tests
            await self.test_dry_run_safety(garbage_items)
            
            # Run report test
            report = await self.test_report_generation()
            
            # Run live cleanup test
            await self.test_live_cleanup()
            
            logger.info("=" * 60)
            logger.info("🎉 All tests passed successfully! 🎉")
            logger.info("=" * 60)
            
            # Print summary
            print("\n" + "="*60)
            print("🧹 GARBAGE COLLECTION ENFORCER - TEST RESULTS")
            print("="*60)
            print(f"✅ All tests passed")
            print(f"📊 Garbage items found: {len(garbage_items)}")
            print(f"📁 Test directory: {self.test_dir}")
            print(f"📋 Report sections: {len(report)}")
            print("="*60)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            raise
        finally:
            self.cleanup_test_environment()

async def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the Garbage Collection Enforcer")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--keep-test-dir", action="store_true", help="Keep test directory after tests")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run tests
    tester = GarbageEnforcerTester()
    
    try:
        success = await tester.run_all_tests()
        return 0 if success else 1
    except Exception as e:
        logger.error(f"Tests failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))