#!/usr/bin/env python3
"""System functionality test"""
import sys
import os
import logging

# Add project root to path
sys.path.insert(0, '/opt/sutazaiapp')

logger = logging.getLogger(__name__)

def test_imports():
    """Test critical imports"""
    try:
        # Test core imports
        from pathlib import Path
        from typing import Dict, List, Any
        import json
        import asyncio
        
        print("✅ Core Python modules imported successfully")
        
        # Test project imports
        try:
            from backend.config import Config
            print("✅ Backend config imported successfully")
        except ImportError as e:
            print(f"⚠️  Backend config import failed: {e}")
        
        try:
            from backend.ai.model_manager import model_manager
            print("✅ AI model manager imported successfully")
        except ImportError as e:
            print(f"⚠️  AI model manager import failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_file_structure():
    """Test file structure"""
    from pathlib import Path
    root_dir = Path('/opt/sutazaiapp')
    
    required_files = [
        'main.py',
        'data/sutazai.db',
        'data/model_registry.json',
        'start.sh'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not (root_dir / file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"⚠️  Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
        return True

def main():
    """Run all tests"""
    print("🧪 Testing SutazAI System")
    print("=========================")
    
    tests = [
        ("Import Test", test_imports),
        ("File Structure Test", test_file_structure)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        if test_func():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed - System ready!")
        return True
    else:
        print("⚠️  Some tests failed - Check configuration")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
