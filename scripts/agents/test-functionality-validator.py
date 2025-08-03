#!/usr/bin/env python3
"""
Test script for the Functionality Preservation Validator

Purpose: Validates that the functionality preservation validator works correctly
Usage: python test-functionality-validator.py
Requirements: The functionality-preservation-validator.py script
"""

import os
import sys
import tempfile
import shutil
import subprocess
import json
from pathlib import Path

def create_test_repo():
    """Create a temporary git repository for testing."""
    test_dir = tempfile.mkdtemp(prefix="func_validator_test_")
    
    # Initialize git repo
    subprocess.run(["git", "init"], cwd=test_dir, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=test_dir, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=test_dir, capture_output=True)
    
    # Create initial Python file
    test_file = os.path.join(test_dir, "test_module.py")
    with open(test_file, 'w') as f:
        f.write('''
def original_function(param1, param2="default"):
    """Original function for testing."""
    return param1 + param2

class OriginalClass:
    """Original class for testing."""
    
    def method1(self):
        return "method1"
    
    def method2(self, arg):
        return f"method2: {arg}"

# API endpoint example
from flask import Flask
app = Flask(__name__)

@app.route('/api/test', methods=['GET'])
def api_test():
    return {"status": "ok"}
''')
    
    # Create test file
    test_test_file = os.path.join(test_dir, "test_test_module.py")
    with open(test_test_file, 'w') as f:
        f.write('''
import pytest
from test_module import original_function, OriginalClass

def test_original_function():
    assert original_function("hello", "world") == "helloworld"
    assert original_function("test") == "testdefault"

def test_original_class():
    obj = OriginalClass()
    assert obj.method1() == "method1"
    assert obj.method2("test") == "method2: test"
''')
    
    # Add and commit files
    subprocess.run(["git", "add", "."], cwd=test_dir, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=test_dir, capture_output=True)
    
    return test_dir

def test_no_changes(validator_path, test_repo):
    """Test validator with no changes."""
    print("🧪 Testing: No changes scenario")
    
    cmd = ["python", validator_path, "validate", "--format", "json", "--repo-path", test_repo]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        report = json.loads(result.stdout)
        if report["summary"]["total_checks"] >= 0:
            print("✅ No changes test passed")
            return True
    
    print(f"❌ No changes test failed: {result.stderr}")
    return False

def test_breaking_function_change(validator_path, test_repo):
    """Test validator with breaking function changes."""
    print("🧪 Testing: Breaking function changes")
    
    # Modify the function signature (breaking change)
    test_file = os.path.join(test_repo, "test_module.py")
    with open(test_file, 'r') as f:
        content = f.read()
    
    # Change function signature - remove parameter
    modified_content = content.replace(
        'def original_function(param1, param2="default"):',
        'def original_function(param1):'
    )
    
    with open(test_file, 'w') as f:
        f.write(modified_content)
    
    cmd = ["python", validator_path, "validate", "--format", "json", "--repo-path", test_repo]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Should detect breaking change
    if result.returncode != 0:  # Should fail due to breaking change
        try:
            report = json.loads(result.stdout)
            breaking_changes = report["summary"]["breaking_changes"]
            if breaking_changes > 0:
                print("✅ Breaking function change test passed")
                return True
        except:
            pass
    
    print(f"❌ Breaking function change test failed")
    return False

def test_function_removal(validator_path, test_repo):
    """Test validator with function removal."""
    print("🧪 Testing: Function removal")
    
    # Reset to original state first
    subprocess.run(["git", "checkout", "HEAD", "--", "test_module.py"], cwd=test_repo, capture_output=True)
    
    # Remove a function
    test_file = os.path.join(test_repo, "test_module.py")
    with open(test_file, 'r') as f:
        content = f.read()
    
    # Remove the entire function
    lines = content.split('\n')
    filtered_lines = []
    skip = False
    for line in lines:
        if line.strip().startswith('def original_function'):
            skip = True
        elif skip and (line.strip() == '' or line.startswith('def ') or line.startswith('class ')):
            skip = False
            if not line.strip() == '':
                filtered_lines.append(line)
        elif not skip:
            filtered_lines.append(line)
    
    with open(test_file, 'w') as f:
        f.write('\n'.join(filtered_lines))
    
    cmd = ["python", validator_path, "validate", "--format", "json", "--repo-path", test_repo]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Should detect function removal
    if result.returncode != 0:
        try:
            report = json.loads(result.stdout)
            if any("removed" in str(r).lower() for r in report["results"]):
                print("✅ Function removal test passed")
                return True
        except:
            pass
    
    print("❌ Function removal test failed")
    return False

def test_class_method_removal(validator_path, test_repo):
    """Test validator with class method removal."""
    print("🧪 Testing: Class method removal")
    
    # Reset to original state first
    subprocess.run(["git", "checkout", "HEAD", "--", "test_module.py"], cwd=test_repo, capture_output=True)
    
    # Remove a method from class
    test_file = os.path.join(test_repo, "test_module.py")
    with open(test_file, 'r') as f:
        content = f.read()
    
    # Remove method2
    lines = content.split('\n')
    filtered_lines = []
    skip = False
    for line in lines:
        if 'def method2' in line:
            skip = True
        elif skip and (line.strip() == '' or line.strip().startswith('def ') or line.strip().startswith('#')):
            skip = False
            if not line.strip() == '':
                filtered_lines.append(line)
        elif not skip:
            filtered_lines.append(line)
    
    with open(test_file, 'w') as f:
        f.write('\n'.join(filtered_lines))
    
    cmd = ["python", validator_path, "validate", "--format", "json", "--repo-path", test_repo]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Should detect method removal
    if result.returncode != 0:
        try:
            report = json.loads(result.stdout)
            if any("method" in str(r).lower() and ("removed" in str(r).lower() or "missing" in str(r).lower()) for r in report["results"]):
                print("✅ Class method removal test passed")
                return True
        except:
            pass
    
    print("❌ Class method removal test failed")
    return False

def test_safe_addition(validator_path, test_repo):
    """Test validator with safe additions (should pass)."""
    print("🧪 Testing: Safe additions")
    
    # Reset to original state first
    subprocess.run(["git", "checkout", "HEAD", "--", "test_module.py"], cwd=test_repo, capture_output=True)
    
    # Add a new function (should be safe)
    test_file = os.path.join(test_repo, "test_module.py")
    with open(test_file, 'a') as f:
        f.write('''

def new_safe_function(param):
    """A new function that doesn't break anything."""
    return f"new: {param}"
''')
    
    cmd = ["python", validator_path, "validate", "--format", "json", "--repo-path", test_repo]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Should pass (additions are generally safe)
    if result.returncode == 0:  # Should succeed
        try:
            report = json.loads(result.stdout)
            if report["summary"]["breaking_changes"] == 0:
                print("✅ Safe addition test passed")
                return True
        except:
            pass
    
    print("❌ Safe addition test failed")
    return False

def run_all_tests():
    """Run all validator tests."""
    # Find the validator script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    validator_path = os.path.join(current_dir, "functionality-preservation-validator.py")
    
    if not os.path.exists(validator_path):
        print(f"❌ Validator script not found at: {validator_path}")
        return False
    
    print(f"🔍 Found validator at: {validator_path}")
    
    # Create test repository
    print("📁 Creating test repository...")
    test_repo = create_test_repo()
    print(f"📁 Test repository created at: {test_repo}")
    
    try:
        # Run tests
        tests = [
            test_no_changes,
            test_safe_addition,
            test_breaking_function_change,
            test_function_removal,
            test_class_method_removal
        ]
        
        passed = 0
        total = len(tests)
        
        for test_func in tests:
            try:
                if test_func(validator_path, test_repo):
                    passed += 1
            except Exception as e:
                print(f"❌ Test {test_func.__name__} failed with exception: {e}")
        
        print(f"\n📊 Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("✅ All tests passed! Functionality Preservation Validator is working correctly.")
            return True
        else:
            print("❌ Some tests failed. Please review the validator implementation.")
            return False
    
    finally:
        # Cleanup
        print(f"🧹 Cleaning up test repository: {test_repo}")
        shutil.rmtree(test_repo, ignore_errors=True)

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)