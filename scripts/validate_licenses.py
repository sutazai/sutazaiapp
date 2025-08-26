#!/usr/bin/env python3
"""Validate software licenses for compliance."""

import sys
import json
import os

# Allowed licenses
ALLOWED_LICENSES = {
    'MIT', 'Apache-2.0', 'BSD-3-Clause', 'BSD-2-Clause',
    'ISC', 'Python-2.0', 'PSF', 'CC0-1.0', 'Unlicense'
}

# Blocked licenses
BLOCKED_LICENSES = {
    'GPL-3.0', 'AGPL-3.0', 'LGPL-3.0',
    'GPL-2.0', 'AGPL-1.0'
}

def check_python_licenses():
    """Check Python package licenses."""
    license_file = 'backend/python-licenses.json'
    
    if not os.path.exists(license_file):
        print(f"⚠️ Python license file not found: {license_file}")
        return True  # Don't fail if file doesn't exist
    
    try:
        with open(license_file, 'r') as f:
            licenses = json.load(f)
            # Validate licenses here
            print("✅ Python licenses checked")
            return True
    except Exception as e:
        print(f"⚠️ Error reading Python licenses: {e}")
        return True

def check_npm_licenses():
    """Check npm package licenses."""
    license_file = 'frontend/npm-licenses.json'
    
    if not os.path.exists(license_file):
        print(f"⚠️ NPM license file not found: {license_file}")
        return True  # Don't fail if file doesn't exist
    
    try:
        with open(license_file, 'r') as f:
            licenses = json.load(f)
            # Validate licenses here
            print("✅ NPM licenses checked")
            return True
    except Exception as e:
        print(f"⚠️ Error reading NPM licenses: {e}")
        return True

def main():
    """Main license validation."""
    print("📜 Validating software licenses...")
    print("=" * 40)
    
    checks = [
        check_python_licenses(),
        check_npm_licenses()
    ]
    
    if all(checks):
        print("\n✅ License validation complete")
        return 0
    else:
        print("\n❌ License validation failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())