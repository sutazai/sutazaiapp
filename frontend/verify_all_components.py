#!/usr/bin/env python3
"""
Comprehensive Streamlit Component Verification Script
Checks all Streamlit custom components for proper installation
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple

def check_component(module_name: str, expected_frontend_dir: str = "frontend/dist") -> Tuple[bool, str, Dict]:
    """
    Check if a Streamlit component is properly installed with all assets.
    
    Args:
        module_name: Name of the Python module to check
        expected_frontend_dir: Expected frontend directory path (dist or build)
    
    Returns:
        Tuple of (success, error_message, details)
    """
    details = {
        'module_found': False,
        'frontend_exists': False,
        'assets_count': 0,
        'module_path': None,
        'frontend_path': None
    }
    
    try:
        # Import the module
        module = __import__(module_name)
        details['module_found'] = True
        
        # Get module path
        module_path = Path(module.__file__).parent
        details['module_path'] = str(module_path)
        
        # Check for frontend directory
        frontend_path = module_path / expected_frontend_dir
        
        if not frontend_path.exists():
            # Try alternative paths
            alt_paths = [
                module_path / "frontend/build",
                module_path / "frontend/dist",
                module_path / "build",
                module_path / "dist"
            ]
            for alt_path in alt_paths:
                if alt_path.exists():
                    frontend_path = alt_path
                    break
        
        details['frontend_path'] = str(frontend_path)
        
        if not frontend_path.exists():
            return False, f"Frontend directory missing: {frontend_path}", details
        
        details['frontend_exists'] = True
        
        # Count assets
        asset_count = 0
        for root, dirs, files in os.walk(frontend_path):
            asset_count += len(files)
        
        details['assets_count'] = asset_count
        
        if asset_count == 0:
            return False, f"No assets found in {frontend_path}", details
        
        return True, f"Component OK: {asset_count} assets", details
        
    except ImportError as e:
        return False, f"Module not installed: {e}", details
    except Exception as e:
        return False, f"Error checking component: {e}", details


def main():
    """Main verification function."""
    
    # Define components to check with their expected frontend directories
    components_to_check = [
        ("streamlit_chat", "frontend/dist"),
        ("streamlit_lottie", "frontend/build"),
        ("streamlit_option_menu", "frontend/dist"),
        ("streamlit_mic_recorder", "frontend/dist"),
    ]
    
    print("=" * 60)
    print("STREAMLIT COMPONENT VERIFICATION")
    print("=" * 60)
    
    all_success = True
    results = []
    
    for component_name, frontend_dir in components_to_check:
        print(f"\n📦 Checking {component_name}...")
        success, message, details = check_component(component_name.replace('-', '_'), frontend_dir)
        
        if success:
            print(f"  ✅ {message}")
            print(f"     Module: {details['module_path']}")
            print(f"     Frontend: {details['frontend_path']}")
            print(f"     Assets: {details['assets_count']} files")
        else:
            print(f"  ❌ {message}")
            all_success = False
            if details['module_path']:
                print(f"     Module: {details['module_path']}")
            print(f"     Issue: {message}")
        
        results.append({
            'component': component_name,
            'success': success,
            'message': message,
            'details': details
        })
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print(f"\n✅ Successful: {successful}/{len(results)}")
    if failed > 0:
        print(f"❌ Failed: {failed}/{len(results)}")
        print("\nFailed components:")
        for r in results:
            if not r['success']:
                print(f"  - {r['component']}: {r['message']}")
        
        print("\n🔧 To fix, run:")
        for r in results:
            if not r['success']:
                print(f"  pip uninstall {r['component']} -y")
                print(f"  pip install {r['component']} --no-cache-dir")
    
    print("\n" + "=" * 60)
    
    # Exit with appropriate code
    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()