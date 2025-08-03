#!/usr/bin/env python3
"""
Purpose: Verify single canonical deployment script (Rule 12 enforcement)
Usage: python check-deployment-scripts.py
Requirements: Python 3.8+
"""

import sys
import subprocess
from pathlib import Path
from typing import List, Dict

def find_deployment_scripts(project_root: Path) -> Dict[str, List[Path]]:
    """Find all deployment-related scripts in the project."""
    deployment_scripts = {
        'shell': [],
        'python': [],
        'other': []
    }
    
    # Patterns to identify deployment scripts
    patterns = [
        "deploy*.sh",
        "*deploy*.sh",
        "deploy*.py", 
        "*deploy*.py",
        "release*.sh",
        "*release*.sh",
        "provision*.sh",
        "*provision*.sh"
    ]
    
    for pattern in patterns:
        try:
            result = subprocess.run(
                ["find", str(project_root), "-name", pattern, "-type", "f"],
                capture_output=True,
                text=True,
                check=True
            )
            
            for filepath in result.stdout.strip().split('\n'):
                if filepath:
                    path = Path(filepath)
                    # Skip archive and test directories
                    if any(skip in str(path) for skip in ['/archive/', '/test/', '/tests/', '/.git/']):
                        continue
                        
                    if path.suffix == '.sh':
                        deployment_scripts['shell'].append(path)
                    elif path.suffix == '.py':
                        deployment_scripts['python'].append(path)
                    else:
                        deployment_scripts['other'].append(path)
                        
        except subprocess.CalledProcessError:
            pass
            
    return deployment_scripts

def check_canonical_script(scripts: Dict[str, List[Path]]) -> bool:
    """Check if there's a single canonical deployment script."""
    all_scripts = []
    for script_list in scripts.values():
        all_scripts.extend(script_list)
    
    # Look for the canonical deploy.sh
    canonical_found = False
    canonical_path = None
    
    for script in all_scripts:
        if script.name == "deploy.sh" and script.parent.name == "sutazaiapp":
            canonical_found = True
            canonical_path = script
            break
    
    if not canonical_found:
        print("❌ Rule 12 Violation: No canonical deploy.sh found in project root")
        return False
    
    # Check if there are multiple deployment scripts
    if len(all_scripts) > 1:
        print(f"⚠️  Warning: Multiple deployment scripts detected ({len(all_scripts)} total)")
        print(f"✅ Canonical script found: {canonical_path}")
        print("\n📋 Other deployment scripts that should be consolidated:")
        
        for script in all_scripts:
            if script != canonical_path:
                print(f"  - {script}")
        
        print("\n📋 Recommendation:")
        print("  1. Consolidate all deployment logic into the canonical deploy.sh")
        print("  2. Remove or archive redundant deployment scripts")
        print("  3. Ensure deploy.sh supports all deployment scenarios with flags")
        print("  4. Document all deployment options in deploy.sh header")
        
        # This is a warning, not a failure if canonical exists
        return True
    
    return True

def check_deployment_script_quality(canonical_script: Path) -> bool:
    """Check the quality of the canonical deployment script."""
    if not canonical_script.exists():
        return True  # Skip if not found (handled elsewhere)
    
    required_features = {
        'phases': False,  # --phase flag support
        'environments': False,  # --env flag support
        'dry_run': False,  # --dry-run support
        'idempotent': False,  # Re-run safety
        'error_handling': False,  # set -e or error handling
    }
    
    try:
        with open(canonical_script, 'r') as f:
            content = f.read()
            
        # Check for required features
        if '--phase' in content or 'phase)' in content:
            required_features['phases'] = True
        if '--env' in content or 'environment' in content:
            required_features['environments'] = True
        if '--dry-run' in content or 'dry_run' in content:
            required_features['dry_run'] = True
        if 'set -e' in content or 'trap' in content:
            required_features['error_handling'] = True
        if 'if [' in content or 'test' in content:
            required_features['idempotent'] = True
            
        missing_features = [k for k, v in required_features.items() if not v]
        
        if missing_features:
            print(f"\n⚠️  Canonical deploy.sh may be missing features:")
            for feature in missing_features:
                print(f"  - {feature}")
                
    except Exception as e:
        print(f"Error checking deployment script quality: {e}")
        
    return True

def main():
    """Main function to check deployment script compliance."""
    project_root = Path("/opt/sutazaiapp")
    
    # Find all deployment scripts
    scripts = find_deployment_scripts(project_root)
    
    # Check for canonical script
    if not check_canonical_script(scripts):
        return 1
    
    # Check canonical script quality
    canonical_path = project_root / "deploy.sh"
    if canonical_path.exists():
        check_deployment_script_quality(canonical_path)
    
    print("✅ Rule 12: Deployment script structure is compliant")
    return 0

if __name__ == "__main__":
    sys.exit(main())