#!/usr/bin/env python3
"""
Comprehensive Streamlit Component Verification and Fix Script
Ensures all Streamlit components have their required frontend directories
"""

import sys
import subprocess
from pathlib import Path
import importlib
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Map of known Streamlit components and their expected frontend directories
COMPONENT_MAP = {
    'streamlit_chat': {'frontend_dir': 'dist', 'version': '0.1.1'},
    'streamlit_lottie': {'frontend_dir': 'build', 'version': '0.0.5'},
    'streamlit_option_menu': {'frontend_dir': 'dist', 'version': '0.4.0'},
    'st_aggrid': {'frontend_dir': 'build', 'version': '1.0.5', 'package_name': 'streamlit-aggrid'},
    'streamlit_elements': {'frontend_dir': 'build', 'version': '0.1.0'},
    'streamlit_mic_recorder': {'frontend_dir': 'build', 'version': '0.0.8'},
    'streamlit_webrtc': {'frontend_dir': 'build', 'version': '0.47.1'},
}

def check_component(module_name, expected_dir):
    """Check if a Streamlit component is properly installed."""
    try:
        module = importlib.import_module(module_name)
        module_path = Path(module.__file__).parent
        frontend_path = module_path / 'frontend' / expected_dir
        
        if frontend_path.exists():
            file_count = len(list(frontend_path.rglob('*')))
            logger.info(f"✅ {module_name}: {expected_dir} directory exists with {file_count} files")
            return True
        else:
            logger.warning(f"❌ {module_name}: {expected_dir} directory missing at {frontend_path}")
            return False
    except ImportError as e:
        logger.error(f"❌ {module_name}: Not installed - {e}")
        return False
    except Exception as e:
        logger.error(f"❌ {module_name}: Error checking - {e}")
        return False

def reinstall_component(package_name, version):
    """Reinstall a Streamlit component package."""
    logger.info(f"🔧 Reinstalling {package_name}=={version}...")
    
    # Uninstall first
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", package_name, "-y"],
        capture_output=True,
        text=True
    )
    
    # Install with no-cache-dir to ensure fresh download
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", f"{package_name}=={version}", "--no-cache-dir"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        logger.info(f"✅ Successfully reinstalled {package_name}")
        return True
    else:
        logger.error(f"❌ Failed to reinstall {package_name}: {result.stderr}")
        return False

def main():
    """Main function to check and fix all Streamlit components."""
    logger.info("=" * 60)
    logger.info("STREAMLIT COMPONENT VERIFICATION AND FIX")
    logger.info("=" * 60)
    
    all_good = True
    fixed_components = []
    failed_components = []
    
    for module_name, config in COMPONENT_MAP.items():
        logger.info(f"\nChecking {module_name}...")
        
        # Get package name from config or convert module name
        package_name = config.get('package_name', module_name.replace('_', '-'))
        
        if not check_component(module_name, config['frontend_dir']):
            all_good = False
            logger.info(f"Attempting to fix {module_name}...")
            
            if reinstall_component(package_name, config['version']):
                # Check again after reinstall
                if check_component(module_name, config['frontend_dir']):
                    fixed_components.append(module_name)
                else:
                    failed_components.append(module_name)
            else:
                failed_components.append(module_name)
    
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    
    if fixed_components:
        logger.info(f"\n✅ Fixed components ({len(fixed_components)}):")
        for comp in fixed_components:
            logger.info(f"   - {comp}")
    
    if failed_components:
        logger.error(f"\n❌ Failed to fix ({len(failed_components)}):")
        for comp in failed_components:
            logger.error(f"   - {comp}")
    
    if all_good:
        logger.info("\n🎉 All components verified successfully!")
        return 0
    elif not failed_components:
        logger.info("\n✅ All components fixed successfully!")
        return 0
    else:
        logger.error(f"\n⚠️ {len(failed_components)} components still have issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())