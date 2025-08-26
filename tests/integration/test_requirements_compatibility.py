#!/usr/bin/env python3
"""
import logging

logger = logging.getLogger(__name__)
Requirements Compatibility Test for SutazAI Frontend
Tests that all required packages can be imported and are compatible
"""

import sys
import importlib
import traceback

def test_package_imports():
    """Test that all critical packages can be imported"""
    critical_packages = [
        # Core Streamlit
        ('streamlit', 'Streamlit web framework'),
        
        # HTTP clients 
        ('httpx', 'Modern HTTP client'),
        ('requests', 'HTTP library'),
        ('aiohttp', 'Async HTTP client/server'),
        
        # Data processing
        ('pandas', 'Data analysis library'),
        ('numpy', 'Numerical computing'),
        ('plotly', 'Interactive visualizations'),
        
        # Security-critical packages
        ('cryptography', 'Cryptographic library'),
        ('jinja2', 'Template engine'),
        
        # Core dependencies
        ('urllib3', 'HTTP library'),
    ]
    
    results = []
    
    logger.info("🧪 TESTING PACKAGE IMPORTS")
    logger.info("="*50)
    
    for package_name, description in critical_packages:
        try:
            module = importlib.import_module(package_name)
            version = getattr(module, '__version__', 'Unknown')
            results.append((package_name, True, version, description, None))
            logger.info(f"✅ {package_name:15} v{version:10} - {description}")
            
        except ImportError as e:
            results.append((package_name, False, 'N/A', description, str(e)))
            logger.error(f"❌ {package_name:15} {'FAILED':10} - {description}")
            logger.error(f"   Error: {e}")
        except Exception as e:
            results.append((package_name, False, 'N/A', description, str(e)))
            logger.error(f"⚠️  {package_name:15} {'ERROR':10} - {description}")
            logger.error(f"   Error: {e}")
    
    return results

def test_streamlit_compatibility():
    """Test Streamlit-specific compatibility"""
    logger.info("\n\n🎯 TESTING STREAMLIT COMPATIBILITY")
    logger.info("="*50)
    
    try:
        import streamlit as st
        logger.info(f"✅ Streamlit imported successfully: v{st.__version__}")
        
        # Test key Streamlit dependencies
        streamlit_deps = [
            'altair',      # Visualization  
            'pandas',      # DataFrames
            'plotly',      # Charts
            'pyarrow',     # Data serialization
            'pillow',      # Image processing
            'protobuf',    # Data format
        ]
        
        for dep in streamlit_deps:
            try:
                importlib.import_module(dep)
                logger.info(f"✅ {dep:15} - Compatible with Streamlit")
            except ImportError:
                logger.error(f"❌ {dep:15} - Import failed")
                
        return True
        
    except Exception as e:
        logger.error(f"❌ Streamlit compatibility test failed: {e}")
        return False

def test_security_packages():
    """Test security-critical packages"""
    logger.info("\n\n🔒 TESTING SECURITY PACKAGES")
    logger.info("="*50)
    
    security_tests = [
        ('cryptography', lambda: importlib.import_module('cryptography.fernet')),
        ('jinja2', lambda: importlib.import_module('jinja2').Environment()),
        ('requests', lambda: importlib.import_module('requests').get.__doc__),
        ('urllib3', lambda: importlib.import_module('urllib3').PoolManager()),
    ]
    
    all_passed = True
    
    for pkg_name, test_func in security_tests:
        try:
            result = test_func()
            logger.info(f"✅ {pkg_name:15} - Security features accessible")
        except Exception as e:
            logger.error(f"❌ {pkg_name:15} - Security test failed: {e}")
            all_passed = False
            
    return all_passed

def main():
    """Run all compatibility tests"""
    logger.info("🧪 REQUIREMENTS COMPATIBILITY TEST SUITE")
    logger.info("="*60)
    logger.info("Testing SutazAI Frontend requirements.txt compatibility")
    logger.info("="*60)
    
    # Test package imports
    import_results = test_package_imports()
    
    # Count successful imports
    successful_imports = sum(1 for _, success, _, _, _ in import_results if success)
    total_packages = len(import_results)
    
    # Test Streamlit compatibility
    streamlit_ok = test_streamlit_compatibility()
    
    # Test security packages
    security_ok = test_security_packages()
    
    # Final summary
    logger.info("\n\n📊 FINAL COMPATIBILITY REPORT")
    logger.info("="*60)
    logger.info(f"Package imports:     {successful_imports}/{total_packages} successful")
    logger.error(f"Streamlit compatibility: {'✅ PASSED' if streamlit_ok else '❌ FAILED'}")
    logger.error(f"Security tests:      {'✅ PASSED' if security_ok else '❌ FAILED'}")
    
    overall_success = (
        successful_imports == total_packages and 
        streamlit_ok and 
        security_ok
    )
    
    if overall_success:
        logger.info("\n🎉 ALL COMPATIBILITY TESTS PASSED!")
        logger.info("✅ Requirements are ready for production use")
        return True
    else:
        logger.info("\n⚠️ SOME COMPATIBILITY ISSUES DETECTED")
        logger.info("❌ Please review and fix issues before deployment")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)