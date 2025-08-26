#!/usr/bin/env python3
"""
import logging

logger = logging.getLogger(__name__)
Security Requirements Validation Script for SutazAI Frontend
Validates that all security vulnerabilities have been addressed in requirements.txt
"""

import re
import sys
from packaging import version

def parse_requirements(file_path):
    """Parse requirements.txt and return package versions"""
    packages = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Extract package name and version
                    match = re.match(r'^([a-zA-Z0-9_-]+)==(.+?)(\s|#|$)', line)
                    if match:
                        pkg_name, pkg_version = match.groups()[:2]
                        packages[pkg_name.lower()] = pkg_version.strip()
    except FileNotFoundError:
        logger.error(f"❌ ERROR: {file_path} not found")
        return {}
    return packages

def validate_security_requirements(packages):
    """Validate that all security requirements are met"""
    results = []
    
    # High Priority Security Fixes
    high_priority = [
        ('setuptools', '70.0.0', 'Path traversal vulnerability fix'),
    ]
    
    # Moderate Priority Security Fixes  
    moderate_priority = [
        ('urllib3', '2.2.2', 'Security vulnerability fixes'),
        ('requests', '2.32.0', 'Security vulnerability fixes'),
        ('jinja2', '3.1.4', 'Security vulnerability fixes'),
    ]
    
    # Low Priority Security Fixes
    low_priority = [
        ('aiohttp', '3.9.4', 'Security vulnerability fixes'),
        ('cryptography', '43.0.1', 'Security vulnerability fixes'),
    ]
    
    all_requirements = [
        ('HIGH', high_priority),
        ('MODERATE', moderate_priority), 
        ('LOW', low_priority)
    ]
    
    for priority, requirements in all_requirements:
        for pkg_name, min_version, description in requirements:
            if pkg_name in packages:
                current_version = packages[pkg_name]
                try:
                    if version.parse(current_version) >= version.parse(min_version):
                        results.append((priority, pkg_name, current_version, min_version, '✅ SECURE', description))
                    else:
                        results.append((priority, pkg_name, current_version, min_version, '❌ VULNERABLE', description))
                except Exception as e:
                    results.append((priority, pkg_name, current_version, min_version, f'⚠️ VERSION PARSE ERROR: {e}', description))
            else:
                results.append((priority, pkg_name, 'NOT FOUND', min_version, '❌ MISSING', description))
    
    return results

def print_results(results):
    """Print validation results in a formatted way"""
    logger.info("\n" + "="*80)
    logger.info("🔒 SECURITY VULNERABILITY VALIDATION REPORT")
    logger.info("="*80)
    
    vulnerable_count = 0
    secure_count = 0
    
    for priority, pkg_name, current, minimum, status, description in results:
        logger.info(f"\n[{priority}] {pkg_name.upper()}")
        logger.info(f"  Required: >= {minimum}")
        logger.info(f"  Current:  {current}")
        logger.info(f"  Status:   {status}")
        logger.info(f"  Reason:   {description}")
        
        if '❌' in status:
            vulnerable_count += 1
        elif '✅' in status:
            secure_count += 1
    
    logger.info("\n" + "="*80)
    logger.info("📊 SUMMARY")
    logger.info("="*80)
    logger.info(f"✅ Secure packages:     {secure_count}")
    logger.info(f"❌ Vulnerable packages: {vulnerable_count}")
    
    if vulnerable_count == 0:
        logger.info("\n🎉 ALL SECURITY REQUIREMENTS SATISFIED!")
        logger.info("✅ Frontend is ready for production deployment")
        return True
    else:
        logger.info(f"\n⚠️  {vulnerable_count} SECURITY ISSUES REMAIN")
        logger.info("❌ Please update vulnerable packages before deployment")
        return False

def main():
    """Main validation function"""
    requirements_file = 'requirements.txt'
    
    logger.info("🔍 Parsing requirements.txt...")
    packages = parse_requirements(requirements_file)
    
    if not packages:
        logger.info("❌ No packages found or file not readable")
        sys.exit(1)
    
    logger.info(f"📦 Found {len(packages)} packages")
    
    logger.info("\n🔒 Validating security requirements...")
    results = validate_security_requirements(packages)
    
    success = print_results(results)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()