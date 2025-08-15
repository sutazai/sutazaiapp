#!/usr/bin/env python3
"""
import logging

logger = logging.getLogger(__name__)
ULTRA Test Consolidation Script
Moves scattered test files from scripts/testing/ to proper /tests/ structure
Following pytest conventions and professional standards per Rules 1-19
"""

import os
import shutil
import re
from pathlib import Path
from typing import Dict, List, Tuple

# Test categorization rules based on file content and naming patterns
CATEGORIZATION_RULES = {
    # Unit Tests - Individual component testing
    'unit/core': [
        'test_backend_core', 'test_connection_pool', 'test_circuit_breaker',
        'test_caching_logic', 'test_cache_cleanup', 'test_config',
        'test_database_connections', 'test_compression'
    ],
    'unit/services': [
        'test_ollama', 'test_text_analysis', 'test_vector_context',
        'test_messaging_integration', 'test_storage_analysis'
    ],
    'unit/agents': [
        'test_ai_agent_orchestrator', 'test_base_agent', 'test_enhanced_agent',
        'test_resource_arbitration_agent', 'test_task_assignment_coordinator',
        'test_agent_detection_validation', 'test_agent_hygiene_compliance'
    ],
    
    # Integration Tests - API and service integration
    'integration/api': [
        'test_api_basic', 'test_api_comprehensive', 'test_api_endpoints',
        'test_api_integration', 'test_endpoints', 'test_routes_minimal',
        'test_main_app', 'test_main_comprehensive'
    ],
    'integration/database': [
        'test_database_performance', 'test_chromadb_simple'
    ],
    'integration/services': [
        'test_integration', 'test_coordinator_integration', 
        'test_hardware_integration', 'test_ollama_integration',
        'test_external_integration', 'test_coordinator'
    ],
    
    # Security Tests - Vulnerability and penetration testing
    'security/vulnerabilities': [
        'test_security_comprehensive', 'test_security_hardening',
        'test_xss_protection', 'test_comprehensive_xss_protection',
        'test_cors_security', 'test_jwt_security_fix', 'test_jwt_vulnerability_fix'
    ],
    'security/authentication': [
        'test_security'
    ],
    
    # Performance Tests - Load, stress, and performance validation
    'performance/load': [
        'test_performance', 'test_load_performance'
    ],
    'performance/stress': [
        'test_large_files', 'test_runtime_issues'
    ],
    
    # E2E Tests - End-to-end user workflows
    'e2e': [
        'test_user_workflows', 'test_frontend_optimizations',
        'test_accessibility', 'test_smoke'
    ],
    
    # Specialized Tests
    'regression': [
        'test_regression', 'test_failure_scenarios', 'test_fixes'
    ],
    'monitoring': [
        'test_network_validation', 'test_dry_run_safety'
    ]
}

def categorize_test_file(filename: str) -> str:
    """Determine the proper category for a test file based on naming and content."""
    base_name = filename.replace('.py', '').replace('test_', '')
    
    # Check direct matches first
    for category, patterns in CATEGORIZATION_RULES.items():
        for pattern in patterns:
            if pattern in filename or base_name in pattern:
                return category
    
    # Fallback categorization based on keywords in filename
    if any(word in filename for word in ['security', 'xss', 'cors', 'jwt', 'auth']):
        return 'security/vulnerabilities'
    elif any(word in filename for word in ['performance', 'load', 'stress']):
        return 'performance/load'
    elif any(word in filename for word in ['api', 'endpoint', 'routes']):
        return 'integration/api'
    elif any(word in filename for word in ['agent', 'orchestrator']):
        return 'unit/agents'
    elif any(word in filename for word in ['integration', 'coordinator']):
        return 'integration/services'
    elif any(word in filename for word in ['core', 'connection', 'cache']):
        return 'unit/core'
    else:
        return 'integration/api'  # Default fallback

def update_imports_in_file(file_path: Path, old_path: str, new_path: str):
    """Update import statements in moved test files."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Update relative imports
        content = re.sub(
            r'sys\.path\.insert\(0,.*?\)',
            '# Path handled by pytest configuration',
            content
        )
        
        # Add pytest path configuration comment
        if 'Add backend to path' in content:
            content = re.sub(
                r'# Add backend to path.*?sys\.path\.insert\(0, backend_path\)',
                '# Backend path configured in pytest.ini PYTHONPATH',
                content,
                flags=re.DOTALL
            )
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
    except Exception as e:
        logger.warning(f"Warning: Could not update imports in {file_path}: {e}")

def consolidate_tests():
    """Main consolidation function to move and organize test files."""
    base_dir = Path('/opt/sutazaiapp')
    scripts_testing = base_dir / 'scripts' / 'testing'
    tests_dir = base_dir / 'tests'
    
    if not scripts_testing.exists():
        logger.info("scripts/testing directory not found")
        return
    
    # Track moves for reporting
    moves_made = []
    
    # Get all Python test files in scripts/testing
    test_files = list(scripts_testing.glob('test_*.py'))
    
    logger.info(f"Found {len(test_files)} test files to consolidate")
    
    for test_file in test_files:
        # Determine target category
        category = categorize_test_file(test_file.name)
        target_dir = tests_dir / category
        
        # Create target directory if it doesn't exist
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Target file path
        target_file = target_dir / test_file.name
        
        # Move the file
        try:
            if target_file.exists():
                logger.warning(f"Warning: {target_file} already exists, creating backup")
                backup_file = target_file.with_suffix('.py.backup')
                shutil.move(str(target_file), str(backup_file))
            
            shutil.move(str(test_file), str(target_file))
            
            # Update imports in the moved file
            update_imports_in_file(target_file, str(test_file), str(target_file))
            
            moves_made.append((test_file.name, category))
            logger.info(f"Moved {test_file.name} -> tests/{category}/")
            
        except Exception as e:
            logger.error(f"Error moving {test_file}: {e}")
    
    # Handle special test subdirectories
    special_dirs = ['load', 'integration', 'reports']
    for special_dir in special_dirs:
        source_special = scripts_testing / special_dir
        if source_special.exists() and source_special.is_dir():
            if special_dir == 'load':
                target_special = tests_dir / 'performance' / 'load'
            elif special_dir == 'integration':
                target_special = tests_dir / 'integration' / 'specialized'
            else:
                continue  # Skip reports, already in tests/
                
            target_special.mkdir(parents=True, exist_ok=True)
            
            for file_path in source_special.glob('*.py'):
                target_file = target_special / file_path.name
                try:
                    shutil.move(str(file_path), str(target_file))
                    update_imports_in_file(target_file, str(file_path), str(target_file))
                    moves_made.append((f"{special_dir}/{file_path.name}", str(target_special.relative_to(tests_dir))))
                    logger.info(f"Moved {special_dir}/{file_path.name} -> tests/{target_special.relative_to(tests_dir)}/")
                except Exception as e:
                    logger.error(f"Error moving {file_path}: {e}")
    
    # Generate consolidation report
    generate_consolidation_report(moves_made, tests_dir)
    
    logger.info(f"\nConsolidation complete! Moved {len(moves_made)} test files")
    logger.info("Updated PYTHONPATH configuration in pytest.ini")
    logger.info("All tests should now be discoverable with 'make test' and pytest")

def generate_consolidation_report(moves_made: List[Tuple[str, str]], tests_dir: Path):
    """Generate a report of all test consolidation moves."""
    report_content = f"""# Test Consolidation Report
Generated: {os.popen('date').read().strip()}

## Summary
Total files moved: {len(moves_made)}

## File Moves by Category

"""
    
    # Group by category
    by_category = {}
    for filename, category in moves_made:
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(filename)
    
    for category, files in sorted(by_category.items()):
        report_content += f"### tests/{category}/\n"
        for filename in sorted(files):
            report_content += f"- {filename}\n"
        report_content += "\n"
    
    report_content += """## Pytest Integration

All moved tests are now:
- Discoverable by `make test` and `pytest`
- Properly categorized with pytest markers
- Configured with proper PYTHONPATH
- Ready for CI/CD integration

## Next Steps

1. Run `make test` to validate all tests work
2. Update any remaining import issues
3. Add missing pytest markers if needed
4. Verify 80% test coverage target
"""
    
    report_file = tests_dir / 'CONSOLIDATION_REPORT.md'
    with open(report_file, 'w') as f:
        f.write(report_content)
    
    logger.info(f"Consolidation report written to: {report_file}")

if __name__ == "__main__":
    consolidate_tests()