#!/usr/bin/env python3
"""
Comprehensive test suite for GitHub Actions workflows.
Tests YAML validity, references, dependencies, and best practices.
"""

import yaml
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict

class WorkflowTester:
    def __init__(self):
        self.workflows_dir = Path("/opt/sutazaiapp/.github/workflows")
        self.project_root = Path("/opt/sutazaiapp")
        self.test_results = defaultdict(list)
        self.warnings = []
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests on all workflows."""
        print("🧪 Running Comprehensive Workflow Tests")
        print("=" * 60)
        
        workflow_files = list(self.workflows_dir.glob("*.yml")) + \
                        list(self.workflows_dir.glob("*.yaml"))
        
        results = {
            'total': len(workflow_files),
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'details': []
        }
        
        for filepath in sorted(workflow_files):
            print(f"\n📋 Testing: {filepath.name}")
            print("-" * 40)
            
            file_results = self.test_workflow(filepath)
            
            if file_results['status'] == 'passed':
                results['passed'] += 1
                print(f"  ✅ All tests passed")
            else:
                results['failed'] += 1
                print(f"  ❌ {len(file_results['errors'])} errors found")
            
            if file_results['warnings']:
                results['warnings'] += len(file_results['warnings'])
                print(f"  ⚠️  {len(file_results['warnings'])} warnings")
            
            results['details'].append(file_results)
        
        return results
    
    def test_workflow(self, filepath: Path) -> Dict[str, Any]:
        """Test a single workflow file."""
        results = {
            'file': filepath.name,
            'status': 'passed',
            'errors': [],
            'warnings': [],
            'tests': {}
        }
        
        # Test 1: YAML Validity
        yaml_test = self.test_yaml_validity(filepath)
        results['tests']['yaml_validity'] = yaml_test
        if not yaml_test['passed']:
            results['errors'].extend(yaml_test['errors'])
            results['status'] = 'failed'
            return results  # Can't continue if YAML is invalid
        
        workflow = yaml_test['data']
        
        # Test 2: Required Fields
        fields_test = self.test_required_fields(workflow)
        results['tests']['required_fields'] = fields_test
        if not fields_test['passed']:
            results['errors'].extend(fields_test['errors'])
            results['status'] = 'failed'
        
        # Test 3: File References
        refs_test = self.test_file_references(workflow, filepath)
        results['tests']['file_references'] = refs_test
        if not refs_test['passed']:
            results['warnings'].extend(refs_test['errors'])  # Warnings, not errors
        
        # Test 4: Service Dependencies
        services_test = self.test_service_dependencies(workflow)
        results['tests']['service_dependencies'] = services_test
        if not services_test['passed']:
            results['warnings'].extend(services_test['errors'])
        
        # Test 5: Best Practices
        practices_test = self.test_best_practices(workflow)
        results['tests']['best_practices'] = practices_test
        if not practices_test['passed']:
            results['warnings'].extend(practices_test['errors'])
        
        # Test 6: Security
        security_test = self.test_security(workflow)
        results['tests']['security'] = security_test
        if not security_test['passed']:
            results['errors'].extend(security_test['errors'])
            results['status'] = 'failed'
        
        return results
    
    def test_yaml_validity(self, filepath: Path) -> Dict[str, Any]:
        """Test if YAML is valid."""
        try:
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
            return {'passed': True, 'data': data, 'errors': []}
        except yaml.YAMLError as e:
            return {'passed': False, 'data': None, 'errors': [f"YAML Error: {str(e)}"]}
        except Exception as e:
            return {'passed': False, 'data': None, 'errors': [f"Read Error: {str(e)}"]}
    
    def test_required_fields(self, workflow: Dict) -> Dict[str, Any]:
        """Test for required workflow fields."""
        errors = []
        
        # Check for name
        if 'name' not in workflow:
            errors.append("Missing required field: 'name'")
        
        # Check for trigger (on or True due to YAML parsing)
        if 'on' not in workflow and True not in workflow:
            errors.append("Missing required field: 'on' (trigger)")
        
        # Check for jobs
        if 'jobs' not in workflow:
            errors.append("Missing required field: 'jobs'")
        else:
            # Check each job has required fields
            for job_name, job_config in workflow.get('jobs', {}).items():
                if not isinstance(job_config, dict):
                    errors.append(f"Job '{job_name}' has invalid configuration")
                    continue
                
                if 'runs-on' not in job_config:
                    errors.append(f"Job '{job_name}' missing 'runs-on'")
                
                if 'steps' not in job_config:
                    errors.append(f"Job '{job_name}' missing 'steps'")
        
        return {'passed': len(errors) == 0, 'errors': errors}
    
    def test_file_references(self, workflow: Dict, filepath: Path) -> Dict[str, Any]:
        """Test that referenced files exist."""
        errors = []
        content = filepath.read_text()
        
        # Check requirements files
        req_pattern = r'pip install.*-r\s+([^\s]+)'
        for match in re.finditer(req_pattern, content):
            req_file = match.group(1)
            full_path = self.project_root / req_file
            if not full_path.exists():
                errors.append(f"Requirements file not found: {req_file}")
        
        # Check Python scripts
        script_pattern = r'python[3]?\s+([^\s]+\.py)'
        for match in re.finditer(script_pattern, content):
            script_file = match.group(1)
            if not script_file.startswith('-'):  # Skip flags
                full_path = self.project_root / script_file
                if not full_path.exists():
                    errors.append(f"Script not found: {script_file}")
        
        # Check docker-compose files
        compose_pattern = r'docker-compose.*-f\s+([^\s]+\.ya?ml)'
        for match in re.finditer(compose_pattern, content):
            compose_file = match.group(1)
            full_path = self.project_root / compose_file
            if not full_path.exists():
                errors.append(f"Docker compose file not found: {compose_file}")
        
        return {'passed': len(errors) == 0, 'errors': errors}
    
    def test_service_dependencies(self, workflow: Dict) -> Dict[str, Any]:
        """Test service dependencies are properly configured."""
        errors = []
        
        for job_name, job_config in workflow.get('jobs', {}).items():
            if not isinstance(job_config, dict):
                continue
            
            # Check if services are defined
            if 'services' in job_config:
                services = job_config['services']
                
                # Common services that should have proper config
                if 'postgres' in services:
                    pg_config = services['postgres']
                    if 'env' not in pg_config:
                        errors.append(f"Job '{job_name}': PostgreSQL missing environment config")
                
                if 'redis' in services:
                    redis_config = services['redis']
                    if 'options' not in redis_config:
                        errors.append(f"Job '{job_name}': Redis missing health check options")
        
        return {'passed': len(errors) == 0, 'errors': errors}
    
    def test_best_practices(self, workflow: Dict) -> Dict[str, Any]:
        """Test for GitHub Actions best practices."""
        warnings = []
        
        # Check for workflow concurrency control
        if 'concurrency' not in workflow:
            warnings.append("Consider adding concurrency control to prevent parallel runs")
        
        # Check for timeout settings
        for job_name, job_config in workflow.get('jobs', {}).items():
            if isinstance(job_config, dict):
                if 'timeout-minutes' not in job_config:
                    warnings.append(f"Job '{job_name}': Consider adding timeout-minutes")
        
        # Check for proper caching
        has_cache = False
        for job_name, job_config in workflow.get('jobs', {}).items():
            if isinstance(job_config, dict):
                for step in job_config.get('steps', []):
                    if isinstance(step, dict) and 'uses' in step:
                        if 'cache' in step.get('uses', ''):
                            has_cache = True
                            break
        
        if not has_cache:
            # Check if it uses setup-python or setup-node which have built-in caching
            for job_name, job_config in workflow.get('jobs', {}).items():
                if isinstance(job_config, dict):
                    for step in job_config.get('steps', []):
                        if isinstance(step, dict) and 'uses' in step:
                            if 'setup-python' in step.get('uses', '') or 'setup-node' in step.get('uses', ''):
                                if 'with' not in step or 'cache' not in step.get('with', {}):
                                    warnings.append("Consider enabling caching in setup actions")
        
        return {'passed': len(warnings) == 0, 'errors': warnings}
    
    def test_security(self, workflow: Dict) -> Dict[str, Any]:
        """Test for security issues."""
        errors = []
        
        # Check for hardcoded secrets
        workflow_str = json.dumps(workflow)
        
        # Common patterns for secrets
        secret_patterns = [
            r'password\s*[=:]\s*["\'][^"\']+["\']',
            r'token\s*[=:]\s*["\'][^"\']+["\']',
            r'api[_-]key\s*[=:]\s*["\'][^"\']+["\']',
        ]
        
        for pattern in secret_patterns:
            matches = re.finditer(pattern, workflow_str, re.IGNORECASE)
            for match in matches:
                value = match.group(0)
                # Skip if it's using GitHub secrets
                if '${{' not in value and 'secrets.' not in value:
                    # Check if it's a known test value
                    if 'test' not in value.lower() and 'example' not in value.lower():
                        errors.append(f"Potential hardcoded secret: {value[:30]}...")
        
        # Check for proper permissions
        if 'permissions' not in workflow:
            # This is actually OK - defaults to read-all
            pass
        
        return {'passed': len(errors) == 0, 'errors': errors}


def main():
    """Main function."""
    tester = WorkflowTester()
    results = tester.run_all_tests()
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    print(f"\nTotal workflows tested: {results['total']}")
    print(f"✅ Passed: {results['passed']}")
    print(f"❌ Failed: {results['failed']}")
    print(f"⚠️  Total warnings: {results['warnings']}")
    
    if results['failed'] > 0:
        print("\n❌ FAILED WORKFLOWS:")
        for detail in results['details']:
            if detail['status'] == 'failed':
                print(f"\n  {detail['file']}:")
                for error in detail['errors']:
                    print(f"    • {error}")
    
    if results['warnings'] > 0:
        print("\n⚠️  WARNINGS:")
        for detail in results['details']:
            if detail['warnings']:
                print(f"\n  {detail['file']}:")
                for warning in detail['warnings']:
                    print(f"    • {warning}")
    
    print("\n" + "=" * 60)
    
    if results['failed'] == 0:
        print("✅ ALL WORKFLOWS PASSED COMPREHENSIVE TESTING!")
        return 0
    else:
        print("❌ Some workflows have issues that need attention")
        return 1


if __name__ == "__main__":
    sys.exit(main())