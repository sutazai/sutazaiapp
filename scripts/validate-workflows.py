#!/usr/bin/env python3
"""
Validate GitHub Actions workflow files for syntax and consistency.
"""
import yaml
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

def validate_workflow(file_path: str) -> Tuple[bool, List[str]]:
    """Validate a single workflow file."""
    errors = []
    
    try:
        with open(file_path, 'r') as f:
            workflow = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return False, [f"YAML parse error: {e}"]
    except Exception as e:
        return False, [f"Could not read file: {e}"]
    
    # Check required fields
    if not workflow:
        errors.append("Empty workflow file")
    
    if 'name' not in workflow:
        errors.append("Missing 'name' field")
    
    # Check for trigger field - 'on' or True (YAML interprets 'on' as boolean)
    if 'on' not in workflow and True not in workflow:
        errors.append("Missing 'on' trigger field")
    
    # Check for common issues
    if 'jobs' in workflow:
        for job_name, job_config in workflow['jobs'].items():
            # Check job configuration
            if not isinstance(job_config, dict):
                errors.append(f"Job '{job_name}' has invalid configuration")
                continue
                
            # Check for steps
            if 'steps' in job_config:
                for i, step in enumerate(job_config['steps']):
                    if not isinstance(step, dict):
                        errors.append(f"Job '{job_name}' step {i} has invalid configuration")
                        continue
                    
                    # Check for common step issues
                    if 'run' in step and 'uses' in step:
                        errors.append(f"Job '{job_name}' step {i} has both 'run' and 'uses'")
                    
                    if 'run' not in step and 'uses' not in step:
                        if 'name' in step:
                            step_name = step['name']
                        else:
                            step_name = f"step {i}"
                        errors.append(f"Job '{job_name}' {step_name} missing 'run' or 'uses'")
    else:
        errors.append("No jobs defined in workflow")
    
    return len(errors) == 0, errors


def check_consistency(workflows_dir: str) -> Dict[str, List[str]]:
    """Check consistency across all workflows."""
    issues = {}
    workflows = list(Path(workflows_dir).glob("*.yml")) + list(Path(workflows_dir).glob("*.yaml"))
    
    python_versions = set()
    node_versions = set()
    
    for workflow_path in workflows:
        with open(workflow_path, 'r') as f:
            try:
                workflow = yaml.safe_load(f)
                
                # Extract Python versions
                content = f.read()
                import re
                py_matches = re.findall(r"python-version:\s*['\"]?([\d.]+)['\"]?", content)
                python_versions.update(py_matches)
                
                # Extract Node versions  
                node_matches = re.findall(r"node-version:\s*['\"]?([\d.]+)['\"]?", content)
                node_versions.update(node_matches)
                
            except:
                pass
    
    if len(python_versions) > 1:
        issues["Python versions"] = [f"Inconsistent Python versions found: {python_versions}"]
    
    if len(node_versions) > 1:
        issues["Node versions"] = [f"Inconsistent Node versions found: {node_versions}"]
    
    return issues


def main():
    """Main validation function."""
    workflows_dir = "/opt/sutazaiapp/.github/workflows"
    
    if not os.path.exists(workflows_dir):
        print(f"❌ Workflows directory not found: {workflows_dir}")
        sys.exit(1)
    
    print("🔍 Validating GitHub Actions Workflows")
    print("=" * 50)
    
    # Get all workflow files
    workflow_files = list(Path(workflows_dir).glob("*.yml")) + list(Path(workflows_dir).glob("*.yaml"))
    
    if not workflow_files:
        print("❌ No workflow files found")
        sys.exit(1)
    
    print(f"Found {len(workflow_files)} workflow files\n")
    
    total_errors = 0
    valid_workflows = 0
    
    # Validate each workflow
    for workflow_path in sorted(workflow_files):
        filename = workflow_path.name
        is_valid, errors = validate_workflow(str(workflow_path))
        
        if is_valid:
            print(f"✅ {filename}")
            valid_workflows += 1
        else:
            print(f"❌ {filename}")
            for error in errors:
                print(f"   - {error}")
                total_errors += 1
    
    # Check consistency
    print("\n🔄 Checking Consistency")
    print("-" * 30)
    consistency_issues = check_consistency(workflows_dir)
    
    if consistency_issues:
        for category, issues in consistency_issues.items():
            print(f"⚠️  {category}:")
            for issue in issues:
                print(f"   - {issue}")
                total_errors += 1
    else:
        print("✅ All workflows are consistent")
    
    # Summary
    print("\n📊 Summary")
    print("=" * 50)
    print(f"Total workflows: {len(workflow_files)}")
    print(f"Valid workflows: {valid_workflows}")
    print(f"Invalid workflows: {len(workflow_files) - valid_workflows}")
    print(f"Total errors: {total_errors}")
    
    if total_errors == 0:
        print("\n✅ All workflows validated successfully!")
        return 0
    else:
        print(f"\n❌ Found {total_errors} errors in workflows")
        return 1


if __name__ == "__main__":
    sys.exit(main())