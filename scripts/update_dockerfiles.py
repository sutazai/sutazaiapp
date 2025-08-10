#!/usr/bin/env python3
"""
ULTRA-THINKING GARBAGE-COLLECTOR EXPERT - Dockerfile Requirements Updater
Updates all Dockerfiles to use the new consolidated requirements structure.
"""

import os
import re
import glob
from pathlib import Path

def update_dockerfile(dockerfile_path):
    """Update a single Dockerfile to use new requirements structure."""
    try:
        with open(dockerfile_path, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Pattern to find COPY requirements lines
        copy_patterns = [
            (r'COPY\s+requirements\.txt\s+\.', 'COPY requirements/base.txt requirements.txt'),
            (r'COPY\s+requirements-dev\.txt\s+\.', 'COPY requirements/dev.txt requirements.txt'),
            (r'COPY\s+requirements_dev\.txt\s+\.', 'COPY requirements/dev.txt requirements.txt'),
            (r'COPY\s+requirements-test\.txt\s+\.', 'COPY requirements/dev.txt requirements.txt'),
            (r'COPY\s+requirements_test\.txt\s+\.', 'COPY requirements/dev.txt requirements.txt'),
            (r'COPY\s+requirements-optional\.txt\s+\.', 'COPY requirements/prod.txt requirements.txt'),
            (r'COPY\s+requirements_optional\.txt\s+\.', 'COPY requirements/prod.txt requirements.txt'),
            (r'COPY\s+requirements-minimal\.txt\s+\.', 'COPY requirements/base.txt requirements.txt'),
            (r'COPY\s+requirements_minimal\.txt\s+\.', 'COPY requirements/base.txt requirements.txt'),
            (r'COPY\s+requirements_consolidated\.txt\s+\.', 'COPY requirements/base.txt requirements.txt'),
            (r'COPY\s+requirements-core\.txt\s+\.', 'COPY requirements/base.txt requirements.txt'),
            (r'COPY\s+requirements_performance\.txt\s+\.', 'COPY requirements/prod.txt requirements.txt'),
        ]
        
        # Apply replacements
        for pattern, replacement in copy_patterns:
            content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
        
        # Handle relative path references to requirements
        relative_patterns = [
            (r'COPY\s+\.\./requirements\.txt\s+\.', 'COPY ../requirements/base.txt requirements.txt'),
            (r'COPY\s+\.\./\.\./requirements\.txt\s+\.', 'COPY ../../requirements/base.txt requirements.txt'),
            (r'COPY\s+\.\./\.\./\.\./requirements\.txt\s+\.', 'COPY ../../../requirements/base.txt requirements.txt'),
        ]
        
        for pattern, replacement in relative_patterns:
            content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
        
        # Determine appropriate requirements file based on context
        dockerfile_dir = str(dockerfile_path).lower()
        
        if 'test' in dockerfile_dir or 'dev' in dockerfile_dir:
            # Use dev requirements for test/dev contexts
            content = re.sub(r'COPY\s+requirements/base\.txt\s+requirements\.txt', 
                           'COPY requirements/dev.txt requirements.txt', content)
        elif 'prod' in dockerfile_dir or 'production' in dockerfile_dir:
            # Use production requirements for production contexts
            content = re.sub(r'COPY\s+requirements/base\.txt\s+requirements\.txt', 
                           'COPY requirements/prod.txt requirements.txt', content)
        
        # Write back if changed
        if content != original_content:
            with open(dockerfile_path, 'w') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error updating {dockerfile_path}: {e}")
        return False

def main():
    """Update all Dockerfiles in the project."""
    root_dir = Path('/opt/sutazaiapp')
    updated_count = 0
    error_count = 0
    
    # Find all Dockerfiles
    dockerfile_patterns = [
        '**/Dockerfile',
        '**/Dockerfile.*'
    ]
    
    dockerfiles = []
    for pattern in dockerfile_patterns:
        dockerfiles.extend(root_dir.glob(pattern))
    
    print(f"Found {len(dockerfiles)} Dockerfiles to update")
    
    for dockerfile_path in dockerfiles:
        try:
            if update_dockerfile(dockerfile_path):
                print(f"✓ Updated: {dockerfile_path}")
                updated_count += 1
            else:
                print(f"- No change: {dockerfile_path}")
        except Exception as e:
            print(f"✗ Error: {dockerfile_path} - {e}")
            error_count += 1
    
    print(f"\nSummary:")
    print(f"  Updated: {updated_count}")
    print(f"  No change: {len(dockerfiles) - updated_count - error_count}")
    print(f"  Errors: {error_count}")
    
    if updated_count > 0:
        print(f"\n✅ Successfully updated {updated_count} Dockerfiles to use consolidated requirements!")
    
    return updated_count > 0

if __name__ == "__main__":
    main()