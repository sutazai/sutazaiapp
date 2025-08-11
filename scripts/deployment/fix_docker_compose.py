#!/usr/bin/env python3
"""
Script to fix docker-compose.yml by removing all deploy: sections
that conflict with mem_limit/cpus directives
"""

import re
import sys

def remove_deploy_sections(content):
    """Remove all deploy: sections from docker-compose content"""
    # Pattern to match deploy: sections with their full content
    deploy_pattern = r'    deploy:\s*\n(?:      .*\n)*?(?=    [a-zA-Z_]|\n  [a-zA-Z_]|\n[a-zA-Z_]|\Z)'
    
    # Remove deploy sections
    cleaned_content = re.sub(deploy_pattern, '', content, flags=re.MULTILINE)
    
    return cleaned_content

def main():
    # Read the docker-compose.yml file
    with open('/opt/sutazaiapp/docker-compose.yml', 'r') as f:
        content = f.read()
    
    print("Original file size:", len(content), "characters")
    
    # Remove deploy sections
    cleaned_content = remove_deploy_sections(content)
    
    print("Cleaned file size:", len(cleaned_content), "characters")
    
    # Write the cleaned content back
    with open('/opt/sutazaiapp/docker-compose.yml', 'w') as f:
        f.write(cleaned_content)
    
    print("✅ Removed all conflicting deploy: sections from docker-compose.yml")

if __name__ == "__main__":
    main()