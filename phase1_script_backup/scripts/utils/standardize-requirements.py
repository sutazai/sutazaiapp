#!/usr/bin/env python3
"""
Standardize all requirements.txt files to use consistent versions
"""
import glob

# Base versions to use
BASE_VERSIONS = {
    'fastapi': '0.115.6',
    'flask': '3.1.0',
    'uvicorn': '0.32.1',
    'pydantic': '2.10.4',
    'pydantic-settings': '2.8.1',
    'sqlalchemy': '2.0.36',
    'psycopg2-binary': '2.9.10',
    'redis': '5.2.1',
    'requests': '2.32.3',
    'httpx': '0.28.1',
    'aiohttp': '3.11.11',
    'cryptography': '44.0.0',
    'prometheus-client': '0.21.1',
    'psutil': '6.1.0',
    'python-multipart': '0.0.19',
    'structlog': '24.4.0',
}

def update_requirements_file(filepath):
    """Update a single requirements file with standardized versions"""
    print(f"Processing: {filepath}")
    
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        updated_lines = []
        changes_made = False
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                updated_lines.append(line)
                continue
            
            # Check if this is a package==version line
            if '==' in line:
                package = line.split('==')[0].strip()
                if package.lower() in BASE_VERSIONS:
                    new_line = f"{package}=={BASE_VERSIONS[package.lower()]}"
                    if new_line != line:
                        print(f"  Updated: {line} -> {new_line}")
                        updated_lines.append(new_line)
                        changes_made = True
                    else:
                        updated_lines.append(line)
                else:
                    updated_lines.append(line)
            else:
                updated_lines.append(line)
        
        if changes_made:
            with open(filepath, 'w') as f:
                f.write('\n'.join(updated_lines) + '\n')
            print(f"  ✓ Updated {filepath}")
        else:
            print(f"  - No changes needed")
            
    except Exception as e:
        print(f"  ✗ Error: {e}")

def main():
    """Find and update all requirements files"""
    # Find all requirements files
    patterns = [
        '/opt/sutazaiapp/**/requirements*.txt',
        '/opt/sutazaiapp/agents/*/requirements.txt',
        '/opt/sutazaiapp/docker/*/requirements.txt',
    ]
    
    all_files = set()
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        all_files.update(files)
    
    # Filter out archives and backups
    requirements_files = [
        f for f in all_files 
        if 'archive' not in f and 'backup' not in f
    ]
    
    print(f"Found {len(requirements_files)} requirements files to process\n")
    
    # Update each file
    for filepath in sorted(requirements_files):
        update_requirements_file(filepath)
    
    print(f"\n✅ Processed {len(requirements_files)} files")
    print("\nNext steps:")
    print("1. Review changes with: git diff")
    print("2. Test services to ensure compatibility")
    print("3. Update Dockerfiles to rebuild with new versions")

if __name__ == "__main__":
    main()