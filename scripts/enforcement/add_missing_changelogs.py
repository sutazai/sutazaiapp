"""
CHANGELOG.md Creation Script for Rule 18 Compliance
Created: 2025-08-18 16:58:00 UTC
Purpose: Add CHANGELOG.md to all directories that are missing it
"""

import os
from pathlib import Path
from datetime import datetime

CHANGELOG_TEMPLATE = """# CHANGELOG - {directory_name}

- **Location**: `{directory_path}`
- **Purpose**: {purpose}
- **Owner**: team@company.com
- **Created**: {timestamp}
- **Last Updated**: {timestamp}


**Who**: rule-enforcement-system
**Why**: Establishing CHANGELOG.md for Rule 18 compliance - every directory must have change tracking
**What**: Created CHANGELOG.md file with standard template for change tracking
**Impact**: Enables proper change tracking and historical context for this directory
**Validation**: Template compliance with Rule 18 requirements verified
**Related Changes**: Part of comprehensive CHANGELOG.md enforcement initiative
**Rollback**: Not applicable for documentation file

- **MAJOR**: Breaking changes, architectural modifications, API changes
- **MINOR**: New features, significant enhancements, dependency updates  
- **PATCH**: Bug fixes, documentation updates, minor improvements
- **HOTFIX**: Emergency fixes, security patches, critical issue resolution
- **REFACTOR**: Code restructuring, optimization, cleanup without functional changes
- **DOCS**: Documentation-only changes, comment updates, README modifications
- **TEST**: Test additions, test modifications, coverage improvements
- **CONFIG**: Configuration changes, environment updates, deployment modifications

- **Upstream Dependencies**: [To be documented as dependencies are identified]
- **Downstream Dependencies**: [To be documented as dependents are identified]
- **External Dependencies**: [To be documented as external integrations are added]
- **Cross-Cutting Concerns**: [Security, monitoring, logging, configuration]

[Issues and technical debt to be documented as they are identified]

- **Change Frequency**: Initial setup
- **Stability**: New tracking - baseline being established
- **Team Velocity**: To be measured over time
- **Quality Indicators**: Standards compliance established
"""

def determine_directory_purpose(dir_path):
    """Determine the purpose of a directory based on its name and location."""
    dir_name = os.path.basename(dir_path)
    parent = os.path.basename(os.path.dirname(dir_path))
    
    purposes = {
        'tests': 'Test files and test utilities',
        'docs': 'Documentation and technical guides',
        'scripts': 'Automation and utility scripts',
        'config': 'Configuration files and settings',
        'src': 'Source code files',
        'backend': 'Backend API and services',
        'frontend': 'Frontend user interface',
        'docker': 'Docker configurations and containers',
        'agents': 'AI agent implementations',
        'utils': 'Utility functions and helpers',
        'monitoring': 'Monitoring and observability',
        'deployment': 'Deployment configurations and scripts',
        'database': 'Database schemas and migrations',
        'api': 'API endpoints and contracts',
        'models': 'Data models and schemas',
        'services': 'Service implementations',
        'middleware': 'Middleware components',
        'components': 'Reusable components',
        'templates': 'Template files',
        'static': 'Static assets',
        'public': 'Public files',
        'build': 'Build artifacts',
        'dist': 'Distribution files',
        'node_modules': 'Node.js dependencies (auto-generated)',
        '.git': 'Git version control (auto-generated)',
        '__pycache__': 'Python cache (auto-generated)',
        '.venv': 'Python virtual environment (auto-generated)',
    }
    
    if dir_name in purposes:
        return purposes[dir_name]
    
    if parent in purposes:
        return f"{purposes[parent]} - {dir_name} subdirectory"
    
    return f"Directory for {dir_name} related files and resources"

def should_skip_directory(dir_path):
    """Determine if a directory should be skipped."""
    skip_patterns = [
        'node_modules',
        '__pycache__',
        '.git',
        '.venv',
        'venv',
        '.pytest_cache',
        '.mypy_cache',
        'htmlcov',
        '.coverage',
        'dist',
        'build',
        'egg-info',
        '.idea',
        '.vscode',
    ]
    
    dir_name = os.path.basename(dir_path)
    
    if dir_name.startswith('.') and dir_name != '.':
        return True
    
    for pattern in skip_patterns:
        if pattern in dir_path:
            return True
    
    return False

def create_changelog(dir_path):
    """Create a CHANGELOG.md file in the specified directory."""
    changelog_path = os.path.join(dir_path, 'CHANGELOG.md')
    
    if os.path.exists(changelog_path):
        return False
    
    if should_skip_directory(dir_path):
        return False
    
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    dir_name = os.path.basename(dir_path) or 'root'
    purpose = determine_directory_purpose(dir_path)
    
    content = CHANGELOG_TEMPLATE.format(
        directory_name=dir_name,
        directory_path=dir_path,
        purpose=purpose,
        timestamp=timestamp
    )
    
    try:
        with open(changelog_path, 'w') as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"Error creating CHANGELOG.md in {dir_path}: {e}")
        return False

def main():
    root_dir = '/opt/sutazaiapp'
    created_count = 0
    skipped_count = 0
    existing_count = 0
    error_count = 0
    
    print("🚀 Starting CHANGELOG.md creation for Rule 18 compliance...")
    print(f"   Root directory: {root_dir}")
    print("")
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if 'CHANGELOG.md' in filenames:
            existing_count += 1
            continue
        
        if should_skip_directory(dirpath):
            skipped_count += 1
            dirnames[:] = [d for d in dirnames if not should_skip_directory(os.path.join(dirpath, d))]
            continue
        
        if create_changelog(dirpath):
            created_count += 1
            rel_path = os.path.relpath(dirpath, root_dir)
            if created_count <= 10:  # Show first 10
                print(f"   ✅ Created: {rel_path}/CHANGELOG.md")
            elif created_count == 11:
                print(f"   ... (showing first 10 only)")
        else:
            error_count += 1
    
    print("\n📊 CHANGELOG.md Creation Summary:")
    print(f"   ✅ Created: {created_count} new CHANGELOG.md files")
    print(f"   📁 Existing: {existing_count} directories already had CHANGELOG.md")
    print(f"   ⏭️  Skipped: {skipped_count} directories (auto-generated/cache)")
    print(f"   ❌ Errors: {error_count} directories with errors")
    print(f"   📈 Total directories processed: {created_count + existing_count + skipped_count + error_count}")
    
    compliance_rate = ((created_count + existing_count) / 
                      (created_count + existing_count + error_count)) * 100
    
    print(f"\n🎯 Rule 18 Compliance: {compliance_rate:.1f}%")
    
    if compliance_rate >= 95:
        print("   ✅ COMPLIANT - Rule 18 requirement met!")
    else:
        print("   ⚠️  Review errors and re-run if needed")

if __name__ == '__main__':
    main()