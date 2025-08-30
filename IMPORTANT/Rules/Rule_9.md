 Rule 9: Single Source Frontend/Backend - Zero Duplication Architecture
Requirement: Maintain absolute architectural clarity with single, authoritative frontend and backend directories, eliminating all duplication and confusion through disciplined version control and feature management.
CRITICAL: One Source of Truth for Each Layer

Single Frontend: Only /frontend directory for all UI code
Single Backend: Only /backend directory for all server code
Zero Tolerance: No v1/, v2/, old/, backup/, deprecated/, or duplicate directories
Version Control: Git branches and tags for versioning, not directory duplication
Feature Management: Feature flags for experiments, not separate codebases

✅ Required Practices:
Mandatory Investigation Before Consolidation:
bash# Comprehensive duplicate detection
find . -type d \( -name "*frontend*" -o -name "*backend*" -o -name "*client*" -o -name "*server*" -o -name "*api*" -o -name "*ui*" \) | grep -v node_modules

# Search for version indicators
find . -type d | grep -E "(v[0-9]+|old|backup|deprecated|legacy|archive|previous|copy)"

# Analyze directory contents for duplication
for dir in $(find . -name "package.json" -o -name "requirements.txt"); do
    echo "Found project root: $(dirname $dir)"
done

# Git history analysis for branching points
git log --all --graph --decorate --oneline | grep -E "(frontend|backend)"
Consolidation Process:
bashconsolidate_to_single_source() {
    local timestamp=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
    
    # Step 1: Inventory all variants
    echo "[$timestamp] Starting consolidation inventory..."
    find . -type d -name "*frontend*" > frontend_variants.txt
    find . -type d -name "*backend*" > backend_variants.txt
    
    # Step 2: Analyze each variant
    for variant in $(cat frontend_variants.txt backend_variants.txt); do
        analyze_variant_purpose "$variant"
        check_unique_features "$variant"
        assess_migration_complexity "$variant"
    done
    
    # Step 3: Create consolidation plan
    create_migration_strategy
    document_feature_differences
    plan_git_branch_structure
    
    # Step 4: Execute consolidation
    merge_to_canonical_directories
    implement_feature_flags
    archive_deprecated_versions
    
    # Step 5: Validate consolidation
    run_comprehensive_tests
    verify_no_functionality_lost
    update_all_references
}
Feature Flag Implementation:
javascript// frontend/src/config/features.js
const FEATURES = {
  EXPERIMENTAL_UI: process.env.REACT_APP_EXPERIMENTAL_UI === 'true',
  BETA_DASHBOARD: process.env.REACT_APP_BETA_DASHBOARD === 'true',
  NEW_AUTH_FLOW: process.env.REACT_APP_NEW_AUTH_FLOW === 'true',
  ADVANCED_ANALYTICS: process.env.REACT_APP_ADVANCED_ANALYTICS === 'true'
};

// Usage in components
if (FEATURES.EXPERIMENTAL_UI) {
  return <ExperimentalComponent />;
} else {
  return <StableComponent />;
}
python# backend/app/config/features.py
from enum import Enum
import os

class FeatureFlags(Enum):
    EXPERIMENTAL_API = os.getenv('EXPERIMENTAL_API', 'false').lower() == 'true'
    BETA_ENDPOINTS = os.getenv('BETA_ENDPOINTS', 'false').lower() == 'true'
    NEW_AUTH_SYSTEM = os.getenv('NEW_AUTH_SYSTEM', 'false').lower() == 'true'
    ADVANCED_CACHING = os.getenv('ADVANCED_CACHING', 'false').lower() == 'true'

# Usage in routes
if FeatureFlags.EXPERIMENTAL_API.value:
    app.include_router(experimental_routes)
Git Branch Strategy:
bash# Main branches
main           # Production-ready code
develop        # Integration branch
staging        # Pre-production testing

# Feature branches
feature/new-dashboard
feature/api-v2-endpoints
experiment/ai-integration
hotfix/critical-bug-fix

# Version tags instead of directories
git tag -a v1.0.0 -m "Version 1.0.0 release"
git tag -a v2.0.0-beta -m "Version 2.0.0 beta"
Directory Structure Enforcement:
/opt/sutazaiapp/
├── frontend/                    # ONLY frontend directory
│   ├── src/
│   │   ├── components/         # Reusable UI components
│   │   ├── pages/             # Page components
│   │   ├── services/          # API service layers
│   │   ├── hooks/             # Custom React hooks
│   │   ├── utils/             # Utility functions
│   │   ├── config/            # Configuration including features
│   │   └── experimental/      # Feature-flagged experimental code
│   ├── public/
│   ├── package.json
│   └── README.md
│
├── backend/                     # ONLY backend directory
│   ├── app/
│   │   ├── api/               # API endpoints
│   │   ├── models/            # Data models
│   │   ├── services/          # Business logic
│   │   ├── utils/             # Utility functions
│   │   ├── config/            # Configuration including features
│   │   └── experimental/      # Feature-flagged experimental code
│   ├── tests/
│   ├── requirements.txt
│   └── README.md
│
└── [NO OTHER FRONTEND/BACKEND DIRECTORIES ALLOWED]
Migration from Duplicates:
bashmigrate_duplicate_codebases() {
    local source_dir="$1"
    local target_dir="$2"
    local timestamp=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
    
    # Create migration record
    cat >> MIGRATION_LOG.md << EOF
## Migration: $source_dir -> $target_dir
**Date**: $timestamp
**Reason**: Consolidating duplicate codebases per Rule 9

### Pre-Migration Analysis
- Unique features identified: [list]
- Dependencies specific to source: [list]
- Configuration differences: [list]

### Migration Steps
1. Backed up source to: /archives/$timestamp/$(basename $source_dir)
2. Identified unique features for feature flags
3. Merged code into $target_dir
4. Updated all references and imports
5. Tested all functionality preserved

### Post-Migration Validation
- All tests passing: ✓
- Feature flags working: ✓
- No functionality lost: ✓
EOF

    # Perform migration
    create_archive "$source_dir"
    extract_unique_features "$source_dir"
    merge_into_target "$target_dir"
    update_all_references
    run_validation_tests
}
🚫 Forbidden Practices:
Directory Duplication Violations:

Creating frontend_v2/, backend_old/, or any versioned directories
Maintaining multiple frontend or backend directories simultaneously
Using directory names to indicate versions or experiments
Creating "backup" directories instead of using Git
Duplicating code instead of using shared libraries
Creating separate directories for different deployment environments
Using copy/paste for experiments instead of branches
Maintaining deprecated code in separate directories

Version Control Violations:

Not using Git branches for experiments and features
Creating new directories for major version changes
Avoiding Git tags for version marking
Not documenting branch purposes and lifespans
Keeping experimental code in main/master branch
Using comments to "version" code instead of Git
Not cleaning up merged feature branches
Creating "archive" directories instead of using Git history

Feature Management Violations:

Hardcoding experimental features without flags
Creating separate apps for A/B testing
Not documenting feature flag purposes
Leaving feature flags permanently enabled
Using compile-time instead of runtime flags
Not having a feature flag retirement plan
Creating duplicate components without flags
Not centralizing feature flag configuration

Investigation Methodology:
Duplicate Detection Process:
bashdetect_all_duplicates() {
    echo "=== Searching for Frontend Duplicates ==="
    find . -type d -name "*frontend*" -o -name "*client*" -o -name "*ui*" | \
        grep -v node_modules | while read dir; do
        echo "Found: $dir"
        echo "  - Size: $(du -sh "$dir" | cut -f1)"
        echo "  - Last modified: $(stat -c %y "$dir" | cut -d' ' -f1)"
        echo "  - Package.json: $([ -f "$dir/package.json" ] && echo "Yes" || echo "No")"
    done
    
    echo "=== Searching for Backend Duplicates ==="
    find . -type d -name "*backend*" -o -name "*server*" -o -name "*api*" | \
        grep -v node_modules | while read dir; do
        echo "Found: $dir"
        echo "  - Size: $(du -sh "$dir" | cut -f1)"
        echo "  - Last modified: $(stat -c %y "$dir" | cut -d' ' -f1)"
        echo "  - Requirements.txt: $([ -f "$dir/requirements.txt" ] && echo "Yes" || echo "No")"
    done
}
Unique Feature Analysis:
bashanalyze_unique_features() {
    local dir1="$1"
    local dir2="$2"
    
    echo "Comparing $dir1 vs $dir2"
    
    # Compare file structures
    diff -qr "$dir1" "$dir2" | grep "Only in" > unique_files.txt
    
    # Compare dependencies
    if [ -f "$dir1/package.json" ] && [ -f "$dir2/package.json" ]; then
        diff <(jq -S '.dependencies' "$dir1/package.json") \
             <(jq -S '.dependencies' "$dir2/package.json")
    fi
    
    # Compare configurations
    find "$dir1" -name "*.config.*" -o -name "*.env*" > config1.txt
    find "$dir2" -name "*.config.*" -o -name "*.env*" > config2.txt
    diff config1.txt config2.txt
    
    # Document findings
    document_unique_features_for_migration
}
Documentation Requirements:
CHANGELOG.md Entry for Consolidation:
markdown### [2024-12-20 15:45:30 UTC] - v2.0.0 - ARCHITECTURE - MAJOR - Frontend/Backend Consolidation
**Who**: DevOps Team (devops@company.com)
**Why**: Eliminate confusion from multiple frontend/backend directories per Rule 9
**What**: 
  - Consolidated frontend_v1/, frontend_v2/, frontend_old/ into /frontend
  - Merged backend/, backend_v2/, api_old/ into /backend  
  - Implemented feature flags for experimental features
  - Created Git tags for version history
  - Archived deprecated code with restoration procedures
**Impact**: 
  - All imports and references updated
  - CI/CD pipelines reconfigured
  - Documentation updated
  - No functionality lost
**Validation**: 
  - All tests passing
  - Feature flags tested
  - Performance benchmarks maintained
Validation Criteria:
Structure Validation:

✓ Only one /frontend directory exists
✓ Only one /backend directory exists
✓ No versioned directories (v1, v2, old, etc.)
✓ No duplicate codebases
✓ Git branches used for versions
✓ Feature flags properly implemented
✓ All references updated
✓ Documentation current

Functionality Validation:

✓ All features preserved during consolidation
✓ No regression in functionality
✓ Feature flags working correctly
✓ Performance maintained or improved
✓ All tests passing
✓ Build processes working
✓ Deployment successful
✓ No broken imports or references

Process Validation:

✓ Investigation completed before consolidation
✓ Unique features identified and preserved
✓ Migration plan documented
✓ Backups created before changes
✓ Team notified of changes
✓ Git history preserved
✓ Feature flag documentation complete
✓ Rollback procedures tested

This expanded Rule 9 provides comprehensive guidance for maintaining single-source architecture while preserving the ability to experiment and version through proper Git usage and feature flags.


*Last Updated: 2025-08-30 00:00:00 UTC - For the infrastructure based in /opt/sutazaiapp/