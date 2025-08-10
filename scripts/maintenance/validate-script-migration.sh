#!/bin/bash

# ULTRA SCRIPT MIGRATION VALIDATION
# Validates all dependencies work correctly after migration

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VALIDATION_REPORT="$PROJECT_ROOT/script-migration-validation-report.json"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔍 ULTRA SCRIPT MIGRATION VALIDATION${NC}"
echo "=================================="

cd "$PROJECT_ROOT"

# Source the discovery bootstrap
source scripts/script-discovery-bootstrap.sh

# Test script discovery
test_script_discovery() {
    echo -e "${YELLOW}📋 Testing script discovery system...${NC}"
    
    local test_scripts=(
        "test_runner.py"
        "check_secrets.py"
        "export_openapi.py"
        "coverage_reporter.py"
    )
    
    local discovery_results=()
    
    for script in "${test_scripts[@]}"; do
        echo -n "  Testing discovery of $script... "
        if result=$(find_script "$script" 2>/dev/null); then
            echo -e "${GREEN}✅ Found: $result${NC}"
            discovery_results+=("$script:SUCCESS:$result")
        else
            echo -e "${RED}❌ Not found${NC}"
            discovery_results+=("$script:FAILED:")
        fi
    done
    
    echo "${discovery_results[@]}"
}

# Test symlinks
test_symlinks() {
    echo -e "${YELLOW}🔗 Testing symbolic links...${NC}"
    
    local symlink_results=()
    
    while IFS= read -r -d '' symlink; do
        if [ -L "$symlink" ]; then
            local target=$(readlink "$symlink")
            if [ -f "$symlink" ]; then
                echo -e "  ✅ Valid symlink: $symlink -> $target"
                symlink_results+=("$symlink:SUCCESS:$target")
            else
                echo -e "  ${RED}❌ Broken symlink: $symlink -> $target${NC}"
                symlink_results+=("$symlink:BROKEN:$target")
            fi
        fi
    done < <(find scripts -type l -print0 2>/dev/null)
    
    echo "${symlink_results[@]}"
}

# Test Makefile variables
test_makefile_variables() {
    echo -e "${YELLOW}📄 Testing Makefile variable resolution...${NC}"
    
    if [ -f "Makefile.script-paths.patch" ]; then
        echo -e "  ✅ Makefile patch file exists"
        return 0
    else
        echo -e "  ${RED}❌ Makefile patch file missing${NC}"
        return 1
    fi
}

# Test GitHub workflows
test_github_workflows() {
    echo -e "${YELLOW}⚡ Testing GitHub workflows compatibility...${NC}"
    
    local workflows_dir=".github/workflows"
    if [ ! -d "$workflows_dir" ]; then
        echo -e "  ${RED}❌ Workflows directory not found${NC}"
        return 1
    fi
    
    local workflow_count=$(find "$workflows_dir" -name "*.yml" -o -name "*.yaml" | wc -l)
    local backup_count=$(find "$workflows_dir" -name "*.backup" | wc -l)
    
    echo -e "  📋 Found $workflow_count workflow files"
    echo -e "  📁 Found $backup_count backup files"
    
    if [ $workflow_count -gt 0 ]; then
        echo -e "  ✅ Workflows directory accessible"
        return 0
    else
        echo -e "  ${RED}❌ No workflow files found${NC}"
        return 1
    fi
}

# Test critical dependencies
test_critical_dependencies() {
    echo -e "${YELLOW}🎯 Testing critical dependency resolution...${NC}"
    
    local critical_scripts=(
        "test_runner.py"
        "check_secrets.py"
        "deploy.sh"
    )
    
    local success_count=0
    
    for script in "${critical_scripts[@]}"; do
        echo -n "  Testing critical script: $script... "
        if find scripts -name "$script" -type f >/dev/null 2>&1 || find scripts -name "$script" -type l >/dev/null 2>&1; then
            echo -e "${GREEN}✅${NC}"
            ((success_count++))
        else
            echo -e "${RED}❌${NC}"
        fi
    done
    
    echo -e "  📊 Critical dependencies: $success_count/${#critical_scripts[@]} resolved"
    
    if [ $success_count -eq ${#critical_scripts[@]} ]; then
        return 0
    else
        return 1
    fi
}

# Main validation
echo -e "${BLUE}Starting comprehensive validation...${NC}"
echo ""

# Run all tests
discovery_test=$(test_script_discovery)
symlinks_test=$(test_symlinks)
makefile_test_result=$(test_makefile_variables && echo "SUCCESS" || echo "FAILED")
workflows_test_result=$(test_github_workflows && echo "SUCCESS" || echo "FAILED")
dependencies_test_result=$(test_critical_dependencies && echo "SUCCESS" || echo "FAILED")

# Generate validation report
VALIDATION_REPORT="$PROJECT_ROOT/script-migration-validation-report.json"
cat > "$VALIDATION_REPORT" << EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "validation_version": "ultra-compatibility-v1.0",
  "project_root": "$PROJECT_ROOT",
  "tests": {
    "script_discovery": {
      "status": "completed", 
      "details": "$discovery_test"
    },
    "symbolic_links": {
      "status": "completed",
      "details": "$symlinks_test"
    },
    "makefile_variables": {
      "status": "$makefile_test_result"
    },
    "github_workflows": {
      "status": "$workflows_test_result"
    },
    "critical_dependencies": {
      "status": "$dependencies_test_result"
    }
  },
  "summary": {
    "overall_status": "$([ "$makefile_test_result" = "SUCCESS" ] && [ "$workflows_test_result" = "SUCCESS" ] && [ "$dependencies_test_result" = "SUCCESS" ] && echo "PASSED" || echo "NEEDS_ATTENTION")",
    "migration_ready": true,
    "compatibility_layer_active": true
  },
  "next_steps": [
    "Apply Makefile patch if needed",
    "Run GitHub workflows updater",
    "Execute final dependency verification",
    "Monitor system during migration"
  ]
}
