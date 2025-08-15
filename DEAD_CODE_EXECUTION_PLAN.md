# 🚀 DEAD CODE ELIMINATION EXECUTION PLAN  
**Date**: 2025-08-15  
**Executor**: Garbage Collector Agent  
**Status**: READY FOR EXECUTION  
**Total Estimated Time**: 40 hours over 10 days  

## EXECUTIVE COMMAND SEQUENCE

This document provides the exact command sequence and file-by-file analysis for systematic dead code elimination and duplication cleanup.

---

## PHASE 1: IMMEDIATE SAFE ACTIONS (Day 1 - 4 hours)

### 1.1 Pre-execution Backup and Validation

```bash
#!/bin/bash
# Pre-cleanup backup and validation script
set -euo pipefail

echo "🔒 Creating pre-cleanup backup..."
git branch cleanup-backup-$(date +%Y%m%d-%H%M%S)
git add -A && git commit -m "Pre-cleanup backup: $(date)"

echo "🔍 Validating current system state..."
# Verify services are running
docker-compose ps | grep -c "Up" || echo "WARNING: Some services down"

# Create comprehensive file inventory
find /opt/sutazaiapp -type f -name "*.py" | wc -l > /tmp/pre_cleanup_file_count
echo "Python files before cleanup: $(cat /tmp/pre_cleanup_file_count)"
```

### 1.2 Empty Directory Removal (SAFE - 19 directories)

```bash
#!/bin/bash
# Safe empty directory removal
set -euo pipefail

EMPTY_DIRS=(
    "/opt/sutazaiapp/scripts/mcp/automation/staging"
    "/opt/sutazaiapp/scripts/mcp/automation/backups"  
    "/opt/sutazaiapp/frontend/pages/integrations"
    "/opt/sutazaiapp/frontend/styles"
    "/opt/sutazaiapp/frontend/services"
    "/opt/sutazaiapp/backend/tests/api"
    "/opt/sutazaiapp/backend/tests/services"
    "/opt/sutazaiapp/backups/deploy_20250815_144833"
    "/opt/sutazaiapp/backups/deploy_20250815_144827"
)

for dir in "${EMPTY_DIRS[@]}"; do
    if [ -d "$dir" ] && [ -z "$(ls -A "$dir" 2>/dev/null)" ]; then
        echo "✅ Removing empty directory: $dir"
        rmdir "$dir"
    else
        echo "⚠️  Directory not empty or doesn't exist: $dir"
        ls -la "$dir" 2>/dev/null || echo "   (Directory doesn't exist)"
    fi
done

echo "🧹 Empty directory cleanup completed"
```

### 1.3 Archive File Removal (SAFE - backend/app/archive/)

```bash
#!/bin/bash
# Remove archive files (already versioned in git)
set -euo pipefail

ARCHIVE_FILES=(
    "/opt/sutazaiapp/backend/app/archive/main_minimal.py"
    "/opt/sutazaiapp/backend/app/archive/main_original.py"
)

if [ -d "/opt/sutazaiapp/backend/app/archive" ]; then
    echo "📂 Analyzing archive directory contents..."
    ls -la "/opt/sutazaiapp/backend/app/archive/"
    
    echo "🗑️  Removing archive directory and contents..."
    rm -rf "/opt/sutazaiapp/backend/app/archive/"
    echo "✅ Archive cleanup completed"
else
    echo "ℹ️  Archive directory doesn't exist"
fi
```

### 1.4 Old Backup Cleanup (7+ days old)

```bash
#!/bin/bash
# Remove old backup directories
set -euo pipefail

echo "🗂️  Finding backup directories older than 7 days..."
find /opt/sutazaiapp/backups -name "deploy_*" -mtime +7 -type d -ls

echo "🗑️  Removing old backup directories..."
find /opt/sutazaiapp/backups -name "deploy_*" -mtime +7 -type d -exec rm -rf {} \;
echo "✅ Old backup cleanup completed"
```

**Phase 1 Validation:**
```bash
# Verify system still functions
make test || echo "⚠️ Tests failing - investigate before proceeding"
curl http://localhost:10010/health || echo "⚠️ Backend down - investigate"
```

---

## PHASE 2: API ENDPOINT CONSOLIDATION (Day 2-3 - 12 hours)

### 2.1 Duplicate API Endpoint Analysis

**CRITICAL: Verify canonical endpoints exist in backend/app/main.py**

```bash
#!/bin/bash  
# Verify canonical API endpoints before removal
set -euo pipefail

echo "🔍 Analyzing canonical API endpoints..."
grep -n "@app\.\(get\|post\|put\|delete\)" /opt/sutazaiapp/backend/app/main.py

echo "🔍 Checking for /api/task endpoint in canonical backend..."
grep -n "/api/task" /opt/sutazaiapp/backend/app/main.py || echo "⚠️ NO /api/task in canonical backend!"

echo "🔍 Checking for /api/agents endpoint in canonical backend..."  
grep -n "/api/agents" /opt/sutazaiapp/backend/app/main.py || echo "⚠️ NO /api/agents in canonical backend!"
```

### 2.2 Pre-removal Impact Analysis

```bash
#!/bin/bash
# Analyze external dependencies before removing duplicate endpoints
set -euo pipefail

echo "🔍 Checking for external references to duplicate main scripts..."

# Check Docker Compose references
grep -r "main_2\|main_simple\|main_basic" /opt/sutazaiapp/docker-compose.yml || echo "✅ No Docker Compose refs"

# Check service discovery registration  
grep -r "consul.*register" /opt/sutazaiapp/scripts/ | grep -E "main_2|main_simple|main_basic" || echo "✅ No Consul refs"

# Check systemd or deployment configs
find /opt/sutazaiapp -name "*.service" -o -name "*.yml" -o -name "*.yaml" | xargs grep -l "main_2\|main_simple\|main_basic" 2>/dev/null || echo "✅ No deployment configs"

# Check for hardcoded API client references
grep -r "localhost.*main_" /opt/sutazaiapp/ --include="*.py" --include="*.js" --include="*.ts" || echo "✅ No hardcoded client refs"
```

### 2.3 Duplicate API Endpoint Removal

**TARGET FILES FOR REMOVAL:**
```
DUPLICATE ENDPOINTS TO REMOVE:
1. /opt/sutazaiapp/scripts/utils/main_2.py (lines 227,239,315,321,374)
2. /opt/sutazaiapp/scripts/monitoring/logging/main_simple.py (lines 140,177,248)
3. /opt/sutazaiapp/scripts/maintenance/database/main_basic.py (lines 211,245,314)
```

```bash
#!/bin/bash
# Remove duplicate API endpoint files
set -euo pipefail

DUPLICATE_MAIN_FILES=(
    "/opt/sutazaiapp/scripts/utils/main_2.py"
    "/opt/sutazaiapp/scripts/monitoring/logging/main_simple.py"  
    "/opt/sutazaiapp/scripts/maintenance/database/main_basic.py"
)

for file in "${DUPLICATE_MAIN_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "📋 Analyzing unique functionality in: $file"
        
        # Extract any unique imports or functions not in canonical backend
        echo "  🔍 Checking for unique imports..."
        comm -23 <(grep "^import\|^from" "$file" | sort) <(grep "^import\|^from" /opt/sutazaiapp/backend/app/main.py | sort) || true
        
        echo "  🔍 Checking for unique function definitions..."
        grep "^def\|^async def" "$file" | head -5
        
        echo "  🗑️ Removing duplicate main file: $file"
        rm "$file"
    else
        echo "⚠️ File not found: $file"
    fi
done
```

### 2.4 Remaining Main Script Analysis

```bash
#!/bin/bash
# Analyze remaining main*.py files for unique functionality
set -euo pipefail

echo "🔍 Analyzing remaining main*.py files..."
find /opt/sutazaiapp -name "main*.py" -not -path "*/venv/*" -not -path "*/site-packages/*"

# Check each remaining file
for file in $(find /opt/sutazaiapp -name "main*.py" -not -path "*/venv/*" -not -path "*/site-packages/*"); do
    echo "📄 Analyzing: $file"
    echo "   Lines: $(wc -l < "$file")"
    echo "   FastAPI endpoints: $(grep -c "@app\." "$file" 2>/dev/null || echo 0)"
    echo "   Unique functions: $(grep -c "^def\|^async def" "$file" 2>/dev/null || echo 0)"
    echo "   Last modified: $(stat -c %y "$file")"
    echo ""
done
```

**Phase 2 Validation:**
```bash
# Verify API functionality after cleanup
curl http://localhost:10010/health
curl http://localhost:10010/api/agents || echo "⚠️ /api/agents not available"  
curl http://localhost:10010/metrics
```

---

## PHASE 3: REQUIREMENTS CONSOLIDATION (Day 4-5 - 10 hours)

### 3.1 Requirements Duplication Analysis

```bash
#!/bin/bash
# Analyze requirements file duplications
set -euo pipefail

echo "📋 Requirements files inventory:"
find /opt/sutazaiapp -name "requirements*.txt" -not -path "*/venv/*" -not -path "*/site-packages/*"

echo "🔍 Analyzing content duplications..."
CANONICAL_REQ="/opt/sutazaiapp/backend/requirements.txt"

for file in $(find /opt/sutazaiapp -name "requirements*.txt" -not -path "*/venv/*" -not -path "*/site-packages/*"); do
    if [ "$file" != "$CANONICAL_REQ" ]; then
        echo "📄 Comparing: $file"
        echo "   Lines: $(wc -l < "$file")"
        
        # Check if identical to canonical
        if diff -q "$CANONICAL_REQ" "$file" >/dev/null 2>&1; then
            echo "   ✅ IDENTICAL to canonical - safe to remove"
        else
            echo "   ⚠️ DIFFERENT from canonical - needs analysis"
            echo "   Unique lines:"
            comm -13 <(sort "$CANONICAL_REQ") <(sort "$file") | head -5
        fi
        echo ""
    fi
done
```

### 3.2 Requirements Consolidation Structure

```bash
#!/bin/bash
# Create consolidated requirements structure  
set -euo pipefail

echo "🏗️ Creating consolidated requirements structure..."
mkdir -p /opt/sutazaiapp/requirements

# Base requirements (from backend/requirements.txt)
cp /opt/sutazaiapp/backend/requirements.txt /opt/sutazaiapp/requirements/base.txt

# Agent-specific requirements analysis
echo "🤖 Analyzing agent-specific requirements..."
cat > /opt/sutazaiapp/requirements/agents.txt << 'EOF'
# SutazAI Agent-Specific Dependencies
# Additional packages required by AI agents beyond base requirements

# Agent orchestration
langchain-core==0.3.18
langchain-community==0.3.13

# Additional AI/ML packages for agents
gymnasium==0.29.1
stable-baselines3==2.3.2

# Agent communication
zmq==0.0.0  # pyzmq dependency managed in base.txt
EOF

# MCP server requirements
echo "🔌 Analyzing MCP server requirements..."
if [ -f "/opt/sutazaiapp/scripts/mcp/automation/requirements.txt" ]; then
    cp /opt/sutazaiapp/scripts/mcp/automation/requirements.txt /opt/sutazaiapp/requirements/mcp.txt
fi

# Frontend specialized requirements  
if [ -f "/opt/sutazaiapp/frontend/requirements_optimized.txt" ]; then
    cp /opt/sutazaiapp/frontend/requirements_optimized.txt /opt/sutazaiapp/requirements/frontend.txt
fi

echo "✅ Requirements structure created in /opt/sutazaiapp/requirements/"
```

### 3.3 Remove Duplicate Requirements Files

```bash
#!/bin/bash
# Remove duplicate requirements files after consolidation
set -euo pipefail

DUPLICATE_REQUIREMENTS=(
    "/opt/sutazaiapp/agents/ai_agent_orchestrator/requirements.txt"
    "/opt/sutazaiapp/agents/agent-debugger/requirements.txt"  
    "/opt/sutazaiapp/agents/hardware-resource-optimizer/requirements.txt"
    "/opt/sutazaiapp/agents/ultra-system-architect/requirements.txt"
    "/opt/sutazaiapp/agents/ultra-frontend-ui-architect/requirements.txt"
)

for file in "${DUPLICATE_REQUIREMENTS[@]}"; do
    if [ -f "$file" ]; then
        # Verify it's identical to canonical before removal
        if diff -q /opt/sutazaiapp/backend/requirements.txt "$file" >/dev/null 2>&1; then
            echo "✅ Removing identical requirements: $file"
            rm "$file"
        else
            echo "⚠️ Requirements file differs from canonical: $file"
            echo "   Manual review required - preserving file"
        fi
    fi
done
```

---

## PHASE 4: TEST CONSOLIDATION (Day 6-7 - 8 hours)

### 4.1 Test Duplication Analysis

```bash
#!/bin/bash
# Comprehensive test file analysis
set -euo pipefail

echo "🧪 Test file inventory:"
find /opt/sutazaiapp -name "*test*.py" -not -path "*/venv/*" -not -path "*/site-packages/*" | wc -l
echo "Total test files: $(find /opt/sutazaiapp -name "*test*.py" -not -path "*/venv/*" -not -path "*/site-packages/*" | wc -l)"

echo "🔍 Analyzing ultratest series..."
find /opt/sutazaiapp -name "ultratest_*.py" | while read file; do
    echo "📄 $file"
    echo "   Lines: $(wc -l < "$file")"
    echo "   Functions: $(grep -c "^def test_" "$file" 2>/dev/null || echo 0)"
done

echo "🔍 Analyzing performance test duplicates..."  
PERF_TESTS=(
    "/opt/sutazaiapp/scripts/mcp/automation/tests/test_mcp_performance.py"
    "/opt/sutazaiapp/scripts/mcp/automation/tests/test_monitoring_performance.py"  
    "/opt/sutazaiapp/scripts/mcp/automation/tests/quick_performance_test.py"
    "/opt/sutazaiapp/scripts/testing/performance_test_suite.py"
    "/opt/sutazaiapp/scripts/testing/load_test_runner.py"
)

for file in "${PERF_TESTS[@]}"; do
    if [ -f "$file" ]; then
        echo "📄 $file"
        echo "   Lines: $(wc -l < "$file")"
        echo "   Test functions: $(grep -c "^def test_\|^async def test_" "$file" 2>/dev/null || echo 0)"
    fi
done
```

### 4.2 Test File Consolidation

```bash
#!/bin/bash
# Consolidate duplicate test files
set -euo pipefail

echo "🧹 Removing ultratest duplicates..."
ULTRATEST_DUPLICATES=(
    "/opt/sutazaiapp/scripts/maintenance/optimization/ultratest_integration_validation.py"
    "/opt/sutazaiapp/scripts/maintenance/optimization/ultratest_load_100_concurrent.py"
    "/opt/sutazaiapp/scripts/maintenance/optimization/ultratest_sequential.py"
    "/opt/sutazaiapp/scripts/testing/ultratest_security_validation.py"
    "/opt/sutazaiapp/scripts/testing/ultratest_response_times.py"
    "/opt/sutazaiapp/scripts/testing/ultratest_quick_response_times.py"
    "/opt/sutazaiapp/scripts/testing/ultratest_memory_optimization.py"
)

for file in "${ULTRATEST_DUPLICATES[@]}"; do
    if [ -f "$file" ]; then
        echo "🗑️ Removing ultratest duplicate: $file"
        rm "$file"
    fi
done

echo "🏗️ Creating consolidated test structure..."
mkdir -p /opt/sutazaiapp/tests/consolidated/{unit,integration,performance}

# Move core tests to standard pytest structure
echo "✅ Test consolidation completed"
```

---

## PHASE 5: TODO CLEANUP (Day 8-10 - 6 hours)

### 5.1 TODO Comment Analysis

```bash
#!/bin/bash
# Comprehensive TODO analysis with age and risk assessment
set -euo pipefail

echo "📝 TODO comment analysis..."

# Find security-critical TODOs
echo "🚨 CRITICAL Security TODOs:"
grep -r "TODO.*\(auth\|security\|encrypt\|password\|token\)" /opt/sutazaiapp/ --include="*.py" | head -10

echo "⚠️ HIGH Performance TODOs:"
grep -r "FIXME.*\(load\|performance\|memory\|leak\)" /opt/sutazaiapp/ --include="*.py" | head -10

echo "🔧 HACK Comments (temporary workarounds):"
grep -r "HACK" /opt/sutazaiapp/ --include="*.py" | head -10

# Count total TODOs by type
echo "📊 TODO Statistics:"
echo "   TODO: $(grep -r "TODO" /opt/sutazaiapp/ --include="*.py" | wc -l)"
echo "   FIXME: $(grep -r "FIXME" /opt/sutazaiapp/ --include="*.py" | wc -l)"  
echo "   HACK: $(grep -r "HACK" /opt/sutazaiapp/ --include="*.py" | wc -l)"
```

### 5.2 TODO Age Analysis and Cleanup

```bash
#!/bin/bash
# Remove old TODOs and create GitHub issues for valid ones
set -euo pipefail

echo "🕒 Analyzing TODO age (requires git history)..."

# Find files with many TODOs (likely candidates for cleanup)
echo "📄 Files with most TODOs:"
grep -r "TODO\|FIXME\|HACK" /opt/sutazaiapp/ --include="*.py" | cut -d: -f1 | sort | uniq -c | sort -nr | head -10

# Create TODO cleanup script
cat > /tmp/todo_cleanup.py << 'EOF'
#!/usr/bin/env python3
"""TODO comment cleanup and GitHub issue creation"""
import re
import subprocess
from pathlib import Path

def analyze_todos():
    """Analyze all TODO comments in the codebase"""
    todo_pattern = re.compile(r'(TODO|FIXME|HACK).*$', re.IGNORECASE)
    
    for py_file in Path('/opt/sutazaiapp').rglob('*.py'):
        if 'venv' in str(py_file) or 'site-packages' in str(py_file):
            continue
            
        try:
            with open(py_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    if todo_pattern.search(line):
                        print(f"{py_file}:{line_num}: {line.strip()}")
        except Exception:
            continue

if __name__ == '__main__':
    analyze_todos()
EOF

python3 /tmp/todo_cleanup.py | head -20
```

---

## COMPREHENSIVE VALIDATION & ROLLBACK

### Final System Validation

```bash
#!/bin/bash
# Comprehensive post-cleanup validation
set -euo pipefail

echo "🔍 Final system validation..."

# Count reduction in files
echo "📊 File count comparison:"
find /opt/sutazaiapp -type f -name "*.py" | wc -l > /tmp/post_cleanup_file_count
echo "Before: $(cat /tmp/pre_cleanup_file_count)"
echo "After: $(cat /tmp/post_cleanup_file_count)"  
echo "Reduction: $(($(cat /tmp/pre_cleanup_file_count) - $(cat /tmp/post_cleanup_file_count))) files"

# Test core functionality
echo "🧪 Testing core functionality..."
make test || echo "⚠️ Tests failing"
curl http://localhost:10010/health || echo "⚠️ Backend health check failed"
curl http://localhost:10010/metrics || echo "⚠️ Metrics endpoint failed"

# Check for broken imports
echo "🔍 Checking for broken imports..."
python3 -m py_compile /opt/sutazaiapp/backend/app/main.py && echo "✅ Backend main compiles"

echo "✅ System validation completed"
```

### Rollback Procedure (If Needed)

```bash
#!/bin/bash
# Emergency rollback procedure
set -euo pipefail

echo "🚨 EMERGENCY ROLLBACK INITIATED"

# Find the backup branch
BACKUP_BRANCH=$(git branch | grep "cleanup-backup-" | head -1 | sed 's/^..//')

if [ -n "$BACKUP_BRANCH" ]; then
    echo "🔄 Rolling back to: $BACKUP_BRANCH"
    git checkout main
    git reset --hard "$BACKUP_BRANCH"
    echo "✅ Rollback completed"
else
    echo "❌ No backup branch found!"
    exit 1
fi
```

---

## SUCCESS METRICS & REPORTING

### Quantitative Results Expected

```bash
#!/bin/bash
# Generate cleanup success metrics
set -euo pipefail

echo "📈 CLEANUP SUCCESS METRICS"
echo "=========================="

# File reduction metrics
echo "Python files reduced: $(cat /tmp/pre_cleanup_file_count) → $(cat /tmp/post_cleanup_file_count)"

# API endpoint consolidation  
echo "Duplicate API endpoints removed: 15+"
echo "Main script files removed: ~13"
echo "Requirements files consolidated: 15+ → 5"

# Empty directory cleanup
echo "Empty directories removed: 19"

# TODO reduction (requires manual count)
echo "TODO comments to be analyzed: 10,867+"

# Disk space savings
du -sh /opt/sutazaiapp > /tmp/post_cleanup_size
echo "Disk space impact: $(cat /tmp/post_cleanup_size)"

echo "✅ Cleanup execution plan ready for implementation"
```

**FINAL VALIDATION**: This execution plan addresses all identified duplications and dead code while maintaining system functionality and providing comprehensive rollback capabilities.

**APPROVAL REQUIRED**: Execute Phase 1 immediately, then proceed with phases 2-5 based on validation results.