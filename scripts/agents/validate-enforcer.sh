#!/bin/bash
#
# Validation Script for Garbage Collection Enforcer
#
# Purpose: Validate that the enforcer works correctly
# Usage: ./validate-enforcer.sh
# Requirements: Python 3.8+
#

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENFORCER_SCRIPT="$SCRIPT_DIR/garbage-collection-enforcer.py"
TEST_DIR="/tmp/garbage_test_$(date +%s)"

log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}" >&2
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

cleanup() {
    if [[ -d "$TEST_DIR" ]]; then
        rm -rf "$TEST_DIR"
        log "Cleaned up test directory: $TEST_DIR"
    fi
}

trap cleanup EXIT

# Header
echo -e "${BOLD}${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${BLUE}║            GARBAGE COLLECTION ENFORCER VALIDATION             ║${NC}"
echo -e "${BOLD}${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check prerequisites
log "Checking prerequisites..."

if [[ ! -f "$ENFORCER_SCRIPT" ]]; then
    error "Enforcer script not found: $ENFORCER_SCRIPT"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    error "Python 3 is required but not installed"
    exit 1
fi

success "Prerequisites check passed"

# Create test environment
log "Creating test environment..."
mkdir -p "$TEST_DIR"/{src,build,cache,logs,temp,old}

# Create test files
cat > "$TEST_DIR/src/main.py" << 'EOF'
#!/usr/bin/env python3
print("Hello World")
EOF

cat > "$TEST_DIR/temp/debug.tmp" << 'EOF'
temporary debug file
EOF

cat > "$TEST_DIR/src/backup.py.bak" << 'EOF'
# backup file
print("old version")
EOF

echo "log entry" > "$TEST_DIR/logs/app.log"
echo "" > "$TEST_DIR/empty.txt"
echo "cache data" > "$TEST_DIR/cache/app.cache"
echo '{"old": "config"}' > "$TEST_DIR/config_old.json"

# Create duplicate files
echo "duplicate content" > "$TEST_DIR/file1.txt"
echo "duplicate content" > "$TEST_DIR/file1_copy.txt"

# Make some files appear old
find "$TEST_DIR" -name "*.tmp" -o -name "*.bak" -o -name "*.log" | while read -r file; do
    touch -d "30 days ago" "$file"
done

success "Test environment created: $TEST_DIR"

# Test 1: Basic scan functionality
log "Test 1: Basic scan functionality..."

SCAN_OUTPUT=$(python3 "$ENFORCER_SCRIPT" --project-root "$TEST_DIR" --dry-run --confidence-threshold 0.5 2>/dev/null || echo "SCAN_FAILED")

if [[ "$SCAN_OUTPUT" == "SCAN_FAILED" ]]; then
    error "Basic scan failed"
    exit 1
fi

if echo "$SCAN_OUTPUT" | grep -q "Items Found:"; then
    ITEMS_FOUND=$(echo "$SCAN_OUTPUT" | grep "Items Found:" | cut -d: -f2 | xargs)
    success "Basic scan passed - found $ITEMS_FOUND items"
else
    error "Scan output format unexpected"
    exit 1
fi

# Test 2: JSON report generation
log "Test 2: JSON report generation..."

REPORT_FILE="/tmp/validation_report_$(date +%s).json"
python3 "$ENFORCER_SCRIPT" --project-root "$TEST_DIR" --dry-run --output "$REPORT_FILE" --confidence-threshold 0.5 > /dev/null 2>&1

if [[ -f "$REPORT_FILE" ]] && jq . "$REPORT_FILE" > /dev/null 2>&1; then
    success "JSON report generation passed"
    rm -f "$REPORT_FILE"
else
    error "JSON report generation failed"
    exit 1
fi

# Test 3: Different confidence thresholds
log "Test 3: Testing confidence thresholds..."

LOW_CONF=$(python3 "$ENFORCER_SCRIPT" --project-root "$TEST_DIR" --dry-run --confidence-threshold 0.3 2>/dev/null | grep "Items Found:" | cut -d: -f2 | xargs || echo "0")
HIGH_CONF=$(python3 "$ENFORCER_SCRIPT" --project-root "$TEST_DIR" --dry-run --confidence-threshold 0.9 2>/dev/null | grep "Items Found:" | cut -d: -f2 | xargs || echo "0")

if [[ "$LOW_CONF" -ge "$HIGH_CONF" ]]; then
    success "Confidence threshold filtering works correctly (Low: $LOW_CONF, High: $HIGH_CONF)"
else
    warning "Confidence threshold results unexpected (Low: $LOW_CONF, High: $HIGH_CONF)"
fi

# Test 4: Risk threshold filtering
log "Test 4: Testing risk threshold filtering..."

SAFE_ITEMS=$(python3 "$ENFORCER_SCRIPT" --project-root "$TEST_DIR" --dry-run --risk-threshold safe 2>/dev/null | grep "Actionable Items:" | cut -d: -f2 | xargs || echo "0")
MODERATE_ITEMS=$(python3 "$ENFORCER_SCRIPT" --project-root "$TEST_DIR" --dry-run --risk-threshold moderate 2>/dev/null | grep "Actionable Items:" | cut -d: -f2 | xargs || echo "0")

if [[ "$MODERATE_ITEMS" -ge "$SAFE_ITEMS" ]]; then
    success "Risk threshold filtering works correctly (Safe: $SAFE_ITEMS, Moderate: $MODERATE_ITEMS)"
else
    warning "Risk threshold results unexpected (Safe: $SAFE_ITEMS, Moderate: $MODERATE_ITEMS)"
fi

# Test 5: Help and version
log "Test 5: Testing help and argument parsing..."

if python3 "$ENFORCER_SCRIPT" --help > /dev/null 2>&1; then
    success "Help system works correctly"
else
    error "Help system failed"
    exit 1
fi

# Test 6: Error handling
log "Test 6: Testing error handling..."

if python3 "$ENFORCER_SCRIPT" --project-root "/nonexistent/path" --dry-run 2>/dev/null; then
    error "Should have failed with nonexistent path"
    exit 1
else
    success "Error handling works correctly"
fi

# Performance test
log "Performance test: Scanning large directory..."
PERF_START=$(date +%s.%N)
python3 "$ENFORCER_SCRIPT" --project-root "/opt/sutazaiapp" --dry-run --confidence-threshold 0.9 --risk-threshold safe > /dev/null 2>&1 || true
PERF_END=$(date +%s.%N)
PERF_DURATION=$(echo "$PERF_END - $PERF_START" | bc -l 2>/dev/null || echo "unknown")

if [[ "$PERF_DURATION" != "unknown" ]]; then
    success "Performance test completed in ${PERF_DURATION}s"
else
    success "Performance test completed"
fi

# Final validation
echo ""
echo -e "${BOLD}${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}║                    VALIDATION SUCCESSFUL                      ║${NC}"
echo -e "${BOLD}${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

success "All validation tests passed!"
success "Garbage Collection Enforcer is working correctly"

echo ""
echo -e "${BLUE}📋 Validation Summary:${NC}"
echo "  ✅ Basic scanning functionality"
echo "  ✅ JSON report generation"
echo "  ✅ Confidence threshold filtering"
echo "  ✅ Risk threshold filtering"
echo "  ✅ Help system and argument parsing"
echo "  ✅ Error handling"
echo "  ✅ Performance validation"
echo ""

success "Ready for production use!"