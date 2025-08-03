#!/bin/bash
#
# Quick Garbage Scan - Rule 13 Enforcement Helper
#
# Purpose: Quick scan for immediate garbage detection
# Usage: ./quick-garbage-scan.sh [options]
# Requirements: Python 3.8+, ripgrep
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings
PROJECT_ROOT="/opt/sutazaiapp"
CONFIDENCE_THRESHOLD="0.8"
RISK_THRESHOLD="safe"
VERBOSE=false
OUTPUT_FORMAT="summary"

# Help function
show_help() {
    cat << EOF
Quick Garbage Scan - Rule 13 Enforcement Helper

USAGE:
    ./quick-garbage-scan.sh [OPTIONS]

OPTIONS:
    -p, --project-root PATH     Project root directory (default: /opt/sutazaiapp)
    -c, --confidence FLOAT      Confidence threshold 0.0-1.0 (default: 0.8)
    -r, --risk LEVEL           Risk threshold: safe|moderate|risky (default: safe)
    -v, --verbose              Enable verbose output
    -f, --format FORMAT        Output format: summary|detailed|json (default: summary)
    -h, --help                 Show this help message

EXAMPLES:
    # Quick scan with default settings
    ./quick-garbage-scan.sh
    
    # Scan specific directory
    ./quick-garbage-scan.sh -p /path/to/project
    
    # More aggressive scan
    ./quick-garbage-scan.sh -c 0.6 -r moderate
    
    # Detailed output
    ./quick-garbage-scan.sh -f detailed -v

EOF
}

# Parse command line arguments
while (( "$#" )); do
    case "$1" in
        -p|--project-root)
            if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
                PROJECT_ROOT="$2"
                shift 2
            else
                echo -e "${RED}Error: Argument for $1 is missing${NC}" >&2
                exit 1
            fi
            ;;
        -c|--confidence)
            if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
                CONFIDENCE_THRESHOLD="$2"
                shift 2
            else
                echo -e "${RED}Error: Argument for $1 is missing${NC}" >&2
                exit 1
            fi
            ;;
        -r|--risk)
            if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
                RISK_THRESHOLD="$2"
                shift 2
            else
                echo -e "${RED}Error: Argument for $1 is missing${NC}" >&2
                exit 1
            fi
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -f|--format)
            if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
                OUTPUT_FORMAT="$2"
                shift 2
            else
                echo -e "${RED}Error: Argument for $1 is missing${NC}" >&2
                exit 1
            fi
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        -*|--*=)
            echo -e "${RED}Error: Unsupported flag $1${NC}" >&2
            exit 1
            ;;
        *)
            echo -e "${RED}Error: Unsupported argument $1${NC}" >&2
            exit 1
            ;;
    esac
done

# Validation
if [[ ! -d "$PROJECT_ROOT" ]]; then
    echo -e "${RED}Error: Project root directory does not exist: $PROJECT_ROOT${NC}" >&2
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is required but not installed${NC}" >&2
    exit 1
fi

if ! command -v rg &> /dev/null; then
    echo -e "${YELLOW}Warning: ripgrep (rg) not found. Reference checking will be slower${NC}" >&2
fi

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENFORCER_SCRIPT="$SCRIPT_DIR/garbage-collection-enforcer.py"

if [[ ! -f "$ENFORCER_SCRIPT" ]]; then
    echo -e "${RED}Error: Garbage collection enforcer script not found: $ENFORCER_SCRIPT${NC}" >&2
    exit 1
fi

# Build command
ENFORCER_CMD=(
    python3 "$ENFORCER_SCRIPT"
    --project-root "$PROJECT_ROOT"
    --dry-run
    --confidence-threshold "$CONFIDENCE_THRESHOLD"
    --risk-threshold "$RISK_THRESHOLD"
)

if [[ "$VERBOSE" == "true" ]]; then
    ENFORCER_CMD+=(--verbose)
fi

# Header
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                   QUICK GARBAGE SCAN - RULE 13                ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Project Root:${NC} $PROJECT_ROOT"
echo -e "${YELLOW}Confidence Threshold:${NC} $CONFIDENCE_THRESHOLD"
echo -e "${YELLOW}Risk Threshold:${NC} $RISK_THRESHOLD"
echo -e "${YELLOW}Output Format:${NC} $OUTPUT_FORMAT"
echo ""

# Run the enforcer
echo -e "${BLUE}Starting garbage scan...${NC}"
echo ""

if [[ "$OUTPUT_FORMAT" == "json" ]]; then
    # JSON output - run silently and output JSON
    "${ENFORCER_CMD[@]}" --output /tmp/quick-garbage-scan.json > /dev/null 2>&1
    cat /tmp/quick-garbage-scan.json
    rm -f /tmp/quick-garbage-scan.json
elif [[ "$OUTPUT_FORMAT" == "detailed" ]]; then
    # Detailed output - show full enforcer output
    "${ENFORCER_CMD[@]}"
else
    # Summary output - capture and format
    TEMP_OUTPUT=$(mktemp)
    "${ENFORCER_CMD[@]}" > "$TEMP_OUTPUT" 2>&1
    
    # Extract key information from output
    ITEMS_FOUND=$(grep "Items Found:" "$TEMP_OUTPUT" | cut -d: -f2 | xargs || echo "0")
    ACTIONABLE_ITEMS=$(grep "Actionable Items:" "$TEMP_OUTPUT" | cut -d: -f2 | xargs || echo "0")
    SPACE_POTENTIAL=$(grep "Space Recovered:" "$TEMP_OUTPUT" | cut -d: -f2 | xargs || echo "0.00 MB")
    SCAN_DURATION=$(grep "Scan Duration:" "$TEMP_OUTPUT" | cut -d: -f2 | xargs || echo "0.00s")
    
    # Show summary
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                        SCAN RESULTS                           ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}📊 Summary:${NC}"
    echo -e "   Total Items Found: ${ITEMS_FOUND}"
    echo -e "   Actionable Items:  ${ACTIONABLE_ITEMS}"
    echo -e "   Potential Space:   ${SPACE_POTENTIAL}"
    echo -e "   Scan Duration:     ${SCAN_DURATION}"
    echo ""
    
    # Show top violations if any
    if grep -q "📊 Top Violations by Size:" "$TEMP_OUTPUT"; then
        echo -e "${YELLOW}🔍 Top Violations:${NC}"
        grep -A 5 "📊 Top Violations by Size:" "$TEMP_OUTPUT" | tail -n +2 | head -n 5
        echo ""
    fi
    
    # Show duplicates if any
    if grep -q "📁 Duplicate Files:" "$TEMP_OUTPUT"; then
        DUPLICATE_COUNT=$(grep "📁 Duplicate Files:" "$TEMP_OUTPUT" | cut -d: -f2 | xargs)
        if [[ "$DUPLICATE_COUNT" != "0" ]]; then
            echo -e "${YELLOW}📁 Duplicate Files Found:${NC} $DUPLICATE_COUNT"
            echo ""
        fi
    fi
    
    # Recommendations
    echo -e "${BLUE}💡 Recommendations:${NC}"
    if [[ "$ACTIONABLE_ITEMS" -gt 0 ]]; then
        echo -e "   • Run with --live to perform actual cleanup"
        echo -e "   • Review detailed report for specific items"
        echo -e "   • Consider running: ${GREEN}./scripts/agents/garbage-collection-enforcer.py --live --risk-threshold safe${NC}"
    else
        echo -e "   • ✅ No immediate action needed"
        echo -e "   • Codebase appears clean according to current thresholds"
    fi
    
    # Show report location
    REPORT_FILE=$(grep "📋 Full report:" "$TEMP_OUTPUT" | cut -d: -f2- | xargs || echo "")
    if [[ -n "$REPORT_FILE" ]]; then
        echo -e "   • Full report: ${REPORT_FILE}"
    fi
    
    rm -f "$TEMP_OUTPUT"
fi

echo ""
echo -e "${GREEN}✅ Quick garbage scan completed${NC}"

# Exit with appropriate code
if [[ "$ACTIONABLE_ITEMS" -gt 10 ]]; then
    echo -e "${YELLOW}⚠️  Warning: High number of actionable items found${NC}"
    exit 2
elif [[ "$ACTIONABLE_ITEMS" -gt 0 ]]; then
    echo -e "${BLUE}ℹ️  Info: Some cleanup opportunities found${NC}"
    exit 1
else
    echo -e "${GREEN}✨ Clean codebase - no garbage detected${NC}"
    exit 0
fi