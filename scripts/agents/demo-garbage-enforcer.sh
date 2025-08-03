#!/bin/bash
#
# Demonstration Script for Garbage Collection Enforcer
#
# Purpose: Show different usage scenarios and capabilities
# Usage: ./demo-garbage-enforcer.sh [scenario]
# Requirements: Python 3.8+, ripgrep
#

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENFORCER_SCRIPT="$SCRIPT_DIR/garbage-collection-enforcer.py"
PROJECT_ROOT="/opt/sutazaiapp"

print_header() {
    echo -e "${BOLD}${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${BLUE}║                GARBAGE COLLECTION ENFORCER DEMO               ║${NC}"
    echo -e "${BOLD}${BLUE}║                    Rule 13: No Garbage, No Rot                ║${NC}"
    echo -e "${BOLD}${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_scenario() {
    echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}${BOLD} $1${NC}"
    echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

wait_for_keypress() {
    echo -e "${YELLOW}Press any key to continue...${NC}"
    read -n 1 -s
    echo ""
}

run_demo_scenario() {
    local scenario="$1"
    local description="$2"
    local command="$3"
    
    print_scenario "SCENARIO: $scenario"
    echo -e "${PURPLE}Description:${NC} $description"
    echo ""
    echo -e "${YELLOW}Command:${NC} $command"
    echo ""
    
    if [[ "${DEMO_INTERACTIVE:-true}" == "true" ]]; then
        wait_for_keypress
    fi
    
    echo -e "${BLUE}Executing...${NC}"
    echo ""
    
    # Execute the command
    eval "$command"
    
    echo ""
    echo -e "${GREEN}✅ Scenario completed${NC}"
    echo ""
    
    if [[ "${DEMO_INTERACTIVE:-true}" == "true" ]]; then
        wait_for_keypress
    fi
}

demo_quick_scan() {
    run_demo_scenario \
        "Quick Safety Scan" \
        "Perform a quick, safe scan to identify obvious garbage files" \
        "python3 '$ENFORCER_SCRIPT' --project-root '$PROJECT_ROOT' --dry-run --confidence-threshold 0.9 --risk-threshold safe"
}

demo_comprehensive_scan() {
    run_demo_scenario \
        "Comprehensive Analysis" \
        "Detailed scan with medium confidence to find more potential garbage" \
        "python3 '$ENFORCER_SCRIPT' --project-root '$PROJECT_ROOT' --dry-run --confidence-threshold 0.6 --risk-threshold moderate --verbose"
}

demo_specific_directory() {
    local target_dir="$PROJECT_ROOT/data"
    
    if [[ -d "$target_dir" ]]; then
        run_demo_scenario \
            "Specific Directory Scan" \
            "Scan only the data directory for garbage files" \
            "python3 '$ENFORCER_SCRIPT' --project-root '$target_dir' --dry-run --confidence-threshold 0.7"
    else
        echo -e "${YELLOW}Skipping specific directory demo - $target_dir not found${NC}"
    fi
}

demo_json_report() {
    local report_file="/tmp/garbage_demo_report_$(date +%s).json"
    
    run_demo_scenario \
        "JSON Report Generation" \
        "Generate a detailed JSON report for integration with other tools" \
        "python3 '$ENFORCER_SCRIPT' --project-root '$PROJECT_ROOT' --dry-run --output '$report_file' --confidence-threshold 0.8"
    
    if [[ -f "$report_file" ]]; then
        echo -e "${CYAN}Report generated at: $report_file${NC}"
        
        if command -v jq &> /dev/null; then
            echo -e "${YELLOW}Sample report content:${NC}"
            jq -r '.analysis | to_entries[] | "\(.key): \(.value)"' "$report_file" | head -10
        fi
        
        rm -f "$report_file"
    fi
}

demo_different_thresholds() {
    print_scenario "Threshold Comparison"
    echo -e "${PURPLE}Description:${NC} Compare results with different confidence and risk thresholds"
    echo ""
    
    local results_file="/tmp/threshold_comparison.txt"
    
    echo -e "${BLUE}Running scans with different thresholds...${NC}"
    echo ""
    
    # Conservative scan
    echo -e "${YELLOW}Conservative (High Confidence, Safe Risk):${NC}"
    CONSERVATIVE=$(python3 "$ENFORCER_SCRIPT" --project-root "$PROJECT_ROOT" --dry-run --confidence-threshold 0.9 --risk-threshold safe 2>/dev/null | grep -E "(Items Found|Actionable Items)" || echo "No data")
    echo "$CONSERVATIVE"
    echo ""
    
    # Balanced scan
    echo -e "${YELLOW}Balanced (Medium Confidence, Moderate Risk):${NC}"
    BALANCED=$(python3 "$ENFORCER_SCRIPT" --project-root "$PROJECT_ROOT" --dry-run --confidence-threshold 0.7 --risk-threshold moderate 2>/dev/null | grep -E "(Items Found|Actionable Items)" || echo "No data")
    echo "$BALANCED"
    echo ""
    
    # Aggressive scan
    echo -e "${YELLOW}Aggressive (Low Confidence, Risky):${NC}"
    AGGRESSIVE=$(python3 "$ENFORCER_SCRIPT" --project-root "$PROJECT_ROOT" --dry-run --confidence-threshold 0.5 --risk-threshold risky 2>/dev/null | grep -E "(Items Found|Actionable Items)" || echo "No data")
    echo "$AGGRESSIVE"
    echo ""
    
    echo -e "${GREEN}✅ Threshold comparison completed${NC}"
    echo ""
    
    if [[ "${DEMO_INTERACTIVE:-true}" == "true" ]]; then
        wait_for_keypress
    fi
}

demo_git_integration() {
    if ! git -C "$PROJECT_ROOT" rev-parse --git-dir &> /dev/null; then
        echo -e "${YELLOW}Skipping git integration demo - not a git repository${NC}"
        return
    fi
    
    run_demo_scenario \
        "Git Integration" \
        "Demonstrate git-aware scanning that considers file tracking status" \
        "python3 '$ENFORCER_SCRIPT' --project-root '$PROJECT_ROOT' --dry-run --confidence-threshold 0.7 --verbose | grep -A 5 -B 5 'git\\|tracked\\|staged'"
}

demo_performance_analysis() {
    print_scenario "Performance Analysis"
    echo -e "${PURPLE}Description:${NC} Analyze the performance of different scan configurations"
    echo ""
    
    echo -e "${BLUE}Running performance tests...${NC}"
    echo ""
    
    # Quick scan timing
    echo -e "${YELLOW}Quick Scan (High Threshold):${NC}"
    time (python3 "$ENFORCER_SCRIPT" --project-root "$PROJECT_ROOT" --dry-run --confidence-threshold 0.9 --risk-threshold safe > /dev/null 2>&1)
    echo ""
    
    # Comprehensive scan timing
    echo -e "${YELLOW}Comprehensive Scan (Low Threshold):${NC}"
    time (python3 "$ENFORCER_SCRIPT" --project-root "$PROJECT_ROOT" --dry-run --confidence-threshold 0.5 --risk-threshold moderate > /dev/null 2>&1)
    echo ""
    
    echo -e "${GREEN}✅ Performance analysis completed${NC}"
    echo ""
    
    if [[ "${DEMO_INTERACTIVE:-true}" == "true" ]]; then
        wait_for_keypress
    fi
}

demo_helper_scripts() {
    print_scenario "Helper Scripts Demonstration"
    echo -e "${PURPLE}Description:${NC} Show the helper scripts in action"
    echo ""
    
    # Quick scan helper
    if [[ -f "$SCRIPT_DIR/quick-garbage-scan.sh" ]]; then
        echo -e "${YELLOW}Quick Garbage Scan Helper:${NC}"
        echo "Command: ./scripts/agents/quick-garbage-scan.sh --confidence 0.9 --risk safe"
        echo ""
        
        if [[ "${DEMO_INTERACTIVE:-true}" == "true" ]]; then
            echo -e "${CYAN}Run quick scan helper? [y/N]:${NC} "
            read -r response
            if [[ "$response" =~ ^[Yy]$ ]]; then
                "$SCRIPT_DIR/quick-garbage-scan.sh" --confidence 0.9 --risk safe
            fi
        fi
    fi
    
    echo ""
    
    # Automation helper
    if [[ -f "$SCRIPT_DIR/enforce-rule13-automation.sh" ]]; then
        echo -e "${YELLOW}Automation Script Helper:${NC}"
        echo "Command: ./scripts/agents/enforce-rule13-automation.sh scan"
        echo ""
        
        if [[ "${DEMO_INTERACTIVE:-true}" == "true" ]]; then
            echo -e "${CYAN}Run automation helper? [y/N]:${NC} "
            read -r response
            if [[ "$response" =~ ^[Yy]$ ]]; then
                "$SCRIPT_DIR/enforce-rule13-automation.sh" scan
            fi
        fi
    fi
    
    echo -e "${GREEN}✅ Helper scripts demonstration completed${NC}"
    echo ""
    
    if [[ "${DEMO_INTERACTIVE:-true}" == "true" ]]; then
        wait_for_keypress
    fi
}

show_best_practices() {
    print_scenario "Best Practices and Recommendations"
    
    cat << 'EOF'
🏆 GARBAGE COLLECTION ENFORCER - BEST PRACTICES

📋 Before Running:
  • Always start with dry-run mode (--dry-run)
  • Commit your current work to git
  • Review the confidence and risk thresholds
  • Check that protected paths are configured

🔍 During Analysis:
  • Use high confidence (0.8+) for initial scans
  • Start with 'safe' risk threshold
  • Review the detailed reports before cleanup
  • Pay attention to items with references

🧹 During Cleanup:
  • Create a git branch for cleanup operations
  • Enable archiving (default) for safety
  • Monitor space recovery metrics
  • Run in batches for large codebases

✅ After Cleanup:
  • Run tests to ensure nothing broke
  • Review git diff for unexpected changes
  • Check application functionality
  • Update .gitignore to prevent future garbage

⚙️ Automation:
  • Set up weekly cron jobs with safe thresholds
  • Integrate with CI/CD for prevention
  • Use pre-commit hooks for developers
  • Monitor metrics over time

🚨 Safety Features:
  • Automatic reference checking
  • Git integration for tracked files
  • Risk-based decision making
  • Comprehensive archiving system
  • Rollback instructions provided

EOF
    
    if [[ "${DEMO_INTERACTIVE:-true}" == "true" ]]; then
        wait_for_keypress
    fi
}

show_menu() {
    echo -e "${BOLD}${CYAN}Select a demonstration scenario:${NC}"
    echo ""
    echo "  1. Quick Safety Scan"
    echo "  2. Comprehensive Analysis" 
    echo "  3. Specific Directory Scan"
    echo "  4. JSON Report Generation"
    echo "  5. Threshold Comparison"
    echo "  6. Git Integration"
    echo "  7. Performance Analysis"
    echo "  8. Helper Scripts"
    echo "  9. Best Practices"
    echo "  0. Run All Scenarios"
    echo "  q. Quit"
    echo ""
    echo -e "${YELLOW}Enter your choice:${NC} "
}

run_interactive_demo() {
    while true; do
        print_header
        show_menu
        read -r choice
        
        case "$choice" in
            1) demo_quick_scan ;;
            2) demo_comprehensive_scan ;;
            3) demo_specific_directory ;;
            4) demo_json_report ;;
            5) demo_different_thresholds ;;
            6) demo_git_integration ;;
            7) demo_performance_analysis ;;
            8) demo_helper_scripts ;;
            9) show_best_practices ;;
            0) run_all_demos ;;
            q|Q) break ;;
            *) echo -e "${RED}Invalid choice. Please try again.${NC}" ;;
        esac
    done
}

run_all_demos() {
    DEMO_INTERACTIVE=false
    
    demo_quick_scan
    demo_comprehensive_scan
    demo_specific_directory
    demo_json_report
    demo_different_thresholds
    demo_git_integration
    demo_performance_analysis
    demo_helper_scripts
    show_best_practices
    
    echo -e "${BOLD}${GREEN}🎉 All demonstrations completed! 🎉${NC}"
}

main() {
    # Check prerequisites
    if [[ ! -f "$ENFORCER_SCRIPT" ]]; then
        echo -e "${RED}❌ Enforcer script not found: $ENFORCER_SCRIPT${NC}"
        exit 1
    fi
    
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python 3 is required but not installed${NC}"
        exit 1
    fi
    
    # Parse arguments  
    case "${1:-interactive}" in
        "quick") demo_quick_scan ;;
        "comprehensive") demo_comprehensive_scan ;;
        "directory") demo_specific_directory ;;
        "json") demo_json_report ;;
        "thresholds") demo_different_thresholds ;;
        "git") demo_git_integration ;;
        "performance") demo_performance_analysis ;;
        "helpers") demo_helper_scripts ;;
        "practices") show_best_practices ;;
        "all")
            DEMO_INTERACTIVE=false
            run_all_demos
            ;;
        "interactive"|*)
            run_interactive_demo
            ;;
    esac
    
    echo ""
    echo -e "${BLUE}Thank you for trying the Garbage Collection Enforcer demo!${NC}"
    echo -e "${CYAN}For more information, see: README-garbage-collection-enforcer.md${NC}"
}

main "$@"