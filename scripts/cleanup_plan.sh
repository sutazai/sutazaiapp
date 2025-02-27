#!/bin/bash

# SutazAI Cleanup Plan Script
# This script identifies and lists redundant files in the SutazAI project
# It creates a list of files that can potentially be removed to clean up the codebase

echo -e "\e[1;34m==== SutazAI Codebase Cleanup Plan ====\e[0m"
echo "This script will analyze the codebase and suggest files for removal"
echo "No files will be deleted automatically - this is just a recommendation"

# Create output directory for reports
REPORT_DIR="/tmp/sutazai_cleanup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$REPORT_DIR"
echo "Reports will be saved to: $REPORT_DIR"

# Check for Python 3.11
if ! command -v python3.11 &> /dev/null; then
    echo -e "\e[1;31mPython 3.11 is not installed. This cleanup plan is designed for Python 3.11 compatibility.\e[0m"
    echo "Please install Python 3.11 before proceeding."
    exit 1
fi

echo -e "\n\e[1;34m1. Identifying duplicate system audit scripts\e[0m"
{
    echo "The following system audit scripts appear to be redundant:"
    echo "------------------------------------------------------"
    echo "Primary script to KEEP: sutazaiapp/scripts/system_audit.py"
    echo
    echo "Redundant scripts that can be removed:"
    find sutazaiapp/scripts -name "*system*audit*.py" | grep -v "system_audit.py"
    echo
    echo "Duplicate scripts in misc directory:"
    find sutazaiapp/misc -name "*system*audit*.py"
} > "$REPORT_DIR/duplicate_audit_scripts.txt"

echo -e "\n\e[1;34m2. Identifying duplicate dependency management scripts\e[0m"
{
    echo "The following dependency management scripts appear to be redundant:"
    echo "------------------------------------------------------"
    echo "Primary script to KEEP: sutazaiapp/scripts/unified_dependency_manager.py"
    echo
    echo "Redundant scripts that can be removed:"
    find sutazaiapp/scripts -name "*dependency*.py" | grep -v "unified_dependency_manager.py"
    echo
    echo "Duplicate scripts in misc directory:"
    find sutazaiapp/misc -name "*dependency*.py"
} > "$REPORT_DIR/duplicate_dependency_scripts.txt"

echo -e "\n\e[1;34m3. Identifying duplicate system health check scripts\e[0m"
{
    echo "The following system health check scripts appear to be redundant:"
    echo "------------------------------------------------------"
    echo "Primary script to KEEP: sutazaiapp/scripts/system_health_check.py"
    echo
    echo "Redundant scripts that can be removed:"
    find sutazaiapp/scripts -name "*health*.py" | grep -v "system_health_check.py"
    echo
    echo "Duplicate scripts in misc directory:"
    find sutazaiapp/misc -name "*health*.py"
} > "$REPORT_DIR/duplicate_health_scripts.txt"

echo -e "\n\e[1;34m4. Identifying duplicate system optimizer scripts\e[0m"
{
    echo "The following system optimizer scripts appear to be redundant:"
    echo "------------------------------------------------------"
    echo "Primary script to KEEP: sutazaiapp/scripts/system_optimizer.py"
    echo
    echo "Redundant scripts that can be removed:"
    find sutazaiapp/scripts -name "*optimizer*.py" | grep -v "system_optimizer.py"
    echo
    echo "Duplicate scripts in misc directory:"
    find sutazaiapp/misc -name "*optimizer*.py"
} > "$REPORT_DIR/duplicate_optimizer_scripts.txt"

echo -e "\n\e[1;34m5. Identifying duplicate setup/initialization scripts\e[0m"
{
    echo "The following setup/initialization scripts appear to be redundant:"
    echo "------------------------------------------------------"
    echo "Primary script to KEEP: sutazaiapp/scripts/system_initializer.py"
    echo
    echo "Redundant scripts that can be removed:"
    find sutazaiapp/scripts -name "*setup*.py" -o -name "*initializer*.py" | grep -v "system_initializer.py"
    echo
    echo "Duplicate scripts in misc directory:"
    find sutazaiapp/misc -name "*setup*.py" -o -name "*initializer*.py"
} > "$REPORT_DIR/duplicate_setup_scripts.txt"

echo -e "\n\e[1;34m6. Identifying potential redundant directories\e[0m"
{
    echo "The following directories appear to contain redundant or duplicate code:"
    echo "------------------------------------------------------"
    echo "1. sutazaiapp/misc/core_system/ - Contains duplicates of system scripts"
    echo "2. sutazaiapp/misc/sutazai/ - Contains duplicates of system scripts"
    echo "3. sutazaiapp/misc/ - Contains many duplicate/outdated scripts"
    echo
    echo "Recommendation: Keep the primary scripts in sutazaiapp/scripts/ and sutazaiapp/core_system/"
    echo "Consider archiving or removing the misc directory after verifying no unique code is lost"
} > "$REPORT_DIR/redundant_directories.txt"

echo -e "\n\e[1;34m7. Python 3.11 Compatibility Check\e[0m"
{
    echo "Checking scripts for Python 3.11 compatibility issues:"
    echo "------------------------------------------------------"
    echo "Scripts explicitly checking for Python 3.11:"
    grep -r "python3.11" --include="*.py" --include="*.sh" sutazaiapp/
    echo
    echo "Scripts that may need Python 3.11 compatibility updates:"
    grep -r "python3" --include="*.py" --include="*.sh" sutazaiapp/ | grep -v "python3.11"
} > "$REPORT_DIR/python311_compatibility.txt"

# Create a summary file with cleanup recommendations
{
    echo "==== SutazAI Codebase Cleanup Recommendations ===="
    echo "Generated on: $(date)"
    echo
    echo "1. Primary Scripts to Keep:"
    echo "   - sutazaiapp/scripts/system_audit.py"
    echo "   - sutazaiapp/scripts/unified_dependency_manager.py"
    echo "   - sutazaiapp/scripts/system_health_check.py"
    echo "   - sutazaiapp/scripts/system_optimizer.py"
    echo "   - sutazaiapp/scripts/system_initializer.py"
    echo "   - sutazaiapp/scripts/install_dependencies.sh"
    echo "   - sutazaiapp/scripts/install_python311.sh"
    echo
    echo "2. Directories to Keep:"
    echo "   - sutazaiapp/core_system/"
    echo "   - sutazaiapp/backend/"
    echo "   - sutazaiapp/scripts/ (after cleanup)"
    echo
    echo "3. Recommended for Removal or Archiving:"
    echo "   - sutazaiapp/misc/ (after verifying no unique code is lost)"
    echo "   - Redundant scripts listed in the individual reports"
    echo
    echo "4. Next Steps:"
    echo "   1. Review each report file to confirm redundant scripts"
    echo "   2. Backup any files before removing them"
    echo "   3. Test the system after removing redundant files"
    echo "   4. Update documentation to reflect the new streamlined structure"
} > "$REPORT_DIR/cleanup_summary.txt"

echo -e "\n\e[1;32mCleanup plan completed successfully!\e[0m"
echo "Reports have been saved to: $REPORT_DIR"
echo "Please review the reports before taking any action."
echo -e "Main summary: $REPORT_DIR/cleanup_summary.txt\n" 