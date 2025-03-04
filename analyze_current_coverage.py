#!/usr/bin/env python3
"""
Script to analyze the current coverage report and identify what still needs to be tested.
Uses the status.json file from the coverage directory.
"""

import json
import os
import sys
from pathlib import Path

def load_coverage_data():
    """Load the coverage data from status.json."""
    coverage_file = Path("coverage/status.json")
    
    if not coverage_file.exists():
        print(f"Error: Coverage file not found: {coverage_file}")
        print("Please run the tests with coverage before running this script.")
        sys.exit(1)
    
    with open(coverage_file, "r") as f:
        return json.load(f)

def analyze_coverage(data):
    """Analyze the coverage data and print out statistics."""
    print("Coverage Analysis:")
    print("-----------------")
    
    # Check if the data has the expected structure
    if 'files' not in data:
        print("Error: Unexpected coverage data format. 'files' key not found.")
        sys.exit(1)
    
    files_data = data['files']
    
    # Calculate total statistics
    total_statements = 0
    total_missing = 0
    total_excluded = 0
    total_covered = 0
    
    # Detailed file statistics
    file_stats = []
    
    for file_id, file_info in files_data.items():
        if 'index' in file_info and 'nums' in file_info['index']:
            nums = file_info['index']['nums']
            file_path = file_info['index'].get('file', file_id)
            statements = nums.get('n_statements', 0)
            missing = nums.get('n_missing', 0)
            excluded = nums.get('n_excluded', 0)
            covered = statements - missing
            
            total_statements += statements
            total_missing += missing
            total_excluded += excluded
            total_covered += covered
            
            # Only include actual Python files, not HTML or other non-source files
            if file_path.endswith('.py'):
                coverage_percent = 0 if statements == 0 else (covered / statements) * 100
                file_stats.append((file_path, coverage_percent, covered, statements, missing))
    
    # Calculate total coverage percentage
    total_coverage_percent = 0 if total_statements == 0 else (total_covered / total_statements) * 100
    
    print(f"Total Coverage: {total_coverage_percent:.2f}%")
    print(f"Lines Covered: {total_covered} of {total_statements}")
    print(f"Missing Lines: {total_missing}")
    print(f"Excluded Lines: {total_excluded}")
    print()
    
    # Display file-by-file coverage
    print("File Coverage:")
    print("-------------")
    
    for file_path, coverage_percent, covered, statements, missing in file_stats:
        # Extract just the file name for cleaner output
        file_name = file_path.split('/')[-1] if '/' in file_path else file_path
        
        print(f"{file_name}: {coverage_percent:.2f}% covered")
        print(f"  Lines: {covered} of {statements} covered")
        
        if missing > 0:
            print(f"  Missing Lines: {missing}")
        print()
        
    # Focus on files with lowest coverage
    print("Files with Lowest Coverage:")
    print("-------------------------")
    
    # Sort by coverage percentage (ascending)
    sorted_files = sorted(file_stats, key=lambda x: x[1])
    
    for file_path, coverage_percent, covered, statements, missing in sorted_files[:3]:
        file_name = file_path.split('/')[-1] if '/' in file_path else file_path
        print(f"{file_name}: {coverage_percent:.2f}% covered ({covered}/{statements} lines)")

def main():
    """Main function."""
    try:
        coverage_data = load_coverage_data()
        analyze_coverage(coverage_data)
    except Exception as e:
        print(f"Error analyzing coverage: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 