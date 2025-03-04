#!/usr/bin/env python3
"""
Script to analyze the current coverage report and identify what still needs to be tested.
Uses the status.json file from the coverage directory.
"""

import json
import os
import sys
from pathlib import Path
from pprint import pprint

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
    
    if 'totals' in data:
        print(f"Total Coverage: {data['totals']['percent_covered']:.2f}%")
        print(f"Lines Covered: {data['totals']['covered_lines']} of {data['totals']['num_statements']}")
        print(f"Missing Lines: {data['totals']['missing_lines']}")
        print()
    
    # Display file-by-file coverage
    print("File Coverage:")
    print("-------------")
    
    for file_path, file_data in data.items():
        if file_path == 'totals':
            continue
        
        # Extract just the file name
        file_name = file_path.split('/')[-1] if '/' in file_path else file_path
        
        print(f"{file_name}: {file_data['percent_covered']:.2f}% covered")
        print(f"  Lines: {file_data['covered_lines']} of {file_data['num_statements']} covered")
        
        if file_data['missing_lines']:
            print(f"  Missing Lines: {file_data['missing_lines']}")
        print()
        
    # Focus on files with lowest coverage
    print("Files with Lowest Coverage:")
    print("-------------------------")
    
    files_by_coverage = sorted(
        [(f, d['percent_covered']) for f, d in data.items() if f != 'totals'],
        key=lambda x: x[1]
    )
    
    for file_path, percent in files_by_coverage[:3]:
        file_name = file_path.split('/')[-1] if '/' in file_path else file_path
        print(f"{file_name}: {percent:.2f}% covered")

def main():
    """Main function."""
    try:
        coverage_data = load_coverage_data()
        analyze_coverage(coverage_data)
    except Exception as e:
        print(f"Error analyzing coverage: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 