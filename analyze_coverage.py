import os
import subprocess
import re

def run_coverage_report():
    """Run coverage for all files and parse the detailed report."""
    cmd = "python -m pytest --cov=core_system.orchestrator --cov-report=term-missing"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout + result.stderr

def parse_coverage_output(output):
    """Parse coverage output to extract uncovered lines for each file."""
    file_pattern = re.compile(r'(core_system/orchestrator/[a-z_]+\.py)\s+\d+\s+\d+\s+(\d+)%\s+(.*)')
    uncovered_files = {}
    
    for line in output.split('\n'):
        match = file_pattern.search(line)
        if match:
            file_path = match.group(1)
            coverage_percent = match.group(2)
            missing_lines = match.group(3).strip()
            uncovered_files[file_path] = {
                'coverage': coverage_percent,
                'missing': missing_lines
            }
    
    return uncovered_files

def convert_to_ranges(missing_str):
    """Convert comma-separated ranges to a list of line numbers."""
    missing_ranges = []
    if missing_str:
        for part in missing_str.split(','):
            part = part.strip()
            if '-' in part:
                start, end = map(int, part.split('-'))
                missing_ranges.extend(range(start, end+1))
            else:
                try:
                    missing_ranges.append(int(part))
                except ValueError:
                    pass
    return missing_ranges

def analyze_files():
    """Analyze files for uncovered code and display the results."""
    print("Running coverage tests...")
    coverage_output = run_coverage_report()
    
    print("\nParsing results...")
    uncovered_files = parse_coverage_output(coverage_output)
    
    for file_path, data in uncovered_files.items():
        print(f"\n\n{'='*80}")
        print(f"File: {file_path}")
        print(f"Coverage: {data['coverage']}%")
        print(f"Missing lines: {data['missing']}")
        print(f"{'='*80}")
        
        if data['missing']:
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                
                missing_ranges = convert_to_ranges(data['missing'])
                display_uncovered_code(lines, missing_ranges)
            except FileNotFoundError:
                print(f"Could not open file: {file_path}")

def display_uncovered_code(lines, missing_ranges):
    """Display uncovered code sections with context."""
    if not missing_ranges:
        return
    
    print("\nUncovered Code Sections:")
    section_start = None
    
    for i, line_num in enumerate(missing_ranges):
        # Check if this starts a new section
        if i == 0 or line_num != missing_ranges[i-1] + 1:
            if section_start is not None:
                print(f"\n--- End section (lines {section_start}-{missing_ranges[i-1]}) ---")
            section_start = line_num
            context_start = max(0, line_num - 3)
            print(f"\n--- Start section (line {line_num}) ---")
            # Print context before
            for j in range(context_start, line_num):
                if j not in missing_ranges and 0 <= j-1 < len(lines):
                    print(f"{j}: (COVERED) {lines[j-1].rstrip()}")
        
        # Print the uncovered line
        if 0 <= line_num-1 < len(lines):
            print(f"{line_num}: (UNCOVERED) {lines[line_num-1].rstrip()}")
        
        # Check if this ends the current section
        if i == len(missing_ranges)-1 or line_num != missing_ranges[i+1] - 1:
            context_end = min(len(lines), line_num + 3)
            # Print context after
            for j in range(line_num + 1, context_end + 1):
                if j not in missing_ranges and 0 <= j-1 < len(lines):
                    print(f"{j}: (COVERED) {lines[j-1].rstrip()}")
            print(f"\n--- End section (lines {section_start}-{line_num}) ---")
            section_start = None

# Run the analysis
if __name__ == "__main__":
    os.chdir("/opt/sutazaiapp")
    analyze_files()
    
    print("\nAnalysis complete. Add targeted tests to cover the missing lines.") 