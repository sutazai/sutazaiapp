#!/usr/bin/env python3
"""
Run Browser Compatibility Tests for JARVIS Frontend

This script orchestrates browser compatibility testing and generates reports.
"""

import sys
import os
import subprocess
import time
from pathlib import Path

# Add frontend directory to path
sys.path.insert(0, '/opt/sutazaiapp/frontend')
sys.path.insert(0, '/opt/sutazaiapp/frontend/tests')

def check_dependencies():
    """Check if required dependencies are installed"""
    print("Checking dependencies...")
    
    dependencies = {
        'playwright': 'pip install playwright',
        'pandas': 'pip install pandas',
        'pytest': 'pip install pytest'
    }
    
    missing = []
    for package, install_cmd in dependencies.items():
        try:
            __import__(package)
        except ImportError:
            missing.append((package, install_cmd))
    
    if missing:
        print("\n⚠️  Missing dependencies detected:")
        for package, cmd in missing:
            print(f"  - {package}: Run '{cmd}'")
        
        print("\nInstalling missing dependencies...")
        for package, cmd in missing:
            subprocess.run(cmd.split(), check=False)
        
        # Install Playwright browsers
        print("\nInstalling Playwright browsers...")
        subprocess.run(['playwright', 'install'], check=False)
    else:
        print("✅ All dependencies installed")

def ensure_streamlit_running():
    """Ensure Streamlit app is running"""
    print("\nChecking if Streamlit is running...")
    
    try:
        import requests
        response = requests.get('http://localhost:11000', timeout=2)
        if response.status_code == 200:
            print("✅ Streamlit is running")
            return True
    except:
        pass
    
    print("⚠️  Streamlit not running. Please start it with:")
    print("    cd /opt/sutazaiapp/frontend && streamlit run app.py --server.port 11000")
    return False

def run_compatibility_tests():
    """Run the browser compatibility tests"""
    print("\n" + "="*80)
    print("Running Browser Compatibility Tests")
    print("="*80)
    
    test_file = Path('/opt/sutazaiapp/frontend/tests/e2e/test_browser_compatibility.py')
    
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return False
    
    # Run the test
    result = subprocess.run(
        [sys.executable, str(test_file)],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("Errors/Warnings:", result.stderr)
    
    return result.returncode == 0

def generate_summary():
    """Generate a summary of test results"""
    print("\n" + "="*80)
    print("Test Results Summary")
    print("="*80)
    
    report_dir = Path('/opt/sutazaiapp/frontend/tests/reports')
    
    # Check for generated reports
    reports = {
        'HTML Report': report_dir / 'compatibility_report.html',
        'Markdown Report': report_dir / 'compatibility_report.md',
        'JSON Matrix': report_dir / 'compatibility_matrix.json'
    }
    
    print("\n📊 Generated Reports:")
    for name, path in reports.items():
        if path.exists():
            print(f"  ✅ {name}: {path}")
        else:
            print(f"  ❌ {name}: Not found")
    
    # Check for screenshots
    screenshot_dir = Path('/opt/sutazaiapp/frontend/tests/screenshots')
    if screenshot_dir.exists():
        screenshots = list(screenshot_dir.glob('*.png'))
        print(f"\n📸 Screenshots captured: {len(screenshots)}")
        if screenshots:
            print("  Recent screenshots:")
            for screenshot in sorted(screenshots)[-5:]:
                print(f"    - {screenshot.name}")
    
    # Parse and display key findings from markdown report
    md_report = report_dir / 'compatibility_report.md'
    if md_report.exists():
        print("\n🔍 Key Findings:")
        with open(md_report, 'r') as f:
            content = f.read()
            
            # Extract summary section
            if '## Summary' in content:
                summary_start = content.index('## Summary')
                summary_end = content.index('\n## ', summary_start + 1) if '\n## ' in content[summary_start + 1:] else len(content)
                summary = content[summary_start:summary_end]
                
                for line in summary.split('\n'):
                    if 'Total Tests:' in line or 'Passed:' in line or 'Failed:' in line:
                        print(f"  {line.strip()}")

def main():
    """Main execution function"""
    print("🤖 JARVIS Frontend Browser Compatibility Testing Suite")
    print("="*80)
    
    # Check dependencies
    check_dependencies()
    
    # Ensure Streamlit is running
    if not ensure_streamlit_running():
        print("\n⚠️  Please start Streamlit first, then run this script again.")
        return 1
    
    # Run tests
    success = run_compatibility_tests()
    
    # Generate summary
    generate_summary()
    
    # Print final status
    print("\n" + "="*80)
    if success:
        print("✅ Browser compatibility testing completed successfully!")
        print("\nTo view the detailed reports:")
        print("  - HTML: Open /opt/sutazaiapp/frontend/tests/reports/compatibility_report.html")
        print("  - Markdown: View /opt/sutazaiapp/frontend/tests/reports/compatibility_report.md")
        print("  - Screenshots: Browse /opt/sutazaiapp/frontend/tests/screenshots/")
        
        print("\nTo apply browser fixes, include this in your HTML:")
        print('  <script src="/tests/e2e/browser_fixes.js"></script>')
    else:
        print("❌ Some tests failed. Check the reports for details.")
    
    print("="*80)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())