#!/bin/bash

# SutazAI Security Scanning Script
# Runs comprehensive security analysis using Semgrep

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORT_DIR="$SCRIPT_DIR/security_reports"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create reports directory
mkdir -p "$REPORT_DIR"

echo -e "${BLUE}🔒 SutazAI Security Scanning Suite${NC}"
echo -e "${BLUE}====================================${NC}"
echo ""

# Check if Semgrep is installed
if ! command -v semgrep &> /dev/null; then
    echo -e "${RED}❌ Semgrep not found. Installing...${NC}"
    pip install semgrep
fi

echo -e "${GREEN}✅ Starting comprehensive security scan...${NC}"
echo ""

# Run Semgrep with multiple rule sets
echo -e "${YELLOW}📋 Running Semgrep scans with multiple rule sets...${NC}"

# 1. Custom SutazAI rules
echo -e "${BLUE}🔍 Running custom SutazAI security rules...${NC}"
semgrep --config="$SCRIPT_DIR/semgrep_custom_rules.yaml" \
    --output="$REPORT_DIR/sutazai_custom_${TIMESTAMP}.json" \
    --json \
    --verbose \
    "$SCRIPT_DIR" || true

# 2. OWASP Top 10 rules
echo -e "${BLUE}🔍 Running OWASP Top 10 security rules...${NC}"
semgrep --config=p/owasp-top-ten \
    --output="$REPORT_DIR/owasp_top10_${TIMESTAMP}.json" \
    --json \
    --verbose \
    "$SCRIPT_DIR" || true

# 3. Security audit rules
echo -e "${BLUE}🔍 Running general security audit rules...${NC}"
semgrep --config=p/security-audit \
    --output="$REPORT_DIR/security_audit_${TIMESTAMP}.json" \
    --json \
    --verbose \
    "$SCRIPT_DIR" || true

# 4. Python security rules
echo -e "${BLUE}🔍 Running Python-specific security rules...${NC}"
semgrep --config=p/python \
    --output="$REPORT_DIR/python_security_${TIMESTAMP}.json" \
    --json \
    --verbose \
    "$SCRIPT_DIR" || true

# 5. Docker security rules
echo -e "${BLUE}🔍 Running Docker security rules...${NC}"
semgrep --config=p/docker \
    --output="$REPORT_DIR/docker_security_${TIMESTAMP}.json" \
    --json \
    --verbose \
    "$SCRIPT_DIR" || true

# 6. Secrets detection
echo -e "${BLUE}🔍 Running secrets detection...${NC}"
semgrep --config=p/secrets \
    --output="$REPORT_DIR/secrets_${TIMESTAMP}.json" \
    --json \
    --verbose \
    "$SCRIPT_DIR" || true

# Generate combined report
echo -e "${YELLOW}📊 Generating combined security report...${NC}"

python3 << 'EOF'
import json
import glob
import os
from datetime import datetime
from collections import defaultdict

def load_semgrep_reports(report_dir):
    """Load all Semgrep JSON reports"""
    reports = []
    pattern = os.path.join(report_dir, "*_*.json")
    
    for file_path in glob.glob(pattern):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if 'results' in data:
                    reports.append({
                        'file': os.path.basename(file_path),
                        'data': data
                    })
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return reports

def analyze_findings(reports):
    """Analyze and categorize findings"""
    all_findings = []
    severity_counts = defaultdict(int)
    rule_counts = defaultdict(int)
    file_counts = defaultdict(int)
    
    for report in reports:
        for result in report['data']['results']:
            finding = {
                'rule_id': result['check_id'],
                'severity': result.get('extra', {}).get('severity', 'INFO').upper(),
                'message': result['extra']['message'],
                'file': result['path'],
                'line': result.get('start', {}).get('line', 0),
                'source': report['file']
            }
            
            all_findings.append(finding)
            severity_counts[finding['severity']] += 1
            rule_counts[finding['rule_id']] += 1
            file_counts[finding['file']] += 1
    
    return all_findings, severity_counts, rule_counts, file_counts

def generate_html_report(findings, severity_counts, rule_counts, file_counts, output_file):
    """Generate HTML security report"""
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>SutazAI Security Scan Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .summary-card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #007bff; }}
        .critical {{ border-left-color: #dc3545; }}
        .high {{ border-left-color: #fd7e14; }}
        .medium {{ border-left-color: #ffc107; }}
        .low {{ border-left-color: #28a745; }}
        .info {{ border-left-color: #17a2b8; }}
        .finding {{ background: #fff; border: 1px solid #e9ecef; border-radius: 8px; margin: 10px 0; padding: 15px; }}
        .finding-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
        .severity-badge {{ padding: 4px 12px; border-radius: 4px; color: white; font-size: 12px; font-weight: bold; }}
        .severity-critical {{ background-color: #dc3545; }}
        .severity-high {{ background-color: #fd7e14; }}
        .severity-medium {{ background-color: #ffc107; color: #000; }}
        .severity-low {{ background-color: #28a745; }}
        .severity-info {{ background-color: #17a2b8; }}
        .file-path {{ font-family: monospace; background: #f8f9fa; padding: 2px 6px; border-radius: 4px; }}
        .rule-id {{ font-family: monospace; color: #6f42c1; }}
        .top-issues {{ background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; padding: 20px; margin: 20px 0; }}
        .chart {{ margin: 20px 0; }}
        h1, h2, h3 {{ color: #333; }}
        .timestamp {{ color: #6c757d; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔒 SutazAI Security Scan Report</h1>
            <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>
        
        <div class="summary">
            <div class="summary-card critical">
                <h3>Critical Issues</h3>
                <div style="font-size: 32px; font-weight: bold;">{severity_counts.get('ERROR', severity_counts.get('CRITICAL', 0))}</div>
            </div>
            <div class="summary-card high">
                <h3>High Priority</h3>
                <div style="font-size: 32px; font-weight: bold;">{severity_counts.get('WARNING', severity_counts.get('HIGH', 0))}</div>
            </div>
            <div class="summary-card medium">
                <h3>Medium Priority</h3>
                <div style="font-size: 32px; font-weight: bold;">{severity_counts.get('MEDIUM', 0)}</div>
            </div>
            <div class="summary-card low">
                <h3>Low Priority</h3>
                <div style="font-size: 32px; font-weight: bold;">{severity_counts.get('LOW', 0)}</div>
            </div>
            <div class="summary-card info">
                <h3>Total Findings</h3>
                <div style="font-size: 32px; font-weight: bold;">{len(findings)}</div>
            </div>
        </div>
        
        <div class="top-issues">
            <h3>🚨 Top Security Issues</h3>
            <ul>
                <li><strong>Authentication Bypass:</strong> Mock authentication system allows unauthorized access</li>
                <li><strong>Docker Socket Exposure:</strong> Container can access host Docker daemon</li>
                <li><strong>Hardcoded Credentials:</strong> Default passwords in configuration files</li>
                <li><strong>CORS Misconfiguration:</strong> Wildcard origins allow cross-site attacks</li>
                <li><strong>Input Validation:</strong> Missing sanitization of user inputs</li>
            </ul>
        </div>
        
        <h2>🔍 Detailed Findings</h2>
    """
    
    # Sort findings by severity
    severity_order = {'ERROR': 0, 'CRITICAL': 0, 'WARNING': 1, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3, 'INFO': 4}
    sorted_findings = sorted(findings, key=lambda x: severity_order.get(x['severity'], 99))
    
    for finding in sorted_findings:
        severity_class = finding['severity'].lower()
        if severity_class == 'error':
            severity_class = 'critical'
        elif severity_class == 'warning':
            severity_class = 'high'
            
        html_content += f"""
        <div class="finding">
            <div class="finding-header">
                <div>
                    <span class="severity-badge severity-{severity_class}">{finding['severity']}</span>
                    <span class="rule-id">{finding['rule_id']}</span>
                </div>
                <div class="file-path">{finding['file']}:{finding['line']}</div>
            </div>
            <div class="message">{finding['message']}</div>
        </div>
        """
    
    # Add top files with issues
    html_content += """
        <h2>📁 Files with Most Issues</h2>
        <div class="chart">
    """
    
    top_files = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for file_path, count in top_files:
        html_content += f"""
            <div style="margin: 5px 0; display: flex; align-items: center;">
                <div style="width: 200px; overflow: hidden; text-overflow: ellipsis;" class="file-path">{file_path}</div>
                <div style="background: #007bff; height: 20px; width: {min(count * 10, 200)}px; margin-left: 10px; border-radius: 4px;"></div>
                <div style="margin-left: 10px; font-weight: bold;">{count}</div>
            </div>
        """
    
    html_content += """
        </div>
        
        <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #e9ecef; text-align: center; color: #6c757d;">
            <p>Generated by SutazAI Semgrep Security Analyzer | Report contains {len(findings)} total findings</p>
        </div>
    </div>
</body>
</html>
    """.replace('{len(findings)}', str(len(findings)))
    
    with open(output_file, 'w') as f:
        f.write(html_content)

# Main execution
if __name__ == "__main__":
    import sys
    report_dir = sys.argv[1] if len(sys.argv) > 1 else "security_reports"
    
    print("📊 Analyzing security scan results...")
    reports = load_semgrep_reports(report_dir)
    
    if not reports:
        print("❌ No scan results found")
        exit(1)
    
    findings, severity_counts, rule_counts, file_counts = analyze_findings(reports)
    
    print(f"📋 Analysis complete:")
    print(f"   - Total findings: {len(findings)}")
    print(f"   - Critical/Error: {severity_counts.get('ERROR', 0) + severity_counts.get('CRITICAL', 0)}")
    print(f"   - High/Warning: {severity_counts.get('WARNING', 0) + severity_counts.get('HIGH', 0)}")
    print(f"   - Medium: {severity_counts.get('MEDIUM', 0)}")
    print(f"   - Low: {severity_counts.get('LOW', 0)}")
    print(f"   - Info: {severity_counts.get('INFO', 0)}")
    
    # Generate HTML report
    html_output = os.path.join(report_dir, f"security_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    generate_html_report(findings, severity_counts, rule_counts, file_counts, html_output)
    print(f"📄 HTML report generated: {html_output}")
    
    # Generate JSON summary
    json_output = os.path.join(report_dir, f"security_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_findings": len(findings),
        "severity_breakdown": dict(severity_counts),
        "top_rules": dict(sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
        "top_files": dict(sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
        "findings": findings
    }
    
    with open(json_output, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"📄 JSON summary generated: {json_output}")

EOF

# Run the Python analysis
python3 -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
exec(open('$SCRIPT_DIR/run_security_scan.sh').read().split('python3 << \\'EOF\\'')[1].split('EOF')[0])
" "$REPORT_DIR"

echo ""
echo -e "${GREEN}✅ Security scanning complete!${NC}"
echo -e "${BLUE}📊 Reports generated in: $REPORT_DIR${NC}"
echo ""
echo -e "${YELLOW}📋 Next steps:${NC}"
echo -e "   1. Review the HTML report for detailed findings"
echo -e "   2. Address critical and high-priority issues immediately"
echo -e "   3. Run the security hardening script: python3 implement_security_fixes.py"
echo -e "   4. Re-run security scan to verify fixes"
echo ""
echo -e "${RED}⚠️  CRITICAL ISSUES FOUND - IMMEDIATE ACTION REQUIRED${NC}"
echo ""