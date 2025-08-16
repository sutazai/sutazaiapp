#!/usr/bin/env python3
"""
RULE 1 PREVENTION FRAMEWORK - Elite Anti-Fantasy Code System
Prevents fantasy implementations and ensures real functionality

This framework provides:
1. Static analysis to detect mock/fantasy code patterns
2. CI/CD hooks to prevent Rule 1 violations
3. Runtime validation of real functionality
4. Automated testing for fantasy code detection

Usage:
    python scripts/debugging/rule1_prevention_framework.py --check-all
    python scripts/debugging/rule1_prevention_framework.py --file path/to/file.py
    python scripts/debugging/rule1_prevention_framework.py --setup-hooks
"""

import ast
import logging
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class FantasyPattern:
    """Definition of a fantasy code pattern"""
    name: str
    pattern: str
    description: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    regex: bool = False
    language: str = "python"  # python, javascript, typescript, all

@dataclass
class Violation:
    """A detected Rule 1 violation"""
    file_path: str
    line_number: int
    pattern_name: str
    severity: str
    description: str
    evidence: str
    fix_suggestion: str

class FantasyCodeDetector:
    """Detects fantasy code patterns that violate Rule 1"""
    
    def __init__(self):
        self.patterns = self._load_fantasy_patterns()
        self.violations: List[Violation] = []
    
    def _load_fantasy_patterns(self) -> List[FantasyPattern]:
        """Load all known fantasy code patterns"""
        return [
            # Mock/Hardcoded Response Patterns
            FantasyPattern(
                name="hardcoded_health_response",
                pattern=r'return\s*{\s*["\']status["\']\s*:\s*["\']healthy["\']',
                description="Hardcoded health check response instead of real check",
                severity="CRITICAL",
                regex=True
            ),
            FantasyPattern(
                name="mock_api_response",
                pattern=r'#.*[Mm]ock.*response|#.*[Ff]ake.*response',
                description="Comment indicating mock or fake response",
                severity="CRITICAL",
                regex=True
            ),
            FantasyPattern(
                name="hardcoded_status_data",
                pattern=r'return\s*{\s*["\']cpu_percent["\']\s*:\s*\d+',
                description="Hardcoded system status data instead of real metrics",
                severity="CRITICAL",
                regex=True
            ),
            FantasyPattern(
                name="mock_todo_comment",
                pattern="# Remove Remove Remove Mocks",
                description="Explicit mock TODO comment indicating fake implementation",
                severity="CRITICAL",
                regex=False
            ),
            FantasyPattern(
                name="subprocess_http_bridge",
                pattern=r'subprocess\.Popen.*stdin=subprocess\.PIPE.*HTTP',
                description="Attempting to bridge STDIO subprocess to HTTP (impossible)",
                severity="CRITICAL",
                regex=True
            ),
            
            # Fantasy Integration Patterns
            FantasyPattern(
                name="unused_service_config",
                pattern=r'url:\s*http://[^/]+/.*#.*not.*used|#.*not.*integrated',
                description="Service configuration that is commented as not used/integrated",
                severity="HIGH",
                regex=True
            ),
            FantasyPattern(
                name="fantasy_connection",
                pattern=r'self\.[a-z_]*_client\s*=\s*None.*#.*never.*connect',
                description="Client set to None with comment about never connecting",
                severity="HIGH",
                regex=True
            ),
            FantasyPattern(
                name="empty_cache_pattern",
                pattern=r'self\.[a-z_]*_cache\s*=\s*{}.*#.*empty|#.*no.*services',
                description="Empty cache/services with comment indicating non-functionality",
                severity="HIGH",
                regex=True
            ),
            
            # Performance Anti-Patterns
            FantasyPattern(
                name="blocking_sleep_in_async",
                pattern=r'time\.sleep\s*\(',
                description="Blocking sleep in async context (performance issue)",
                severity="MEDIUM",
                regex=True
            ),
            FantasyPattern(
                name="sync_in_async_context",
                pattern=r'async\s+def.*:.*(?!await).*requests\.',
                description="Synchronous requests in async function",
                severity="MEDIUM",
                regex=True
            ),
            
            # Security Anti-Patterns
            FantasyPattern(
                name="hardcoded_credentials",
                pattern=r'password["\']?\s*=\s*["\'][^"\']*["\']|secret["\']?\s*=\s*["\'][^"\']*["\']',
                description="Hardcoded credentials in code",
                severity="HIGH",
                regex=True
            ),
            FantasyPattern(
                name="cors_wildcard",
                pattern=r'allow_origins\s*=\s*\[.*\*.*\]',
                description="CORS wildcard allowing all origins",
                severity="HIGH",
                regex=True
            ),
            
            # Docker/Infrastructure Fantasy Patterns
            FantasyPattern(
                name="unused_env_var",
                pattern=r'ENV\s+[A-Z_]+.*#.*unused|#.*not.*used',
                description="Environment variable marked as unused",
                severity="MEDIUM",
                regex=True
            ),
            FantasyPattern(
                name="disabled_health_check",
                pattern=r'#.*HEALTHCHECK|HEALTHCHECK.*NONE',
                description="Disabled or commented health checks",
                severity="MEDIUM",
                regex=True
            ),
        ]
    
    def scan_file(self, file_path: Path) -> List[Violation]:
        """Scan a single file for fantasy patterns"""
        violations = []
        
        if not file_path.exists() or file_path.is_dir():
            return violations
        
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            lines = content.split('\n')
            
            for pattern in self.patterns:
                if pattern.language != "all" and not self._matches_language(file_path, pattern.language):
                    continue
                
                if pattern.regex:
                    matches = re.finditer(pattern.pattern, content, re.MULTILINE | re.IGNORECASE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        evidence = lines[line_num - 1].strip() if line_num <= len(lines) else ""
                        
                        violations.append(Violation(
                            file_path=str(file_path),
                            line_number=line_num,
                            pattern_name=pattern.name,
                            severity=pattern.severity,
                            description=pattern.description,
                            evidence=evidence,
                            fix_suggestion=self._get_fix_suggestion(pattern.name)
                        ))
                else:
                    # Simple string search
                    for line_num, line in enumerate(lines, 1):
                        if pattern.pattern in line:
                            violations.append(Violation(
                                file_path=str(file_path),
                                line_number=line_num,
                                pattern_name=pattern.name,
                                severity=pattern.severity,
                                description=pattern.description,
                                evidence=line.strip(),
                                fix_suggestion=self._get_fix_suggestion(pattern.name)
                            ))
        
        except Exception as e:
            logger.warning(f"Error scanning {file_path}: {e}")
        
        return violations
    
    def _matches_language(self, file_path: Path, language: str) -> bool:
        """Check if file matches the specified language"""
        suffix = file_path.suffix.lower()
        language_map = {
            "python": [".py", ".pyx"],
            "javascript": [".js", ".jsx"],
            "typescript": [".ts", ".tsx"],
            "yaml": [".yml", ".yaml"],
            "dockerfile": ["dockerfile", ".dockerfile"]
        }
        
        if language in language_map:
            return suffix in language_map[language] or file_path.name.lower() in language_map[language]
        return True
    
    def _get_fix_suggestion(self, pattern_name: str) -> str:
        """Get specific fix suggestion for pattern"""
        fixes = {
            "hardcoded_health_response": "Replace with real HTTP call to backend health endpoint",
            "mock_api_response": "Implement real API call using httpx or requests",
            "hardcoded_status_data": "Implement real system metrics collection using psutil",
            "mock_todo_comment": "Remove mock implementation and implement real functionality",
            "subprocess_http_bridge": "Use proper HTTP service architecture instead of STDIO bridge",
            "unused_service_config": "Either implement integration or remove unused configuration",
            "fantasy_connection": "Implement real client connection initialization",
            "empty_cache_pattern": "Implement real caching mechanism with actual data",
            "blocking_sleep_in_async": "Replace time.sleep with asyncio.sleep",
            "sync_in_async_context": "Replace requests with httpx async client",
            "hardcoded_credentials": "Move credentials to environment variables or secrets manager",
            "cors_wildcard": "Specify exact allowed origins instead of wildcard",
            "unused_env_var": "Remove unused environment variable or implement usage",
            "disabled_health_check": "Implement proper Docker health check"
        }
        return fixes.get(pattern_name, "Review and fix according to Rule 1 requirements")
    
    def scan_directory(self, directory: Path, exclude_patterns: List[str] = None) -> List[Violation]:
        """Scan entire directory for fantasy patterns"""
        if exclude_patterns is None:
            exclude_patterns = [
                "*.git*",
                "*__pycache__*",
                "*.pyc",
                "*node_modules*",
                "*.egg-info*",
                "*venv*",
                "*virtualenv*"
            ]
        
        violations = []
        
        # File extensions to scan
        extensions = {".py", ".js", ".ts", ".tsx", ".jsx", ".yml", ".yaml", ".json", ".sh", ".dockerfile"}
        
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                # Check exclusions
                if any(file_path.match(pattern) for pattern in exclude_patterns):
                    continue
                
                # Check extensions
                if file_path.suffix.lower() in extensions or file_path.name.lower() in ["dockerfile", "makefile"]:
                    file_violations = self.scan_file(file_path)
                    violations.extend(file_violations)
        
        return violations
    
    def generate_report(self, violations: List[Violation]) -> Dict:
        """Generate comprehensive violation report"""
        if not violations:
            return {
                "status": "CLEAN",
                "total_violations": 0,
                "by_severity": {},
                "by_file": {},
                "summary": "No Rule 1 violations detected"
            }
        
        # Group by severity
        by_severity = {}
        for violation in violations:
            if violation.severity not in by_severity:
                by_severity[violation.severity] = []
            by_severity[violation.severity].append(violation)
        
        # Group by file
        by_file = {}
        for violation in violations:
            if violation.file_path not in by_file:
                by_file[violation.file_path] = []
            by_file[violation.file_path].append(violation)
        
        # Calculate risk score
        severity_weights = {"CRITICAL": 10, "HIGH": 5, "MEDIUM": 2, "LOW": 1}
        risk_score = sum(severity_weights.get(v.severity, 1) for v in violations)
        
        return {
            "status": "VIOLATIONS_FOUND",
            "total_violations": len(violations),
            "risk_score": risk_score,
            "by_severity": {
                severity: len(viols) for severity, viols in by_severity.items()
            },
            "by_file": {
                file_path: len(viols) for file_path, viols in by_file.items()
            },
            "critical_files": [
                file_path for file_path, viols in by_file.items()
                if any(v.severity == "CRITICAL" for v in viols)
            ],
            "violations": [
                {
                    "file": v.file_path,
                    "line": v.line_number,
                    "pattern": v.pattern_name,
                    "severity": v.severity,
                    "description": v.description,
                    "evidence": v.evidence,
                    "fix": v.fix_suggestion
                }
                for v in violations
            ],
            "summary": f"Found {len(violations)} Rule 1 violations across {len(by_file)} files"
        }

class CICDHooksInstaller:
    """Installs CI/CD hooks to prevent fantasy code"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
    
    def install_pre_commit_hook(self) -> bool:
        """Install pre-commit hook to check for fantasy patterns"""
        hook_content = '''#!/bin/bash
# Pre-commit hook to prevent Rule 1 violations
echo "🔍 Checking for Rule 1 violations (fantasy code)..."

# Run fantasy code detector on staged files
python3 scripts/debugging/rule1_prevention_framework.py --check-staged

exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "❌ COMMIT BLOCKED: Rule 1 violations detected"
    echo "Fix the violations above before committing"
    exit 1
fi

echo "✅ No Rule 1 violations detected"
exit 0
'''
        
        hooks_dir = self.project_root / ".git" / "hooks"
        if not hooks_dir.exists():
            logger.error("Git hooks directory not found. Is this a git repository?")
            return False
        
        hook_file = hooks_dir / "pre-commit"
        hook_file.write_text(hook_content)
        hook_file.chmod(0o755)
        
        logger.info(f"✅ Pre-commit hook installed at {hook_file}")
        return True
    
    def install_github_action(self) -> bool:
        """Install GitHub Action for Rule 1 checking"""
        action_content = '''name: Rule 1 Compliance Check
on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  rule1-check:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install httpx requests
    
    - name: Check Rule 1 Compliance
      run: |
        python3 scripts/debugging/rule1_prevention_framework.py --check-all --strict
    
    - name: Upload violation report
      if: failure()
      uses: actions/upload-artifact@v3
      with:
        name: rule1-violations
        path: reports/rule1_violations.json
'''
        
        workflows_dir = self.project_root / ".github" / "workflows"
        workflows_dir.mkdir(parents=True, exist_ok=True)
        
        action_file = workflows_dir / "rule1-compliance.yml"
        action_file.write_text(action_content)
        
        logger.info(f"✅ GitHub Action installed at {action_file}")
        return True

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Rule 1 Prevention Framework")
    parser.add_argument("--check-all", action="store_true", help="Check entire project")
    parser.add_argument("--check-staged", action="store_true", help="Check only staged files")
    parser.add_argument("--file", help="Check specific file")
    parser.add_argument("--setup-hooks", action="store_true", help="Install CI/CD hooks")
    parser.add_argument("--strict", action="store_true", help="Exit with error on any violation")
    parser.add_argument("--output", default="reports/rule1_violations.json", help="Output file")
    
    args = parser.parse_args()
    
    project_root = Path("/opt/sutazaiapp")
    detector = FantasyCodeDetector()
    violations = []
    
    if args.setup_hooks:
        installer = CICDHooksInstaller(project_root)
        logger.info("Installing CI/CD hooks...")
        installer.install_pre_commit_hook()
        installer.install_github_action()
        return
    
    if args.check_staged:
        # Get staged files
        try:
            result = subprocess.run(
                ["git", "diff", "--cached", "--name-only"],
                capture_output=True,
                text=True,
                cwd=project_root
            )
            staged_files = [project_root / f for f in result.stdout.strip().split('\n') if f]
            
            for file_path in staged_files:
                if file_path.exists():
                    violations.extend(detector.scan_file(file_path))
        except Exception as e:
            logger.error(f"Error getting staged files: {e}")
            sys.exit(1)
    
    elif args.file:
        file_path = Path(args.file)
        violations = detector.scan_file(file_path)
    
    elif args.check_all:
        logger.info("Scanning entire project for Rule 1 violations...")
        violations = detector.scan_directory(project_root)
    
    else:
        parser.print_help()
        return
    
    # Generate report
    report = detector.generate_report(violations)
    
    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print("🚨 RULE 1 PREVENTION FRAMEWORK REPORT")
    print(f"{'='*80}")
    
    if report["status"] == "CLEAN":
        print("✅ No Rule 1 violations detected")
        print(f"📄 Report saved to: {output_path}")
        sys.exit(0)
    
    print(f"❌ Found {report['total_violations']} Rule 1 violations")
    print(f"🔥 Risk Score: {report['risk_score']}")
    
    if report["by_severity"]:
        print(f"\nBy Severity:")
        for severity, count in report["by_severity"].items():
            emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵"}.get(severity, "⚪")
            print(f"  {emoji} {severity}: {count}")
    
    if report["critical_files"]:
        print(f"\n🚨 Critical Files ({len(report['critical_files'])}):")
        for file_path in report["critical_files"][:10]:  # Show first 10
            print(f"  ❌ {file_path}")
        if len(report["critical_files"]) > 10:
            print(f"  ... and {len(report['critical_files']) - 10} more")
    
    print(f"\n📊 Top Violations:")
    critical_violations = [v for v in violations if v.severity == "CRITICAL"][:5]
    for v in critical_violations:
        print(f"  🔴 {Path(v.file_path).name}:{v.line_number} - {v.description}")
        print(f"     Evidence: {v.evidence[:80]}...")
        print(f"     Fix: {v.fix_suggestion}")
        print()
    
    print(f"📄 Full report saved to: {output_path}")
    
    if args.strict or report["by_severity"].get("CRITICAL", 0) > 0:
        print(f"\n🚨 RULE 1 PREVENTION CHECK FAILED")
        sys.exit(1)
    else:
        print(f"\n⚠️  RULE 1 PREVENTION CHECK COMPLETED WITH WARNINGS")
        sys.exit(0)

if __name__ == "__main__":
    main()