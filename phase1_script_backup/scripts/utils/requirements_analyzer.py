#!/usr/bin/env python3
"""
Requirements Analysis Tool for SutazAI Codebase

Analyzes all requirements files for version conflicts, security issues,
and maintainability problems.
"""

import re
import json
from pathlib import Path
from collections import defaultdict
import requests
from datetime import datetime

class RequirementsAnalyzer:
    def __init__(self, root_dir: str = "/opt/sutazaiapp"):
        self.root_dir = Path(root_dir)
        self.packages = defaultdict(list)  # package_name -> [(version, file_path)]
        self.issues = []
        self.active_files = []  # Non-backup, non-archive files
        self.backup_files = []  # Backup/archive files
        
    def find_requirements_files(self) -> List[Path]:
        """Find all requirements files in the codebase."""
        patterns = [
            "**/requirements*.txt",
            "**/pyproject.toml"
        ]
        
        files = []
        for pattern in patterns:
            files.extend(self.root_dir.glob(pattern))
        
        # Categorize active vs backup files
        for file_path in files:
            path_str = str(file_path)
            if any(marker in path_str for marker in [
                "backup", "archive", "dependency-backups", 
                "docs_requirements_", "_pre_cleanup"
            ]):
                self.backup_files.append(file_path)
            else:
                self.active_files.append(file_path)
                
        return files
    
    def parse_requirements_txt(self, file_path: Path) -> List[Tuple[str, str]]:
        """Parse a requirements.txt file and return list of (package, version) tuples."""
        packages = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Handle different version specifier formats
                    # Example patterns: package==1.0.0, package>=1.0.0, package~=1.0
                    match = re.match(r'^([a-zA-Z0-9_-]+(?:\[[^\]]*\])?)\s*([><=!~]+)\s*(.+)$', line)
                    if match:
                        package, operator, version = match.groups()
                        packages.append((package.lower().strip(), f"{operator}{version}"))
                    elif '==' in line or '>=' in line or '<=' in line or '~=' in line:
                        # Fallback parsing
                        for op in ['==', '>=', '<=', '~=', '!=', '>', '<']:
                            if op in line:
                                parts = line.split(op, 1)
                                if len(parts) == 2:
                                    package = parts[0].strip().lower()
                                    version = f"{op}{parts[1].strip()}"
                                    packages.append((package, version))
                                break
                    else:
                        # Package without version specifier
                        if line and not line.startswith('-'):
                            packages.append((line.lower().strip(), "unspecified"))
                            
        except Exception as e:
            self.issues.append({
                "type": "parse_error",
                "file": str(file_path),
                "error": str(e)
            })
            
        return packages
    
    def parse_pyproject_toml(self, file_path: Path) -> List[Tuple[str, str]]:
        """Parse a pyproject.toml file for dependencies."""
        packages = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Look for dependencies section
            in_dependencies = False
            for line in content.split('\n'):
                line = line.strip()
                
                if line.startswith('dependencies = ['):
                    in_dependencies = True
                    continue
                elif in_dependencies and line == ']':
                    in_dependencies = False
                    continue
                elif in_dependencies:
                    # Extract package from quoted string
                    match = re.search(r'"([^"]+)"', line)
                    if match:
                        dep_line = match.group(1)
                        # Parse similar to requirements.txt
                        for op in ['==', '>=', '<=', '~=', '!=', '>', '<']:
                            if op in dep_line:
                                parts = dep_line.split(op, 1)
                                if len(parts) == 2:
                                    package = parts[0].strip().lower()
                                    version = f"{op}{parts[1].strip()}"
                                    packages.append((package, version))
                                break
                        else:
                            if dep_line and not dep_line.startswith('#'):
                                packages.append((dep_line.lower().strip(), "unspecified"))
                                
        except Exception as e:
            self.issues.append({
                "type": "parse_error", 
                "file": str(file_path),
                "error": str(e)
            })
            
        return packages
    
    def analyze_files(self):
        """Analyze all requirements files for conflicts and issues."""
        files = self.find_requirements_files()
        
        for file_path in files:
            if file_path.name == "pyproject.toml":
                packages = self.parse_pyproject_toml(file_path)
            else:
                packages = self.parse_requirements_txt(file_path)
                
            for package, version in packages:
                self.packages[package].append((version, str(file_path)))
    
    def find_version_conflicts(self) -> List[Dict]:
        """Find packages with conflicting version requirements."""
        conflicts = []
        
        for package, versions in self.packages.items():
            if len(versions) <= 1:
                continue
                
            # Group by version specification
            version_groups = defaultdict(list)
            for version, file_path in versions:
                version_groups[version].append(file_path)
            
            # Check for actual conflicts (different pinned versions)
            pinned_versions = {}
            for version, files in version_groups.items():
                if version.startswith('==') and version != "==LATEST_SECURE":
                    actual_version = version[2:]
                    if actual_version in pinned_versions:
                        pinned_versions[actual_version].extend(files)
                    else:
                        pinned_versions[actual_version] = files
            
            # Report conflicts if multiple different pinned versions exist
            if len(pinned_versions) > 1:
                conflicts.append({
                    "package": package,
                    "versions": dict(version_groups),
                    "pinned_versions": pinned_versions,
                    "severity": "high" if len(pinned_versions) > 2 else "medium"
                })
            elif len(version_groups) > 3:  # Many different version specs
                conflicts.append({
                    "package": package,
                    "versions": dict(version_groups),
                    "severity": "low",
                    "issue": "inconsistent_version_specs"
                })
                
        return conflicts
    
    def find_duplicate_files(self) -> List[Dict]:
        """Find duplicate or redundant requirements files."""
        duplicates = []
        
        # Check for files with identical content
        file_hashes = {}
        for file_path in self.active_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read().strip()
                content_hash = hash(content)
                if content_hash in file_hashes:
                    duplicates.append({
                        "type": "identical_content",
                        "files": [str(file_hashes[content_hash]), str(file_path)],
                        "severity": "medium"
                    })
                else:
                    file_hashes[content_hash] = file_path
            except Exception:
                continue
                
        # Check for files with very similar package lists
        file_packages = {}
        for file_path in self.active_files:
            if file_path.name == "pyproject.toml":
                packages = set(pkg for pkg, _ in self.parse_pyproject_toml(file_path))
            else:
                packages = set(pkg for pkg, _ in self.parse_requirements_txt(file_path))
            file_packages[str(file_path)] = packages
            
        # Find files with >80% package overlap
        checked_pairs = set()
        for file1, packages1 in file_packages.items():
            for file2, packages2 in file_packages.items():
                if file1 >= file2 or (file1, file2) in checked_pairs:
                    continue
                checked_pairs.add((file1, file2))
                
                if packages1 and packages2:
                    overlap = len(packages1 & packages2)
                    total = len(packages1 | packages2)
                    similarity = overlap / total if total > 0 else 0
                    
                    if similarity > 0.8 and overlap > 3:
                        duplicates.append({
                            "type": "similar_packages",
                            "files": [file1, file2],
                            "similarity": similarity,
                            "overlap": overlap,
                            "severity": "low"
                        })
                        
        return duplicates
    
    def check_security_issues(self) -> List[Dict]:
        """Check for known security vulnerabilities (simplified check)."""
        security_issues = []
        
        # Known vulnerable versions (simplified - in production use safety/pip-audit)
        known_vulnerabilities = {
            "requests": ["2.20.0", "2.19.1"],
            "urllib3": ["1.24.1", "1.25.2"],
            "pyyaml": ["3.12", "5.1"],
            "jinja2": ["2.10", "2.10.1"],
            "werkzeug": ["0.15.5"],
        }
        
        for package, versions in self.packages.items():
            if package in known_vulnerabilities:
                vulnerable_versions = known_vulnerabilities[package]
                for version, file_path in versions:
                    version_num = version.lstrip('>=<=~!').strip()
                    if version_num in vulnerable_versions:
                        security_issues.append({
                            "package": package,
                            "version": version,
                            "file": file_path,
                            "vulnerability": f"Known vulnerability in {package} {version_num}",
                            "severity": "high"
                        })
                        
        return security_issues
    
    def check_outdated_packages(self) -> List[Dict]:
        """Check for packages that might be outdated (simplified check)."""
        outdated = []
        
        # Check for obviously old versions
        old_patterns = {
            "requests": ("2.31.0", "Consider updating - newer versions available"),
            "fastapi": ("0.104.1", "Rapidly evolving framework - check for updates"),
            "pydantic": ("2.5.0", "Version 2.x series has frequent updates"),
            "pytest": ("7.4.3", "Test framework - consider latest stable"),
        }
        
        for package, versions in self.packages.items():
            if package in old_patterns:
                target_version, message = old_patterns[package]
                for version, file_path in versions:
                    if version.startswith('==') and version[2:] <= target_version:
                        outdated.append({
                            "package": package,
                            "current_version": version,
                            "file": file_path,
                            "suggestion": message,
                            "severity": "low"
                        })
                        
        return outdated
    
    def generate_report(self) -> Dict:
        """Generate comprehensive analysis report."""
        self.analyze_files()
        
        return {
            "analysis_date": datetime.now().isoformat(),
            "summary": {
                "total_files": len(self.active_files),
                "backup_files": len(self.backup_files),
                "unique_packages": len(self.packages),
                "total_issues": len(self.issues)
            },
            "active_files": [str(f) for f in self.active_files],
            "backup_files": [str(f) for f in self.backup_files],
            "version_conflicts": self.find_version_conflicts(),
            "duplicate_files": self.find_duplicate_files(),
            "security_issues": self.check_security_issues(),
            "outdated_packages": self.check_outdated_packages(),
            "parse_errors": [issue for issue in self.issues if issue["type"] == "parse_error"],
            "package_inventory": {
                pkg: [(v, f.split('/')[-1]) for v, f in versions] 
                for pkg, versions in self.packages.items()
            }
        }

def main():
    analyzer = RequirementsAnalyzer()
    report = analyzer.generate_report()
    
    # Print summary
    print("=== REQUIREMENTS ANALYSIS REPORT ===")
    print(f"Analysis Date: {report['analysis_date']}")
    print(f"Total Active Files: {report['summary']['total_files']}")
    print(f"Backup Files: {report['summary']['backup_files']}")
    print(f"Unique Packages: {report['summary']['unique_packages']}")
    print()
    
    # Version Conflicts
    if report['version_conflicts']:
        print("🔴 VERSION CONFLICTS FOUND:")
        for conflict in report['version_conflicts']:
            print(f"  Package: {conflict['package']} ({conflict['severity']} severity)")
            for version, files in conflict['versions'].items():
                print(f"    {version}: {len(files)} files")
                for file in files[:3]:  # Show first 3 files
                    print(f"      - {file.split('/')[-2:]}")
                if len(files) > 3:
                    print(f"      ... and {len(files) - 3} more")
            print()
    else:
        print("✅ No version conflicts found")
    
    # Duplicate Files
    if report['duplicate_files']:
        print(f"⚠️  DUPLICATE/SIMILAR FILES ({len(report['duplicate_files'])}):")
        for dup in report['duplicate_files']:
            print(f"  {dup['type']} ({dup['severity']} severity)")
            for file in dup['files']:
                print(f"    - {file}")
            print()
    else:
        print("✅ No duplicate files found")
        
    # Security Issues
    if report['security_issues']:
        print(f"🔒 SECURITY ISSUES ({len(report['security_issues'])}):")
        for issue in report['security_issues']:
            print(f"  {issue['package']} {issue['version']} in {issue['file'].split('/')[-1]}")
            print(f"    {issue['vulnerability']}")
            print()
    else:
        print("✅ No known security issues found")
    
    # Save detailed report
    report_file = "/opt/sutazaiapp/requirements_analysis_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"📄 Detailed report saved to: {report_file}")

if __name__ == "__main__":
    main()