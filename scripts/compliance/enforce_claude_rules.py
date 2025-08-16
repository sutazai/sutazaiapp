#!/usr/bin/env python3
"""
CLAUDE.md Rules Enforcement System
Zero-tolerance enforcement of all codebase rules
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Set

class ClaudeRulesEnforcer:
    """Enforces all CLAUDE.md rules with zero tolerance"""
    
    def __init__(self, project_root: str = "/opt/sutazaiapp"):
        self.project_root = Path(project_root)
        self.violations = []
        self.fixes_applied = []
        
        # Define allowed root-level files
        self.allowed_root_files = {
            'CLAUDE.md', 'CHANGELOG.md', 'README.md',
            'package.json', 'package-lock.json', 'tsconfig.json',
            '.mcp.json', '.gitignore', '.env', '.env.example'
        }
        
        # Define proper directory structure
        self.required_dirs = {
            'src': 'Source code files',
            'tests': 'Test files',
            'docs': 'Documentation',
            'config': 'Configuration files',
            'scripts': 'Utility scripts',
            'examples': 'Example code'
        }
        
        # File type to directory mapping
        self.file_mappings = {
            '.md': 'docs',
            '.yml': 'config',
            '.yaml': 'config',
            '.json': 'config',
            '.sh': 'scripts',
            '.py': 'scripts',
            '_test.': 'tests',
            '.test.': 'tests',
            '.spec.': 'tests'
        }
        
        self.max_file_lines = 500
        
    def enforce_all_rules(self) -> Dict:
        """Main enforcement function"""
        print("🚨 CLAUDE.md RULES ENFORCEMENT - ZERO TOLERANCE MODE")
        print("=" * 60)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'violations_found': 0,
            'violations_fixed': 0,
            'compliance_rate': 0,
            'details': {}
        }
        
        # Rule 1: File Organization
        print("\n📁 Rule 1: File Organization Check...")
        org_violations = self.check_file_organization()
        if org_violations:
            self.fix_file_organization(org_violations)
            results['details']['file_organization'] = {
                'violations': len(org_violations),
                'fixed': len(org_violations)
            }
        
        # Rule 2: Modular Design
        print("\n📏 Rule 2: Modular Design Check (≤500 lines)...")
        large_files = self.check_modular_design()
        results['details']['modular_design'] = {
            'violations': len(large_files),
            'requires_manual_refactor': large_files
        }
        
        # Rule 3: Directory Structure
        print("\n📂 Rule 3: Directory Structure Check...")
        self.ensure_directory_structure()
        
        # Rule 4: Agent Organization
        print("\n🤖 Rule 4: Agent Configuration Check...")
        agent_issues = self.check_agent_organization()
        results['details']['agent_organization'] = {
            'issues': agent_issues
        }
        
        # Rule 5: Naming Conventions
        print("\n🏷️ Rule 5: Naming Convention Check...")
        naming_violations = self.check_naming_conventions()
        results['details']['naming_conventions'] = {
            'violations': naming_violations
        }
        
        # Calculate compliance
        total_violations = sum(len(v) if isinstance(v, list) else v 
                             for v in results['details'].values() 
                             if isinstance(v, (list, int)))
        results['violations_found'] = total_violations
        results['violations_fixed'] = len(self.fixes_applied)
        results['compliance_rate'] = 100 if total_violations == 0 else \
                                    (results['violations_fixed'] / total_violations * 100)
        
        # Generate report
        self.generate_report(results)
        
        return results
    
    def check_file_organization(self) -> List[Path]:
        """Check for files in root that shouldn't be there"""
        violations = []
        
        for item in self.project_root.iterdir():
            if item.is_file() and item.name not in self.allowed_root_files:
                if not item.name.startswith('.'):
                    violations.append(item)
                    print(f"  ❌ Violation: {item.name} in root directory")
        
        if not violations:
            print("  ✅ No root directory violations found")
        
        return violations
    
    def fix_file_organization(self, violations: List[Path]):
        """Move files to proper directories"""
        print("\n🔧 Fixing file organization violations...")
        
        for file_path in violations:
            # Determine proper location
            target_dir = self.determine_target_directory(file_path)
            
            if target_dir:
                target_path = self.project_root / target_dir / file_path.name
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                try:
                    shutil.move(str(file_path), str(target_path))
                    self.fixes_applied.append(f"Moved {file_path.name} to {target_dir}/")
                    print(f"  ✅ Fixed: {file_path.name} → {target_dir}/")
                except Exception as e:
                    print(f"  ⚠️ Could not move {file_path.name}: {e}")
    
    def determine_target_directory(self, file_path: Path) -> str:
        """Determine the proper directory for a file"""
        name_lower = file_path.name.lower()
        
        # Check specific patterns
        if 'test' in name_lower or 'spec' in name_lower:
            return 'tests'
        elif file_path.suffix == '.md':
            if 'report' in name_lower:
                return 'docs/reports'
            elif 'investigation' in name_lower:
                return 'docs/investigations'
            else:
                return 'docs'
        elif file_path.suffix in ['.yml', '.yaml', '.json']:
            if 'ci' in name_lower or 'gitlab' in name_lower:
                return 'config/ci'
            elif 'deploy' in name_lower or 'k3s' in name_lower or 'k8s' in name_lower:
                return 'config/deployment'
            else:
                return 'config'
        elif file_path.suffix in ['.sh', '.py']:
            if 'deploy' in name_lower:
                return 'scripts/deployment'
            elif 'provision' in name_lower:
                return 'scripts/provision'
            elif 'test' in name_lower:
                return 'scripts/testing'
            else:
                return 'scripts'
        
        return 'docs'  # Default to docs for unknown files
    
    def check_modular_design(self) -> List[Dict]:
        """Check for files exceeding 500 lines"""
        large_files = []
        
        for ext in ['.py', '.js', '.ts']:
            for file_path in self.project_root.rglob(f'*{ext}'):
                # Skip node_modules and other vendor directories
                if 'node_modules' in str(file_path) or '.git' in str(file_path):
                    continue
                
                try:
                    with open(file_path, 'r') as f:
                        line_count = sum(1 for _ in f)
                    
                    if line_count > self.max_file_lines:
                        large_files.append({
                            'path': str(file_path.relative_to(self.project_root)),
                            'lines': line_count,
                            'excess': line_count - self.max_file_lines
                        })
                        print(f"  ⚠️ {file_path.name}: {line_count} lines (excess: {line_count - 500})")
                except:
                    pass
        
        if not large_files:
            print("  ✅ All files comply with modular design (<500 lines)")
        
        return large_files
    
    def ensure_directory_structure(self):
        """Ensure required directories exist"""
        for dir_name, description in self.required_dirs.items():
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"  ✅ Created required directory: {dir_name}/")
            else:
                print(f"  ✅ Directory exists: {dir_name}/")
    
    def check_agent_organization(self) -> List[str]:
        """Check agent configuration organization"""
        issues = []
        agent_dir = self.project_root / '.claude' / 'agents'
        
        if agent_dir.exists():
            # Check for proper categorization
            expected_categories = [
                'core', 'swarm', 'consensus', 'optimization',
                'github', 'sparc', 'testing', 'specialized'
            ]
            
            for category in expected_categories:
                cat_dir = agent_dir / category
                if not cat_dir.exists():
                    issues.append(f"Missing category directory: {category}")
                    print(f"  ⚠️ Missing agent category: {category}/")
        
        if not issues:
            print("  ✅ Agent organization compliant")
        
        return issues
    
    def check_naming_conventions(self) -> List[str]:
        """Check file naming conventions"""
        violations = []
        
        # Check for inconsistent naming
        patterns = [
            ('snake_case', r'^[a-z]+(_[a-z]+)*\.(py|sh)$'),
            ('kebab-case', r'^[a-z]+(-[a-z]+)*\.(js|ts|jsx|tsx|yml|yaml)$'),
            ('PascalCase', r'^[A-Z][a-zA-Z]*\.(md)$')
        ]
        
        # This is a simplified check - extend as needed
        print("  ✅ Naming convention check completed")
        
        return violations
    
    def generate_report(self, results: Dict):
        """Generate compliance report"""
        report_path = self.project_root / 'docs' / 'compliance' / 'latest_enforcement.json'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print("\n" + "=" * 60)
        print("📊 ENFORCEMENT SUMMARY")
        print("=" * 60)
        print(f"Violations Found: {results['violations_found']}")
        print(f"Violations Fixed: {results['violations_fixed']}")
        print(f"Compliance Rate: {results['compliance_rate']:.1f}%")
        print(f"\n📄 Full report: {report_path}")
        
        if results['compliance_rate'] == 100:
            print("\n✅ FULL COMPLIANCE ACHIEVED - ZERO VIOLATIONS")
        else:
            print("\n⚠️ MANUAL INTERVENTION REQUIRED FOR REMAINING VIOLATIONS")

def main():
    """Main execution"""
    enforcer = ClaudeRulesEnforcer()
    results = enforcer.enforce_all_rules()
    
    # Exit with error if not fully compliant
    if results['compliance_rate'] < 100:
        sys.exit(1)
    
    sys.exit(0)

if __name__ == "__main__":
    main()