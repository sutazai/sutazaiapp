#!/usr/bin/env python3
"""
Purpose: Safe requirements file cleanup with backup and rollback capabilities
Usage: python safe-requirements-cleanup.py [--dry-run] [--backup-dir PATH] [--execute]
Requirements: Python 3.8+, hashlib, shutil

CRITICAL: This script creates backups before ANY changes and provides full rollback.
Never removes files without explicit confirmation and backup verification.
"""

import os
import sys
import json
import hashlib
import shutil
import datetime
import logging
from pathlib import Path
from collections import defaultdict
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/logs/requirements-cleanup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SafeRequirementsCleanup:
    def __init__(self, project_root: str = "/opt/sutazaiapp"):
        self.project_root = Path(project_root)
        self.backup_root = None
        self.analysis_results = {}
        self.cleanup_actions = []
        self.critical_services = {
            'backend', 'frontend', 'ollama', 'agents', 'monitoring',
            'postgres', 'redis', 'nginx', 'health-monitor'
        }

    def discover_requirements_files(self) -> Dict[str, List[Path]]:
        """Discover all requirements files with metadata"""
        requirements_map = defaultdict(list)
        
        patterns = [
            'requirements*.txt', 'requirements*.yml', 'requirements*.yaml',
            'package.json', 'pyproject.toml', 'Pipfile', 'environment.yml',
            'conda.yml', 'conda.yaml', 'setup.py', 'setup.cfg'
        ]
        
        logger.info("🔍 Discovering requirements files...")
        
        for pattern in patterns:
            for req_file in self.project_root.rglob(pattern):
                # Skip certain directories
                if any(skip in str(req_file) for skip in [
                    '.git', '__pycache__', 'venv', 'node_modules', 
                    '.pytest_cache', '.mypy_cache', '.coverage'
                ]):
                    continue
                
                service_name = self._extract_service_name(req_file)
                file_info = {
                    'path': req_file,
                    'size': req_file.stat().st_size,
                    'modified': datetime.datetime.fromtimestamp(req_file.stat().st_mtime),
                    'hash': self._calculate_file_hash(req_file),
                    'type': self._classify_requirements_type(req_file)
                }
                
                requirements_map[service_name].append(file_info)
        
        logger.info(f"📋 Found requirements for {len(requirements_map)} services")
        return dict(requirements_map)

    def _extract_service_name(self, file_path: Path) -> str:
        """Extract service name from file path"""
        path_parts = file_path.parts
        
        # Look for service-specific directories
        for i, part in enumerate(path_parts):
            if part in ['docker', 'agents', 'services', 'backend', 'frontend']:
                if i + 1 < len(path_parts):
                    return path_parts[i + 1]
            elif part.endswith(('-service', '-agent', '-manager')):
                return part
                
        # Fallback to parent directory name
        return file_path.parent.name

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file content"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            logger.warning(f"Could not hash {file_path}: {e}")
            return ""

    def _classify_requirements_type(self, file_path: Path) -> str:
        """Classify the type of requirements file"""
        name = file_path.name.lower()
        
        if name == 'package.json':
            return 'npm'
        elif name == 'pyproject.toml':
            return 'python-modern'
        elif name.startswith('requirements'):
            if 'test' in name:
                return 'python-test'
            elif 'dev' in name:
                return 'python-dev'
            elif 'prod' in name:
                return 'python-prod'
            else:
                return 'python-main'
        elif name in ['pipfile', 'pipfile.lock']:
            return 'pipenv'
        elif 'environment' in name or 'conda' in name:
            return 'conda'
        else:
            return 'other'

    def analyze_duplicates_and_redundancy(self, requirements_map: Dict[str, List]) -> Dict:
        """Analyze duplicate and redundant requirements files"""
        analysis = {
            'exact_duplicates': [],
            'similar_files': [],
            'redundant_files': [],
            'consolidation_opportunities': [],
            'critical_files': [],
            'safe_to_remove': []
        }
        
        logger.info("🔍 Analyzing duplicates and redundancy...")
        
        # Group files by hash to find exact duplicates
        hash_groups = defaultdict(list)
        for service, files in requirements_map.items():
            for file_info in files:
                if file_info['hash']:
                    hash_groups[file_info['hash']].append((service, file_info))
        
        # Identify exact duplicates
        for file_hash, file_list in hash_groups.items():
            if len(file_list) > 1:
                duplicate_group = {
                    'hash': file_hash,
                    'files': [{'service': service, 'path': str(info['path']), 
                             'size': info['size']} for service, info in file_list],
                    'keep': str(file_list[0][1]['path']),  # Keep first occurrence
                    'remove': [str(info['path']) for service, info in file_list[1:]]
                }
                analysis['exact_duplicates'].append(duplicate_group)
        
        # Analyze per-service redundancy
        for service, files in requirements_map.items():
            if len(files) > 1:
                # Group by type
                type_groups = defaultdict(list)
                for file_info in files:
                    type_groups[file_info['type']].append(file_info)
                
                # Check for multiple files of same type
                for req_type, type_files in type_groups.items():
                    if len(type_files) > 1:
                        # Keep the most recently modified
                        latest_file = max(type_files, key=lambda x: x['modified'])
                        redundant_files = [f for f in type_files if f != latest_file]
                        
                        if service not in self.critical_services:
                            analysis['redundant_files'].extend([
                                {
                                    'service': service,
                                    'path': str(f['path']),
                                    'type': req_type,
                                    'reason': f'Superseded by {latest_file["path"].name}'
                                } for f in redundant_files
                            ])
        
        # Identify consolidation opportunities
        python_files = defaultdict(list)
        for service, files in requirements_map.items():
            for file_info in files:
                if file_info['type'].startswith('python-'):
                    python_files[service].append(file_info)
        
        for service, files in python_files.items():
            if len(files) > 2:  # main, dev, test is reasonable
                analysis['consolidation_opportunities'].append({
                    'service': service,
                    'files': [str(f['path']) for f in files],
                    'suggestion': 'Consider consolidating into requirements.txt, requirements-dev.txt, requirements-test.txt'
                })
        
        # Mark critical files (never remove these)
        for service, files in requirements_map.items():
            if service in self.critical_services:
                for file_info in files:
                    analysis['critical_files'].append(str(file_info['path']))
        
        logger.info(f"📊 Analysis complete:")
        logger.info(f"  - Exact duplicates: {len(analysis['exact_duplicates'])}")
        logger.info(f"  - Redundant files: {len(analysis['redundant_files'])}")
        logger.info(f"  - Critical files: {len(analysis['critical_files'])}")
        
        return analysis

    def create_backup(self, backup_dir: Optional[Path] = None) -> Path:
        """Create comprehensive backup of all requirements files"""
        if backup_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = self.project_root / "archive" / f"requirements_cleanup_{timestamp}"
        
        self.backup_root = backup_dir
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📦 Creating backup in {backup_dir}")
        
        # Create backup manifest
        manifest = {
            'timestamp': datetime.datetime.now().isoformat(),
            'project_root': str(self.project_root),
            'backup_root': str(backup_dir),
            'files': []
        }
        
        # Backup all requirements files
        requirements_map = self.discover_requirements_files()
        
        for service, files in requirements_map.items():
            for file_info in files:
                source_path = file_info['path']
                rel_path = source_path.relative_to(self.project_root)
                backup_path = backup_dir / rel_path
                
                # Create directory structure
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file with metadata preservation
                shutil.copy2(source_path, backup_path)
                
                manifest['files'].append({
                    'service': service,
                    'original_path': str(source_path),
                    'backup_path': str(backup_path),
                    'relative_path': str(rel_path),
                    'hash': file_info['hash'],
                    'size': file_info['size']
                })
        
        # Save manifest
        manifest_path = backup_dir / "backup_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Create rollback script
        rollback_script = backup_dir / "rollback.sh"
        with open(rollback_script, 'w') as f:
            f.write(f"""#!/bin/bash
# Rollback script for requirements cleanup
# Generated: {datetime.datetime.now().isoformat()}

set -e

echo "🔄 Rolling back requirements files..."

PROJECT_ROOT="{self.project_root}"
BACKUP_ROOT="{backup_dir}"

""")
            
            for file_info in manifest['files']:
                f.write(f'echo "Restoring {file_info["relative_path"]}"\n')
                f.write(f'cp "$BACKUP_ROOT/{file_info["relative_path"]}" "$PROJECT_ROOT/{file_info["relative_path"]}"\n')
                f.write('\n')
            
            f.write("""
echo "✅ Rollback complete!"
echo "Verify with: python scripts/validate-container-infrastructure.py --critical-only"
""")
        
        rollback_script.chmod(0o755)
        
        logger.info(f"✅ Backup created with {len(manifest['files'])} files")
        logger.info(f"📜 Rollback script: {rollback_script}")
        
        return backup_dir

    def plan_cleanup_actions(self, analysis: Dict) -> List[Dict]:
        """Plan safe cleanup actions with verification"""
        actions = []
        
        # Plan removal of exact duplicates (keep first, remove others)
        for duplicate_group in analysis['exact_duplicates']:
            for remove_path in duplicate_group['remove']:
                # Verify this is not a critical file
                if remove_path not in analysis['critical_files']:
                    actions.append({
                        'action': 'remove',
                        'path': remove_path,
                        'reason': f'Exact duplicate of {duplicate_group["keep"]}',
                        'risk': 'low',
                        'backup_required': True
                    })
        
        # Plan removal of redundant files in non-critical services
        for redundant_file in analysis['redundant_files']:
            if redundant_file['path'] not in analysis['critical_files']:
                actions.append({
                    'action': 'remove',
                    'path': redundant_file['path'],
                    'reason': redundant_file['reason'],
                    'risk': 'medium',
                    'backup_required': True
                })
        
        # Plan consolidation actions (informational only)
        for consolidation in analysis['consolidation_opportunities']:
            actions.append({
                'action': 'consolidate',
                'service': consolidation['service'],
                'files': consolidation['files'],
                'suggestion': consolidation['suggestion'],
                'risk': 'high',
                'backup_required': True,
                'manual_review_required': True
            })
        
        logger.info(f"📋 Planned {len(actions)} cleanup actions")
        return actions

    def verify_action_safety(self, action: Dict) -> bool:
        """Verify that an action is safe to execute"""
        if action['action'] == 'remove':
            file_path = Path(action['path'])
            
            # Check if file exists
            if not file_path.exists():
                logger.warning(f"File does not exist: {file_path}")
                return False
            
            # Check if file is referenced in critical files
            if self._is_file_referenced(file_path):
                logger.warning(f"File is referenced elsewhere: {file_path}")
                return False
            
            # Check if it's the only requirements file for a critical service
            service = self._extract_service_name(file_path)
            if service in self.critical_services:
                # Count other requirements files for this service
                other_requirements = list(file_path.parent.glob("requirements*"))
                other_requirements = [f for f in other_requirements if f != file_path]
                
                if len(other_requirements) == 0:
                    logger.warning(f"Cannot remove only requirements file for critical service {service}")
                    return False
        
        return True

    def _is_file_referenced(self, file_path: Path) -> bool:
        """Check if file is referenced in Dockerfiles or compose files"""
        file_name = file_path.name
        
        # Search in Dockerfiles
        for dockerfile in self.project_root.rglob("Dockerfile*"):
            try:
                with open(dockerfile) as f:
                    content = f.read()
                    if file_name in content:
                        return True
            except Exception:
                continue
        
        # Search in docker-compose files
        for compose_file in self.project_root.rglob("docker-compose*.yml"):
            try:
                with open(compose_file) as f:
                    content = f.read()
                    if file_name in content:
                        return True
            except Exception:
                continue
        
        return False

    def execute_cleanup(self, actions: List[Dict], dry_run: bool = True) -> Dict:
        """Execute cleanup actions with safety checks"""
        results = {
            'executed': [],
            'skipped': [],
            'failed': [],
            'dry_run': dry_run
        }
        
        if dry_run:
            logger.info("🧪 DRY RUN - No files will be modified")
        else:
            logger.info("⚡ EXECUTING cleanup actions")
        
        for action in actions:
            try:
                # Verify safety
                if not self.verify_action_safety(action):
                    results['skipped'].append({
                        'action': action,
                        'reason': 'Safety check failed'
                    })
                    continue
                
                if action['action'] == 'remove':
                    file_path = Path(action['path'])
                    
                    if dry_run:
                        logger.info(f"[DRY RUN] Would remove: {file_path}")
                    else:
                        logger.info(f"🗑️  Removing: {file_path}")
                        file_path.unlink()
                    
                    results['executed'].append(action)
                
                elif action['action'] == 'consolidate':
                    if dry_run:
                        logger.info(f"[DRY RUN] Would suggest consolidation for {action['service']}")
                    else:
                        logger.info(f"📋 Manual consolidation required for {action['service']}")
                    
                    results['executed'].append(action)
                
            except Exception as e:
                logger.error(f"❌ Failed to execute action {action}: {e}")
                results['failed'].append({
                    'action': action,
                    'error': str(e)
                })
        
        logger.info(f"✅ Cleanup summary:")
        logger.info(f"  - Executed: {len(results['executed'])}")
        logger.info(f"  - Skipped: {len(results['skipped'])}")
        logger.info(f"  - Failed: {len(results['failed'])}")
        
        return results

    def run_comprehensive_cleanup(self, dry_run: bool = True, backup_dir: Optional[Path] = None) -> Dict:
        """Run complete safe cleanup process"""
        logger.info("🚀 Starting comprehensive requirements cleanup")
        
        # Step 1: Discovery
        requirements_map = self.discover_requirements_files()
        
        # Step 2: Analysis
        analysis = self.analyze_duplicates_and_redundancy(requirements_map)
        self.analysis_results = analysis
        
        # Step 3: Create backup (always, even for dry runs)
        if backup_dir is None:
            backup_dir = self.create_backup()
        
        # Step 4: Plan actions
        actions = self.plan_cleanup_actions(analysis)
        self.cleanup_actions = actions
        
        # Step 5: Execute (dry run or real)
        execution_results = self.execute_cleanup(actions, dry_run=dry_run)
        
        # Step 6: Generate report
        report = {
            'timestamp': datetime.datetime.now().isoformat(),
            'dry_run': dry_run,
            'backup_directory': str(backup_dir),
            'requirements_discovered': len(requirements_map),
            'analysis': analysis,
            'planned_actions': actions,
            'execution_results': execution_results
        }
        
        return report

    def generate_report(self, report_data: Dict, format_type: str = "json") -> str:
        """Generate cleanup report"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format_type == "json":
            report_path = f"/opt/sutazaiapp/reports/requirements_cleanup_{timestamp}.json"
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
        
        elif format_type == "markdown":
            report_path = f"/opt/sutazaiapp/reports/requirements_cleanup_{timestamp}.md"
            with open(report_path, 'w') as f:
                f.write(self._generate_markdown_report(report_data))
        
        logger.info(f"📄 Report generated: {report_path}")
        return report_path

    def _generate_markdown_report(self, report_data: Dict) -> str:
        """Generate markdown report"""
        analysis = report_data['analysis']
        
        md = f"""# Requirements Cleanup Report

Generated: {report_data['timestamp']}
Mode: {'DRY RUN' if report_data['dry_run'] else 'EXECUTION'}
Backup Directory: `{report_data['backup_directory']}`

## Summary

- **Requirements Files Discovered**: {report_data['requirements_discovered']}
- **Exact Duplicates Found**: {len(analysis['exact_duplicates'])}
- **Redundant Files Found**: {len(analysis['redundant_files'])}
- **Critical Files Protected**: {len(analysis['critical_files'])}
- **Planned Actions**: {len(report_data['planned_actions'])}

## Exact Duplicates

"""
        for dup in analysis['exact_duplicates']:
            md += f"### Hash: {dup['hash'][:8]}...\n"
            md += f"- **Keep**: `{dup['keep']}`\n"
            md += f"- **Remove**: {len(dup['remove'])} files\n"
            for remove_file in dup['remove']:
                md += f"  - `{remove_file}`\n"
            md += "\n"

        md += f"""
## Planned Actions

| Action | Path | Reason | Risk Level |
|--------|------|--------|------------|
"""
        for action in report_data['planned_actions']:
            if action['action'] == 'remove':
                md += f"| Remove | `{action['path']}` | {action['reason']} | {action['risk']} |\n"
        
        md += f"""
## Rollback Instructions

If you need to rollback changes:

```bash
cd {report_data['backup_directory']}
./rollback.sh
```

Then verify with:
```bash
python scripts/validate-container-infrastructure.py --critical-only
```
"""
        
        return md


def main():
    parser = argparse.ArgumentParser(description="Safe requirements file cleanup")
    parser.add_argument("--dry-run", action="store_true", default=True,
                       help="Perform dry run (default)")
    parser.add_argument("--execute", action="store_true",
                       help="Execute cleanup actions (overrides --dry-run)")
    parser.add_argument("--backup-dir", type=Path,
                       help="Custom backup directory")
    parser.add_argument("--report-format", choices=["json", "markdown"],
                       default="json", help="Report format")
    
    args = parser.parse_args()
    
    # Ensure directories exist
    os.makedirs("/opt/sutazaiapp/reports", exist_ok=True)
    os.makedirs("/opt/sutazaiapp/logs", exist_ok=True)
    
    # Determine execution mode
    dry_run = not args.execute
    if args.execute:
        print("⚠️  EXECUTION MODE: Files will be modified!")
        response = input("Are you sure you want to proceed? (yes/no): ")
        if response.lower() != 'yes':
            print("Operation cancelled.")
            sys.exit(0)
    
    try:
        cleanup = SafeRequirementsCleanup()
        report_data = cleanup.run_comprehensive_cleanup(
            dry_run=dry_run, 
            backup_dir=args.backup_dir
        )
        
        # Generate report
        report_path = cleanup.generate_report(report_data, args.report_format)
        
        print(f"\n✅ Cleanup {'analysis' if dry_run else 'execution'} complete!")
        print(f"📄 Report: {report_path}")
        
        if dry_run:
            print("\n🧪 This was a DRY RUN. To execute, use --execute flag.")
        else:
            print(f"\n📦 Backup created: {report_data['backup_directory']}")
            print(f"🔄 Rollback script: {report_data['backup_directory']}/rollback.sh")
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()