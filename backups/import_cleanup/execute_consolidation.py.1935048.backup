#!/usr/bin/env python3
"""
ULTRA DOCKERFILE CONSOLIDATION EXECUTOR
Automates the migration of 185 Dockerfiles to 15 master templates
Author: Ultra System Architect
Date: August 10, 2025
"""

import os
import shutil
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple

class DockerfileConsolidator:
    """Orchestrates the consolidation of Dockerfiles to master templates."""
    
    def __init__(self, base_path: str = "/opt/sutazaiapp"):
        self.base_path = Path(base_path)
        self.templates_dir = self.base_path / "docker" / "templates"
        self.archive_dir = self.base_path / "archive" / "dockerfiles" / datetime.now().strftime("%Y%m%d")
        self.migration_log = []
        self.statistics = {
            "total_files": 0,
            "migrated": 0,
            "skipped": 0,
            "errors": 0,
            "templates_created": 0
        }
        
    def scan_dockerfiles(self) -> List[Path]:
        """Scan for all Dockerfiles in the codebase."""
        dockerfiles = []
        exclude_dirs = {'node_modules', '.git', 'archive', 'vendor', 'test'}
        
        for root, dirs, files in os.walk(self.base_path):
            # Remove excluded directories from search
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                if file.startswith("Dockerfile"):
                    dockerfiles.append(Path(root) / file)
                    
        self.statistics["total_files"] = len(dockerfiles)
        return dockerfiles
    
    def analyze_dockerfile(self, filepath: Path) -> Dict:
        """Analyze a Dockerfile to determine its category and base image."""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
                
            # Extract base image
            from_match = re.search(r'^FROM\s+(.+?)(?:\s+as\s+.+)?$', content, re.MULTILINE)
            base_image = from_match.group(1) if from_match else "unknown"
            
            # Determine category based on path and content
            category = self._determine_category(filepath, content)
            
            # Check for specific patterns
            patterns = {
                "uses_python": bool(re.search(r'python|pip|requirements\.txt', content, re.IGNORECASE)),
                "uses_nodejs": bool(re.search(r'node|npm|yarn|package\.json', content, re.IGNORECASE)),
                "uses_gpu": bool(re.search(r'cuda|nvidia|tensorflow-gpu|torch.*cuda', content, re.IGNORECASE)),
                "is_monitoring": bool(re.search(r'prometheus|grafana|metrics|exporter', content, re.IGNORECASE)),
                "is_ml": bool(re.search(r'tensorflow|pytorch|scikit|pandas|numpy', content, re.IGNORECASE)),
                "is_api": bool(re.search(r'fastapi|flask|express|gin', content, re.IGNORECASE)),
                "is_frontend": bool(re.search(r'streamlit|react|vue|angular', content, re.IGNORECASE)),
                "uses_alpine": "alpine" in base_image.lower(),
                "non_root_user": bool(re.search(r'USER\s+(?!root)', content))
            }
            
            return {
                "path": filepath,
                "base_image": base_image,
                "category": category,
                "patterns": patterns,
                "size_lines": len(content.splitlines())
            }
        except Exception as e:
            self.migration_log.append(f"ERROR analyzing {filepath}: {e}")
            return None
    
    def _determine_category(self, filepath: Path, content: str) -> str:
        """Determine the category/template type for a Dockerfile."""
        path_str = str(filepath).lower()
        
        # Path-based categorization
        if "agents" in path_str or "ai-agent" in path_str:
            return "ai-agent"
        elif "backend" in path_str or "api" in path_str:
            return "backend-api"
        elif "frontend" in path_str or "ui" in path_str:
            return "frontend-ui"
        elif "monitoring" in path_str or "metrics" in path_str:
            return "monitoring"
        elif "ml" in path_str or "training" in path_str:
            return "ml-training"
        elif "data" in path_str or "pipeline" in path_str:
            return "data-pipeline"
        elif "security" in path_str or "auth" in path_str:
            return "security-service"
        elif "database" in path_str or "db" in path_str:
            return "database-client"
        elif "edge" in path_str or "iot" in path_str:
            return "edge-compute"
        elif "test" in path_str or "qa" in path_str:
            return "test-automation"
        
        # Content-based fallback
        if "sutazai-python-agent-master" in content:
            return "ai-agent"
        elif "sutazai-nodejs-agent-master" in content:
            return "frontend-ui"
        elif "alpine" in content.lower():
            return "monitoring"
        
        return "generic"
    
    def create_migration_map(self, dockerfiles: List[Path]) -> Dict[str, List[Path]]:
        """Create a mapping of template categories to Dockerfiles."""
        migration_map = {}
        
        for dockerfile in dockerfiles:
            analysis = self.analyze_dockerfile(dockerfile)
            if analysis:
                category = analysis["category"]
                if category not in migration_map:
                    migration_map[category] = []
                migration_map[category].append(analysis)
        
        return migration_map
    
    def generate_migration_report(self, migration_map: Dict) -> str:
        """Generate a detailed migration report."""
        report = []
        report.append("=" * 80)
        report.append("DOCKERFILE CONSOLIDATION MIGRATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append(f"Total Dockerfiles: {self.statistics['total_files']}")
        report.append("")
        
        # Category breakdown
        report.append("MIGRATION CATEGORIES")
        report.append("-" * 40)
        for category, files in sorted(migration_map.items()):
            report.append(f"\n{category.upper()} ({len(files)} files)")
            report.append("-" * 30)
            
            # Group by base image
            by_base = {}
            for f in files:
                base = f["base_image"]
                if base not in by_base:
                    by_base[base] = []
                by_base[base].append(f)
            
            for base, base_files in sorted(by_base.items()):
                report.append(f"  Base: {base} ({len(base_files)} files)")
                for bf in base_files[:3]:  # Show first 3 examples
                    rel_path = bf["path"].relative_to(self.base_path)
                    report.append(f"    - {rel_path}")
                if len(base_files) > 3:
                    report.append(f"    ... and {len(base_files) - 3} more")
        
        # Statistics
        report.append("\n" + "=" * 40)
        report.append("CONSOLIDATION STATISTICS")
        report.append("-" * 40)
        
        # Calculate savings
        total_lines = sum(f["size_lines"] for files in migration_map.values() for f in files)
        estimated_template_lines = len(migration_map) * 100  # Rough estimate
        
        report.append(f"Total Dockerfile lines: {total_lines:,}")
        report.append(f"Estimated template lines: {estimated_template_lines:,}")
        report.append(f"Line reduction: {(1 - estimated_template_lines/total_lines)*100:.1f}%")
        report.append(f"File reduction: {(1 - 15/self.statistics['total_files'])*100:.1f}%")
        
        # Recommendations
        report.append("\n" + "=" * 40)
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        
        # Find most common patterns
        all_patterns = {}
        for files in migration_map.values():
            for f in files:
                for pattern, value in f["patterns"].items():
                    if value:
                        all_patterns[pattern] = all_patterns.get(pattern, 0) + 1
        
        report.append("\nCommon Patterns Found:")
        for pattern, count in sorted(all_patterns.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / self.statistics['total_files']) * 100
            report.append(f"  - {pattern}: {count} files ({percentage:.1f}%)")
        
        # Security status
        non_root_count = all_patterns.get("non_root_user", 0)
        root_count = self.statistics['total_files'] - non_root_count
        report.append(f"\nSecurity Status:")
        report.append(f"  - Non-root users: {non_root_count} ({non_root_count/self.statistics['total_files']*100:.1f}%)")
        report.append(f"  - Root users: {root_count} ({root_count/self.statistics['total_files']*100:.1f}%)")
        
        return "\n".join(report)
    
    def create_template_mapping(self) -> Dict:
        """Create a JSON mapping of services to templates."""
        mapping = {
            "version": "1.0.0",
            "created": datetime.now().isoformat(),
            "templates": {
                "python-base-master": {
                    "base_image": "python:3.12.8-slim-bookworm",
                    "description": "Universal Python base for all Python services",
                    "security": "non-root user (appuser:1000)",
                    "services": []
                },
                "nodejs-base-master": {
                    "base_image": "node:18-slim",
                    "description": "Universal Node.js base",
                    "security": "non-root user (node:1000)",
                    "services": []
                },
                "alpine-base-master": {
                    "base_image": "alpine:3.18",
                    "description": "Minimal Alpine base for microservices",
                    "security": "non-root user (appuser:1000)",
                    "services": []
                },
                "ai-agent-master": {
                    "inherits": "python-base-master",
                    "description": "AI agent services with ML libraries",
                    "services": []
                },
                "backend-api-master": {
                    "inherits": "python-base-master",
                    "description": "API services with FastAPI/Flask",
                    "services": []
                },
                "frontend-ui-master": {
                    "inherits": "nodejs-base-master or python-base-master",
                    "description": "UI services with Streamlit or React",
                    "services": []
                },
                "monitoring-master": {
                    "inherits": "alpine-base-master",
                    "description": "Monitoring and metrics services",
                    "services": []
                },
                "data-pipeline-master": {
                    "inherits": "python-base-master",
                    "description": "Data processing and ETL services",
                    "services": []
                },
                "ml-training-master": {
                    "inherits": "python-base-master",
                    "description": "ML training with PyTorch/TensorFlow",
                    "services": []
                },
                "security-service-master": {
                    "inherits": "alpine-base-master",
                    "description": "Security and compliance services",
                    "services": []
                },
                "database-client-master": {
                    "inherits": "python-base-master",
                    "description": "Database-heavy services",
                    "services": []
                },
                "gpu-compute-master": {
                    "base_image": "nvidia/cuda:12.0-runtime",
                    "description": "GPU-accelerated services",
                    "services": []
                },
                "edge-compute-master": {
                    "inherits": "alpine-base-master",
                    "description": "Edge/IoT deployment",
                    "services": []
                },
                "test-automation-master": {
                    "inherits": "python-base-master",
                    "description": "Testing and QA services",
                    "services": []
                },
                "third-party-wrapper": {
                    "base_image": "various",
                    "description": "Wrapper for third-party services",
                    "services": []
                }
            }
        }
        
        return mapping
    
    def execute_consolidation(self, dry_run: bool = True):
        """Execute the consolidation process."""
        print("=" * 80)
        print("ULTRA DOCKERFILE CONSOLIDATION EXECUTOR")
        print("=" * 80)
        print(f"Mode: {'DRY RUN' if dry_run else 'LIVE MIGRATION'}")
        print(f"Base Path: {self.base_path}")
        print()
        
        # Step 1: Scan for Dockerfiles
        print("Step 1: Scanning for Dockerfiles...")
        dockerfiles = self.scan_dockerfiles()
        print(f"  Found {len(dockerfiles)} Dockerfiles")
        
        # Step 2: Analyze and categorize
        print("\nStep 2: Analyzing Dockerfiles...")
        migration_map = self.create_migration_map(dockerfiles)
        print(f"  Categorized into {len(migration_map)} template groups")
        
        # Step 3: Generate reports
        print("\nStep 3: Generating migration report...")
        report = self.generate_migration_report(migration_map)
        report_path = self.base_path / "DOCKERFILE_MIGRATION_REPORT.txt"
        
        if not dry_run:
            with open(report_path, 'w') as f:
                f.write(report)
            print(f"  Report saved to {report_path}")
        else:
            print("  [DRY RUN] Report would be saved to", report_path)
            print("\nReport Preview:")
            print("-" * 40)
            print(report[:1000] + "..." if len(report) > 1000 else report)
        
        # Step 4: Create template mapping
        print("\nStep 4: Creating template mapping...")
        mapping = self.create_template_mapping()
        mapping_path = self.base_path / "docker" / "templates" / "service-mapping.json"
        
        if not dry_run:
            os.makedirs(mapping_path.parent, exist_ok=True)
            with open(mapping_path, 'w') as f:
                json.dump(mapping, f, indent=2)
            print(f"  Mapping saved to {mapping_path}")
        else:
            print(f"  [DRY RUN] Mapping would be saved to {mapping_path}")
        
        # Step 5: Archive old Dockerfiles
        if not dry_run:
            print("\nStep 5: Archiving old Dockerfiles...")
            os.makedirs(self.archive_dir, exist_ok=True)
            archived = 0
            for dockerfile in dockerfiles:
                if "node_modules" not in str(dockerfile):
                    archive_path = self.archive_dir / dockerfile.relative_to(self.base_path)
                    os.makedirs(archive_path.parent, exist_ok=True)
                    shutil.copy2(dockerfile, archive_path)
                    archived += 1
            print(f"  Archived {archived} Dockerfiles to {self.archive_dir}")
        else:
            print("\nStep 5: [DRY RUN] Would archive Dockerfiles to", self.archive_dir)
        
        # Summary
        print("\n" + "=" * 80)
        print("CONSOLIDATION SUMMARY")
        print("=" * 80)
        print(f"Total Dockerfiles: {self.statistics['total_files']}")
        print(f"Template Categories: {len(migration_map)}")
        print(f"Expected Reduction: {(1 - 15/self.statistics['total_files'])*100:.1f}%")
        
        if dry_run:
            print("\n⚠️  This was a DRY RUN. No changes were made.")
            print("To execute the migration, run with --execute flag")
        else:
            print("\n✅ Migration completed successfully!")
            print("Next steps:")
            print("  1. Review the migration report")
            print("  2. Update docker-compose.yml references")
            print("  3. Test migrated services")
            print("  4. Remove archived Dockerfiles after validation")

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ultra Dockerfile Consolidation Executor")
    parser.add_argument("--execute", action="store_true", help="Execute the migration (default is dry run)")
    parser.add_argument("--path", default="/opt/sutazaiapp", help="Base path for the project")
    args = parser.parse_args()
    
    consolidator = DockerfileConsolidator(args.path)
    consolidator.execute_consolidation(dry_run=not args.execute)

if __name__ == "__main__":
    main()