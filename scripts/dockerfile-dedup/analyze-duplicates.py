#!/usr/bin/env python3
"""
Ultra-Deep Dockerfile Deduplication Analyzer
Identifies exact duplicates, near-duplicates, and consolidation opportunities
Author: System Architect
Date: August 10, 2025
"""

import os
import hashlib
import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import difflib
from datetime import datetime

# Configuration
BASE_DIR = Path("/opt/sutazaiapp")
REPORT_DIR = BASE_DIR / "reports" / "dockerfile-dedup"
ARCHIVE_DIR = BASE_DIR / "archive" / "dockerfile-backups"

# Create directories
REPORT_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

class DockerfileAnalyzer:
    """Comprehensive Dockerfile analysis and deduplication tool"""
    
    def __init__(self):
        self.dockerfiles = []
        self.duplicates = defaultdict(list)
        self.near_duplicates = defaultdict(list)
        self.base_images = defaultdict(list)
        self.categories = defaultdict(list)
        self.stats = {
            'total_files': 0,
            'unique_files': 0,
            'exact_duplicates': 0,
            'near_duplicates': 0,
            'migration_backups': 0,
            'total_lines': 0,
            'total_size_mb': 0
        }
    
    def find_all_dockerfiles(self) -> List[Path]:
        """Find all Dockerfiles in the codebase"""
        dockerfiles = []
        for pattern in ['Dockerfile', 'Dockerfile.*', '*.Dockerfile']:
            dockerfiles.extend(BASE_DIR.glob(f"**/{pattern}"))
        
        # Filter out archive and backup directories
        dockerfiles = [
            f for f in dockerfiles 
            if 'archive' not in str(f) and 'backup' not in str(f)
        ]
        
        self.stats['total_files'] = len(dockerfiles)
        return dockerfiles
    
    def calculate_hash(self, filepath: Path) -> str:
        """Calculate MD5 hash of file content"""
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def extract_base_image(self, filepath: Path) -> str:
        """Extract base image from Dockerfile"""
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    if line.strip().startswith('FROM'):
                        return line.split()[1].split(':')[0]
        except:
            pass
        return 'unknown'
    
    def calculate_similarity(self, file1: Path, file2: Path) -> float:
        """Calculate similarity percentage between two files"""
        try:
            with open(file1, 'r') as f1, open(file2, 'r') as f2:
                lines1 = f1.readlines()
                lines2 = f2.readlines()
                
            matcher = difflib.SequenceMatcher(None, lines1, lines2)
            return matcher.ratio()
        except:
            return 0.0
    
    def categorize_dockerfile(self, filepath: Path) -> str:
        """Categorize Dockerfile based on content and location"""
        path_str = str(filepath)
        content = filepath.read_text()
        
        # Check location-based categories
        if 'agents' in path_str:
            if 'ai' in path_str.lower() or 'ml' in content.lower():
                return 'ai-agent'
            return 'agent'
        elif 'frontend' in path_str:
            return 'frontend'
        elif 'backend' in path_str:
            return 'backend'
        elif 'monitoring' in path_str:
            return 'monitoring'
        elif 'docker/base' in path_str:
            return 'base-image'
        
        # Check content-based categories
        if 'tensorflow' in content.lower() or 'pytorch' in content.lower():
            return 'ml-heavy'
        elif 'node' in content.lower() or 'npm' in content.lower():
            return 'nodejs'
        elif 'python' in content.lower():
            return 'python'
        elif 'nginx' in content.lower():
            return 'nginx'
        
        return 'other'
    
    def analyze_dockerfiles(self):
        """Perform comprehensive analysis"""
        print("🔍 Starting Dockerfile analysis...")
        
        dockerfiles = self.find_all_dockerfiles()
        
        # Group by hash for exact duplicates
        hash_groups = defaultdict(list)
        
        for dockerfile in dockerfiles:
            # Skip security migration backups
            if 'security_migration' in str(dockerfile):
                self.stats['migration_backups'] += 1
                continue
            
            file_hash = self.calculate_hash(dockerfile)
            hash_groups[file_hash].append(dockerfile)
            
            # Extract metadata
            base_image = self.extract_base_image(dockerfile)
            self.base_images[base_image].append(dockerfile)
            
            category = self.categorize_dockerfile(dockerfile)
            self.categories[category].append(dockerfile)
            
            # Calculate file stats
            self.stats['total_lines'] += len(dockerfile.read_text().splitlines())
            self.stats['total_size_mb'] += dockerfile.stat().st_size / 1024 / 1024
        
        # Identify exact duplicates
        for file_hash, files in hash_groups.items():
            if len(files) > 1:
                self.duplicates[file_hash] = files
                self.stats['exact_duplicates'] += len(files) - 1
        
        self.stats['unique_files'] = len(hash_groups)
        
        # Find near-duplicates (>80% similar)
        unique_files = [files[0] for files in hash_groups.values()]
        
        for i, file1 in enumerate(unique_files):
            similar_files = []
            for j, file2 in enumerate(unique_files):
                if i >= j:
                    continue
                
                similarity = self.calculate_similarity(file1, file2)
                if similarity > 0.8 and similarity < 1.0:
                    similar_files.append((file2, similarity))
            
            if similar_files:
                self.near_duplicates[str(file1)] = similar_files
                self.stats['near_duplicates'] += len(similar_files)
    
    def generate_deduplication_plan(self) -> Dict:
        """Generate actionable deduplication plan"""
        plan = {
            'phase1_immediate': [],
            'phase2_exact_duplicates': [],
            'phase3_near_duplicates': [],
            'phase4_category_consolidation': [],
            'phase5_base_image_migration': []
        }
        
        # Phase 1: Immediate cleanup (migration backups)
        for dockerfile in self.find_all_dockerfiles():
            if 'security_migration' in str(dockerfile):
                plan['phase1_immediate'].append({
                    'action': 'archive',
                    'file': str(dockerfile),
                    'reason': 'Security migration backup'
                })
        
        # Phase 2: Exact duplicates
        for file_hash, files in self.duplicates.items():
            if len(files) > 1:
                keep = files[0]
                for duplicate in files[1:]:
                    plan['phase2_exact_duplicates'].append({
                        'action': 'remove',
                        'file': str(duplicate),
                        'duplicate_of': str(keep),
                        'hash': file_hash
                    })
        
        # Phase 3: Near duplicates
        for original, similar_files in self.near_duplicates.items():
            for similar, similarity in similar_files:
                plan['phase3_near_duplicates'].append({
                    'action': 'consolidate',
                    'file': str(similar),
                    'similar_to': original,
                    'similarity': f"{similarity * 100:.1f}%"
                })
        
        # Phase 4: Category consolidation
        for category, files in self.categories.items():
            if len(files) > 5:  # Categories with many files
                plan['phase4_category_consolidation'].append({
                    'category': category,
                    'count': len(files),
                    'action': f'Create sutazai-{category}-base image',
                    'files': [str(f) for f in files[:5]]  # Sample
                })
        
        # Phase 5: Base image standardization
        for base_image, files in self.base_images.items():
            if base_image.startswith('python:3.') and len(files) > 10:
                plan['phase5_base_image_migration'].append({
                    'current_base': base_image,
                    'target_base': 'sutazai-python-agent-master',
                    'count': len(files),
                    'action': 'migrate'
                })
        
        return plan
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = REPORT_DIR / f"dockerfile_analysis_{timestamp}.json"
        
        report = {
            'timestamp': timestamp,
            'statistics': self.stats,
            'deduplication_potential': {
                'files_to_remove': self.stats['exact_duplicates'] + self.stats['migration_backups'],
                'files_to_consolidate': self.stats['near_duplicates'],
                'estimated_final_count': self.stats['unique_files'] - self.stats['exact_duplicates'],
                'reduction_percentage': (1 - (self.stats['unique_files'] / self.stats['total_files'])) * 100
            },
            'base_images': {
                base: len(files) for base, files in self.base_images.items()
            },
            'categories': {
                cat: len(files) for cat, files in self.categories.items()
            },
            'deduplication_plan': self.generate_deduplication_plan()
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report_file
    
    def print_summary(self):
        """Print analysis summary to console"""
        print("\n" + "="*60)
        print("  DOCKERFILE DEDUPLICATION ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"\n📊 Statistics:")
        print(f"  Total Dockerfiles: {self.stats['total_files']}")
        print(f"  Unique files: {self.stats['unique_files']}")
        print(f"  Exact duplicates: {self.stats['exact_duplicates']}")
        print(f"  Near duplicates: {self.stats['near_duplicates']}")
        print(f"  Migration backups: {self.stats['migration_backups']}")
        print(f"  Total size: {self.stats['total_size_mb']:.2f} MB")
        
        print(f"\n🎯 Deduplication Potential:")
        reduction = self.stats['exact_duplicates'] + self.stats['migration_backups']
        print(f"  Immediate removal: {reduction} files")
        print(f"  Files to consolidate: {self.stats['near_duplicates']}")
        final_count = self.stats['total_files'] - reduction
        print(f"  Estimated final count: {final_count}")
        reduction_pct = (reduction / self.stats['total_files']) * 100
        print(f"  Reduction: {reduction_pct:.1f}%")
        
        print(f"\n📦 Top Base Images:")
        sorted_bases = sorted(self.base_images.items(), key=lambda x: len(x[1]), reverse=True)
        for base, files in sorted_bases[:5]:
            print(f"  {base}: {len(files)} files")
        
        print(f"\n🏷️ Categories:")
        for category, files in sorted(self.categories.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"  {category}: {len(files)} files")
        
        print(f"\n✅ Recommendations:")
        print("  1. Archive 55 security migration backups immediately")
        print(f"  2. Remove {self.stats['exact_duplicates']} exact duplicates")
        print("  3. Create 5 category-based master images")
        print("  4. Migrate services to use master images")
        print("  5. Consolidate near-duplicates through inheritance")

def main():
    """Main execution"""
    analyzer = DockerfileAnalyzer()
    
    # Run analysis
    analyzer.analyze_dockerfiles()
    
    # Generate report
    report_file = analyzer.generate_report()
    
    # Print summary
    analyzer.print_summary()
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    # Generate actionable script
    script_file = REPORT_DIR / "deduplication_commands.sh"
    with open(script_file, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Auto-generated Dockerfile deduplication commands\n")
        f.write(f"# Generated: {datetime.now()}\n\n")
        
        plan = analyzer.generate_deduplication_plan()
        
        f.write("# Phase 1: Archive migration backups\n")
        for item in plan['phase1_immediate']:
            f.write(f"mv {item['file']} {ARCHIVE_DIR}/\n")
        
        f.write("\n# Phase 2: Remove exact duplicates\n")
        for item in plan['phase2_exact_duplicates']:
            f.write(f"# Duplicate of {item['duplicate_of']}\n")
            f.write(f"rm {item['file']}\n")
    
    os.chmod(script_file, 0o755)
    print(f"📜 Executable commands saved to: {script_file}")

if __name__ == "__main__":
    main()