#!/usr/bin/env python3
"""
ULTRAORGANIZED Script Deduplication Tool
Purpose: Identify and remove duplicate scripts across the system
Generated: August 11, 2025
Rule Compliance: Rules 1, 2, 4
"""

import os
import sys
import hashlib
import difflib
from pathlib import Path
from collections import defaultdict
import json

class ScriptDeduplicator:
    """Identifies and manages duplicate scripts across the system"""
    
    def __init__(self, project_root="/opt/sutazaiapp"):
        self.project_root = Path(project_root)
        self.script_hashes = defaultdict(list)
        self.duplicate_groups = []
        self.similar_groups = []
        
    def calculate_file_hash(self, file_path):
        """Calculate MD5 hash of file content"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None
    
    def calculate_content_hash(self, file_path):
        """Calculate hash of normalized content (ignoring whitespace/comments)"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # Normalize content: remove comments, empty lines, normalize whitespace
            normalized_lines = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('//'):
                    normalized_lines.append(line.lower())
            
            normalized_content = '\n'.join(normalized_lines)
            return hashlib.md5(normalized_content.encode()).hexdigest()
        except Exception as e:
            print(f"Error normalizing {file_path}: {e}")
            return None
    
    def find_script_files(self):
        """Find all script files in the project"""
        script_extensions = {'.sh', '.py', '.js', '.ts', '.yml', '.yaml'}
        script_files = []
        
        for root, dirs, files in os.walk(self.project_root):
            # Skip certain directories
            skip_dirs = {'.git', '__pycache__', 'node_modules', '.pytest_cache', 'venv', 'env'}
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix in script_extensions:
                    script_files.append(file_path)
        
        return script_files
    
    def analyze_duplicates(self):
        """Analyze all scripts and identify duplicates"""
        print("🔍 ANALYZING SCRIPT DUPLICATES...")
        
        script_files = self.find_script_files()
        print(f"Found {len(script_files)} script files to analyze")
        
        # Group files by exact hash
        exact_hashes = defaultdict(list)
        content_hashes = defaultdict(list)
        
        for file_path in script_files:
            # Exact hash
            exact_hash = self.calculate_file_hash(file_path)
            if exact_hash:
                exact_hashes[exact_hash].append(file_path)
            
            # Content hash (normalized)
            content_hash = self.calculate_content_hash(file_path)
            if content_hash:
                content_hashes[content_hash].append(file_path)
        
        # Find exact duplicates
        exact_duplicates = {h: files for h, files in exact_hashes.items() if len(files) > 1}
        print(f"Found {len(exact_duplicates)} groups of exact duplicates")
        
        # Find content duplicates (similar functionality)
        content_duplicates = {h: files for h, files in content_hashes.items() if len(files) > 1}
        print(f"Found {len(content_duplicates)} groups of content duplicates")
        
        return exact_duplicates, content_duplicates
    
    def analyze_functionality(self, file_path):
        """Analyze what functionality a script provides"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().lower()
            
            # Identify functionality patterns
            functionalities = []
            
            patterns = {
                'deployment': ['deploy', 'start', 'up', 'launch'],
                'health_check': ['health', 'status', 'check', 'ping'],
                'backup': ['backup', 'dump', 'export', 'archive'],
                'cleanup': ['cleanup', 'clean', 'remove', 'delete', 'prune'],
                'monitoring': ['monitor', 'watch', 'observe', 'metrics'],
                'security': ['security', 'auth', 'ssl', 'tls', 'cert'],
                'database': ['postgres', 'redis', 'neo4j', 'db', 'sql'],
                'docker': ['docker', 'container', 'image', 'compose'],
                'testing': ['test', 'spec', 'assert', 'validate'],
                'maintenance': ['maintain', 'update', 'upgrade', 'patch']
            }
            
            for category, keywords in patterns.items():
                if any(keyword in content for keyword in keywords):
                    functionalities.append(category)
            
            return functionalities
        except Exception as e:
            return []
    
    def generate_deduplication_plan(self):
        """Generate a plan for deduplicating scripts"""
        exact_duplicates, content_duplicates = self.analyze_duplicates()
        
        plan = {
            'summary': {
                'total_scripts': 0,
                'exact_duplicate_groups': len(exact_duplicates),
                'content_duplicate_groups': len(content_duplicates),
                'estimated_removal': 0
            },
            'exact_duplicates': [],
            'content_duplicates': [],
            'recommendations': []
        }
        
        # Process exact duplicates
        for hash_key, files in exact_duplicates.items():
            if len(files) < 2:
                continue
                
            # Choose the "master" file (shortest path, or most central location)
            master_file = min(files, key=lambda x: (len(str(x)), str(x)))
            duplicates = [f for f in files if f != master_file]
            
            plan['exact_duplicates'].append({
                'master': str(master_file),
                'duplicates': [str(f) for f in duplicates],
                'hash': hash_key,
                'size': len(duplicates)
            })
            
            plan['summary']['estimated_removal'] += len(duplicates)
        
        # Process content duplicates
        for hash_key, files in content_duplicates.items():
            if len(files) < 2:
                continue
                
            # Analyze functionality
            file_info = []
            for file_path in files:
                functionalities = self.analyze_functionality(file_path)
                file_info.append({
                    'path': str(file_path),
                    'functionalities': functionalities,
                    'size': file_path.stat().st_size if file_path.exists() else 0
                })
            
            plan['content_duplicates'].append({
                'files': file_info,
                'hash': hash_key,
                'count': len(files)
            })
        
        # Generate recommendations
        plan['recommendations'] = [
            "Consolidate exact duplicates by keeping the master file and removing duplicates",
            "Review content duplicates for potential merging opportunities",
            "Create shared library functions for common operations",
            "Establish naming conventions to prevent future duplication",
            "Implement script organization standards"
        ]
        
        return plan
    
    def save_plan(self, plan, output_file="deduplication_plan.json"):
        """Save the deduplication plan to a file"""
        output_path = self.project_root / output_file
        with open(output_path, 'w') as f:
            json.dump(plan, f, indent=2)
        print(f"📄 Deduplication plan saved to: {output_path}")

def main():
    """Main execution function"""
    print("🚀 ULTRAORGANIZED Script Deduplication Tool")
    print("=" * 60)
    
    deduplicator = ScriptDeduplicator()
    plan = deduplicator.generate_deduplication_plan()
    
    # Display summary
    print("\n📊 DEDUPLICATION ANALYSIS SUMMARY")
    print("=" * 40)
    print(f"Exact Duplicate Groups: {plan['summary']['exact_duplicate_groups']}")
    print(f"Content Duplicate Groups: {plan['summary']['content_duplicate_groups']}")
    print(f"Scripts That Can Be Removed: {plan['summary']['estimated_removal']}")
    
    # Show top exact duplicates
    if plan['exact_duplicates']:
        print("\n🎯 TOP EXACT DUPLICATES TO REMOVE:")
        for i, duplicate_group in enumerate(plan['exact_duplicates'][:5], 1):
            print(f"{i}. Master: {duplicate_group['master']}")
            print(f"   Remove: {len(duplicate_group['duplicates'])} duplicates")
    
    # Save detailed plan
    deduplicator.save_plan(plan)
    
    print("\n✅ DEDUPLICATION ANALYSIS COMPLETE")
    print("📋 Review the generated plan and execute removals safely")

if __name__ == "__main__":
    main()