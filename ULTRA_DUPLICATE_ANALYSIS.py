#!/usr/bin/env python3
"""
ULTRA CODE REVIEWER - COMPREHENSIVE DUPLICATE ANALYSIS
Analyzes exact duplicates and near-duplicates in the SutazAI codebase.
Follows CODEBASE RULE 4: Reuse Before Creating
"""

import os
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import subprocess
import difflib
from typing import Dict, List, Tuple, Set

class UltraDuplicateAnalyzer:
    def __init__(self, root_path: str = "/opt/sutazaiapp"):
        self.root_path = Path(root_path)
        self.exact_duplicates = defaultdict(list)
        self.near_duplicates = defaultdict(list)
        self.stats = {
            "total_files": 0,
            "shell_scripts": 0,
            "python_files": 0,
            "dockerfiles": 0,
            "exact_duplicates": 0,
            "near_duplicates": 0,
            "docker_compose_references": 0
        }

    def calculate_md5(self, file_path: Path) -> str:
        """Calculate MD5 hash of file content."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return ""

    def get_file_content(self, file_path: Path) -> str:
        """Get file content as string."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return ""

    def similarity_ratio(self, content1: str, content2: str) -> float:
        """Calculate similarity ratio between two text contents."""
        return difflib.SequenceMatcher(None, content1, content2).ratio()

    def find_exact_duplicates(self, file_patterns: List[str]) -> Dict[str, List[str]]:
        """Find exact duplicates based on MD5 hash."""
        hash_to_files = defaultdict(list)
        
        for pattern in file_patterns:
            try:
                # Use find command to locate files
                result = subprocess.run(
                    ["find", str(self.root_path), "-name", pattern, "-type", "f"],
                    capture_output=True, text=True
                )
                
                files = result.stdout.strip().split('\n')
                if files == ['']:
                    files = []
                    
                for file_path in files:
                    if os.path.exists(file_path):
                        file_path_obj = Path(file_path)
                        md5_hash = self.calculate_md5(file_path_obj)
                        if md5_hash:
                            hash_to_files[md5_hash].append(str(file_path))
                            self.stats["total_files"] += 1
                            
                            if pattern == "*.sh":
                                self.stats["shell_scripts"] += 1
                            elif pattern == "*.py":
                                self.stats["python_files"] += 1
                            elif pattern.startswith("Dockerfile"):
                                self.stats["dockerfiles"] += 1
                                
            except Exception as e:
                print(f"Error processing pattern {pattern}: {e}")
        
        # Filter to only duplicates (more than one file with same hash)
        duplicates = {h: files for h, files in hash_to_files.items() if len(files) > 1}
        self.stats["exact_duplicates"] = len(duplicates)
        
        return duplicates

    def find_near_duplicates(self, file_patterns: List[str], similarity_threshold: float = 0.9) -> Dict[str, List[Tuple[str, str, float]]]:
        """Find near-duplicates with similarity > threshold."""
        near_dupes = defaultdict(list)
        file_contents = {}
        
        # Collect all files and their contents
        for pattern in file_patterns:
            try:
                result = subprocess.run(
                    ["find", str(self.root_path), "-name", pattern, "-type", "f"],
                    capture_output=True, text=True
                )
                
                files = result.stdout.strip().split('\n')
                if files == ['']:
                    files = []
                    
                for file_path in files:
                    if os.path.exists(file_path):
                        content = self.get_file_content(Path(file_path))
                        if content and len(content) > 50:  # Skip very small files
                            file_contents[file_path] = content
                            
            except Exception as e:
                print(f"Error processing near-duplicates for {pattern}: {e}")
        
        # Compare all pairs
        files_list = list(file_contents.keys())
        for i, file1 in enumerate(files_list):
            for j, file2 in enumerate(files_list[i+1:], i+1):
                similarity = self.similarity_ratio(file_contents[file1], file_contents[file2])
                if similarity >= similarity_threshold:
                    key = f"{Path(file1).suffix}_{similarity:.2f}"
                    near_dupes[key].append((file1, file2, similarity))
        
        self.stats["near_duplicates"] = sum(len(pairs) for pairs in near_dupes.values())
        return near_dupes

    def find_docker_compose_script_references(self) -> List[Dict]:
        """Find docker-compose script references and duplicates."""
        docker_compose_files = []
        
        try:
            result = subprocess.run(
                ["find", str(self.root_path), "-name", "docker-compose*.yml", "-type", "f"],
                capture_output=True, text=True
            )
            
            files = result.stdout.strip().split('\n')
            if files == ['']:
                files = []
                
            for file_path in files:
                if os.path.exists(file_path):
                    content = self.get_file_content(Path(file_path))
                    # Count script references (simplified)
                    script_refs = content.count('./scripts/') + content.count('scripts/')
                    docker_compose_files.append({
                        "file": file_path,
                        "script_references": script_refs,
                        "size": len(content)
                    })
                    self.stats["docker_compose_references"] += script_refs
                    
        except Exception as e:
            print(f"Error analyzing docker-compose files: {e}")
            
        return docker_compose_files

    def analyze_test_scripts(self) -> Dict:
        """Analyze test script duplicates."""
        test_patterns = ["test*.py", "*test*.py", "test*.sh", "*test*.sh"]
        test_duplicates = {}
        
        for pattern in test_patterns:
            dupes = self.find_exact_duplicates([pattern])
            if dupes:
                test_duplicates[pattern] = dupes
                
        return test_duplicates

    def analyze_build_scripts(self) -> Dict:
        """Analyze build script duplicates."""
        build_patterns = ["build*.sh", "*build*.sh", "deploy*.sh", "*deploy*.sh"]
        build_duplicates = {}
        
        for pattern in build_patterns:
            dupes = self.find_exact_duplicates([pattern])
            if dupes:
                build_duplicates[pattern] = dupes
                
        return build_duplicates

    def generate_mergeable_files_report(self, exact_duplicates: Dict) -> Dict:
        """Generate report of files that can be safely merged."""
        mergeable = {}
        
        for hash_val, files in exact_duplicates.items():
            if len(files) > 1:
                # Categorize by type
                file_type = "unknown"
                first_file = Path(files[0])
                if first_file.suffix == ".sh":
                    file_type = "shell_script"
                elif first_file.suffix == ".py":
                    file_type = "python_script"
                elif first_file.name.startswith("Dockerfile"):
                    file_type = "dockerfile"
                
                mergeable[hash_val] = {
                    "type": file_type,
                    "files": files,
                    "count": len(files),
                    "can_merge": True,  # All exact duplicates can be merged
                    "recommended_keeper": files[0],  # Keep first, shortest path
                    "files_to_remove": files[1:]
                }
        
        return mergeable

    def run_comprehensive_analysis(self) -> Dict:
        """Run complete duplicate analysis."""
        print("🔍 Starting ULTRA DUPLICATE ANALYSIS...")
        
        # 1. Find exact duplicates
        print("📋 Finding exact duplicates...")
        shell_duplicates = self.find_exact_duplicates(["*.sh"])
        python_duplicates = self.find_exact_duplicates(["*.py"])
        dockerfile_duplicates = self.find_exact_duplicates(["Dockerfile*"])
        
        # 2. Find near-duplicates
        print("📋 Finding near-duplicates (>90% similar)...")
        near_dupes_shell = self.find_near_duplicates(["*.sh"], 0.9)
        near_dupes_python = self.find_near_duplicates(["*.py"], 0.9)
        near_dupes_dockerfile = self.find_near_duplicates(["Dockerfile*"], 0.9)
        
        # 3. Analyze docker-compose references
        print("📋 Analyzing docker-compose script references...")
        docker_compose_refs = self.find_docker_compose_script_references()
        
        # 4. Analyze test and build scripts
        print("📋 Analyzing test and build script duplicates...")
        test_duplicates = self.analyze_test_scripts()
        build_duplicates = self.analyze_build_scripts()
        
        # 5. Generate mergeable files report
        all_exact_duplicates = {**shell_duplicates, **python_duplicates, **dockerfile_duplicates}
        mergeable_report = self.generate_mergeable_files_report(all_exact_duplicates)
        
        # Compile final report
        report = {
            "analysis_timestamp": subprocess.run(["date"], capture_output=True, text=True).stdout.strip(),
            "statistics": self.stats,
            "exact_duplicates": {
                "shell_scripts": shell_duplicates,
                "python_files": python_duplicates,
                "dockerfiles": dockerfile_duplicates,
                "total_exact_groups": len(all_exact_duplicates),
                "total_duplicate_files": sum(len(files) for files in all_exact_duplicates.values())
            },
            "near_duplicates": {
                "shell_scripts": dict(near_dupes_shell),
                "python_files": dict(near_dupes_python), 
                "dockerfiles": dict(near_dupes_dockerfile)
            },
            "docker_compose_analysis": {
                "files": docker_compose_refs,
                "total_script_references": self.stats["docker_compose_references"]
            },
            "specialized_duplicates": {
                "test_scripts": test_duplicates,
                "build_scripts": build_duplicates
            },
            "mergeable_files_report": mergeable_report,
            "recommendations": self.generate_recommendations(mergeable_report)
        }
        
        return report

    def generate_recommendations(self, mergeable_report: Dict) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        total_mergeable = len(mergeable_report)
        total_files_to_remove = sum(len(item["files_to_remove"]) for item in mergeable_report.values())
        
        recommendations.append(f"🎯 CRITICAL: {total_mergeable} groups of exact duplicates found")
        recommendations.append(f"🗑️ SAFE TO REMOVE: {total_files_to_remove} duplicate files can be safely deleted")
        
        # Count by type
        type_counts = defaultdict(int)
        for item in mergeable_report.values():
            type_counts[item["type"]] += len(item["files_to_remove"])
            
        for file_type, count in type_counts.items():
            recommendations.append(f"📁 {file_type}: {count} duplicate files to remove")
            
        recommendations.append("⚠️ FOLLOW RULE 4: Reuse existing code before creating new duplicates")
        recommendations.append("🔄 Create symlinks or imports instead of copying files")
        
        return recommendations


if __name__ == "__main__":
    analyzer = UltraDuplicateAnalyzer()
    
    print("=" * 80)
    print("🚀 ULTRA CODE REVIEWER - DUPLICATE ANALYSIS")
    print("=" * 80)
    
    # Run comprehensive analysis
    results = analyzer.run_comprehensive_analysis()
    
    # Save results
    output_file = "/opt/sutazaiapp/ULTRA_DUPLICATE_ANALYSIS_REPORT.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📊 Results saved to: {output_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("📈 ANALYSIS SUMMARY")
    print("=" * 80)
    
    stats = results["statistics"]
    print(f"📁 Total files analyzed: {stats['total_files']}")
    print(f"🐚 Shell scripts: {stats['shell_scripts']}")
    print(f"🐍 Python files: {stats['python_files']}")
    print(f"🐳 Dockerfiles: {stats['dockerfiles']}")
    print(f"🔀 Exact duplicate groups: {results['exact_duplicates']['total_exact_groups']}")
    print(f"📦 Total duplicate files: {results['exact_duplicates']['total_duplicate_files']}")
    print(f"⚖️ Near-duplicate groups: {stats['near_duplicates']}")
    print(f"🐳 Docker-compose script refs: {stats['docker_compose_references']}")
    
    print("\n" + "=" * 80)
    print("🎯 RECOMMENDATIONS")
    print("=" * 80)
    
    for rec in results["recommendations"]:
        print(rec)
        
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)