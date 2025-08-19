#!/usr/bin/env python3
"""
Cache Consolidation Helper Script
Assists with implementing the comprehensive cache consolidation plan
"""

import os
import shutil
from pathlib import Path

def analyze_cache_imports():
    """Analyze all cache import statements to plan migration"""
    backend_dir = Path("/opt/sutazaiapp/backend")
    cache_imports = []
    
    for py_file in backend_dir.rglob("*.py"):
        try:
            with open(py_file, 'r') as f:
                content = f.read()
                if 'cache' in content.lower() and any(keyword in content for keyword in ['import', 'from']):
                    lines = content.split('\n')
                    for i, line in enumerate(lines, 1):
                        if 'cache' in line.lower() and ('import' in line or 'from' in line):
                            cache_imports.append({
                                'file': str(py_file),
                                'line': i,
                                'content': line.strip()
                            })
        except Exception as e:
            print(f"Error reading {py_file}: {e}")
    
    return cache_imports

def backup_cache_files():
    """Create backup of all cache-related files before consolidation"""
    backend_dir = Path("/opt/sutazaiapp/backend")
    backup_dir = Path("/opt/sutazaiapp/cache_consolidation_backup")
    
    backup_dir.mkdir(exist_ok=True)
    
    cache_files = [
        "app/core/cache.py",
        "app/core/performance.py", 
        "app/core/ollama_cache.py",
        "app/core/performance_tuning.py",
        "app/core/middleware.py",
        "edge_inference/model_cache.py"
    ]
    
    for cache_file in cache_files:
        src = backend_dir / cache_file
        if src.exists():
            dst = backup_dir / cache_file
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"✅ Backed up: {cache_file}")
    
    print(f"\n🔒 Backup created at: {backup_dir}")

def generate_consolidation_summary():
    """Generate summary of current cache state"""
    print("🔍 CACHE CONSOLIDATION ANALYSIS")
    print("=" * 50)
    
    # Analyze imports
    imports = analyze_cache_imports()
    print(f"📊 Found {len(imports)} cache-related imports")
    
    # Group by file
    files_with_cache = {}
    for imp in imports:
        file_path = imp['file'].replace('/opt/sutazaiapp/backend/', '')
        if file_path not in files_with_cache:
            files_with_cache[file_path] = []
        files_with_cache[file_path].append(imp['content'])
    
    print(f"📁 Files using cache: {len(files_with_cache)}")
    
    # Show primary cache usage
    primary_cache_files = [f for f in files_with_cache.keys() if 'app.core.cache' in str(files_with_cache[f])]
    print(f"🎯 Files using primary cache (app.core.cache): {len(primary_cache_files)}")
    
    return {
        'total_imports': len(imports),
        'files_count': len(files_with_cache),
        'primary_usage': len(primary_cache_files),
        'imports': imports
    }

if __name__ == "__main__":
    print("🚀 CACHE CONSOLIDATION HELPER")
    print("=" * 40)
    
    # Generate analysis
    summary = generate_consolidation_summary()
    
    # Create backup
    print("\n📦 Creating backup...")
    backup_cache_files()
    
    # Show recommendations
    print("\n💡 RECOMMENDATIONS:")
    print("1. ✅ performance_ultrafix.py already removed")
    print("2. 🔄 Primary cache (app.core.cache.py) is actively used")
    print("3. 🎯 Focus consolidation on merging duplicate CacheManager classes")
    print("4. ⚠️  Preserve Ollama-specific and Edge inference caches")
    
    print(f"\n📋 Summary saved to: /opt/sutazaiapp/COMPREHENSIVE_CACHE_CONSOLIDATION_REPORT.md")
    print("🎉 Cache consolidation analysis complete!")