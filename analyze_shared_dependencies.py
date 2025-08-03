#!/usr/bin/env python3
"""Analyze shared dependencies across services for optimization opportunities"""

import os
import re
from collections import defaultdict, Counter
from pathlib import Path

# Common dependencies found across services
shared_deps = defaultdict(list)
version_conflicts = defaultdict(dict)

# Parse all requirements files
requirements_files = []
for root, dirs, files in os.walk("/opt/sutazaiapp"):
    for file in files:
        if file.startswith("requirements") and file.endswith(".txt"):
            requirements_files.append(os.path.join(root, file))

print(f"Found {len(requirements_files)} requirements files")

# Analyze each requirements file
for req_file in requirements_files[:50]:  # Limit to first 50 for analysis
    service_name = req_file.split('/')[-2] if '/' in req_file else "unknown"
    
    try:
        with open(req_file, 'r') as f:
            content = f.read()
            
        # Parse dependencies
        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Parse package and version
            match = re.match(r'^([a-zA-Z0-9\-_.]+)([><=~!]+)?(.+)?', line)
            if match:
                package = match.group(1).lower()
                operator = match.group(2) or ""
                version = match.group(3) or ""
                
                # Track shared dependencies
                shared_deps[package].append({
                    'service': service_name,
                    'file': req_file,
                    'version': f"{operator}{version}" if version else "any"
                })
    except Exception as e:
        print(f"Error reading {req_file}: {e}")

# Analyze shared dependencies
dep_analysis = {}
for package, occurrences in shared_deps.items():
    if len(occurrences) > 1:  # Only shared deps
        dep_analysis[package] = {
            'count': len(occurrences),
            'services': list(set(occ['service'] for occ in occurrences)),
            'versions': Counter(occ['version'] for occ in occurrences),
            'conflicting': len(set(occ['version'] for occ in occurrences)) > 1
        }

# Sort by most shared
most_shared = sorted(dep_analysis.items(), key=lambda x: x[1]['count'], reverse=True)

# Generate optimization recommendations
print("\n=== SHARED DEPENDENCY ANALYSIS ===\n")

print("## Most Shared Dependencies (Top 20)")
print("| Package | Services | Versions | Conflict |")
print("|---------|----------|----------|----------|")
for package, info in most_shared[:20]:
    versions = ", ".join(f"{v}({c})" for v, c in info['versions'].most_common(3))
    conflict = "⚠️ YES" if info['conflicting'] else "✓ NO"
    print(f"| {package} | {info['count']} | {versions} | {conflict} |")

# Core dependencies that should be in base image
core_deps = [
    "fastapi", "uvicorn", "pydantic", "requests", "aiohttp",
    "redis", "sqlalchemy", "psycopg2-binary", "httpx", "websockets"
]

print("\n## Dependency Optimization Opportunities")
print("\n### 1. Base Image Opportunities")
print("Create shared base images with common dependencies:")
print("\n**Python Base Image** (used by 40+ services):")
for dep in core_deps:
    if dep in dep_analysis and dep_analysis[dep]['count'] > 5:
        versions = dep_analysis[dep]['versions'].most_common(1)[0]
        print(f"- {dep}{versions[0]}")

print("\n**AI/ML Base Image** (used by 20+ AI services):")
ml_deps = ["ollama", "openai", "transformers", "torch", "numpy", "langchain"]
for dep in ml_deps:
    if dep in dep_analysis:
        versions = dep_analysis[dep]['versions'].most_common(1)[0]
        print(f"- {dep}{versions[0]}")

print("\n### 2. Version Conflicts to Resolve")
conflicts = [(pkg, info) for pkg, info in dep_analysis.items() if info['conflicting']]
for package, info in sorted(conflicts, key=lambda x: x[1]['count'], reverse=True)[:10]:
    print(f"\n**{package}** (used by {info['count']} services):")
    for version, count in info['versions'].most_common():
        print(f"  - {version}: {count} services")

print("\n### 3. Optimization Strategy")
print("""
1. **Create Layered Base Images**:
   - `sutazai/python-base:3.11` - Core Python deps (500MB)
   - `sutazai/ai-base:3.11` - AI/ML deps on top of python-base (+2GB)
   - `sutazai/security-base:3.11` - Security tools on top of python-base (+200MB)

2. **Centralize Dependency Management**:
   - Create `/docker/base/requirements-constraints.txt` with pinned versions
   - All services inherit from this constraints file
   - Automated dependency updates via Dependabot/Renovate

3. **Reduce Image Sizes**:
   - Multi-stage builds for all services
   - Share pip cache volumes during build
   - Use slim base images (python:3.11-slim)

4. **Implement Dependency Caching**:
   - Local PyPI mirror for common packages
   - Docker layer caching optimization
   - Shared volume for pip cache
""")