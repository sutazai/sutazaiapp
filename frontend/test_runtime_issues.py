#!/usr/bin/env python3
"""
Test for potential runtime issues in the optimized frontend
"""

import ast
import re
from pathlib import Path

def analyze_potential_runtime_issues():
    """Analyze code for potential runtime issues"""
    print("🔍 Analyzing Potential Runtime Issues...")
    
    issues_found = []
    warnings_found = []
    
    # Test 1: Check for hardcoded URLs/ports
    print("\n1️⃣ Checking for hardcoded URLs/ports...")
    
    files_to_check = [
        "app_optimized.py",
        "utils/optimized_api_client.py",
        "utils/performance_cache.py"
    ]
    
    for file_path in files_to_check:
        if not Path(file_path).exists():
            continue
            
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Look for hardcoded localhost URLs
        if re.search(r'http://127\.0\.0\.1:\d+|http://localhost:\d+', content):
            warnings_found.append(f"Hardcoded localhost URL found in {file_path}")
        
        # Look for hardcoded ports
        hardcoded_ports = re.findall(r':(\d{4,5})', content)
        if hardcoded_ports:
            warnings_found.append(f"Hardcoded ports {hardcoded_ports} found in {file_path}")
    
    # Test 2: Check for missing error handling
    print("2️⃣ Checking error handling patterns...")
    
    with open("app_optimized.py", 'r') as f:
        app_content = f.read()
    
    # Count try/except blocks vs async/await usage
    try_count = len(re.findall(r'try:', app_content))
    async_count = len(re.findall(r'async def|await ', app_content))
    
    if async_count > try_count:
        warnings_found.append(f"Many async operations ({async_count}) but few try/except blocks ({try_count})")
    
    # Test 3: Check for potential memory leaks
    print("3️⃣ Checking for potential memory issues...")
    
    # Look for session state accumulation without cleanup
    if 'navigation_history' in app_content:
        if 'navigation_history[-5:]' not in app_content:
            if 'len(navigation_history) >' not in app_content:
                issues_found.append("navigation_history may accumulate without limit")
    
    # Test 4: Check import dependencies
    print("4️⃣ Analyzing import dependencies...")
    
    with open("utils/optimized_api_client.py", 'r') as f:
        api_client_content = f.read()
    
    # Check for circular import risk
    if "from utils.performance_cache import" in api_client_content:
        with open("utils/performance_cache.py", 'r') as f:
            cache_content = f.read()
        
        if "from utils.api_client import" in cache_content:
            issues_found.append("Potential circular import between api_client and cache")
    
    # Test 5: Check async/await compatibility
    print("5️⃣ Checking async/await patterns...")
    
    # Look for asyncio.run() in Streamlit context
    if "asyncio.run(" in app_content:
        warnings_found.append("Using asyncio.run() in Streamlit may cause issues")
    
    # Test 6: Check for resource cleanup
    print("6️⃣ Checking resource cleanup...")
    
    if "httpx.AsyncClient(" in api_client_content:
        if "await.*close()" not in api_client_content and "aclose()" not in api_client_content:
            warnings_found.append("HTTP client may not be properly closed")
    
    # Test 7: Check for performance bottlenecks
    print("7️⃣ Checking for performance bottlenecks...")
    
    # Look for synchronous operations in async code
    sync_in_async_patterns = [
        r'def.*sync.*\(.*\):\s*return asyncio\.run\(',
        r'\.sync\(',
    ]
    
    for pattern in sync_in_async_patterns:
        if re.search(pattern, api_client_content):
            warnings_found.append("Synchronous wrappers around async code may block the event loop")
    
    # Test 8: Check session state usage
    print("8️⃣ Checking session state usage patterns...")
    
    session_state_patterns = re.findall(r'st\.session_state\.(\w+)', app_content)
    session_state_keys = set(session_state_patterns)
    
    if len(session_state_keys) > 20:
        warnings_found.append(f"Many session state keys ({len(session_state_keys)}) may impact performance")
    
    # Test 9: Check for uncaught exceptions
    print("9️⃣ Checking exception handling coverage...")
    
    # Look for bare except clauses
    if re.search(r'except:\s*\n', app_content):
        warnings_found.append("Bare except clauses found - may hide important errors")
    
    # Test 10: Check for blocking operations
    print("🔟 Checking for potentially blocking operations...")
    
    blocking_patterns = [
        r'time\.sleep\(',
        r'requests\.get\(',
        r'requests\.post\(',
    ]
    
    for pattern in blocking_patterns:
        for file_path in files_to_check:
            if not Path(file_path).exists():
                continue
            with open(file_path, 'r') as f:
                content = f.read()
            if re.search(pattern, content):
                warnings_found.append(f"Potentially blocking operation found in {file_path}: {pattern}")
    
    # Generate report
    print("\n" + "="*60)
    print("📊 RUNTIME ANALYSIS RESULTS")
    print("="*60)
    
    if issues_found:
        print("🔥 CRITICAL ISSUES:")
        for issue in issues_found:
            print(f"   ❌ {issue}")
    else:
        print("✅ No critical runtime issues found")
    
    if warnings_found:
        print("\n⚠️  WARNINGS:")
        for warning in warnings_found:
            print(f"   ⚠️  {warning}")
    else:
        print("✅ No runtime warnings")
    
    total_issues = len(issues_found) + len(warnings_found)
    
    if total_issues == 0:
        print("\n🎉 RUNTIME ANALYSIS PASSED!")
        print("✅ Code appears to be runtime-safe")
    elif len(issues_found) == 0:
        print(f"\n✅ RUNTIME ANALYSIS PASSED WITH {len(warnings_found)} WARNINGS")
        print("⚠️  Code should work but may have performance or robustness issues")
    else:
        print(f"\n❌ RUNTIME ANALYSIS FAILED")
        print(f"🔥 {len(issues_found)} critical issues found")
    
    return {
        "issues": issues_found,
        "warnings": warnings_found,
        "total_problems": total_issues
    }

if __name__ == "__main__":
    analyze_potential_runtime_issues()