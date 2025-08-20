#!/usr/bin/env python3
"""
Security Audit: Extended Memory Persistence Reality Test
Validates that the extended-memory service is REAL and not a mock
"""

import requests
import sqlite3
import subprocess
import json
import time
import hashlib
import uuid
from datetime import datetime

def security_audit_extended_memory():
    """Comprehensive security audit of extended memory persistence"""
    
    print("=" * 80)
    print("SECURITY AUDIT: Extended Memory Persistence Implementation")
    print("=" * 80)
    
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "auditor": "security-auditor",
        "tests": [],
        "verdict": None
    }
    
    # Test 1: Verify no mock classes in source code
    print("\n[1/10] Checking for mock implementations in source code...")
    try:
        result = subprocess.run(
            ["grep", "-r", "Mock\\|mock\\|Stub\\|stub\\|Fake\\|fake", 
             "/opt/sutazaiapp/docker/mcp-services/extended-memory-persistent/server.py"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            results["tests"].append({
                "test": "source_code_check",
                "passed": False,
                "reason": "Found mock/stub/fake patterns in code"
            })
            print("  ❌ FAILED: Found mock patterns in source")
        else:
            results["tests"].append({
                "test": "source_code_check",
                "passed": True,
                "reason": "No mock patterns found"
            })
            print("  ✅ PASSED: No mock patterns found in source")
    except Exception as e:
        print(f"  ⚠ ERROR: {e}")
    
    # Test 2: Verify SQLite database file exists
    print("\n[2/10] Checking SQLite database file existence...")
    import os
    db_path = "/opt/sutazaiapp/data/mcp/extended-memory/extended_memory.db"
    if os.path.exists(db_path):
        file_size = os.path.getsize(db_path)
        results["tests"].append({
            "test": "database_file_exists",
            "passed": True,
            "db_path": db_path,
            "file_size": file_size
        })
        print(f"  ✅ PASSED: Database exists at {db_path} ({file_size} bytes)")
    else:
        results["tests"].append({
            "test": "database_file_exists",
            "passed": False
        })
        print(f"  ❌ FAILED: Database not found at {db_path}")
    
    # Test 3: Verify SQLite database structure
    print("\n[3/10] Validating SQLite database structure...")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = ["memory_store", "metadata"]
        if all(t in tables for t in expected_tables):
            results["tests"].append({
                "test": "database_structure",
                "passed": True,
                "tables": tables
            })
            print(f"  ✅ PASSED: Required tables found: {tables}")
        else:
            results["tests"].append({
                "test": "database_structure",
                "passed": False,
                "tables": tables
            })
            print(f"  ❌ FAILED: Missing tables. Found: {tables}")
        
        conn.close()
    except Exception as e:
        print(f"  ⚠ ERROR: {e}")
    
    # Test 4: API Health Check
    print("\n[4/10] Testing API health endpoint...")
    try:
        response = requests.get("http://localhost:3009/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("persistence", {}).get("type") == "SQLite":
                results["tests"].append({
                    "test": "api_health",
                    "passed": True,
                    "persistence_type": "SQLite",
                    "db_path": data.get("persistence", {}).get("path")
                })
                print(f"  ✅ PASSED: API healthy with SQLite persistence")
            else:
                results["tests"].append({
                    "test": "api_health",
                    "passed": False,
                    "reason": "Not using SQLite"
                })
                print(f"  ❌ FAILED: Not using SQLite persistence")
        else:
            results["tests"].append({
                "test": "api_health",
                "passed": False,
                "status_code": response.status_code
            })
            print(f"  ❌ FAILED: Health check returned {response.status_code}")
    except Exception as e:
        print(f"  ⚠ ERROR: {e}")
    
    # Test 5: Write unique test data
    print("\n[5/10] Writing unique test data...")
    test_key = f"security_audit_{uuid.uuid4()}"
    test_value = {
        "audit_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "hash": hashlib.sha256(str(time.time()).encode()).hexdigest(),
        "data": list(range(1000))  # Large enough to be non-trivial
    }
    
    try:
        response = requests.post(
            "http://localhost:3009/store",
            json={"key": test_key, "value": test_value}
        )
        if response.status_code == 200:
            results["tests"].append({
                "test": "write_data",
                "passed": True,
                "key": test_key
            })
            print(f"  ✅ PASSED: Wrote test data with key: {test_key}")
        else:
            results["tests"].append({
                "test": "write_data",
                "passed": False
            })
            print(f"  ❌ FAILED: Could not write data")
    except Exception as e:
        print(f"  ⚠ ERROR: {e}")
    
    # Test 6: Verify data in SQLite directly
    print("\n[6/10] Verifying data directly in SQLite...")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT value, type FROM memory_store WHERE key = ?", (test_key,))
        row = cursor.fetchone()
        
        if row:
            stored_value = json.loads(row[0])
            if stored_value == test_value:
                results["tests"].append({
                    "test": "sqlite_verification",
                    "passed": True,
                    "data_intact": True
                })
                print(f"  ✅ PASSED: Data found and intact in SQLite")
            else:
                results["tests"].append({
                    "test": "sqlite_verification",
                    "passed": False,
                    "reason": "Data mismatch"
                })
                print(f"  ❌ FAILED: Data mismatch in SQLite")
        else:
            results["tests"].append({
                "test": "sqlite_verification",
                "passed": False,
                "reason": "Data not found"
            })
            print(f"  ❌ FAILED: Data not found in SQLite")
        
        conn.close()
    except Exception as e:
        print(f"  ⚠ ERROR: {e}")
    
    # Test 7: Container restart persistence
    print("\n[7/10] Testing container restart persistence...")
    try:
        # Restart container
        subprocess.run(["docker", "restart", "mcp-extended-memory"], check=True, capture_output=True)
        time.sleep(5)  # Wait for container to be ready
        
        # Try to retrieve data after restart
        response = requests.get(f"http://localhost:3009/retrieve/{test_key}", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("value") == test_value and data.get("source") == "database":
                results["tests"].append({
                    "test": "restart_persistence",
                    "passed": True,
                    "source": "database"
                })
                print(f"  ✅ PASSED: Data persisted across restart (source: database)")
            else:
                results["tests"].append({
                    "test": "restart_persistence",
                    "passed": False,
                    "reason": "Data changed or wrong source"
                })
                print(f"  ❌ FAILED: Data not properly persisted")
        else:
            results["tests"].append({
                "test": "restart_persistence",
                "passed": False
            })
            print(f"  ❌ FAILED: Could not retrieve after restart")
    except Exception as e:
        print(f"  ⚠ ERROR: {e}")
    
    # Test 8: Volume mount verification
    print("\n[8/10] Verifying Docker volume mounts...")
    try:
        result = subprocess.run(
            ["docker", "inspect", "mcp-extended-memory"],
            capture_output=True,
            text=True,
            check=True
        )
        inspect_data = json.loads(result.stdout)[0]
        mounts = inspect_data.get("Mounts", [])
        
        correct_mount = False
        for mount in mounts:
            if mount.get("Destination") == "/var/lib/mcp" and \
               "/opt/sutazaiapp/data/mcp/extended-memory" in mount.get("Source", ""):
                correct_mount = True
                break
        
        if correct_mount:
            results["tests"].append({
                "test": "volume_mount",
                "passed": True,
                "mount": mount
            })
            print(f"  ✅ PASSED: Correct volume mount to host filesystem")
        else:
            results["tests"].append({
                "test": "volume_mount",
                "passed": False
            })
            print(f"  ❌ FAILED: Incorrect volume mounting")
    except Exception as e:
        print(f"  ⚠ ERROR: {e}")
    
    # Test 9: Performance test (not mock-like)
    print("\n[9/10] Testing realistic performance characteristics...")
    try:
        # Real SQLite should have measurable write latency
        latencies = []
        for i in range(10):
            start = time.time()
            response = requests.post(
                "http://localhost:3009/store",
                json={"key": f"perf_test_{i}", "value": f"data_{i}"}
            )
            latencies.append(time.time() - start)
        
        avg_latency = sum(latencies) / len(latencies)
        # Real persistence should have some latency (not instant like mock)
        if 0.001 < avg_latency < 1.0:  # Between 1ms and 1s is realistic
            results["tests"].append({
                "test": "performance_characteristics",
                "passed": True,
                "avg_latency_ms": avg_latency * 1000
            })
            print(f"  ✅ PASSED: Realistic latency: {avg_latency*1000:.2f}ms")
        else:
            results["tests"].append({
                "test": "performance_characteristics",
                "passed": False,
                "avg_latency_ms": avg_latency * 1000
            })
            print(f"  ❌ FAILED: Unrealistic latency: {avg_latency*1000:.2f}ms")
    except Exception as e:
        print(f"  ⚠ ERROR: {e}")
    
    # Test 10: Backup functionality
    print("\n[10/10] Testing backup functionality...")
    try:
        response = requests.post("http://localhost:3009/backup")
        if response.status_code == 200:
            data = response.json()
            backup_path = data.get("path", "")
            # Check if backup file was created
            if "backup" in backup_path:
                # Check if backup exists in container
                result = subprocess.run(
                    ["docker", "exec", "mcp-extended-memory", "ls", "-la", backup_path],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    results["tests"].append({
                        "test": "backup_functionality",
                        "passed": True,
                        "backup_path": backup_path
                    })
                    print(f"  ✅ PASSED: Backup created: {backup_path}")
                else:
                    results["tests"].append({
                        "test": "backup_functionality",
                        "passed": False,
                        "reason": "Backup file not found"
                    })
                    print(f"  ❌ FAILED: Backup file not found")
            else:
                results["tests"].append({
                    "test": "backup_functionality",
                    "passed": False
                })
                print(f"  ❌ FAILED: Invalid backup response")
        else:
            results["tests"].append({
                "test": "backup_functionality",
                "passed": False
            })
            print(f"  ❌ FAILED: Backup endpoint failed")
    except Exception as e:
        print(f"  ⚠ ERROR: {e}")
    
    # Final verdict
    print("\n" + "=" * 80)
    print("SECURITY AUDIT RESULTS")
    print("=" * 80)
    
    passed_tests = sum(1 for t in results["tests"] if t.get("passed", False))
    total_tests = len(results["tests"])
    
    print(f"\nTests Passed: {passed_tests}/{total_tests}")
    
    if passed_tests >= 8:  # At least 80% pass rate
        results["verdict"] = "VERIFIED_REAL"
        print("\n✅ VERDICT: IMPLEMENTATION IS REAL")
        print("The extended-memory service is using genuine SQLite persistence.")
        print("No mock or facade implementations detected.")
    else:
        results["verdict"] = "SUSPICIOUS"
        print("\n❌ VERDICT: IMPLEMENTATION IS SUSPICIOUS")
        print("Multiple tests failed. This may be a mock or incomplete implementation.")
    
    # Write detailed report
    report_path = f"/opt/sutazaiapp/reports/security_audit_extended_memory_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed report saved to: {report_path}")
    
    return results["verdict"] == "VERIFIED_REAL"


if __name__ == "__main__":
    import sys
    success = security_audit_extended_memory()
    sys.exit(0 if success else 1)