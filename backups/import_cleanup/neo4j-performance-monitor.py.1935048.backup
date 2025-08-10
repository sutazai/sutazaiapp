#!/usr/bin/env python3
"""
Neo4j Performance Monitoring Script
Tracks memory usage, CPU consumption, and connection metrics before/after optimization
"""

import subprocess
import json
import time
import datetime
import os

def get_container_stats():
    """Get Neo4j container resource usage"""
    try:
        cmd = ["docker", "stats", "sutazai-neo4j", "--no-stream", "--format", 
               "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.BlockIO}}"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError:
        return "Neo4j container not running"

def get_neo4j_process_info():
    """Get detailed process information from inside the container"""
    try:
        cmd = ["docker", "exec", "sutazai-neo4j", "sh", "-c", "ps aux | grep java"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError:
        return "Unable to get process info"

def get_database_metrics():
    """Get Neo4j internal metrics via cypher-shell"""
    try:
        password_file = "/opt/sutazaiapp/secrets_secure/neo4j_password.txt"
        with open(password_file, 'r') as f:
            password = f.read().strip()
        
        # Basic database info query
        cmd = ["docker", "exec", "sutazai-neo4j", "cypher-shell", "-u", "neo4j", "-p", password,
               "CALL dbms.listConfig() YIELD name, value WHERE name CONTAINS 'memory' RETURN name, value LIMIT 10"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.stdout if result.returncode == 0 else "Database query failed"
    except Exception as e:
        return f"Database metrics unavailable: {str(e)}"

def get_heap_dump_analysis():
    """Analyze Java heap usage"""
    try:
        cmd = ["docker", "exec", "sutazai-neo4j", "sh", "-c", 
               "jstat -gc -t $(pgrep -f 'org.neo4j') 2>/dev/null | tail -1"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "GC stats unavailable"

def main():
    """Main monitoring function"""
    print("=" * 80)
    print(f"Neo4j Performance Report - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Container-level metrics
    print("\n🐳 Container Resource Usage:")
    print(get_container_stats())
    
    # Process-level metrics
    print("\n⚙️ Java Process Information:")
    process_info = get_neo4j_process_info()
    if "java" in process_info:
        lines = process_info.split('\n')
        for line in lines:
            if 'java' in line and 'grep' not in line:
                parts = line.split()
                if len(parts) >= 6:
                    print(f"   PID: {parts[1]}, CPU: {parts[2]}%, MEM: {parts[3]}%, Command: {' '.join(parts[10:13])}")
    else:
        print(f"   {process_info}")
    
    # Database metrics
    print("\n📊 Database Configuration:")
    db_metrics = get_database_metrics()
    if "memory" in db_metrics.lower():
        print("   Memory-related configuration:")
        for line in db_metrics.split('\n'):
            if line.strip() and '|' in line:
                print(f"   {line.strip()}")
    else:
        print(f"   {db_metrics}")
    
    # Garbage Collection stats
    print("\n🗑️ JVM Garbage Collection:")
    gc_stats = get_heap_dump_analysis()
    if gc_stats and gc_stats != "GC stats unavailable":
        print(f"   {gc_stats}")
    else:
        print(f"   {gc_stats}")
    
    # Recommendations
    print("\n💡 Performance Recommendations:")
    stats = get_container_stats()
    if "%" in stats:
        lines = stats.split('\n')
        for line in lines:
            if 'sutazai-neo4j' in line:
                parts = line.split()
                if len(parts) >= 4:
                    cpu_percent = float(parts[1].replace('%', ''))
                    memory_usage = parts[2]
                    
                    print(f"   Current CPU: {cpu_percent}%")
                    print(f"   Current Memory: {memory_usage}")
                    
                    if cpu_percent > 50:
                        print("   ⚠️ HIGH CPU: Consider reducing concurrent connections or query complexity")
                    elif cpu_percent < 5:
                        print("   ✅ LOW CPU: Excellent performance")
                    
                    if "GiB" in memory_usage and float(memory_usage.split("GiB")[0]) > 0.8:
                        print("   ⚠️ HIGH MEMORY: Memory usage approaching limits")
                    else:
                        print("   ✅ MEMORY: Usage within acceptable range")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()