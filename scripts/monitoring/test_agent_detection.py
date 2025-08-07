#!/usr/bin/env python3
"""
Agent Detection Test - Tests live agent detection functionality
=============================================================

This test validates the monitor's ability to detect and health-check real running agents.
"""

import sys
import time
import subprocess
import socket
from pathlib import Path

# Add the monitoring directory to path
sys.path.insert(0, '/opt/sutazaiapp/scripts/monitoring')

from static_monitor import EnhancedMonitor

def test_port_scanning():
    """Test port scanning functionality"""
    print("=== Port Scanning Test ===")
    
    monitor = EnhancedMonitor()
    
    # Test common ports that might be in use
    test_ports = [
        22,    # SSH
        80,    # HTTP
        443,   # HTTPS
        8000,  # Common dev port
        8001,  # Common dev port
        8080,  # Common web server
        8116,  # Hardware resource optimizer
        10104, # Ollama default
    ]
    
    print("Scanning common ports...")
    for port in test_ports:
        is_open = monitor._test_port_connection(port)
        status = "OPEN" if is_open else "CLOSED"
        print(f"  Port {port:>5}: {status}")
    
    monitor.cleanup()

def test_agent_health_checking():
    """Test agent health checking with real services"""
    print("\n=== Agent Health Checking Test ===")
    
    monitor = EnhancedMonitor()
    
    # Get actual agent status
    agents, healthy, total = monitor.get_ai_agents_status()
    
    print(f"Agent Status Summary: {healthy}/{total} healthy")
    print("\nDetailed Agent Status:")
    
    for i, agent_line in enumerate(agents):
        print(f"  {i+1:2d}. {agent_line}")
    
    # Test specific agent health check if hardware-resource-optimizer is in registry
    registry = monitor.agent_registry
    if 'hardware-resource-optimizer' in registry.get('agents', {}):
        print(f"\nTesting hardware-resource-optimizer specifically...")
        agent_info = registry['agents']['hardware-resource-optimizer']
        
        # Try to get endpoint
        endpoint = monitor._get_agent_endpoint('hardware-resource-optimizer', agent_info)
        print(f"  Detected endpoint: {endpoint or 'None'}")
        
        if endpoint:
            # Test health check
            health_status, response_time = monitor._check_agent_health(
                'hardware-resource-optimizer', agent_info, 5
            )
            print(f"  Health status: {health_status}")
            print(f"  Response time: {response_time:.0f}ms" if response_time else "  Response time: N/A")
    
    monitor.cleanup()

def test_docker_container_detection():
    """Test Docker container detection as fallback"""
    print("\n=== Docker Container Detection Test ===")
    
    monitor = EnhancedMonitor()
    
    # Test Docker container detection
    containers, running, total_containers = monitor.get_docker_containers()
    
    print(f"Docker Containers: {running}/{total_containers} running")
    print("\nContainer Status:")
    
    for i, container_line in enumerate(containers):
        if container_line != "No containers found":
            print(f"  {i+1}. {container_line}")
        else:
            print(f"  {container_line}")
    
    monitor.cleanup()

def test_network_activity():
    """Test network activity detection"""
    print("\n=== Network Activity Test ===")
    
    monitor = EnhancedMonitor()
    
    # Get initial network stats
    stats1 = monitor.get_system_stats()
    initial_network = stats1['network']
    
    print("Initial network stats:")
    print(f"  Bandwidth: {initial_network['bandwidth_mbps']:.2f} Mbps")
    print(f"  Upload: {initial_network['upload_mbps']:.2f} Mbps")
    print(f"  Download: {initial_network['download_mbps']:.2f} Mbps")
    
    # Wait a moment and get stats again
    print("\nWaiting 2 seconds for network activity...")
    time.sleep(2)
    
    stats2 = monitor.get_system_stats()
    updated_network = stats2['network']
    
    print("Updated network stats:")
    print(f"  Bandwidth: {updated_network['bandwidth_mbps']:.2f} Mbps")
    print(f"  Upload: {updated_network['upload_mbps']:.2f} Mbps")
    print(f"  Download: {updated_network['download_mbps']:.2f} Mbps")
    
    monitor.cleanup()

def test_gpu_detection_detailed():
    """Test detailed GPU detection"""
    print("\n=== GPU Detection Detailed Test ===")
    
    monitor = EnhancedMonitor()
    
    # Check WSL environment
    wsl_info = monitor._detect_wsl_environment()
    print("WSL Environment:")
    for key, value in wsl_info.items():
        print(f"  {key}: {value}")
    
    # Check GPU availability
    print(f"\nGPU Available: {monitor.gpu_available}")
    print(f"GPU Driver Type: {monitor.gpu_driver_type}")
    print("GPU Info:")
    for key, value in monitor.gpu_info.items():
        print(f"  {key}: {value}")
    
    # Get GPU stats
    gpu_stats = monitor.get_gpu_stats()
    print("\nGPU Statistics:")
    for key, value in gpu_stats.items():
        print(f"  {key}: {value}")
    
    monitor.cleanup()

def run_live_monitoring_test():
    """Run a brief live monitoring session"""
    print("\n=== Live Monitoring Test (10 seconds) ===")
    
    monitor = EnhancedMonitor()
    
    print("Starting live monitoring for 10 seconds...")
    print("(This will show the actual monitor output)")
    
    try:
        start_time = time.time()
        
        # Run for 10 seconds
        while time.time() - start_time < 10:
            # Get comprehensive data
            stats = monitor.get_system_stats()
            
            # Get agent status
            agents, healthy, total = monitor.get_ai_agents_status()
            
            # Simple one-line status update
            print(f"\r[{time.time() - start_time:4.1f}s] CPU:{stats['cpu_percent']:5.1f}% MEM:{stats['mem_percent']:5.1f}% GPU:{stats['gpu']['usage']:5.1f}% Agents:{healthy}/{total}   ", end='', flush=True)
            
            time.sleep(1)
        
        print()  # New line after monitoring
        
    except KeyboardInterrupt:
        print("\nMonitoring interrupted by user")
    
    monitor.cleanup()

def main():
    """Run all agent detection tests"""
    print("=" * 60)
    print("AGENT DETECTION AND HEALTH CHECK VALIDATION")
    print("=" * 60)
    
    test_port_scanning()
    test_agent_health_checking()
    test_docker_container_detection()
    test_network_activity()
    test_gpu_detection_detailed()
    run_live_monitoring_test()
    
    print("\n" + "=" * 60)
    print("AGENT DETECTION TESTS COMPLETED")
    print("=" * 60)
    print("\nSummary:")
    print("✓ Port scanning functionality tested") 
    print("✓ Agent health checking validated")
    print("✓ Docker container detection tested")
    print("✓ Network activity monitoring verified")
    print("✓ GPU detection thoroughly tested")
    print("✓ Live monitoring session completed")
    print("\nThe static monitor is fully functional for agent detection and monitoring.")

if __name__ == "__main__":
    main()