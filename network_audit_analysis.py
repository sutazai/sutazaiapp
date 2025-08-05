#!/usr/bin/env python3
"""
Network and Port Audit Analysis Script
Analyzes Docker Compose files for port conflicts and network issues
"""

import re
import os
import glob
from collections import defaultdict, Counter
import yaml

def extract_ports_from_compose_files():
    """Extract port mappings from all Docker Compose files"""
    port_mappings = defaultdict(list)
    compose_files = []
    
    # Find all docker-compose files
    for pattern in ['docker-compose*.yml', '**/docker-compose*.yml']:
        compose_files.extend(glob.glob(pattern, recursive=True))
    
    port_pattern = re.compile(r'^\s*-\s*(\d+):(\d+)(?:/\w+)?')
    
    for compose_file in compose_files:
        try:
            with open(compose_file, 'r') as f:
                content = f.read()
                
            # Extract port mappings
            for line_num, line in enumerate(content.split('\n'), 1):
                match = port_pattern.match(line)
                if match:
                    host_port, container_port = match.groups()
                    port_mappings[int(host_port)].append({
                        'file': compose_file,
                        'line': line_num,
                        'host_port': int(host_port),
                        'container_port': int(container_port),
                        'mapping': f"{host_port}:{container_port}"
                    })
        except Exception as e:
            print(f"Error reading {compose_file}: {e}")
    
    return port_mappings

def analyze_running_containers():
    """Analyze currently running containers and their ports"""
    import subprocess
    
    try:
        result = subprocess.run(
            ['docker', 'ps', '--format', 'table {{.Names}}\t{{.Ports}}'],
            capture_output=True, text=True, check=True
        )
        
        running_containers = {}
        lines = result.stdout.strip().split('\n')[1:]  # Skip header
        
        for line in lines:
            if '\t' in line:
                name, ports = line.split('\t', 1)
                running_containers[name] = ports
        
        return running_containers
    except subprocess.CalledProcessError:
        return {"error": "Docker not accessible"}

def check_network_configs():
    """Check network configurations across compose files"""
    networks = defaultdict(list)
    
    compose_files = glob.glob('**/docker-compose*.yml', recursive=True)
    
    for compose_file in compose_files:
        try:
            with open(compose_file, 'r') as f:
                data = yaml.safe_load(f)
                
            if data and 'networks' in data:
                for network_name, config in data['networks'].items():
                    networks[network_name].append({
                        'file': compose_file,
                        'config': config
                    })
        except Exception as e:
            print(f"Error parsing {compose_file}: {e}")
    
    return networks

def main():
    print("=" * 80)
    print("SUTAZAI NETWORK AND PORT AUDIT REPORT")
    print("=" * 80)
    
    # 1. Port Analysis
    print("\n1. PORT MAPPING ANALYSIS")
    print("-" * 40)
    
    port_mappings = extract_ports_from_compose_files()
    
    # Find conflicts
    conflicts = {port: mappings for port, mappings in port_mappings.items() if len(mappings) > 1}
    
    if conflicts:
        print(f"\n🚨 CRITICAL: Found {len(conflicts)} port conflicts!")
        for port, mappings in conflicts.items():
            print(f"\nPort {port} conflicts:")
            for mapping in mappings:
                print(f"  - {mapping['file']}:{mapping['line']} -> {mapping['mapping']}")
    else:
        print("✅ No port conflicts found in compose files")
    
    # Port usage summary
    print(f"\n📊 Port Usage Summary:")
    print(f"Total unique host ports configured: {len(port_mappings)}")
    
    port_ranges = {
        "System (0-1023)": 0,
        "User (1024-49151)": 0,
        "Dynamic (49152-65535)": 0
    }
    
    for port in port_mappings.keys():
        if port <= 1023:
            port_ranges["System (0-1023)"] += 1
        elif port <= 49151:
            port_ranges["User (1024-49151)"] += 1
        else:
            port_ranges["Dynamic (49152-65535)"] += 1
    
    for range_name, count in port_ranges.items():
        print(f"  {range_name}: {count} ports")
    
    # 2. Running Container Analysis
    print("\n\n2. RUNNING CONTAINER ANALYSIS")
    print("-" * 40)
    
    running_containers = analyze_running_containers()
    
    if "error" in running_containers:
        print(f"❌ {running_containers['error']}")
    else:
        print(f"📦 Currently running containers: {len(running_containers)}")
        
        # Extract ports from running containers
        running_ports = set()
        for name, ports in running_containers.items():
            if ports and ports != "":
                # Extract port numbers from Docker port format
                port_matches = re.findall(r'(\d+)->\d+', ports)
                for port in port_matches:
                    running_ports.add(int(port))
        
        print(f"🔌 Active ports: {len(running_ports)} ports in use")
        
        # Check for discrepancies
        configured_ports = set(port_mappings.keys())
        extra_ports = running_ports - configured_ports
        missing_ports = configured_ports - running_ports
        
        if extra_ports:
            print(f"⚠️  Ports running but not in main compose: {sorted(extra_ports)}")
        
        if missing_ports:
            print(f"⚠️  Ports configured but not running: {sorted(missing_ports)}")
    
    # 3. Network Analysis
    print("\n\n3. NETWORK CONFIGURATION ANALYSIS")
    print("-" * 40)
    
    networks = check_network_configs()
    
    print(f"🌐 Network configurations found: {len(networks)}")
    
    for network_name, configs in networks.items():
        print(f"\nNetwork '{network_name}':")
        for config in configs:
            print(f"  - {config['file']}")
            if config['config']:
                if 'driver' in config['config']:
                    print(f"    Driver: {config['config']['driver']}")
                if 'ipam' in config['config']:
                    print(f"    IPAM: {config['config']['ipam']}")
    
    # 4. Hardcoded IP/URL Analysis
    print("\n\n4. HARDCODED IP/URL ANALYSIS")
    print("-" * 40)
    
    # Check for localhost references
    localhost_files = []
    for file_path in glob.glob('**/*.yml', recursive=True) + glob.glob('**/*.py', recursive=True):
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                if re.search(r'localhost|127\.0\.0\.1', content):
                    localhost_files.append(file_path)
        except:
            pass
    
    if localhost_files:
        print(f"⚠️  Files with localhost references: {len(localhost_files)}")
        for f in localhost_files[:10]:  # Show first 10
            print(f"  - {f}")
        if len(localhost_files) > 10:
            print(f"  ... and {len(localhost_files) - 10} more")
    else:
        print("✅ No localhost references found")
    
    # 5. Recommendations
    print("\n\n5. RECOMMENDATIONS")
    print("-" * 40)
    
    if conflicts:
        print("🔧 CRITICAL - Resolve port conflicts:")
        for port in conflicts.keys():
            print(f"  - Port {port}: Choose one compose file or use different ports")
    
    if localhost_files:
        print("🔧 Replace localhost with service names in Docker networks")
    
    print("🔧 Consider port range standardization:")
    print("  - Core services: 10000-10099")
    print("  - AI agents: 10100-10199") 
    print("  - Monitoring: 10200-10299")
    print("  - Applications: 10300-10499")
    print("  - Development: 10500-10999")
    print("  - Phase deployments: 11000+")
    
    print("\n" + "=" * 80)
    print("END OF AUDIT REPORT")
    print("=" * 80)

if __name__ == "__main__":
    main()