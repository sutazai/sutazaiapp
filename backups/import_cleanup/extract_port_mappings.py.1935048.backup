#!/usr/bin/env python3
"""Extract port mappings from docker-compose files."""

import yaml
import glob
import json
from pathlib import Path

def extract_ports_from_compose(file_path):
    """Extract service names and their port mappings from a docker-compose file."""
    mappings = {}
    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
            if data and 'services' in data:
                for service_name, service_config in data.get('services', {}).items():
                    if 'ports' in service_config:
                        ports = service_config['ports']
                        container_name = service_config.get('container_name', service_name)
                        mapped_ports = []
                        for port in ports:
                            if isinstance(port, str):
                                # Format: "host:container" or "host:container/protocol"
                                port_parts = port.split(':')
                                if len(port_parts) >= 2:
                                    host_port = port_parts[0].strip('"')
                                    container_port = port_parts[1].split('/')[0]
                                    mapped_ports.append({
                                        'host': int(host_port) if host_port.isdigit() else host_port,
                                        'container': int(container_port) if container_port.isdigit() else container_port
                                    })
                        if mapped_ports:
                            mappings[container_name] = {
                                'service': service_name,
                                'ports': mapped_ports,
                                'file': str(file_path)
                            }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    return mappings

def main():
    """Main function to extract all port mappings."""
    all_mappings = {}
    
    # Find all docker-compose files
    compose_files = glob.glob('docker-compose*.yml')
    
    for file_path in sorted(compose_files):
        print(f"Processing {file_path}...")
        mappings = extract_ports_from_compose(file_path)
        all_mappings.update(mappings)
    
    # Organize by port ranges
    infrastructure_ports = {}  # 10000-10199
    monitoring_ports = {}      # 10200-10299
    agent_ports = {}          # 11000-11148
    other_ports = {}
    
    for container, info in all_mappings.items():
        for port_map in info['ports']:
            host_port = port_map['host']
            if isinstance(host_port, int):
                if 10000 <= host_port < 10200:
                    infrastructure_ports[host_port] = {
                        'container': container,
                        'service': info['service'],
                        'container_port': port_map['container'],
                        'file': info['file']
                    }
                elif 10200 <= host_port < 10300:
                    monitoring_ports[host_port] = {
                        'container': container,
                        'service': info['service'],
                        'container_port': port_map['container'],
                        'file': info['file']
                    }
                elif 11000 <= host_port < 11149:
                    agent_ports[host_port] = {
                        'container': container,
                        'service': info['service'],
                        'container_port': port_map['container'],
                        'file': info['file']
                    }
                else:
                    other_ports[host_port] = {
                        'container': container,
                        'service': info['service'],
                        'container_port': port_map['container'],
                        'file': info['file']
                    }
    
    # Print summary
    print("\n=== PORT ALLOCATION SUMMARY ===")
    print(f"\nInfrastructure Ports (10000-10199): {len(infrastructure_ports)} allocated")
    for port in sorted(infrastructure_ports.keys()):
        info = infrastructure_ports[port]
        print(f"  {port}: {info['service']} ({info['container']})")
    
    print(f"\nMonitoring Ports (10200-10299): {len(monitoring_ports)} allocated")
    for port in sorted(monitoring_ports.keys()):
        info = monitoring_ports[port]
        print(f"  {port}: {info['service']} ({info['container']})")
    
    print(f"\nAgent Ports (11000-11148): {len(agent_ports)} allocated")
    for port in sorted(agent_ports.keys()):
        info = agent_ports[port]
        print(f"  {port}: {info['service']} ({info['container']})")
    
    print(f"\nOther Ports: {len(other_ports)} allocated")
    for port in sorted(other_ports.keys()):
        info = other_ports[port]
        print(f"  {port}: {info['service']} ({info['container']})")
    
    # Save to JSON for further processing
    with open('port_mappings.json', 'w') as f:
        json.dump({
            'infrastructure': infrastructure_ports,
            'monitoring': monitoring_ports,
            'agents': agent_ports,
            'other': other_ports
        }, f, indent=2, default=str)
    
    print("\nPort mappings saved to port_mappings.json")

if __name__ == "__main__":
    main()