#!/usr/bin/env python3
"""
Port Registry Verification and Update Script
Purpose: Verify and update PortRegistry.md with actual port usage
Created: 2024-11-13 22:35:00 UTC
Version: 1.0.0

Performs:
1. Scans all docker-compose files for port mappings
2. Compares with PortRegistry.md
3. Identifies conflicts and discrepancies
4. Generates updated PortRegistry.md
5. Creates detailed port audit report
"""

import os
import re
import yaml
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Set, Tuple
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s UTC - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class PortRegistryVerifier:
    """Verify and update port registry"""
    
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.port_registry_path = repo_root / "IMPORTANT" / "ports" / "PortRegistry.md"
        self.docker_compose_files = []
        self.port_mappings = {}  # port -> {service, file, internal_port}
        self.conflicts = []
        self.discrepancies = []
        
    def scan_docker_compose_files(self):
        """Scan all docker-compose files for port mappings"""
        logger.info("Scanning docker-compose files...")
        
        # Find all docker-compose files
        self.docker_compose_files = list(self.repo_root.glob("docker-compose*.yml"))
        logger.info(f"Found {len(self.docker_compose_files)} docker-compose files")
        
        for compose_file in self.docker_compose_files:
            self._parse_compose_file(compose_file)
            
        logger.info(f"Found {len(self.port_mappings)} unique port mappings")
        
    def _parse_compose_file(self, compose_file: Path):
        """Parse a single docker-compose file"""
        try:
            with open(compose_file, 'r') as f:
                content = yaml.safe_load(f)
                
            if not content or 'services' not in content:
                return
                
            for service_name, service_config in content['services'].items():
                if 'ports' not in service_config:
                    continue
                    
                for port_mapping in service_config['ports']:
                    # Handle both string and dict formats
                    if isinstance(port_mapping, str):
                        match = re.match(r'"?(\d+):(\d+)"?', port_mapping)
                        if match:
                            external_port = match.group(1)
                            internal_port = match.group(2)
                        else:
                            continue
                    else:
                        # Dictionary format
                        external_port = str(port_mapping.get('published', ''))
                        internal_port = str(port_mapping.get('target', ''))
                        
                    if not external_port:
                        continue
                        
                    # Check for conflicts
                    if external_port in self.port_mappings:
                        existing = self.port_mappings[external_port]
                        if existing['service'] != service_name:
                            self.conflicts.append({
                                'port': external_port,
                                'service1': existing['service'],
                                'file1': str(existing['file']),
                                'service2': service_name,
                                'file2': str(compose_file)
                            })
                            logger.warning(f"Port conflict: {external_port} used by {existing['service']} and {service_name}")
                    else:
                        self.port_mappings[external_port] = {
                            'service': service_name,
                            'file': compose_file,
                            'internal_port': internal_port,
                            'container_name': service_config.get('container_name', service_name)
                        }
                        
        except yaml.YAMLError as e:
            logger.error(f"Error parsing {compose_file}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error parsing {compose_file}: {e}")
            
    def compare_with_registry(self):
        """Compare actual ports with PortRegistry.md"""
        logger.info("Comparing with PortRegistry.md...")
        
        if not self.port_registry_path.exists():
            logger.error("PortRegistry.md not found!")
            return
            
        with open(self.port_registry_path, 'r') as f:
            registry_content = f.read()
            
        # Extract ports from registry
        registry_ports = set()
        for line in registry_content.split('\n'):
            match = re.search(r'\*\*(\d+)\*\*:', line)
            if match:
                registry_ports.add(match.group(1))
                
        actual_ports = set(self.port_mappings.keys())
        
        # Find discrepancies
        in_registry_not_actual = registry_ports - actual_ports
        in_actual_not_registry = actual_ports - registry_ports
        
        if in_registry_not_actual:
            logger.warning(f"Ports in registry but not in docker-compose: {in_registry_not_actual}")
            for port in in_registry_not_actual:
                self.discrepancies.append({
                    'type': 'in_registry_not_actual',
                    'port': port
                })
                
        if in_actual_not_registry:
            logger.warning(f"Ports in docker-compose but not in registry: {in_actual_not_registry}")
            for port in in_actual_not_registry:
                self.discrepancies.append({
                    'type': 'in_actual_not_registry',
                    'port': port,
                    'service': self.port_mappings[port]['service']
                })
                
        logger.info(f"Registry ports: {len(registry_ports)}")
        logger.info(f"Actual ports: {len(actual_ports)}")
        logger.info(f"Discrepancies: {len(self.discrepancies)}")
        
    def generate_updated_registry(self):
        """Generate updated PortRegistry.md"""
        logger.info("Generating updated PortRegistry.md...")
        
        # Group ports by range
        core_infra = {}
        ai_vector = {}
        monitoring = {}
        agents = {}
        mcp_bridge = {}
        other = {}
        
        for port, info in sorted(self.port_mappings.items(), key=lambda x: int(x[0])):
            port_num = int(port)
            service_name = info['service']
            container = info['container_name']
            
            if 10000 <= port_num < 10100:
                core_infra[port] = (service_name, container)
            elif 10100 <= port_num < 10200:
                ai_vector[port] = (service_name, container)
            elif 10200 <= port_num < 10300:
                monitoring[port] = (service_name, container)
            elif 11000 <= port_num < 11900:
                if 11100 <= port_num < 11200:
                    mcp_bridge[port] = (service_name, container)
                else:
                    agents[port] = (service_name, container)
            else:
                other[port] = (service_name, container)
                
        # Generate content
        content = f"""# SutazaiApp Port Registry - Auto-Generated
## Multi-Agent AI System Port Allocation
### Last Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
### Generated By: Port Registry Verification Script v1.0.0

---

## Core Infrastructure (10000-10099)
"""
        for port, (service, container) in sorted(core_infra.items(), key=lambda x: int(x[0])):
            content += f"- **{port}**: {container} ({service})\n"
            
        content += "\n## AI & Vector Services (10100-10199)\n"
        for port, (service, container) in sorted(ai_vector.items(), key=lambda x: int(x[0])):
            content += f"- **{port}**: {container} ({service})\n"
            
        content += "\n## Monitoring Stack (10200-10299)\n"
        for port, (service, container) in sorted(monitoring.items(), key=lambda x: int(x[0])):
            content += f"- **{port}**: {container} ({service})\n"
            
        content += "\n## MCP Bridge Services (11100-11199)\n"
        for port, (service, container) in sorted(mcp_bridge.items(), key=lambda x: int(x[0])):
            content += f"- **{port}**: {container} ({service})\n"
            
        content += "\n## Agent Services (11000-11999)\n"
        for port, (service, container) in sorted(agents.items(), key=lambda x: int(x[0])):
            content += f"- **{port}**: {container} ({service})\n"
            
        if other:
            content += "\n## Other Services\n"
            for port, (service, container) in sorted(other.items(), key=lambda x: int(x[0])):
                content += f"- **{port}**: {container} ({service})\n"
                
        content += f"""
---

## Port Analysis Summary
- **Total Ports Mapped**: {len(self.port_mappings)}
- **Core Infrastructure**: {len(core_infra)} ports
- **AI & Vector Services**: {len(ai_vector)} ports
- **Monitoring Stack**: {len(monitoring)} ports
- **MCP Bridge**: {len(mcp_bridge)} ports
- **Agent Services**: {len(agents)} ports
- **Other Services**: {len(other)} ports

## Port Conflicts Detected
"""
        if self.conflicts:
            for conflict in self.conflicts:
                content += f"- **Port {conflict['port']}**: Conflict between {conflict['service1']} and {conflict['service2']}\n"
        else:
            content += "- No port conflicts detected ✅\n"
            
        content += "\n## Network Configuration\n"
        content += "- **Primary Network**: sutazai-network (172.20.0.0/16)\n"
        content += "- **See docker-compose files for detailed network configuration**\n"
        
        # Save updated registry
        updated_path = self.repo_root / "IMPORTANT" / "ports" / "PortRegistry_Updated.md"
        with open(updated_path, 'w') as f:
            f.write(content)
            
        logger.info(f"Updated registry saved to: {updated_path}")
        
        return content
        
    def generate_audit_report(self):
        """Generate comprehensive port audit report"""
        logger.info("Generating port audit report...")
        
        report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'total_ports': len(self.port_mappings),
            'total_conflicts': len(self.conflicts),
            'total_discrepancies': len(self.discrepancies),
            'docker_compose_files_scanned': len(self.docker_compose_files),
            'port_mappings': {
                port: {
                    'service': info['service'],
                    'container': info['container_name'],
                    'internal_port': info['internal_port'],
                    'file': str(info['file'].name)
                }
                for port, info in self.port_mappings.items()
            },
            'conflicts': self.conflicts,
            'discrepancies': self.discrepancies
        }
        
        report_path = self.repo_root / "port_registry_audit.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Audit report saved to: {report_path}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("PORT REGISTRY AUDIT SUMMARY")
        print("=" * 80)
        print(f"Total Ports Mapped: {len(self.port_mappings)}")
        print(f"Port Conflicts: {len(self.conflicts)}")
        print(f"Discrepancies with Registry: {len(self.discrepancies)}")
        print(f"Docker Compose Files Scanned: {len(self.docker_compose_files)}")
        print("=" * 80)
        
        if self.conflicts:
            print("\nPORT CONFLICTS:")
            for conflict in self.conflicts:
                print(f"  ⚠️  Port {conflict['port']}: {conflict['service1']} vs {conflict['service2']}")
                
        if self.discrepancies:
            print("\nDISCREPANCIES:")
            for disc in self.discrepancies[:10]:  # Show first 10
                if disc['type'] == 'in_actual_not_registry':
                    print(f"  ℹ️  Port {disc['port']} ({disc['service']}) in docker-compose but not in registry")
                else:
                    print(f"  ℹ️  Port {disc['port']} in registry but not in docker-compose")
                    
        print("=" * 80)
        
        return report
        
    def run_full_audit(self):
        """Execute complete port registry audit"""
        logger.info("Starting port registry audit...")
        
        self.scan_docker_compose_files()
        self.compare_with_registry()
        updated_content = self.generate_updated_registry()
        report = self.generate_audit_report()
        
        logger.info("Port registry audit completed successfully")
        
        return report

def main():
    """Main execution"""
    repo_root = Path("/home/runner/work/sutazaiapp/sutazaiapp")
    
    verifier = PortRegistryVerifier(repo_root)
    verifier.run_full_audit()
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
