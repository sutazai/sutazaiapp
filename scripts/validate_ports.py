#!/usr/bin/env python3
"""
validate_ports.py

Purpose: Ensure docker-compose published ports match the authoritative mapping
from IMPORTANT and config/port-registry.yaml.

Exit codes:
  0 -> OK
  1 -> Drift detected (mismatched or missing ports)
  2 -> Script error (e.g., YAML parsing)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_port_mapping(compose: dict) -> Dict[str, List[Tuple[int, int]]]:
    services = compose.get("services", {}) or {}
    mapping: Dict[str, List[Tuple[int, int]]] = {}
    for name, cfg in services.items():
        ports = []
        for p in (cfg.get("ports") or []):
            # Formats: "HOST:CONTAINER" or mapping dicts; normalize to tuple[int,int]
            if isinstance(p, str):
                try:
                    host, container = p.split(":", 1)
                    ports.append((int(host), int(container)))
                except Exception:
                    continue
            elif isinstance(p, dict):
                hp = p.get("published")
                cp = p.get("target")
                if isinstance(hp, int) and isinstance(cp, int):
                    ports.append((hp, cp))
        mapping[name] = ports
    return mapping


def merge_mappings(base: Dict[str, List[Tuple[int, int]]], override: Dict[str, List[Tuple[int, int]]]) -> Dict[str, List[Tuple[int, int]]]:
    out = {**base}
    for svc, ports in override.items():
        if ports:
            out[svc] = ports  # override wins
    return out


def expected_service_ports_from_registry(registry: dict) -> Dict[str, List[int]]:
    expected: Dict[str, List[int]] = {}

    def reg(section: str) -> dict:
        return registry.get(section, {}) or {}

    # Infrastructure
    infra = reg("infrastructure")
    infra_expect = {
        "backend": 10010,
        "frontend": 10011,
        "sutazai-postgres": 10000,
        "sutazai-redis": 10001,
        "sutazai-qdrant": [10101, 10102],
        "sutazai-chromadb": 10100,
        "sutazai-faiss": 10103,
        "sutazai-neo4j": [10002, 10003],
        "sutazai-ollama": 10104,
        "sutazai-kong": 10005,
        "sutazai-consul": 10006,
        "sutazai-rabbitmq": [10007, 10008],
    }
    # Monitoring
    mon = reg("monitoring")
    mon_expect = {
        "sutazai-prometheus": 10200,
        "sutazai-grafana": 10201,
        "sutazai-loki": 10202,
        "sutazai-alertmanager": 10203,
        "sutazai-blackbox-exporter": 10204,
        "sutazai-node-exporter": 10220,
        "sutazai-cadvisor": 10221,
        "sutazai-postgres-exporter": 10207,
        "sutazai-redis-exporter": 10208,
        "sutazai-ai-metrics-exporter": 11063,
    }

    def add(expect: Dict[str, List[int]]):
        for k, v in expect.items():
            expected[k] = v if isinstance(v, list) else [v]

    add(infra_expect)
    add(mon_expect)
    return expected


def normalize_container_name(name: str) -> str:
    # docker-compose uses service keys; containers have container_name in many cases
    return name.strip()


def invert_by_container(mapping: Dict[str, List[Tuple[int, int]]]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for svc, pairs in mapping.items():
        out[svc] = [h for h, _ in pairs]
    return out


def main() -> int:
    try:
        compose_path = REPO_ROOT / "docker-compose.yml"
        override_path = REPO_ROOT / "docker-compose.override.yml"
        registry_path = REPO_ROOT / "config" / "port-registry.yaml"

        base = load_yaml(compose_path)
        override = load_yaml(override_path) if override_path.exists() else {"services": {}}
        registry = load_yaml(registry_path)

        base_map = parse_port_mapping(base)
        override_map = parse_port_mapping(override)
        effective = merge_mappings(base_map, override_map)
        effective_by_service = invert_by_container(effective)

        expected = expected_service_ports_from_registry(registry)

        errors: List[str] = []

        # try to match by known container_name in compose when present
        # but we only have service keys here; map heuristically
        alias_map = {
            "backend": "sutazai-backend",
            "frontend": "sutazai-frontend",
            "postgres": "sutazai-postgres",
            "redis": "sutazai-redis",
            "qdrant": "sutazai-qdrant",
            "chromadb": "sutazai-chromadb",
            "faiss": "sutazai-faiss",
            "neo4j": "sutazai-neo4j",
            "ollama": "sutazai-ollama",
            "prometheus": "sutazai-prometheus",
            "grafana": "sutazai-grafana",
            "loki": "sutazai-loki",
            "alertmanager": "sutazai-alertmanager",
            "blackbox-exporter": "sutazai-blackbox-exporter",
            "node-exporter": "sutazai-node-exporter",
            "cadvisor": "sutazai-cadvisor",
            "postgres-exporter": "sutazai-postgres-exporter",
            "redis-exporter": "sutazai-redis-exporter",
            "ai-metrics-exporter": "sutazai-ai-metrics-exporter",
            "kong": "sutazai-kong",
            "consul": "sutazai-consul",
            "rabbitmq": "sutazai-rabbitmq",
        }

        # Build lookup of effective ports by canonical container name
        effective_by_canonical: Dict[str, List[int]] = {}
        for svc_key, ports in effective_by_service.items():
            canonical = alias_map.get(svc_key, svc_key)
            effective_by_canonical[canonical] = ports

        for canonical_container, exp_ports in expected.items():
            eff_ports = effective_by_canonical.get(canonical_container, [])
            # Allow services to be absent (then they won't be compared)
            if not eff_ports:
                continue
            exp_set = set(exp_ports)
            eff_set = set(eff_ports)
            if exp_set != eff_set:
                errors.append(
                    f"Port drift for {canonical_container}: expected {sorted(exp_set)}, found {sorted(eff_set)}"
                )

        if errors:
            print("IMPORTANT alignment check failed (ports):", file=sys.stderr)
            for e in errors:
                print(f" - {e}", file=sys.stderr)
            return 1

        print("OK: Ports match IMPORTANT/config registry where services are present.")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Port Validation Script for SUTAZAIAPP
Validates port allocations across all docker-compose files and checks for conflicts.
"""

import yaml
import glob
import sys
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

# Port range definitions
PORT_RANGES = {
    'infrastructure': (10000, 10199),
    'monitoring': (10200, 10299),
    'integrations': (10300, 10499),
    'agi_system': (10500, 10599),
    'agents': (11000, 11148),
    'ollama': (10104, 11436)
}

# Service type patterns
SERVICE_PATTERNS = {
    'infrastructure': ['postgres', 'redis', 'rabbitmq', 'kong', 'consul', 'vault', 'qdrant'],
    'monitoring': ['prometheus', 'grafana', 'loki', 'jaeger', 'alertmanager', 'exporter', 'cadvisor'],
    'agents': ['ai-', 'agent-', '-agent', 'coordinator', 'orchestrator', 'validator', 'developer']
}

class PortValidator:
    def __init__(self):
        self.port_allocations = {}
        self.conflicts = []
        self.warnings = []
        self.non_compliant = []
        
    def load_port_registry(self, registry_path: str = 'config/port-registry.yaml'):
        """Load the official port registry."""
        try:
            with open(registry_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: Port registry not found at {registry_path}")
            return None
    
    def extract_ports_from_compose(self, file_path: str) -> Dict:
        """Extract port mappings from a docker-compose file."""
        mappings = {}
        try:
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
                if data and 'services' in data:
                    for service_name, service_config in data.get('services', {}).items():
                        if 'ports' in service_config:
                            ports = service_config['ports']
                            container_name = service_config.get('container_name', service_name)
                            for port in ports:
                                if isinstance(port, str):
                                    port_parts = port.split(':')
                                    if len(port_parts) >= 2:
                                        host_port = port_parts[0].strip('"')
                                        if host_port.isdigit():
                                            host_port = int(host_port)
                                            if host_port in mappings:
                                                self.conflicts.append({
                                                    'port': host_port,
                                                    'services': [mappings[host_port]['service'], service_name],
                                                    'files': [mappings[host_port]['file'], str(file_path)]
                                                })
                                            mappings[host_port] = {
                                                'service': service_name,
                                                'container': container_name,
                                                'file': str(file_path)
                                            }
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
        return mappings
    
    def categorize_service(self, service_name: str) -> str:
        """Categorize a service based on its name."""
        service_lower = service_name.lower()
        
        # Check for specific patterns
        for category, patterns in SERVICE_PATTERNS.items():
            for pattern in patterns:
                if pattern in service_lower:
                    return category
        
        # Check for agent-specific patterns
        if any(keyword in service_lower for keyword in ['ai-', 'agent', 'orchestrator', 'validator', 'developer']):
            return 'agents'
        
        return 'other'
    
    def validate_port_range(self, port: int, service_name: str, container: str, file: str):
        """Validate if a port is in the correct range for its service type."""
        service_type = self.categorize_service(service_name)
        
        # Check if it's an agent service
        if service_type == 'agents':
            if not (PORT_RANGES['agents'][0] <= port <= PORT_RANGES['agents'][1]):
                self.non_compliant.append({
                    'port': port,
                    'service': service_name,
                    'container': container,
                    'file': file,
                    'expected_range': PORT_RANGES['agents'],
                    'reason': f"Agent service using port {port} outside standard range {PORT_RANGES['agents']}"
                })
        
        # Check infrastructure services
        elif service_type == 'infrastructure':
            if not (PORT_RANGES['infrastructure'][0] <= port <= PORT_RANGES['infrastructure'][1]):
                self.warnings.append({
                    'port': port,
                    'service': service_name,
                    'container': container,
                    'file': file,
                    'expected_range': PORT_RANGES['infrastructure'],
                    'reason': f"Infrastructure service using port {port} outside recommended range"
                })
        
        # Check monitoring services
        elif service_type == 'monitoring':
            if not (PORT_RANGES['monitoring'][0] <= port <= PORT_RANGES['monitoring'][1]):
                self.warnings.append({
                    'port': port,
                    'service': service_name,
                    'container': container,
                    'file': file,
                    'expected_range': PORT_RANGES['monitoring'],
                    'reason': f"Monitoring service using port {port} outside recommended range"
                })
        
        # Check for use of common/privileged ports
        if port < 1024:
            self.warnings.append({
                'port': port,
                'service': service_name,
                'container': container,
                'file': file,
                'reason': f"Using privileged port {port} (< 1024)"
            })
        elif port in [8080, 8000, 3000] and 'agent' in service_name.lower():
            self.warnings.append({
                'port': port,
                'service': service_name,
                'container': container,
                'file': file,
                'reason': f"Agent using common port {port} instead of agent range"
            })
    
    def check_port_availability(self, start_port: int, end_port: int) -> List[int]:
        """Find available ports in a given range."""
        used_ports = set(self.port_allocations.keys())
        available = []
        for port in range(start_port, end_port + 1):
            if port not in used_ports:
                available.append(port)
        return available
    
    def suggest_port(self, service_name: str) -> int:
        """Suggest an appropriate port for a service."""
        service_type = self.categorize_service(service_name)
        
        if service_type == 'agents':
            available = self.check_port_availability(PORT_RANGES['agents'][0], PORT_RANGES['agents'][1])
            if available:
                return available[0]
        elif service_type == 'infrastructure':
            available = self.check_port_availability(PORT_RANGES['infrastructure'][0], PORT_RANGES['infrastructure'][1])
            if available:
                return available[0]
        elif service_type == 'monitoring':
            available = self.check_port_availability(PORT_RANGES['monitoring'][0], PORT_RANGES['monitoring'][1])
            if available:
                return available[0]
        
        # Fallback to any available port in a safe range
        available = self.check_port_availability(12000, 13000)
        if available:
            return available[0]
        return None
    
    def validate_all(self):
        """Run complete validation across all docker-compose files."""
        compose_files = glob.glob('docker-compose*.yml')
        
        print("=" * 80)
        print("SUTAZAIAPP Port Validation Report")
        print("=" * 80)
        print()
        
        # Load all port allocations
        for file_path in sorted(compose_files):
            mappings = self.extract_ports_from_compose(file_path)
            for port, info in mappings.items():
                if port in self.port_allocations:
                    # Conflict detected
                    existing = self.port_allocations[port]
                    if existing['service'] != info['service']:
                        self.conflicts.append({
                            'port': port,
                            'services': [existing['service'], info['service']],
                            'containers': [existing['container'], info['container']],
                            'files': [existing['file'], info['file']]
                        })
                else:
                    self.port_allocations[port] = info
                
                # Validate port range compliance
                self.validate_port_range(port, info['service'], info['container'], info['file'])
        
        # Print summary
        print(f"Total ports allocated: {len(self.port_allocations)}")
        print(f"Total compose files scanned: {len(compose_files)}")
        print()
        
        # Print conflicts
        if self.conflicts:
            print("❌ PORT CONFLICTS DETECTED:")
            print("-" * 40)
            for conflict in self.conflicts:
                print(f"Port {conflict['port']}:")
                for i, service in enumerate(conflict.get('services', [])):
                    container = conflict.get('containers', [None, None])[i] if 'containers' in conflict else 'unknown'
                    file = conflict.get('files', ['unknown', 'unknown'])[i]
                    print(f"  - {service} ({container}) in {Path(file).name}")
            print()
        else:
            print("✅ No port conflicts detected")
            print()
        
        # Print non-compliant services
        if self.non_compliant:
            print("⚠️ NON-COMPLIANT PORT ALLOCATIONS:")
            print("-" * 40)
            for item in self.non_compliant:
                print(f"Port {item['port']}: {item['service']} ({item['container']})")
                print(f"  File: {Path(item['file']).name}")
                print(f"  Issue: {item['reason']}")
                suggested = self.suggest_port(item['service'])
                if suggested:
                    print(f"  Suggested port: {suggested}")
                print()
        
        # Print warnings
        if self.warnings:
            print("⚠️ WARNINGS:")
            print("-" * 40)
            for warning in self.warnings[:10]:  # Limit to first 10 warnings
                print(f"Port {warning['port']}: {warning['service']}")
                print(f"  Issue: {warning['reason']}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more warnings")
            print()
        
        # Print port range usage
        print("📊 PORT RANGE USAGE:")
        print("-" * 40)
        for range_name, (start, end) in PORT_RANGES.items():
            used_in_range = [p for p in self.port_allocations.keys() if start <= p <= end]
            total_available = end - start + 1
            usage_percent = (len(used_in_range) / total_available) * 100 if total_available > 0 else 0
            print(f"{range_name:15} ({start:5}-{end:5}): {len(used_in_range):3}/{total_available:3} used ({usage_percent:.1f}%)")
        print()
        
        # Print available ports for agents
        agent_range = PORT_RANGES['agents']
        available_agent_ports = self.check_port_availability(agent_range[0], agent_range[1])
        print(f"📝 AVAILABLE AGENT PORTS: {len(available_agent_ports)} ports")
        if available_agent_ports[:10]:
            print(f"   Next available: {', '.join(map(str, available_agent_ports[:10]))}")
            if len(available_agent_ports) > 10:
                print(f"   ... and {len(available_agent_ports) - 10} more")
        print()
        
        # Export results
        self.export_results()
        
        # Return exit code
        if self.conflicts:
            return 1  # Exit with error if conflicts exist
        elif self.non_compliant:
            return 2  # Exit with warning if non-compliant services exist
        else:
            return 0  # Success
    
    def export_results(self):
        """Export validation results to JSON."""
        results = {
            'total_ports': len(self.port_allocations),
            'conflicts': self.conflicts,
            'non_compliant': self.non_compliant,
            'warnings': self.warnings[:20],  # Limit warnings in export
            'port_allocations': {
                str(port): info for port, info in sorted(self.port_allocations.items())
            },
            'available_agent_ports': self.check_port_availability(PORT_RANGES['agents'][0], PORT_RANGES['agents'][1])[:50]
        }
        
        with open('port_validation_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print("Results exported to port_validation_results.json")
    
    def fix_docker_compose(self, file_path: str, dry_run: bool = True):
        """Suggest fixes for a docker-compose file."""
        print(f"\n🔧 SUGGESTED FIXES for {file_path}:")
        print("-" * 40)
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                data = yaml.safe_load(content)
            
            if not data or 'services' not in data:
                return
            
            fixes = []
            for service_name, service_config in data.get('services', {}).items():
                if 'ports' in service_config:
                    for i, port_mapping in enumerate(service_config['ports']):
                        if isinstance(port_mapping, str):
                            port_parts = port_mapping.split(':')
                            if len(port_parts) >= 2:
                                host_port = port_parts[0].strip('"')
                                if host_port.isdigit():
                                    host_port = int(host_port)
                                    service_type = self.categorize_service(service_name)
                                    
                                    # Check if port needs fixing
                                    needs_fix = False
                                    if service_type == 'agents' and not (PORT_RANGES['agents'][0] <= host_port <= PORT_RANGES['agents'][1]):
                                        needs_fix = True
                                        suggested = self.suggest_port(service_name)
                                        if suggested:
                                            fixes.append({
                                                'service': service_name,
                                                'old_port': host_port,
                                                'new_port': suggested,
                                                'old_mapping': port_mapping,
                                                'new_mapping': port_mapping.replace(str(host_port), str(suggested))
                                            })
            
            if fixes:
                for fix in fixes:
                    print(f"Service: {fix['service']}")
                    print(f"  Change: {fix['old_mapping']} → {fix['new_mapping']}")
                
                if not dry_run:
                    # Apply fixes
                    for fix in fixes:
                        content = content.replace(fix['old_mapping'], fix['new_mapping'])
                    
                    backup_path = f"{file_path}.backup"
                    with open(backup_path, 'w') as f:
                        f.write(content)
                    
                    with open(file_path, 'w') as f:
                        f.write(content)
                    
                    print(f"\n✅ Fixes applied! Backup saved to {backup_path}")
                else:
                    print(f"\n(Dry run - no changes made. Run with --fix to apply changes)")
            else:
                print("No fixes needed for this file.")
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate port allocations in SUTAZAIAPP')
    parser.add_argument('--fix', action='store_true', help='Apply suggested fixes to docker-compose files')
    parser.add_argument('--file', type=str, help='Specific docker-compose file to validate/fix')
    parser.add_argument('--check-port', type=int, help='Check if a specific port is available')
    parser.add_argument('--suggest', type=str, help='Suggest a port for a service name')
    
    args = parser.parse_args()
    
    validator = PortValidator()
    
    if args.check_port:
        # Check specific port availability
        validator.port_allocations = {}
        compose_files = glob.glob('docker-compose*.yml')
        for file_path in compose_files:
            mappings = validator.extract_ports_from_compose(file_path)
            validator.port_allocations.update(mappings)
        
        if args.check_port in validator.port_allocations:
            info = validator.port_allocations[args.check_port]
            print(f"Port {args.check_port} is IN USE by:")
            print(f"  Service: {info['service']}")
            print(f"  Container: {info['container']}")
            print(f"  File: {info['file']}")
        else:
            print(f"Port {args.check_port} is AVAILABLE")
        return
    
    if args.suggest:
        # Suggest port for a service
        validator.port_allocations = {}
        compose_files = glob.glob('docker-compose*.yml')
        for file_path in compose_files:
            mappings = validator.extract_ports_from_compose(file_path)
            validator.port_allocations.update(mappings)
        
        suggested = validator.suggest_port(args.suggest)
        if suggested:
            print(f"Suggested port for '{args.suggest}': {suggested}")
        else:
            print(f"No available ports found for '{args.suggest}'")
        return
    
    if args.file:
        # Validate/fix specific file
        if args.fix:
            validator.fix_docker_compose(args.file, dry_run=False)
        else:
            mappings = validator.extract_ports_from_compose(args.file)
            print(f"Ports in {args.file}:")
            for port, info in sorted(mappings.items()):
                print(f"  {port}: {info['service']} ({info['container']})")
    else:
        # Run full validation
        exit_code = validator.validate_all()
        
        if args.fix:
            print("\n" + "=" * 80)
            print("FIX MODE - Suggesting fixes for non-compliant services")
            print("=" * 80)
            
            # Get unique files with non-compliant services
            files_to_fix = set()
            for item in validator.non_compliant:
                files_to_fix.add(item['file'])
            
            for file_path in files_to_fix:
                validator.fix_docker_compose(file_path, dry_run=False)
        
        sys.exit(exit_code)


if __name__ == "__main__":
    main()