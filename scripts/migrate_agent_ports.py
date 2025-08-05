#!/usr/bin/env python3
"""
Port Migration Script for SUTAZAIAPP Agents
Automatically migrates non-compliant agent services to the standard 11000-11148 port range.
"""

import yaml
import glob
import sys
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Standard agent port range
AGENT_PORT_RANGE = (11000, 11148)

# Port mappings for migration
PORT_MIGRATION_MAP = {
    # Critical agents currently using 8xxx ports
    8200: 11069,  # ai-system-architect
    8201: 11070,  # deployment-automation-master
    8202: 11071,  # mega-code-auditor
    8203: 11072,  # system-optimizer-reorganizer
    8204: 11073,  # hardware-resource-optimizer
    8205: 11074,  # ollama-integration-specialist
    8206: 11075,  # infrastructure-devops-manager
    8207: 11076,  # ai-agent-orchestrator
    8208: 11077,  # ai-senior-backend-developer
    8209: 11078,  # ai-senior-frontend-developer
    8210: 11079,  # testing-qa-validator
    8211: 11080,  # document-knowledge-manager
    8212: 11081,  # security-pentesting-specialist
    8213: 11082,  # cicd-pipeline-orchestrator
    
    # Other agents needing migration
    8003: 11083,  # ai-system-architect (duplicate)
    8004: 11084,  # ai-agent-orchestrator (duplicate)
    8005: 11085,  # infrastructure-devops-manager (duplicate)
    8006: 11086,  # deployment-automation-master (duplicate)
    8007: 11087,  # Reserved
    8008: 11088,  # testing-qa-validator
    8009: 11089,  # agent-creator
    8010: 11090,  # agent-debugger
    8011: 11091,  # docker-specialist
    8012: 11092,  # ai-senior-backend-developer (duplicate)
    8013: 11093,  # ai-senior-engineer
    8014: 11094,  # ai-senior-backend-developer (duplicate)
    8015: 11095,  # ai-senior-frontend-developer (duplicate)
    8016: 11096,  # ai-senior-full-stack-developer
    8017: 11097,  # ai-system-validator
    8018: 11098,  # ai-testing-qa-validator
    8019: 11099,  # kali-hacker/mega-code-auditor
    
    # Special agent ports
    8365: 11100,  # edge-computing-optimizer
    8388: 11101,  # data-analysis-engineer
    8551: 11102,  # task-assignment-coordinator
    8586: 11103,  # agentzero-coordinator
    8587: 11104,  # multi-agent-coordinator
    8588: 11105,  # resource-arbitration-agent
    8589: 11106,  # ai-agent-orchestrator (orchestration)
    8726: 11107,  # deep-local-brain-builder
}

class PortMigrator:
    def __init__(self, dry_run=True):
        self.dry_run = dry_run
        self.migrations_performed = []
        self.errors = []
        self.backup_dir = f"backups/port_migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    def create_backup(self, file_path: str):
        """Create a backup of the file before modification."""
        if not self.dry_run:
            os.makedirs(self.backup_dir, exist_ok=True)
            backup_path = os.path.join(self.backup_dir, os.path.basename(file_path))
            shutil.copy2(file_path, backup_path)
            return backup_path
        return None
    
    def is_agent_service(self, service_name: str) -> bool:
        """Check if a service is an agent based on naming patterns."""
        agent_patterns = [
            'agent', 'orchestrator', 'coordinator', 'validator',
            'developer', 'engineer', 'architect', 'manager',
            'specialist', 'optimizer', 'builder', 'creator',
            'debugger', 'auditor', 'ai-'
        ]
        service_lower = service_name.lower()
        return any(pattern in service_lower for pattern in agent_patterns)
    
    def get_next_available_port(self, used_ports: set) -> int:
        """Find the next available port in the agent range."""
        for port in range(AGENT_PORT_RANGE[0], AGENT_PORT_RANGE[1] + 1):
            if port not in used_ports and port not in PORT_MIGRATION_MAP.values():
                return port
        return None
    
    def migrate_compose_file(self, file_path: str, used_ports: set) -> Tuple[bool, List[Dict]]:
        """Migrate ports in a single docker-compose file."""
        migrations = []
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                original_content = content
            
            # Parse YAML
            data = yaml.safe_load(content)
            if not data or 'services' not in data:
                return False, migrations
            
            modified = False
            
            for service_name, service_config in data.get('services', {}).items():
                if not self.is_agent_service(service_name):
                    continue
                
                if 'ports' not in service_config:
                    continue
                
                container_name = service_config.get('container_name', service_name)
                new_ports = []
                
                for port_mapping in service_config['ports']:
                    if isinstance(port_mapping, str):
                        # Parse port mapping
                        if ':' in port_mapping:
                            parts = port_mapping.split(':')
                            host_port = parts[0].strip('"').strip()
                            
                            if host_port.isdigit():
                                host_port = int(host_port)
                                
                                # Check if port needs migration
                                if not (AGENT_PORT_RANGE[0] <= host_port <= AGENT_PORT_RANGE[1]):
                                    # Determine new port
                                    if host_port in PORT_MIGRATION_MAP:
                                        new_port = PORT_MIGRATION_MAP[host_port]
                                    else:
                                        new_port = self.get_next_available_port(used_ports)
                                    
                                    if new_port:
                                        # Create new mapping
                                        new_mapping = port_mapping.replace(str(host_port), str(new_port))
                                        
                                        # Update in content (preserve formatting)
                                        content = content.replace(
                                            f'"{port_mapping}"',
                                            f'"{new_mapping}"'
                                        )
                                        content = content.replace(
                                            f"'{port_mapping}'",
                                            f"'{new_mapping}'"
                                        )
                                        content = content.replace(
                                            f"- {port_mapping}",
                                            f"- {new_mapping}"
                                        )
                                        
                                        migrations.append({
                                            'service': service_name,
                                            'container': container_name,
                                            'old_port': host_port,
                                            'new_port': new_port,
                                            'old_mapping': port_mapping,
                                            'new_mapping': new_mapping,
                                            'file': file_path
                                        })
                                        
                                        used_ports.add(new_port)
                                        modified = True
            
            if modified and not self.dry_run:
                # Create backup
                backup_path = self.create_backup(file_path)
                
                # Write updated content
                with open(file_path, 'w') as f:
                    f.write(content)
                
                print(f"✅ Updated {file_path} (backup: {backup_path})")
            
            return modified, migrations
            
        except Exception as e:
            self.errors.append(f"Error processing {file_path}: {e}")
            return False, []
    
    def scan_and_migrate(self):
        """Scan all docker-compose files and perform migrations."""
        compose_files = glob.glob('docker-compose*.yml')
        used_ports = set(PORT_MIGRATION_MAP.values())
        
        print("=" * 80)
        print("SUTAZAIAPP Agent Port Migration Tool")
        print("=" * 80)
        print(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE MIGRATION'}")
        print(f"Scanning {len(compose_files)} docker-compose files...")
        print()
        
        all_migrations = []
        
        for file_path in sorted(compose_files):
            modified, migrations = self.migrate_compose_file(file_path, used_ports)
            if migrations:
                all_migrations.extend(migrations)
                self.migrations_performed.extend(migrations)
        
        # Print summary
        if all_migrations:
            print("📋 MIGRATION SUMMARY")
            print("-" * 40)
            
            # Group by file
            by_file = {}
            for migration in all_migrations:
                file_name = Path(migration['file']).name
                if file_name not in by_file:
                    by_file[file_name] = []
                by_file[file_name].append(migration)
            
            for file_name, migrations in by_file.items():
                print(f"\n{file_name}:")
                for m in migrations:
                    print(f"  {m['service']} ({m['container']})")
                    print(f"    Port: {m['old_port']} → {m['new_port']}")
            
            print(f"\nTotal migrations: {len(all_migrations)}")
            
            if self.dry_run:
                print("\n⚠️  DRY RUN MODE - No changes were made")
                print("Run with --apply to perform actual migration")
            else:
                print(f"\n✅ Migration complete! Backups saved to {self.backup_dir}")
        else:
            print("✅ No migrations needed - all agents are using compliant ports!")
        
        if self.errors:
            print("\n❌ ERRORS:")
            for error in self.errors:
                print(f"  - {error}")
        
        return len(all_migrations) > 0
    
    def generate_migration_report(self):
        """Generate a detailed migration report."""
        if not self.migrations_performed:
            return
        
        report_path = f"migration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_path, 'w') as f:
            f.write("# Agent Port Migration Report\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Mode**: {'Dry Run' if self.dry_run else 'Applied'}\n\n")
            
            f.write("## Migrations Performed\n\n")
            f.write("| Service | Container | Old Port | New Port | File |\n")
            f.write("|---------|-----------|----------|----------|------|\n")
            
            for m in self.migrations_performed:
                f.write(f"| {m['service']} | {m['container']} | {m['old_port']} | {m['new_port']} | {Path(m['file']).name} |\n")
            
            f.write(f"\n**Total Migrations**: {len(self.migrations_performed)}\n")
            
            if not self.dry_run:
                f.write(f"\n## Backup Location\n\n`{self.backup_dir}`\n")
            
            f.write("\n## Next Steps\n\n")
            f.write("1. Update the port registry at `/opt/sutazaiapp/config/port-registry.yaml`\n")
            f.write("2. Restart affected services:\n")
            f.write("   ```bash\n")
            
            files = set(m['file'] for m in self.migrations_performed)
            for file in files:
                f.write(f"   docker-compose -f {file} down\n")
                f.write(f"   docker-compose -f {file} up -d\n")
            f.write("   ```\n")
            f.write("3. Validate port allocations:\n")
            f.write("   ```bash\n")
            f.write("   python3 scripts/validate_ports.py\n")
            f.write("   ```\n")
        
        print(f"\n📄 Migration report saved to {report_path}")
    
    def rollback(self):
        """Rollback migrations by restoring from backup."""
        if self.dry_run:
            print("No rollback needed in dry run mode")
            return
        
        if not os.path.exists(self.backup_dir):
            print("No backup directory found")
            return
        
        print(f"Rolling back migrations from {self.backup_dir}...")
        
        for backup_file in os.listdir(self.backup_dir):
            backup_path = os.path.join(self.backup_dir, backup_file)
            original_path = backup_file
            
            if os.path.exists(original_path):
                shutil.copy2(backup_path, original_path)
                print(f"  Restored {original_path}")
        
        print("✅ Rollback complete")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Migrate SUTAZAIAPP agent ports to standard range (11000-11148)'
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Apply migrations (default is dry run)'
    )
    parser.add_argument(
        '--rollback',
        action='store_true',
        help='Rollback last migration'
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate migration report'
    )
    
    args = parser.parse_args()
    
    if args.rollback:
        migrator = PortMigrator(dry_run=False)
        migrator.rollback()
        return
    
    # Run migration
    migrator = PortMigrator(dry_run=not args.apply)
    has_migrations = migrator.scan_and_migrate()
    
    if args.report or has_migrations:
        migrator.generate_migration_report()
    
    # Exit with appropriate code
    if has_migrations and not args.apply:
        sys.exit(2)  # Indicate migrations are needed
    elif migrator.errors:
        sys.exit(1)  # Indicate errors occurred
    else:
        sys.exit(0)  # Success


if __name__ == "__main__":
    main()