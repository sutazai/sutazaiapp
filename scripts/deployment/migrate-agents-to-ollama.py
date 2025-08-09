#!/usr/bin/env python3
"""
Purpose: Migrate all 131 agents to use BaseAgentV2 with Ollama integration
Usage: python migrate-agents-to-ollama.py [--dry-run] [--phase PHASE]
Requirements: pyyaml, httpx, asyncio
"""

import os
import sys
import json
import shutil
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import yaml

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from agents.core.ollama_integration import OllamaConfig
from agents.core.migration_helper import validate_migration

class AgentMigrator:
    """Handles migration of agents to BaseAgentV2 with Ollama integration"""
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.agents_dir = Path("/opt/sutazaiapp/agents")
        self.backup_dir = Path(f"/opt/sutazaiapp/backups/migration-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        self.migration_log = []
        self.success_count = 0
        self.failure_count = 0
        
        # Migration phases
        self.phases = {
            "canary": ["health-monitor", "garbage-collector", "resource-visualiser"],
            "development": [
                "agent-debugger", "ai-testing-qa-validator", "testing-qa-team-lead",
                "qa-team-lead", "testing-qa-validator", "code-improver"
            ],
            "balanced": [agent for agent in OllamaConfig.AGENT_MODELS.keys() 
                        if OllamaConfig.AGENT_MODELS[agent] == OllamaConfig.SONNET_MODEL][:20],
            "complex": [agent for agent in OllamaConfig.AGENT_MODELS.keys() 
                       if OllamaConfig.AGENT_MODELS[agent] == OllamaConfig.OPUS_MODEL][:10],
            "production": [],  # Will be filled with remaining agents
            "complete": []     # All agents
        }
        
    def log(self, message: str, level: str = "INFO"):
        """Log migration message"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        print(log_entry)
        self.migration_log.append(log_entry)
        
    def create_backup(self, agent_path: Path) -> Optional[Path]:
        """Create backup of agent before migration"""
        if self.dry_run:
            self.log(f"[DRY RUN] Would backup {agent_path}")
            return None
            
        try:
            backup_path = self.backup_dir / agent_path.name
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Copy all files
            for file in agent_path.glob("*"):
                if file.is_file():
                    shutil.copy2(file, backup_path / file.name)
                    
            self.log(f"Backed up {agent_path.name} to {backup_path}")
            return backup_path
        except Exception as e:
            self.log(f"Failed to backup {agent_path.name}: {e}", "ERROR")
            return None
            
    def update_agent_imports(self, agent_path: Path) -> bool:
        """Update agent to use BaseAgentV2"""
        app_file = agent_path / "app.py"
        if not app_file.exists():
            app_file = agent_path / "agent.py"
            if not app_file.exists():
                self.log(f"No app.py or agent.py found in {agent_path.name}", "WARNING")
                return False
                
        try:
            content = app_file.read_text()
            
            # Check if already migrated
            if "BaseAgentV2" in content or "base_agent_v2" in content:
                self.log(f"{agent_path.name} already migrated")
                return True
                
            # Update imports
            updates = [
                ("from agents.agent_base import BaseAgent", 
                 "from agents.core.base_agent_v2 import BaseAgentV2"),
                ("from agent_base import BaseAgent",
                 "from agents.core.base_agent_v2 import BaseAgentV2"),
                ("from .agent_base import BaseAgent",
                 "from agents.core.base_agent_v2 import BaseAgentV2"),
                ("class.*\\(BaseAgent\\)", "class \\1(BaseAgentV2)"),
            ]
            
            modified = False
            for old_pattern, new_pattern in updates:
                if old_pattern in content:
                    content = content.replace(old_pattern, new_pattern)
                    modified = True
                    
            # Update class definition
            import re
            content = re.sub(
                r'class\s+(\w+)\s*\(\s*BaseAgent\s*\)',
                r'class \1(BaseAgentV2)',
                content
            )
            
            # Add async query_ollama if using sync version
            if "def query_ollama(" in content and "async def query_ollama_async(" not in content:
                content = content.replace(
                    "def query_ollama(",
                    "async def query_ollama_async("
                )
                modified = True
                
            if modified:
                if not self.dry_run:
                    app_file.write_text(content)
                self.log(f"Updated imports in {agent_path.name}")
                return True
            else:
                self.log(f"No updates needed for {agent_path.name}")
                return True
                
        except Exception as e:
            self.log(f"Failed to update {agent_path.name}: {e}", "ERROR")
            return False
            
    def update_dockerfile(self, agent_path: Path) -> bool:
        """Update Dockerfile to include new dependencies"""
        dockerfile = agent_path / "Dockerfile"
        if not dockerfile.exists():
            self.log(f"No Dockerfile in {agent_path.name}", "WARNING")
            return True  # Not critical
            
        try:
            content = dockerfile.read_text()
            
            # Check if already has core module copy
            if "COPY agents/core /app/agents/core" in content:
                return True
                
            # Add core module copy before COPY app.py
            lines = content.split('\n')
            new_lines = []
            for line in lines:
                if line.strip().startswith("COPY app.py") or line.strip().startswith("COPY agent.py"):
                    # Add core module copy before app copy
                    new_lines.append("# Copy enhanced agent core modules")
                    new_lines.append("COPY agents/core /app/agents/core")
                    new_lines.append("")
                new_lines.append(line)
                
            if not self.dry_run:
                dockerfile.write_text('\n'.join(new_lines))
            self.log(f"Updated Dockerfile for {agent_path.name}")
            return True
            
        except Exception as e:
            self.log(f"Failed to update Dockerfile for {agent_path.name}: {e}", "ERROR")
            return False
            
    def update_requirements(self, agent_path: Path) -> bool:
        """Update requirements.txt with new dependencies"""
        req_file = agent_path / "requirements.txt"
        if not req_file.exists():
            self.log(f"No requirements.txt in {agent_path.name}", "WARNING")
            return True  # Not critical
            
        try:
            content = req_file.read_text()
            required_deps = ["httpx>=0.25.0", "prometheus-client>=0.19.0"]
            
            lines = content.strip().split('\n')
            existing = set(line.split('>=')[0].split('==')[0].strip() for line in lines if line.strip())
            
            added = False
            for dep in required_deps:
                dep_name = dep.split('>=')[0]
                if dep_name not in existing:
                    lines.append(dep)
                    added = True
                    
            if added:
                if not self.dry_run:
                    req_file.write_text('\n'.join(lines) + '\n')
                self.log(f"Updated requirements.txt for {agent_path.name}")
                
            return True
            
        except Exception as e:
            self.log(f"Failed to update requirements for {agent_path.name}: {e}", "ERROR")
            return False
            
    def migrate_agent(self, agent_name: str) -> bool:
        """Migrate a single agent"""
        agent_path = self.agents_dir / agent_name
        
        if not agent_path.exists():
            self.log(f"Agent directory not found: {agent_name}", "ERROR")
            return False
            
        self.log(f"Migrating {agent_name}...")
        
        # Create backup
        backup_path = self.create_backup(agent_path)
        
        # Update agent files
        success = True
        success &= self.update_agent_imports(agent_path)
        success &= self.update_dockerfile(agent_path)
        success &= self.update_requirements(agent_path)
        
        if success:
            self.success_count += 1
            self.log(f"Successfully migrated {agent_name}", "SUCCESS")
        else:
            self.failure_count += 1
            self.log(f"Failed to migrate {agent_name}", "ERROR")
            
            # Restore from backup if not dry run
            if not self.dry_run and backup_path and backup_path.exists():
                self.log(f"Restoring {agent_name} from backup...")
                shutil.rmtree(agent_path)
                shutil.copytree(backup_path, agent_path)
                
        return success
        
    def get_agents_for_phase(self, phase: str) -> List[str]:
        """Get list of agents for a specific phase"""
        if phase == "complete":
            # All agents
            return [d.name for d in self.agents_dir.iterdir() 
                   if d.is_dir() and not d.name.startswith('.')]
        elif phase == "production":
            # All agents not in other phases
            all_agents = set(d.name for d in self.agents_dir.iterdir() 
                            if d.is_dir() and not d.name.startswith('.'))
            other_phases = set()
            for p, agents in self.phases.items():
                if p not in ["production", "complete"]:
                    other_phases.update(agents)
            return list(all_agents - other_phases)
        else:
            return self.phases.get(phase, [])
            
    def run_migration(self, phase: Optional[str] = None):
        """Run the migration"""
        self.log(f"Starting migration (dry_run={self.dry_run})")
        
        # Create backup directory
        if not self.dry_run:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            
        # Determine which agents to migrate
        if phase:
            agents = self.get_agents_for_phase(phase)
            self.log(f"Migrating phase '{phase}' with {len(agents)} agents")
        else:
            agents = self.get_agents_for_phase("complete")
            self.log(f"Migrating all {len(agents)} agents")
            
        # Migrate each agent
        for agent_name in agents:
            try:
                self.migrate_agent(agent_name)
            except Exception as e:
                self.log(f"Unexpected error migrating {agent_name}: {e}", "ERROR")
                self.failure_count += 1
                
        # Summary
        self.log("=" * 60)
        self.log(f"Migration completed:")
        self.log(f"  Success: {self.success_count}")
        self.log(f"  Failed: {self.failure_count}")
        self.log(f"  Total: {self.success_count + self.failure_count}")
        
        # Save migration log
        if not self.dry_run:
            log_file = self.backup_dir / "migration.log"
            log_file.write_text('\n'.join(self.migration_log))
            self.log(f"Migration log saved to {log_file}")
            
        # Run validation
        if not self.dry_run and self.success_count > 0:
            self.log("Running migration validation...")
            validation_report = validate_migration()
            
            report_file = self.backup_dir / "validation_report.json"
            with open(report_file, 'w') as f:
                json.dump(validation_report, f, indent=2)
            self.log(f"Validation report saved to {report_file}")
            
    def create_rollback_script(self):
        """Create a rollback script"""
        if self.dry_run:
            return
            
        rollback_script = f"""#!/bin/bash
# Rollback script for migration {self.backup_dir.name}

echo "Rolling back migration from {self.backup_dir}"

# Restore all agents from backup
for agent_dir in {self.backup_dir}/*; do
    if [ -d "$agent_dir" ]; then
        agent_name=$(basename "$agent_dir")
        echo "Restoring $agent_name..."
        rm -rf /opt/sutazaiapp/agents/$agent_name
        cp -r $agent_dir /opt/sutazaiapp/agents/
    fi
done

echo "Rollback completed"
"""
        
        script_path = self.backup_dir / "rollback.sh"
        script_path.write_text(rollback_script)
        script_path.chmod(0o755)
        self.log(f"Rollback script created: {script_path}")

def main():
    parser = argparse.ArgumentParser(description="Migrate agents to BaseAgentV2 with Ollama")
    parser.add_argument("--dry-run", action="store_true", help="Simulate migration without changes")
    parser.add_argument("--phase", choices=["canary", "development", "balanced", "complex", "production", "complete"],
                       help="Migration phase to run")
    parser.add_argument("--validate-only", action="store_true", help="Only run validation")
    
    args = parser.parse_args()
    
    if args.validate_only:
        print("Running migration validation...")
        report = validate_migration()
        print(json.dumps(report, indent=2))
        return
    
    migrator = AgentMigrator(dry_run=args.dry_run)
    
    # Show phase info
    if args.phase:
        agents = migrator.get_agents_for_phase(args.phase)
        print(f"\nPhase '{args.phase}' includes {len(agents)} agents:")
        for agent in agents[:10]:  # Show first 10
            print(f"  - {agent}")
        if len(agents) > 10:
            print(f"  ... and {len(agents) - 10} more")
        print()
        
        response = input("Continue with migration? [y/N] ")
        if response.lower() != 'y':
            print("Migration cancelled")
            return
            
    migrator.run_migration(args.phase)
    migrator.create_rollback_script()

if __name__ == "__main__":
    main()