#!/usr/bin/env python3
"""
Fix Agent Configuration System
Generates missing configuration files and consolidates registries
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

class AgentConfigurationFixer:
    def __init__(self):
        self.base_dir = Path("/opt/sutazaiapp")
        self.agents_dir = self.base_dir / "agents"
        self.configs_dir = self.agents_dir / "configs"
        self.registry_path = self.agents_dir / "agent_registry.json"
        self.backup_dir = self.base_dir / "backups" / f"agent_configs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Statistics
        self.stats = {
            "configs_created": 0,
            "configs_updated": 0,
            "registry_fixed": False,
            "errors": []
        }
    
    def load_registry(self) -> Dict[str, Any]:
        """Load the agent registry"""
        try:
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.stats["errors"].append(f"Failed to load registry: {e}")
            return {}
    
    def create_backup(self):
        """Create backup of existing configurations"""
        print(f"Creating backup in {self.backup_dir}")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup registry
        if self.registry_path.exists():
            import shutil
            shutil.copy2(self.registry_path, self.backup_dir / "agent_registry.json")
        
        # Backup existing configs
        if self.configs_dir.exists():
            for config_file in self.configs_dir.glob("*.json"):
                shutil.copy2(config_file, self.backup_dir / config_file.name)
    
    def generate_config_from_agent_data(self, agent_name: str, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a proper configuration file for an agent"""
        
        # Extract capabilities from agent data
        capabilities = agent_data.get("capabilities", [])
        
        # Map capabilities to proper configuration
        config = {
            "id": agent_name,
            "name": agent_data.get("name", agent_name),
            "version": "1.0.0",
            "description": agent_data.get("description", ""),
            "provider": "universal",
            "type": "system",
            "status": "active",
            "capabilities": capabilities,
            "configuration": {
                "enabled": True,
                "priority": 5,
                "timeout": 300,
                "max_retries": 3,
                "rate_limit": {
                    "requests_per_minute": 60,
                    "burst_size": 10
                }
            },
            "resources": {
                "cpu_limit": "1.0",
                "memory_limit": "512Mi",
                "gpu_required": False
            },
            "endpoints": {
                "health": f"/api/v1/agents/{agent_name}/health",
                "execute": f"/api/v1/agents/{agent_name}/execute",
                "status": f"/api/v1/agents/{agent_name}/status"
            },
            "dependencies": [],
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "created_by": "AgentConfigurationFixer",
                "tags": self._extract_tags_from_capabilities(capabilities),
                "category": self._categorize_agent(agent_name, capabilities)
            }
        }
        
        # Add specialized configurations based on capabilities
        if "orchestration" in capabilities:
            config["orchestration"] = {
                "max_parallel_tasks": 10,
                "coordination_protocol": "event-driven",
                "discovery_enabled": True
            }
        
        if "security_analysis" in capabilities:
            config["security"] = {
                "scan_on_startup": True,
                "vulnerability_threshold": "medium",
                "compliance_checks": ["OWASP", "CIS"]
            }
        
        if "monitoring" in capabilities:
            config["monitoring"] = {
                "metrics_enabled": True,
                "logging_level": "INFO",
                "telemetry": True
            }
        
        if "deployment" in capabilities:
            config["deployment"] = {
                "strategy": "rolling",
                "health_check_interval": 30,
                "rollback_on_failure": True
            }
        
        return config
    
    def _extract_tags_from_capabilities(self, capabilities: List[str]) -> List[str]:
        """Extract relevant tags from capabilities"""
        tags = []
        
        tag_mapping = {
            "code_generation": ["development", "automation"],
            "testing": ["quality", "validation"],
            "deployment": ["infrastructure", "devops"],
            "monitoring": ["observability", "metrics"],
            "security_analysis": ["security", "compliance"],
            "orchestration": ["coordination", "workflow"],
            "optimization": ["performance", "efficiency"],
            "documentation": ["knowledge", "docs"],
            "automation": ["automation", "workflow"],
            "integration": ["integration", "api"],
            "analysis": ["analytics", "insights"]
        }
        
        for capability in capabilities:
            if capability in tag_mapping:
                tags.extend(tag_mapping[capability])
        
        return list(set(tags))  # Remove duplicates
    
    def _categorize_agent(self, agent_name: str, capabilities: List[str]) -> str:
        """Categorize agent based on name and capabilities"""
        
        # Name-based categorization
        if "orchestrat" in agent_name.lower():
            return "orchestration"
        elif "security" in agent_name.lower() or "pentest" in agent_name.lower():
            return "security"
        elif "test" in agent_name.lower() or "qa" in agent_name.lower():
            return "testing"
        elif "deploy" in agent_name.lower() or "devops" in agent_name.lower():
            return "infrastructure"
        elif "frontend" in agent_name.lower() or "ui" in agent_name.lower():
            return "frontend"
        elif "backend" in agent_name.lower() or "api" in agent_name.lower():
            return "backend"
        elif "ai" in agent_name.lower() or "ml" in agent_name.lower():
            return "ai-ml"
        elif "data" in agent_name.lower() or "analys" in agent_name.lower():
            return "data"
        elif "monitor" in agent_name.lower() or "observ" in agent_name.lower():
            return "monitoring"
        
        # Capability-based categorization
        if "orchestration" in capabilities:
            return "orchestration"
        elif "security_analysis" in capabilities:
            return "security"
        elif "testing" in capabilities:
            return "testing"
        elif "deployment" in capabilities:
            return "infrastructure"
        elif "monitoring" in capabilities:
            return "monitoring"
        elif "code_generation" in capabilities:
            return "development"
        
        return "utility"
    
    def fix_configurations(self):
        """Main method to fix all agent configurations"""
        print("=" * 80)
        print("AGENT CONFIGURATION FIXER")
        print("=" * 80)
        
        # Create backup
        self.create_backup()
        
        # Load registry
        registry = self.load_registry()
        if not registry:
            print("ERROR: Could not load registry")
            return self.stats
        
        agents = registry.get("agents", {})
        print(f"Found {len(agents)} agents in registry")
        
        # Ensure configs directory exists
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each agent
        for agent_name, agent_data in agents.items():
            config_path = agent_data.get("config_path", "")
            
            if not config_path:
                print(f"⚠️  {agent_name}: No config_path specified")
                continue
            
            # Extract filename from config_path
            config_filename = Path(config_path).name
            full_config_path = self.configs_dir / config_filename
            
            # Check if config exists
            if full_config_path.exists():
                print(f"✓ {agent_name}: Config exists at {config_filename}")
                continue
            
            # Generate configuration
            print(f"⚡ {agent_name}: Generating config {config_filename}")
            config = self.generate_config_from_agent_data(agent_name, agent_data)
            
            # Write configuration file
            try:
                with open(full_config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                self.stats["configs_created"] += 1
                print(f"✅ {agent_name}: Created {config_filename}")
            except Exception as e:
                self.stats["errors"].append(f"Failed to create config for {agent_name}: {e}")
                print(f"❌ {agent_name}: Failed to create config: {e}")
        
        # Update registry to ensure consistency
        self.update_registry(registry)
        
        # Print summary
        self.print_summary()
        
        return self.stats
    
    def update_registry(self, registry: Dict[str, Any]):
        """Update registry to ensure all paths are correct"""
        print("\nUpdating registry for consistency...")
        
        updated = False
        agents = registry.get("agents", {})
        
        for agent_name, agent_data in agents.items():
            config_path = agent_data.get("config_path", "")
            if config_path:
                # Ensure path uses forward slashes and correct format
                normalized_path = config_path.replace("\\", "/")
                if not normalized_path.startswith("configs/"):
                    normalized_path = f"configs/{Path(normalized_path).name}"
                
                if normalized_path != config_path:
                    agent_data["config_path"] = normalized_path
                    updated = True
        
        if updated:
            # Save updated registry
            try:
                with open(self.registry_path, 'w') as f:
                    json.dump(registry, f, indent=2)
                self.stats["registry_fixed"] = True
                print("✅ Registry updated successfully")
            except Exception as e:
                self.stats["errors"].append(f"Failed to update registry: {e}")
                print(f"❌ Failed to update registry: {e}")
    
    def print_summary(self):
        """Print summary of fixes applied"""
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Configs created: {self.stats['configs_created']}")
        print(f"Configs updated: {self.stats['configs_updated']}")
        print(f"Registry fixed: {self.stats['registry_fixed']}")
        
        if self.stats["errors"]:
            print(f"\n⚠️  Errors encountered: {len(self.stats['errors'])}")
            for error in self.stats["errors"][:5]:  # Show first 5 errors
                print(f"  - {error}")
        
        print("\n✅ Agent configuration fix complete!")
        print(f"Backup saved to: {self.backup_dir}")

def main():
    """Main entry point"""
    fixer = AgentConfigurationFixer()
    stats = fixer.fix_configurations()
    
    # Write stats to file for validation
    stats_file = Path("/opt/sutazaiapp/agent_config_fix_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nStats saved to: {stats_file}")
    
    # Return exit code based on success
    if stats["configs_created"] > 0 and not stats["errors"]:
        return 0  # Success
    elif stats["errors"]:
        return 1  # Partial success with errors
    else:
        return 2  # No changes made

if __name__ == "__main__":
    import sys
    sys.exit(main())