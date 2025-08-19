#!/usr/bin/env python3
"""
Agent Configuration Consolidation Script
Merges all scattered agent configurations into a unified registry
"""

import json
import yaml
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict

# Configuration paths
BASE_DIR = Path("/opt/sutazaiapp")
AGENT_DIR = BASE_DIR / "agents"
CONFIG_DIR = BASE_DIR / "config"

# Source files
SOURCES = {
    "agent_registry": AGENT_DIR / "agent_registry.json",
    "agent_status": AGENT_DIR / "agent_status.json",
    "collective_intelligence": AGENT_DIR / "collective_intelligence.json",
    "unified_registry": CONFIG_DIR / "agents" / "unified_agent_registry.json",
    "universal_agents": CONFIG_DIR / "universal_agents.json",
    "hygiene_agents": CONFIG_DIR / "hygiene-agents.json",
    "essential_agents": CONFIG_DIR / "agents" / "essential_agents.json"
}

# Target files
TARGET_REGISTRY = CONFIG_DIR / "agents" / "registry.yaml"
TARGET_CAPABILITIES = CONFIG_DIR / "agents" / "capabilities.yaml"
TARGET_RUNTIME = CONFIG_DIR / "agents" / "runtime" / "status.json"

def load_json_safe(filepath: Path) -> Dict:
    """Safely load JSON file, return empty dict if error"""
    try:
        if filepath.exists() and filepath.stat().st_size > 0:
            with open(filepath, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {filepath}: {e}")
    return {}

def merge_agent_data() -> Dict[str, Any]:
    """Merge all agent data from various sources"""
    
    # Load all source data
    registry_data = load_json_safe(SOURCES["agent_registry"])
    status_data = load_json_safe(SOURCES["agent_status"])
    collective_data = load_json_safe(SOURCES["collective_intelligence"])
    unified_data = load_json_safe(SOURCES["unified_registry"])
    universal_data = load_json_safe(SOURCES["universal_agents"])
    hygiene_data = load_json_safe(SOURCES["hygiene_agents"])
    essential_data = load_json_safe(SOURCES["essential_agents"])
    
    # Initialize consolidated registry
    consolidated = {
        "version": "3.0",
        "generated": datetime.now().isoformat(),
        "source": "Consolidated from multiple sources",
        "agents": {},
        "categories": {}
    }
    
    # Process main agent registry
    if "agents" in registry_data:
        for agent_id, agent_info in registry_data["agents"].items():
            consolidated["agents"][agent_id] = {
                "id": agent_id,
                "name": agent_info.get("name", agent_id),
                "type": "standard",
                "description": agent_info.get("description", ""),
                "capabilities": agent_info.get("capabilities", []),
                "config_path": agent_info.get("config_path", ""),
                "deployment": {
                    "method": "docker",  # Default assumption
                    "config": {}
                },
                "runtime": {
                    "status": "unknown"
                },
                "metadata": {}
            }
    
    # Merge runtime status from agent_status.json
    if "active_agents" in status_data:
        for agent_id, status_info in status_data["active_agents"].items():
            if agent_id not in consolidated["agents"]:
                # New agent from status file
                consolidated["agents"][agent_id] = {
                    "id": agent_id,
                    "name": status_info.get("name", agent_id),
                    "type": status_info.get("type", "utility"),
                    "description": status_info.get("description", ""),
                    "capabilities": status_info.get("capabilities", ["automation"]),
                    "deployment": {
                        "method": "docker",
                        "config": {
                            "path": status_info.get("path", "")
                        }
                    }
                }
            
            # Update runtime information
            consolidated["agents"][agent_id]["runtime"] = {
                "status": status_info.get("status", "unknown"),
                "port": status_info.get("port"),
                "endpoint": f"http://localhost:{status_info.get('port')}" if status_info.get("port") else None,
                "process_id": status_info.get("process_id"),
                "start_time": status_info.get("start_time"),
                "last_check": status_info.get("last_check")
            }
    
    # Merge collective intelligence data
    if "agent_registry" in collective_data:
        for agent_id, coll_info in collective_data["agent_registry"].items():
            if agent_id in consolidated["agents"]:
                # Update endpoint and capabilities if not already set
                if "endpoint" in coll_info:
                    consolidated["agents"][agent_id]["runtime"]["endpoint"] = coll_info["endpoint"]
                if "capabilities" in coll_info and not consolidated["agents"][agent_id]["capabilities"]:
                    consolidated["agents"][agent_id]["capabilities"] = coll_info["capabilities"]
    
    # Process unified registry (Claude agents)
    if "agents" in unified_data:
        for agent_id, unified_info in unified_data["agents"].items():
            if agent_id not in consolidated["agents"]:
                consolidated["agents"][agent_id] = {
                    "id": agent_id,
                    "name": unified_info.get("name", agent_id),
                    "type": unified_info.get("type", "claude"),
                    "description": unified_info.get("description", ""),
                    "capabilities": unified_info.get("capabilities", []),
                    "priority": unified_info.get("priority", 5),
                    "deployment": unified_info.get("deployment_info", {
                        "method": "task_tool",
                        "agent_file": ""
                    }),
                    "runtime": {
                        "status": "available"
                    },
                    "metadata": unified_info.get("metadata", {})
                }
    
    # Process hygiene agents
    if "agents" in hygiene_data:
        for hygiene_id, hygiene_info in hygiene_data["agents"].items():
            agent_id = f"hygiene_{hygiene_id}"
            consolidated["agents"][agent_id] = {
                "id": agent_id,
                "name": hygiene_info.get("name", hygiene_id),
                "type": "hygiene",
                "description": hygiene_info.get("description", ""),
                "capabilities": ["rule_enforcement"],
                "enforces_rules": hygiene_info.get("enforces_rules", []),
                "deployment": {
                    "method": "internal",
                    "config": {
                        "resource_limits": hygiene_info.get("resource_limits", {}),
                        "enabled": hygiene_info.get("enabled", True),
                        "dependencies": hygiene_info.get("dependencies", [])
                    }
                },
                "runtime": {
                    "status": "enabled" if hygiene_info.get("enabled", True) else "disabled"
                },
                "metadata": {
                    "category": "hygiene"
                }
            }
            if "hygiene" not in consolidated["categories"]:
                consolidated["categories"]["hygiene"] = []
            consolidated["categories"]["hygiene"].append(agent_id)
    
    # Process essential agents
    if "agents" in essential_data:
        for essential_info in essential_data["agents"]:
            agent_id = essential_info.get("id", "")
            if agent_id and agent_id not in consolidated["agents"]:
                consolidated["agents"][agent_id] = {
                    "id": agent_id,
                    "name": essential_info.get("name", agent_id),
                    "type": essential_info.get("type", "general"),
                    "description": "",
                    "capabilities": essential_info.get("capabilities", []),
                    "deployment": {
                        "method": "ollama",
                        "config": {
                            "model": essential_info.get("model", "tinyllama:latest"),
                            "system_prompt": essential_info.get("system_prompt", "")
                        }
                    },
                    "runtime": {
                        "status": "available"
                    },
                    "metadata": {
                        "category": "essential"
                    }
                }
                if "essential" not in consolidated["categories"]:
                    consolidated["categories"]["essential"] = []
                consolidated["categories"]["essential"].append(agent_id)
    
    # Sort agents by ID for consistency
    sorted_agents = dict(sorted(consolidated["agents"].items()))
    consolidated["agents"] = sorted_agents
    
    # Add statistics
    consolidated["statistics"] = {
        "total_agents": len(consolidated["agents"]),
        "active_agents": sum(1 for a in consolidated["agents"].values() 
                           if a["runtime"].get("status") == "healthy"),
        "agent_types": list(set(a["type"] for a in consolidated["agents"].values())),
        "categories": dict(consolidated["categories"])
    }
    
    return consolidated

def extract_capabilities(consolidated: Dict) -> Dict:
    """Extract unique capabilities into separate file"""
    capabilities = {
        "version": "1.0",
        "generated": datetime.now().isoformat(),
        "capabilities": {}
    }
    
    # Collect all unique capabilities
    all_caps = set()
    for agent in consolidated["agents"].values():
        all_caps.update(agent.get("capabilities", []))
    
    # Create capability definitions
    for cap in sorted(all_caps):
        capabilities["capabilities"][cap] = {
            "name": cap.replace("_", " ").title(),
            "description": f"Capability for {cap.replace('_', ' ')}",
            "agents_with_capability": [
                agent_id for agent_id, agent in consolidated["agents"].items()
                if cap in agent.get("capabilities", [])
            ]
        }
    
    return capabilities

def save_consolidated_data(consolidated: Dict, capabilities: Dict):
    """Save consolidated data to target files"""
    
    # Create backup directory
    backup_dir = BASE_DIR / f"backups/agent_configs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Backup existing files
    for source_name, source_path in SOURCES.items():
        if source_path.exists():
            backup_path = backup_dir / f"{source_name}_{source_path.name}"
            print(f"Backing up {source_path} to {backup_path}")
            with open(source_path, 'r') as src, open(backup_path, 'w') as dst:
                dst.write(src.read())
    
    # Save consolidated registry as YAML
    print(f"Saving consolidated registry to {TARGET_REGISTRY}")
    with open(TARGET_REGISTRY, 'w') as f:
        yaml.dump(consolidated, f, default_flow_style=False, sort_keys=False)
    
    # Save capabilities as YAML
    print(f"Saving capabilities to {TARGET_CAPABILITIES}")
    with open(TARGET_CAPABILITIES, 'w') as f:
        yaml.dump(capabilities, f, default_flow_style=False, sort_keys=False)
    
    # Extract and save runtime status
    runtime_status = {
        "version": "1.0",
        "generated": datetime.now().isoformat(),
        "agents": {}
    }
    
    for agent_id, agent in consolidated["agents"].items():
        if agent["runtime"].get("status") == "healthy":
            runtime_status["agents"][agent_id] = agent["runtime"]
    
    print(f"Saving runtime status to {TARGET_RUNTIME}")
    with open(TARGET_RUNTIME, 'w') as f:
        json.dump(runtime_status, f, indent=2)
    
    print(f"\nConsolidation complete!")
    print(f"- Total agents consolidated: {consolidated['statistics']['total_agents']}")
    print(f"- Active agents: {consolidated['statistics']['active_agents']}")
    print(f"- Agent types: {', '.join(consolidated['statistics']['agent_types'])}")
    print(f"- Backup created at: {backup_dir}")

def main():
    """Main consolidation process"""
    print("Starting agent configuration consolidation...")
    print("=" * 60)
    
    # Merge all data
    consolidated = merge_agent_data()
    
    # Extract capabilities
    capabilities = extract_capabilities(consolidated)
    
    # Save consolidated data
    save_consolidated_data(consolidated, capabilities)
    
    print("\nNext steps:")
    print("1. Review the consolidated registry at:")
    print(f"   {TARGET_REGISTRY}")
    print("2. Test the new configuration with backend code")
    print("3. Once verified, remove old configuration files")
    print("4. Update documentation")

if __name__ == "__main__":
    main()