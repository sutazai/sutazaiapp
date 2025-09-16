#!/usr/bin/env python3
import json
import os
import shutil
from pathlib import Path

def compress_agent(agent_data):
    """Compress agent to minimal format"""
    # Extract only essential fields
    compressed = {
        "name": agent_data.get("name", ""),
        "capabilities": agent_data.get("capabilities", [])[:3],  # Only top 3
    }
    
    # Create ultra-short description
    desc = agent_data.get("description", "")
    if desc:
        # Extract first meaningful line
        lines = desc.split('\\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith("Use this") and len(line) > 10:
                compressed["description"] = line[:80]
                break
    
    # Only include config if exists
    if "config_path" in agent_data:
        compressed["config_path"] = agent_data["config_path"]
    
    return compressed

def main():
    registry_path = Path('/root/.claude/agents/agent_registry.json')
    backup_path = registry_path.with_suffix('.json.backup')
    optimized_path = registry_path.with_suffix('.json.optimized')
    
    if not registry_path.exists():
        print(f"Registry not found at {registry_path}")
        return
    
    # Create backup
    if not backup_path.exists():
        shutil.copy2(registry_path, backup_path)
        print(f"Backup created: {backup_path}")
    
    # Load registry
    with open(registry_path, 'r') as f:
        registry = json.load(f)
    
    original_size = len(json.dumps(registry))
    
    # Compress agents
    optimized = {"agents": {}}
    for name, agent in registry.get("agents", {}).items():
        optimized["agents"][name] = compress_agent(agent)
    
    # Save optimized version
    with open(optimized_path, 'w') as f:
        json.dump(optimized, f, separators=(',', ':'))  # Minimal JSON
    
    optimized_size = len(json.dumps(optimized))
    
    print(f"Original size: {original_size:,} bytes (~{original_size//4:,} tokens)")
    print(f"Optimized size: {optimized_size:,} bytes (~{optimized_size//4:,} tokens)")
    print(f"Reduction: {(1 - optimized_size/original_size)*100:.1f}%")
    print(f"Token savings: ~{(original_size - optimized_size)//4:,} tokens")
    
    # Apply optimization
    shutil.copy2(optimized_path, registry_path)
    print(f"Optimization applied to {registry_path}")

if __name__ == "__main__":
    main()
