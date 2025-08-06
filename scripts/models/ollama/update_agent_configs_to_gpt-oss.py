#!/usr/bin/env python3
"""Update all agent Ollama configurations to use tinyllama model."""

import json
import os
import glob

def update_agent_configs():
    """Update all agent _ollama.json files to use tinyllama."""
    configs_dir = "agents/configs"
    updated_count = 0
    
    # Find all _ollama.json files
    ollama_configs = glob.glob(os.path.join(configs_dir, "*_ollama.json"))
    
    print(f"Found {len(ollama_configs)} Ollama configuration files")
    
    for config_path in ollama_configs:
        try:
            # Read the config
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Update the modelfile to use tinyllama
            if 'modelfile' in config:
                # Replace any FROM model line with tinyllama
                lines = config['modelfile'].split('\n')
                for i, line in enumerate(lines):
                    if line.startswith('FROM '):
                        lines[i] = 'FROM tinyllama:latest'
                        break
                config['modelfile'] = '\n'.join(lines)
            
            # Update model preferences for efficiency
            config['model_preference'] = 'ultra_small'
            config['memory_efficient'] = True
            config['max_context_length'] = 2048  # Reduced for tinyllama
            config['temperature'] = 0.7
            config['max_tokens'] = 1024  # Reduced for efficiency
            
            # Update config section if it exists
            if 'config' in config:
                config['config']['num_predict'] = 2048  # Reduced for tinyllama
            
            # Write the updated config
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            updated_count += 1
            print(f"Updated: {os.path.basename(config_path)}")
            
        except Exception as e:
            print(f"Error updating {config_path}: {e}")
    
    print(f"\n✅ Successfully updated {updated_count} agent configurations to use tinyllama")

if __name__ == "__main__":
    update_agent_configs()