#!/usr/bin/env python3
"""
"""
import os
import json
import shutil

def update_agent_configs():
    configs_dir = "agents/configs"
    
    # List all agent config files
    for filename in os.listdir(configs_dir):
        if filename.endswith("_ollama.json"):
            agent_name = filename.replace("_ollama.json", "")
            
            # Check if there's a universal config
            universal_config = os.path.join(configs_dir, f"{agent_name}_universal.json")
            ollama_config = os.path.join(configs_dir, filename)
            
            if os.path.exists(universal_config):
                # Update universal config to use Ollama
                with open(ollama_config, 'r') as f:
                    ollama_data = json.load(f)
                
                with open(universal_config, 'w') as f:
                    json.dump(ollama_data, f, indent=2)
                
                print(f"✅ Updated {agent_name} to use Ollama config")

    configs_dir = "agents/configs"
    removed_count = 0
    
    for filename in os.listdir(configs_dir):
            file_path = os.path.join(configs_dir, filename)
            os.remove(file_path)
            removed_count += 1
            print(f"🗑️  Removed {filename}")
    

if __name__ == "__main__":
    print("Updating agent configurations to use native Ollama...")
    update_agent_configs()
    print("\n✅ All agents now configured for native Ollama!")
