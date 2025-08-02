#!/usr/bin/env python3
"""
Verify and report on model configuration for SutazAI system.
"""

import os
import json
import yaml
import subprocess
import requests
from pathlib import Path

def check_ollama_models():
    """Check what models are available in Ollama."""
    try:
        result = subprocess.run(
            ["docker", "exec", "sutazai-ollama", "ollama", "list"],
            capture_output=True, text=True, check=True
        )
        print("📦 Ollama Models Available:")
        print(result.stdout)
        
        models = []
        for line in result.stdout.strip().split('\n')[1:]:  # Skip header
            if line:
                model_name = line.split()[0]
                models.append(model_name)
        return models
    except Exception as e:
        print(f"❌ Error checking Ollama models: {e}")
        return []

def check_agent_models():
    """Check what models agents are configured to use."""
    agent_dir = Path("/opt/sutazaiapp/.claude/agents")
    model_usage = {}
    
    for agent_file in agent_dir.glob("*.md"):
        if agent_file.name.endswith("_backup") or agent_file.name.endswith(".backup"):
            continue
            
        with open(agent_file, 'r') as f:
            content = f.read()
            
        # Extract model from YAML front matter
        if content.startswith("---"):
            yaml_end = content.find("---", 3)
            if yaml_end != -1:
                yaml_content = content[3:yaml_end]
                try:
                    agent_config = yaml.safe_load(yaml_content)
                    model = agent_config.get('model', 'not specified')
                    agent_name = agent_file.stem
                    
                    if model not in model_usage:
                        model_usage[model] = []
                    model_usage[model].append(agent_name)
                except:
                    pass
    
    print("\n🤖 Agent Model Configuration:")
    for model, agents in model_usage.items():
        print(f"  {model}: {len(agents)} agents")
        if model != "tinyllama:latest":
            print(f"    ⚠️  Non-default model used by: {', '.join(agents[:3])}...")
    
    return model_usage

def check_env_config():
    """Check environment configuration."""
    print("\n⚙️  Environment Configuration:")
    
    # Check .env file
    env_path = Path("/opt/sutazaiapp/.env")
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                if 'MODEL' in line and not line.startswith('#'):
                    print(f"  {line.strip()}")
    
    # Check ollama optimization config
    config_path = Path("/opt/sutazaiapp/config/ollama_optimization.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        print("\n📊 Ollama Optimization Settings:")
        memory_limits = config.get('ollama_config', {}).get('memory_limits', {})
        model_mgmt = config.get('ollama_config', {}).get('model_management', {})
        
        print(f"  Max memory per model: {memory_limits.get('max_model_memory', 'N/A')}")
        print(f"  Max loaded models: {model_mgmt.get('max_loaded_models', 'N/A')}")
        print(f"  Unload after idle: {model_mgmt.get('unload_after_idle', 'N/A')}")
        print(f"  Lazy loading: {model_mgmt.get('lazy_loading', 'N/A')}")

def check_api_model_usage():
    """Check what model the API is actually using."""
    print("\n🌐 API Model Usage:")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"  API Status: {health.get('status', 'unknown')}")
            print(f"  Models loaded: {health.get('services', {}).get('models', {}).get('loaded_count', 0)}")
    except Exception as e:
        print(f"  ⚠️  Could not check API: {e}")

def main():
    """Run all checks and provide recommendations."""
    print("🔍 SutazAI Model Configuration Verification\n")
    
    # Check available models
    available_models = check_ollama_models()
    
    # Check agent configurations
    model_usage = check_agent_models()
    
    # Check environment
    check_env_config()
    
    # Check API
    check_api_model_usage()
    
    # Recommendations
    print("\n📋 Summary & Recommendations:")
    
    if 'tinyllama:latest' in available_models:
        print("  ✅ Default model (tinyllama) is available")
    else:
        print("  ❌ Default model (tinyllama) is NOT available - please pull it")
    
    if all(model == 'tinyllama:latest' for model in model_usage.keys()):
        print("  ✅ All agents are using the default model")
    else:
        print("  ⚠️  Some agents are configured for non-default models")
    
    if 'qwen2.5:3b' in available_models:
        print("  ✅ On-demand model (qwen2.5:3b) is available for complex tasks")
    
    print("\n✨ Model Strategy:")
    print("  • tinyllama:latest - Always active (default for all agents)")
    print("  • qwen2.5:3b - On standby (loaded on-demand for complex tasks)")
    print("  • Other models - Pull only when specifically needed")

if __name__ == "__main__":
    main()