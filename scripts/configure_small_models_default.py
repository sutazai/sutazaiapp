#!/usr/bin/env python3
"""
Configure Small Models as Default Across Entire Codebase
Updates all configuration files to use small models by default (qwen2.5:3b, llama3.2:3b)
"""

import os
import sys
import json
import yaml
import re
from pathlib import Path

# Small model configurations
SMALL_MODELS = {
    'primary': 'qwen2.5:3b',
    'secondary': 'llama3.2:3b',
    'coding': 'qwen2.5-coder:3b'
}

MODEL_REPLACEMENTS = {
    # Replace large models with small models
    'qwen2.5:3b': 'qwen2.5:3b',
    'llama3.2:3b': 'llama3.2:3b',
    'llama3.2:3b': 'llama3.2:3b',
    'qwen2.5:3b': 'qwen2.5:3b',
    'qwen2.5-coder:3b': 'qwen2.5-coder:3b',
    'qwen2.5:3b': 'qwen2.5:3b',
    'qwen2.5:3b': 'qwen2.5:3b',
    'qwen2.5:3b': 'qwen2.5:3b',
    'qwen2.5:3b': 'qwen2.5:3b',  # Too small, upgrade to 3b
}

def update_docker_compose_files():
    """Update Docker Compose files to use small models"""
    print("🔧 Updating Docker Compose files...")
    
    compose_files = [
        'docker-compose.yml',
        'docker-compose-agents-tier1.yml',
        'docker-compose.memory-optimized.yml'
    ]
    
    for compose_file in compose_files:
        file_path = Path(f'/opt/sutazaiapp/{compose_file}')
        if not file_path.exists():
            continue
            
        print(f"   Updating {compose_file}...")
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Replace model references
        for old_model, new_model in MODEL_REPLACEMENTS.items():
            content = content.replace(old_model, new_model)
        
        # Add small model environment variables
        small_model_env = f"""
      # Small model defaults for memory efficiency
      DEFAULT_MODEL: "{SMALL_MODELS['primary']}"
      FALLBACK_MODEL: "{SMALL_MODELS['secondary']}"
      MODEL_PREFERENCE: "small"
      SMALL_MODEL_MODE: "true"
"""
        
        # Update specific services
        services_to_update = [
            'aider', 'crewai', 'autogpt', 'gpt-engineer', 'bigagi',
            'langflow', 'flowise', 'dify', 'agentgpt', 'localagi'
        ]
        
        for service in services_to_update:
            # Find service definition and add small model config
            pattern = f'({service}:.*?environment:.*?)(\\n\\s+ports:)'
            replacement = f'\\1{small_model_env}\\2'
            content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        
        with open(file_path, 'w') as f:
            f.write(content)

def update_agent_configurations():
    """Update agent configuration files"""
    print("🤖 Updating agent configurations...")
    
    # Update agent configs
    config_dir = Path('/opt/sutazaiapp/agents/configs')
    if config_dir.exists():
        for config_file in config_dir.glob('*.json'):
            print(f"   Updating {config_file.name}...")
            
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                # Update model references
                if 'model' in config:
                    old_model = config['model']
                    config['model'] = MODEL_REPLACEMENTS.get(old_model, SMALL_MODELS['primary'])
                
                if 'default_model' in config:
                    config['default_model'] = SMALL_MODELS['primary']
                
                if 'fallback_model' in config:
                    config['fallback_model'] = SMALL_MODELS['secondary']
                
                # Add small model specific settings
                config.update({
                    'model_preference': 'small',
                    'memory_efficient': True,
                    'max_context_length': 4096,
                    'temperature': 0.7,
                    'max_tokens': 2048
                })
                
                with open(config_file, 'w') as f:
                    json.dump(config, f, indent=2)
                    
            except Exception as e:
                print(f"   ⚠️  Error updating {config_file}: {e}")

def update_backend_config():
    """Update backend configuration files"""
    print("🔧 Updating backend configurations...")
    
    # Update backend config files
    backend_configs = [
        '/opt/sutazaiapp/backend/config.py',
        '/opt/sutazaiapp/backend/app/core/config.py',
        '/opt/sutazaiapp/config/litellm_config.yaml'
    ]
    
    for config_file in backend_configs:
        if not os.path.exists(config_file):
            continue
            
        print(f"   Updating {config_file}...")
        
        with open(config_file, 'r') as f:
            content = f.read()
        
        # Replace model references in Python files
        if config_file.endswith('.py'):
            for old_model, new_model in MODEL_REPLACEMENTS.items():
                content = content.replace(f'"{old_model}"', f'"{new_model}"')
                content = content.replace(f"'{old_model}'", f"'{new_model}'")
            
            # Add small model constants if not present
            if 'DEFAULT_MODEL' not in content:
                small_model_constants = f'''
# Small model defaults for memory efficiency
DEFAULT_MODEL = "{SMALL_MODELS['primary']}"
FALLBACK_MODEL = "{SMALL_MODELS['secondary']}"
CODING_MODEL = "{SMALL_MODELS['coding']}"
MODEL_PREFERENCE = "small"
MAX_CONTEXT_LENGTH = 4096
'''
                content = small_model_constants + content
        
        # Handle YAML files
        elif config_file.endswith('.yaml') or config_file.endswith('.yml'):
            try:
                config_data = yaml.safe_load(content)
                if config_data:
                    # Update model references in YAML
                    def update_yaml_models(obj):
                        if isinstance(obj, dict):
                            for key, value in obj.items():
                                if isinstance(value, str) and value in MODEL_REPLACEMENTS:
                                    obj[key] = MODEL_REPLACEMENTS[value]
                                elif isinstance(value, (dict, list)):
                                    update_yaml_models(value)
                        elif isinstance(obj, list):
                            for i, item in enumerate(obj):
                                if isinstance(item, str) and item in MODEL_REPLACEMENTS:
                                    obj[i] = MODEL_REPLACEMENTS[item]
                                elif isinstance(item, (dict, list)):
                                    update_yaml_models(item)
                    
                    update_yaml_models(config_data)
                    content = yaml.dump(config_data, default_flow_style=False)
                    
            except Exception as e:
                print(f"   ⚠️  Error parsing YAML {config_file}: {e}")
        
        with open(config_file, 'w') as f:
            f.write(content)

def update_frontend_config():
    """Update frontend configuration"""
    print("🎨 Updating frontend configurations...")
    
    frontend_files = [
        '/opt/sutazaiapp/frontend/config.py',
        '/opt/sutazaiapp/frontend/app.py',
        '/opt/sutazaiapp/frontend/minimal_app.py'
    ]
    
    for config_file in frontend_files:
        if not os.path.exists(config_file):
            continue
            
        print(f"   Updating {config_file}...")
        
        with open(config_file, 'r') as f:
            content = f.read()
        
        # Replace model references
        for old_model, new_model in MODEL_REPLACEMENTS.items():
            content = content.replace(f'"{old_model}"', f'"{new_model}"')
            content = content.replace(f"'{old_model}'", f"'{new_model}'")
        
        # Add small model defaults for Streamlit
        if 'streamlit' in content.lower() and 'DEFAULT_MODEL' not in content:
            small_model_config = f'''
# Small model configuration for memory efficiency
DEFAULT_MODEL = "{SMALL_MODELS['primary']}"
AVAILABLE_MODELS = ["{SMALL_MODELS['primary']}", "{SMALL_MODELS['secondary']}", "{SMALL_MODELS['coding']}"]
MODEL_DESCRIPTIONS = {{
    "{SMALL_MODELS['primary']}": "Primary small model (2GB RAM)",
    "{SMALL_MODELS['secondary']}": "Backup small model (2GB RAM)", 
    "{SMALL_MODELS['coding']}": "Coding-focused small model (2GB RAM)"
}}
'''
            content = small_model_config + content
        
        with open(config_file, 'w') as f:
            f.write(content)

def update_scripts_and_tools():
    """Update scripts and tool configurations"""
    print("📜 Updating scripts and tools...")
    
    script_dirs = [
        '/opt/sutazaiapp/scripts',
        '/opt/sutazaiapp/agents'
    ]
    
    for script_dir in script_dirs:
        if not os.path.exists(script_dir):
            continue
            
        for root, dirs, files in os.walk(script_dir):
            for file in files:
                if file.endswith(('.py', '.sh', '.yaml', '.yml', '.json')):
                    file_path = os.path.join(root, file)
                    
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                        
                        original_content = content
                        
                        # Replace model references
                        for old_model, new_model in MODEL_REPLACEMENTS.items():
                            content = content.replace(old_model, new_model)
                        
                        # Only write if content changed
                        if content != original_content:
                            print(f"   Updated {file_path}")
                            with open(file_path, 'w') as f:
                                f.write(content)
                                
                    except Exception as e:
                        # Skip binary files or files with encoding issues
                        continue

def create_model_verification_script():
    """Create a script to verify small model configuration"""
    print("✅ Creating model verification script...")
    
    verification_script = '''#!/bin/bash
# Verify Small Model Configuration

echo "🔍 Verifying Small Model Configuration"
echo "====================================="

# Check Ollama models
echo "📦 Available Ollama models:"
curl -s http://localhost:11434/api/tags | jq -r '.models[]?.name // "No models available"' 2>/dev/null || echo "Ollama not available"

echo ""
echo "🔄 Currently loaded models:"
curl -s http://localhost:11434/api/ps | jq -r '.models[]?.name // "No models loaded"' 2>/dev/null || echo "Ollama not available"

echo ""
echo "🔧 Hardware optimizer status:"
curl -s http://localhost:8523/ollama-status | jq -r '.small_model_mode // "Unknown"' 2>/dev/null || echo "Hardware optimizer not available"

echo ""
echo "📊 Memory usage:"
free -h | grep "Mem:"

echo ""
echo "✅ Small model verification complete"
'''
    
    with open('/opt/sutazaiapp/scripts/verify_small_models.sh', 'w') as f:
        f.write(verification_script)
    
    os.chmod('/opt/sutazaiapp/scripts/verify_small_models.sh', 0o755)

def update_environment_files():
    """Update environment files with small model defaults"""
    print("🌍 Updating environment files...")
    
    env_files = [
        '/opt/sutazaiapp/.env',
        '/opt/sutazaiapp/config/litellm.env'
    ]
    
    small_model_env_vars = f'''
# Small model defaults for memory efficiency
DEFAULT_MODEL={SMALL_MODELS['primary']}
FALLBACK_MODEL={SMALL_MODELS['secondary']}
CODING_MODEL={SMALL_MODELS['coding']}
MODEL_PREFERENCE=small
SMALL_MODEL_MODE=true
MAX_CONTEXT_LENGTH=4096
OLLAMA_MAX_LOADED_MODELS=1
OLLAMA_KEEP_ALIVE=30s
MEMORY_EFFICIENT_MODE=true
'''
    
    for env_file in env_files:
        if os.path.exists(env_file):
            print(f"   Updating {env_file}...")
            
            with open(env_file, 'r') as f:
                content = f.read()
            
            # Remove existing small model config
            lines = content.split('\n')
            filtered_lines = []
            skip_section = False
            
            for line in lines:
                if '# Small model defaults' in line:
                    skip_section = True
                elif skip_section and line.strip() == '':
                    skip_section = False
                    continue
                elif not skip_section:
                    filtered_lines.append(line)
            
            # Add updated small model config
            content = '\n'.join(filtered_lines) + small_model_env_vars
            
            with open(env_file, 'w') as f:
                f.write(content)

def main():
    """Main function"""
    print("🚀 Configuring Small Models as Default Across SutazAI System")
    print("=" * 70)
    print(f"Primary Model: {SMALL_MODELS['primary']}")
    print(f"Secondary Model: {SMALL_MODELS['secondary']}")
    print(f"Coding Model: {SMALL_MODELS['coding']}")
    print("=" * 70)
    
    try:
        update_docker_compose_files()
        update_agent_configurations()
        update_backend_config()
        update_frontend_config()
        update_scripts_and_tools()
        update_environment_files()
        create_model_verification_script()
        
        print("")
        print("✅ Small model configuration completed successfully!")
        print("")
        print("📋 Summary of changes:")
        print("   • Docker Compose files updated with small model defaults")
        print("   • Agent configurations set to use small models")
        print("   • Backend and frontend configs updated")
        print("   • Scripts and tools configured for small models")
        print("   • Environment variables set for memory efficiency")
        print("   • Verification script created")
        print("")
        print("🔄 Next steps:")
        print("   1. Restart containers: docker-compose restart")
        print("   2. Verify configuration: ./scripts/verify_small_models.sh")
        print("   3. Monitor system: python3 scripts/memory_monitor_dashboard.py")
        print("")
        
    except Exception as e:
        print(f"❌ Error during configuration: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()