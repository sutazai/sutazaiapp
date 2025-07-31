#!/bin/bash
# Configure all AI agents to use Ollama

echo "=== Configuring AI Agents for Ollama ==="

# Common Ollama configuration
OLLAMA_BASE_URL="http://ollama:11434"
OLLAMA_API_KEY="local"
DEFAULT_MODEL="qwen2.5:3b"
EMBEDDING_MODEL="nomic-embed-text"
CODE_MODEL="qwen2.5-coder:3b"

# Function to configure agent
configure_agent() {
    local agent_name=$1
    local config_type=$2
    local config_path=$3
    
    echo "Configuring $agent_name..."
    
    # Create directory if needed
    local dir=$(dirname "$config_path")
    mkdir -p "$dir"
    
    case $config_type in
        "json")
            cat > "$config_path" << EOF
{
    "llm_provider": "ollama",
    "ollama_base_url": "$OLLAMA_BASE_URL",
    "ollama_api_key": "$OLLAMA_API_KEY",
    "model": "$DEFAULT_MODEL",
    "embedding_model": "$EMBEDDING_MODEL",
    "api_compatibility_mode": "openai",
    "openai_api_base": "$OLLAMA_BASE_URL/v1",
    "openai_api_key": "dummy"
}
EOF
            ;;
            
        "yaml")
            cat > "$config_path" << EOF
llm:
  provider: ollama
  base_url: $OLLAMA_BASE_URL
  api_key: $OLLAMA_API_KEY
  model: $DEFAULT_MODEL
  
embeddings:
  provider: ollama
  model: $EMBEDDING_MODEL
  
api:
  compatibility_mode: openai
  base_url: $OLLAMA_BASE_URL/v1
  key: dummy
EOF
            ;;
            
        "env")
            cat > "$config_path" << EOF
# Ollama Configuration
OLLAMA_BASE_URL=$OLLAMA_BASE_URL
OLLAMA_API_KEY=$OLLAMA_API_KEY
DEFAULT_MODEL=$DEFAULT_MODEL
EMBEDDING_MODEL=$EMBEDDING_MODEL

# OpenAI API Compatibility
OPENAI_API_BASE=$OLLAMA_BASE_URL/v1
OPENAI_API_KEY=dummy
OPENAI_API_HOST=$OLLAMA_BASE_URL/v1

# Model Selection
LLM_MODEL=$DEFAULT_MODEL
CHAT_MODEL=$DEFAULT_MODEL
CODE_MODEL=$CODE_MODEL
EOF
            ;;
            
        "toml")
            cat > "$config_path" << EOF
[model]
kind = "ollama"
model_id = "$DEFAULT_MODEL"
api_endpoint = "$OLLAMA_BASE_URL"

[llm]
provider = "ollama"
base_url = "$OLLAMA_BASE_URL"
model = "$DEFAULT_MODEL"

[embeddings]
provider = "ollama"
model = "$EMBEDDING_MODEL"
EOF
            ;;
            
        "aider")
            cat > "$config_path" << EOF
# Aider configuration for Ollama
openai-api-base: $OLLAMA_BASE_URL/v1
openai-api-key: dummy
model: $CODE_MODEL
edit-format: whole
auto-commits: false
EOF
            ;;
            
        "privategpt")
            cat > "$config_path" << EOF
llm:
  mode: ollama
  
ollama:
  llm_model: $DEFAULT_MODEL
  embedding_model: $EMBEDDING_MODEL
  api_base: $OLLAMA_BASE_URL
  embedding_api_base: $OLLAMA_BASE_URL
  keep_alive: 5m
  request_timeout: 120.0
  
qdrant:
  path: /app/data/qdrant
  
data:
  local_data_folder: /app/data/documents
  
ui:
  enabled: true
  path: /
  default_chat_system_prompt: "You are a helpful AI assistant."
EOF
            ;;
    esac
    
    echo "✓ Configured $agent_name"
}

# Configure each agent
AGENT_DIR="/opt/sutazaiapp/docker"

# JSON-based agents
configure_agent "autogpt" "json" "$AGENT_DIR/autogpt/config.json"
configure_agent "crewai" "json" "$AGENT_DIR/crewai/config.json"
configure_agent "letta" "json" "$AGENT_DIR/letta/config.json"
configure_agent "gpt-engineer" "json" "$AGENT_DIR/gpt-engineer/config.json"
configure_agent "autogen" "json" "$AGENT_DIR/autogen/config.json"
configure_agent "agentzero" "json" "$AGENT_DIR/agentzero/config.json"
configure_agent "browser-use" "json" "$AGENT_DIR/browser-use/config.json"
configure_agent "llamaindex" "json" "$AGENT_DIR/llamaindex/config.json"
configure_agent "finrobot" "json" "$AGENT_DIR/finrobot/config.json"
configure_agent "realtimestt" "json" "$AGENT_DIR/realtimestt/config.json"
configure_agent "documind" "json" "$AGENT_DIR/documind/config.json"

# YAML-based agents
configure_agent "localagi" "yaml" "$AGENT_DIR/localagi/config.yaml"
configure_agent "skyvern" "yaml" "$AGENT_DIR/skyvern/skyvern.yml"

# ENV-based agents
configure_agent "bigagi" "env" "$AGENT_DIR/bigagi/.env.local"
configure_agent "dify" "env" "$AGENT_DIR/dify/.env"
configure_agent "agentgpt" "env" "$AGENT_DIR/agentgpt/.env"
configure_agent "flowise" "env" "$AGENT_DIR/flowise/.env"
configure_agent "shellgpt" "env" "$AGENT_DIR/shellgpt/.env"
configure_agent "pentestgpt" "env" "$AGENT_DIR/pentestgpt/.env"
configure_agent "opendevin" "env" "$AGENT_DIR/opendevin/.env"

# TOML-based agents
configure_agent "tabbyml" "toml" "$AGENT_DIR/tabbyml/config.toml"

# Special configurations
configure_agent "aider" "aider" "$AGENT_DIR/aider/.aider.conf.yml"
configure_agent "privategpt" "privategpt" "$AGENT_DIR/privategpt/settings.yaml"

# Create LiteLLM proxy configuration
echo "Configuring LiteLLM proxy..."
cat > "$AGENT_DIR/litellm/config.yaml" << EOF
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: ollama/$DEFAULT_MODEL
      api_base: $OLLAMA_BASE_URL
  - model_name: gpt-4
    litellm_params:
      model: ollama/qwen2.5:3b
      api_base: $OLLAMA_BASE_URL
  - model_name: text-embedding-ada-002
    litellm_params:
      model: ollama/$EMBEDDING_MODEL
      api_base: $OLLAMA_BASE_URL
  - model_name: code-davinci-002
    litellm_params:
      model: ollama/$CODE_MODEL
      api_base: $OLLAMA_BASE_URL

litellm_settings:
  drop_params: true
  set_verbose: false
  
general_settings:
  master_key: sk-1234
  database_url: postgresql://sutazai:sutazai_password@postgres:5432/sutazai
EOF

echo "✓ Configured LiteLLM proxy"

# Create unified environment file for docker-compose
echo "Creating unified environment configuration..."
cat > "$AGENT_DIR/../.env.agents" << EOF
# Ollama Configuration
OLLAMA_BASE_URL=$OLLAMA_BASE_URL
OLLAMA_API_KEY=$OLLAMA_API_KEY
DEFAULT_MODEL=$DEFAULT_MODEL
EMBEDDING_MODEL=$EMBEDDING_MODEL
CODE_MODEL=$CODE_MODEL

# OpenAI API Compatibility (via Ollama)
OPENAI_API_BASE=$OLLAMA_BASE_URL/v1
OPENAI_API_KEY=dummy
OPENAI_API_HOST=$OLLAMA_BASE_URL/v1

# LiteLLM Proxy
LITELLM_BASE_URL=http://litellm:4000/v1
LITELLM_API_KEY=sk-1234

# Model Configuration
LLM_MODEL=$DEFAULT_MODEL
CHAT_MODEL=$DEFAULT_MODEL
COMPLETION_MODEL=$CODE_MODEL
EMBEDDING_MODEL=$EMBEDDING_MODEL

# Agent Configuration
AGENT_TIMEOUT=300
AGENT_MAX_RETRIES=3
AGENT_LOG_LEVEL=INFO
EOF

echo ""
echo "=== Agent Configuration Complete ==="
echo "All agents configured to use Ollama at: $OLLAMA_BASE_URL"
echo "Default model: $DEFAULT_MODEL"
echo "Code model: $CODE_MODEL"
echo "Embedding model: $EMBEDDING_MODEL"
echo ""
echo "Next steps:"
echo "1. Build agent containers: docker-compose build"
echo "2. Start services: docker-compose up -d"
echo "3. Verify configuration: docker-compose logs"