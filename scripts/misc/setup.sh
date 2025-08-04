#!/bin/bash

# SutazAI MCP Server Setup Script
# Installs and configures the MCP server for integration with Claude Desktop

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="/opt/sutazaiapp"
MCP_SERVER_DIR="${PROJECT_ROOT}/mcp_server"
LOG_FILE="${PROJECT_ROOT}/logs/mcp_setup.log"

# Logging function
log() {
    echo -e "${2:-$BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    log "ERROR: $1" "$RED"
    exit 1
}

success() {
    log "SUCCESS: $1" "$GREEN"
}

warning() {
    log "WARNING: $1" "$YELLOW"
}

# Check if running from correct directory
check_environment() {
    if [[ ! -d "$PROJECT_ROOT" ]]; then
        error "SutazAI project directory not found at $PROJECT_ROOT"
    fi
    
    if [[ ! -f "$PROJECT_ROOT/docker-compose.yml" ]]; then
        error "Docker Compose file not found. Please run from SutazAI project root."
    fi
    
    log "Environment check passed"
}

# Install Node.js dependencies
install_dependencies() {
    log "Installing MCP server dependencies..."
    
    cd "$MCP_SERVER_DIR"
    
    if ! command -v node &> /dev/null; then
        warning "Node.js not found. Installing Node.js 20..."
        curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
        sudo apt-get install -y nodejs
    fi
    
    # Install dependencies
    npm ci --silent
    success "Dependencies installed successfully"
}

# Initialize database schema
init_database() {
    log "Initializing MCP server database schema..."
    
    # Wait for PostgreSQL to be ready
    log "Waiting for PostgreSQL to be ready..."
    timeout=60
    while ! docker exec sutazai-postgres pg_isready -U sutazai > /dev/null 2>&1; do
        if [[ $timeout -le 0 ]]; then
            error "PostgreSQL did not become ready in time"
        fi
        sleep 2
        ((timeout-=2))
    done
    
    # Run database schema
    docker exec -i sutazai-postgres psql -U sutazai -d sutazai < "${MCP_SERVER_DIR}/database/schema.sql"
    success "Database schema initialized"
}

# Create configuration files
setup_configuration() {
    log "Setting up MCP server configuration..."
    
    # Create environment file if it doesn't exist
    if [[ ! -f "${MCP_SERVER_DIR}/.env" ]]; then
        log "Creating .env file from template..."
        cp "${MCP_SERVER_DIR}/config.example.env" "${MCP_SERVER_DIR}/.env"
        
        # Get database credentials from main project
        if [[ -f "${PROJECT_ROOT}/.env" ]]; then
            POSTGRES_PASSWORD=$(grep "POSTGRES_PASSWORD" "${PROJECT_ROOT}/.env" | cut -d'=' -f2)
            REDIS_PASSWORD=$(grep "REDIS_PASSWORD" "${PROJECT_ROOT}/.env" | cut -d'=' -f2)
            
            sed -i "s/sutazai_password/${POSTGRES_PASSWORD}/g" "${MCP_SERVER_DIR}/.env"
            sed -i "s/redis_password/${REDIS_PASSWORD}/g" "${MCP_SERVER_DIR}/.env"
        fi
        
        success "Configuration files created"
    else
        log "Configuration file already exists"
    fi
}

# Build and start MCP server
build_and_start() {
    log "Building and starting MCP server..."
    
    cd "$PROJECT_ROOT"
    
    # Build MCP server container
    docker-compose build mcp-server
    
    # Start MCP server
    docker-compose up -d mcp-server
    
    # Wait for server to be ready
    log "Waiting for MCP server to be ready..."
    timeout=60
    while ! docker logs sutazai-mcp-server 2>&1 | grep -q "started successfully"; do
        if [[ $timeout -le 0 ]]; then
            error "MCP server did not start in time"
        fi
        sleep 2
        ((timeout-=2))
    done
    
    success "MCP server started successfully"
}

# Test MCP server functionality
test_server() {
    log "Testing MCP server functionality..."
    
    # Test database connections
    if docker exec sutazai-mcp-server node -e "
        const pg = require('pg');
        const client = new pg.Client({connectionString: process.env.DATABASE_URL});
        client.connect().then(() => {
            console.log('Database connection successful');
            client.end();
        }).catch(console.error);
    " > /dev/null 2>&1; then
        success "Database connection test passed"
    else
        warning "Database connection test failed"
    fi
    
    # Test Redis connection
    if docker exec sutazai-mcp-server node -e "
        const redis = require('redis');
        const client = redis.createClient({url: process.env.REDIS_URL});
        client.connect().then(() => {
            console.log('Redis connection successful');
            client.disconnect();
        }).catch(console.error);
    " > /dev/null 2>&1; then
        success "Redis connection test passed"
    else
        warning "Redis connection test failed"
    fi
    
    log "Server functionality tests completed"
}

# Configure Claude Desktop
configure_claude_desktop() {
    log "Configuring Claude Desktop integration..."
    
    # Detect operating system
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        CLAUDE_CONFIG_DIR="$HOME/.config/claude"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        CLAUDE_CONFIG_DIR="$HOME/Library/Application Support/claude"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        CLAUDE_CONFIG_DIR="$APPDATA/claude"
    else
        warning "Unknown operating system. Please configure Claude Desktop manually."
        return
    fi
    
    # Create Claude config directory
    mkdir -p "$CLAUDE_CONFIG_DIR"
    
    # Create Claude Desktop configuration
    cat > "$CLAUDE_CONFIG_DIR/claude_desktop_config.json" << EOF
{
  "mcpServers": {
    "sutazai-mcp-server": {
      "command": "node",
      "args": ["${MCP_SERVER_DIR}/index.js"],
      "env": {
        "DATABASE_URL": "postgresql://sutazai:\${POSTGRES_PASSWORD}@localhost:5432/sutazai",
        "REDIS_URL": "redis://:\${REDIS_PASSWORD}@localhost:6379/0",
        "BACKEND_API_URL": "http://localhost:8000",
        "OLLAMA_URL": "http://localhost:11434",
        "CHROMADB_URL": "http://localhost:8000",
        "QDRANT_URL": "http://localhost:6333"
      }
    }
  }
}
EOF

    success "Claude Desktop configuration created at: $CLAUDE_CONFIG_DIR/claude_desktop_config.json"
    
    log ""
    log "IMPORTANT: To complete Claude Desktop setup:"
    log "1. Restart Claude Desktop completely"
    log "2. Go to Settings → Developer → Edit Config"
    log "3. Verify the configuration is loaded"
    log "4. Look for 'sutazai-mcp-server' in your tools/resources"
    log ""
}

# Generate MCP Inspector configuration
setup_mcp_inspector() {
    log "Setting up MCP Inspector for development..."
    
    cat > "${MCP_SERVER_DIR}/mcp_inspector_config.json" << EOF
{
  "mcpServers": {
    "sutazai-mcp-server": {
      "command": "node",
      "args": ["${MCP_SERVER_DIR}/index.js"],
      "env": {
        "DATABASE_URL": "postgresql://sutazai:sutazai_password@localhost:5432/sutazai",
        "REDIS_URL": "redis://:redis_password@localhost:6379/0",
        "BACKEND_API_URL": "http://localhost:8000",
        "OLLAMA_URL": "http://localhost:11434",
        "LOG_LEVEL": "DEBUG"
      }
    }
  }
}
EOF

    success "MCP Inspector configuration created"
    log "You can test the MCP server with MCP Inspector at: http://localhost:6274"
}

# Main installation function
main() {
    log "Starting SutazAI MCP Server Setup"
    log "================================="
    
    # Create logs directory
    mkdir -p "${PROJECT_ROOT}/logs"
    mkdir -p "${PROJECT_ROOT}/data/mcp_server"
    
    check_environment
    install_dependencies
    setup_configuration
    
    # Start required services if not running
    if ! docker ps | grep -q sutazai-postgres; then
        log "Starting required services..."
        cd "$PROJECT_ROOT"
        docker-compose up -d postgres redis
        sleep 10
    fi
    
    init_database
    build_and_start
    test_server
    configure_claude_desktop
    setup_mcp_inspector
    
    log ""
    log "================================="
    success "SutazAI MCP Server setup completed successfully!"
    log "================================="
    log ""
    log "Next steps:"
    log "1. Restart Claude Desktop to load the new MCP server"
    log "2. Test the server with MCP Inspector: http://localhost:6274"
    log "3. Check logs: docker logs sutazai-mcp-server"
    log "4. View resources and tools in Claude Desktop"
    log ""
    log "Available MCP Resources:"
    log "  - sutazai://agents/list - List all AI agents"
    log "  - sutazai://models/available - Available AI models" 
    log "  - sutazai://agents/tasks - Agent task history"
    log "  - sutazai://system/metrics - System metrics"
    log "  - sutazai://knowledge/embeddings - Knowledge base"
    log ""
    log "Available MCP Tools:"
    log "  - deploy_agent - Deploy new AI agents"
    log "  - execute_agent_task - Run agent tasks"
    log "  - manage_model - Manage Ollama models"
    log "  - query_knowledge_base - Search knowledge base"
    log "  - monitor_system - Get system metrics"
    log "  - orchestrate_multi_agent - Coordinate multiple agents"
    log ""
    log "Configuration files:"
    log "  - MCP Server: ${MCP_SERVER_DIR}/.env"
    log "  - Claude Desktop: ${CLAUDE_CONFIG_DIR}/claude_desktop_config.json"
    log "  - Logs: ${LOG_FILE}"
    log ""
}

# Handle script termination
trap 'log "Setup interrupted"; exit 1' INT TERM

# Run main function
main "$@" 