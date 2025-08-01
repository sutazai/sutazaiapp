#!/bin/bash

# SutazAI Autonomous Agents Demo Launcher
# ======================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🤖 SutazAI Autonomous Multi-Agent Demonstration"
echo "=============================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running in project directory
if [[ ! -f "$PROJECT_ROOT/docker-compose.yml" ]]; then
    print_error "Please run this script from the SutazAI project directory"
    exit 1
fi

# Function to check if a service is running
check_service() {
    local service_name=$1
    local port=$2
    
    if docker ps --format "table {{.Names}}" | grep -q "sutazai-$service_name"; then
        if curl -s "http://localhost:$port" > /dev/null 2>&1 || \
           curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

# Check required services
print_status "Checking required services..."

REQUIRED_SERVICES=("redis:6379" "ollama:11434")
MISSING_SERVICES=()

for service_port in "${REQUIRED_SERVICES[@]}"; do
    service=$(echo $service_port | cut -d: -f1)
    port=$(echo $service_port | cut -d: -f2)
    
    if check_service "$service" "$port"; then
        print_success "$service is running ✓"
    else
        print_warning "$service is not running or not accessible"
        MISSING_SERVICES+=("$service")
    fi
done

# Start missing services if needed
if [[ ${#MISSING_SERVICES[@]} -gt 0 ]]; then
    print_status "Starting required services..."
    
    # Start core services
    docker-compose up -d redis ollama
    
    print_status "Waiting for services to be ready..."
    sleep 10
    
    # Check Ollama models
    print_status "Checking Ollama models..."
    if ! docker exec sutazai-ollama ollama list | grep -q "deepseek-r1:8b"; then
        print_status "Pulling deepseek-r1:8b model for demo..."
        docker exec sutazai-ollama ollama pull deepseek-r1:8b || {
            print_warning "Failed to pull deepseek-r1:8b, trying alternative model..."
            docker exec sutazai-ollama ollama pull llama3.2:3b || {
                print_error "Failed to pull required models"
                exit 1
            }
        }
    fi
fi

# Install Python dependencies if needed
print_status "Checking Python dependencies..."

# Create virtual environment if it doesn't exist
if [[ ! -d "$PROJECT_ROOT/venv" ]]; then
    print_status "Creating Python virtual environment..."
    python3 -m venv "$PROJECT_ROOT/venv"
fi

# Activate virtual environment
source "$PROJECT_ROOT/venv/bin/activate"

# Install required packages
pip install -q redis httpx rich > /dev/null 2>&1 || {
    print_error "Failed to install Python dependencies"
    exit 1
}

print_success "All dependencies ready!"

echo ""
echo -e "${CYAN}🚀 Starting Autonomous Agents Demonstration${NC}"
echo "=============================================="
echo ""

# Parse command line arguments
DEMO_TASK="analyze"
NUM_AGENTS=3
CODE_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            DEMO_TASK="$2"
            shift 2
            ;;
        --agents)
            NUM_AGENTS="$2"
            shift 2
            ;;
        --code-file)
            CODE_FILE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --task TASK        Demo task to run (analyze|monitor) [default: analyze]"
            echo "  --agents N         Number of agents to create [default: 3]"
            echo "  --code-file FILE   Path to code file to analyze"
            echo "  --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Run basic demo"
            echo "  $0 --task monitor                     # Monitor agents for 60 seconds"
            echo "  $0 --agents 5                         # Use 5 agents"
            echo "  $0 --code-file backend/app/main.py    # Analyze specific file"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build demo command
DEMO_CMD="python3 $SCRIPT_DIR/autonomous_agents_demo.py --task $DEMO_TASK --agents $NUM_AGENTS"

if [[ -n "$CODE_FILE" ]]; then
    if [[ -f "$CODE_FILE" ]]; then
        DEMO_CMD="$DEMO_CMD --code-file $CODE_FILE"
    else
        print_error "Code file not found: $CODE_FILE"
        exit 1
    fi
fi

# Show demo information
echo -e "${PURPLE}Demo Configuration:${NC}"
echo "  Task: $DEMO_TASK"
echo "  Agents: $NUM_AGENTS"
if [[ -n "$CODE_FILE" ]]; then
    echo "  Code File: $CODE_FILE"
fi
echo ""

# Run the demo
print_status "Launching autonomous agents demo..."
echo ""

# Execute the demo
exec $DEMO_CMD