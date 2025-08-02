#!/bin/bash
# SutazAI Installation Verification Script
# Verifies that all components of the SutazAI application are installed and running correctly

# Navigate to the project root directory
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log file
VERIFY_LOG="${PROJECT_ROOT}/logs/verify.log"
mkdir -p "$(dirname "$VERIFY_LOG")"

# Logging function
log() {
    local message="$1"
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo -e "$message"
    echo "[$timestamp] $message" >> "$VERIFY_LOG"
}

log "${BLUE}Starting SutazAI installation verification...${NC}"

# Function to check if a process is running
check_process() {
    local process_name="$1"
    local process_check="$2"
    
    if pgrep -f "$process_check" > /dev/null; then
        log "${GREEN}✓ $process_name is running${NC}"
        return 0
    else
        log "${RED}✗ $process_name is not running${NC}"
        return 1
    fi
}

# Function to check if a port is in use (service is listening)
check_port() {
    local service_name="$1"
    local port="$2"
    
    if command -v nc &> /dev/null; then
        if nc -z localhost $port &> /dev/null; then
            log "${GREEN}✓ $service_name is listening on port $port${NC}"
            return 0
        else
            log "${RED}✗ $service_name is not listening on port $port${NC}"
            return 1
        fi
    elif command -v lsof &> /dev/null; then
        if lsof -i:$port &> /dev/null; then
            log "${GREEN}✓ $service_name is listening on port $port${NC}"
            return 0
        else
            log "${RED}✗ $service_name is not listening on port $port${NC}"
            return 1
        fi
    else
        log "${YELLOW}⚠ Cannot check if $service_name is listening on port $port (nc and lsof not available)${NC}"
        return 2
    fi
}

# Function to check if a file exists
check_file() {
    local file_name="$1"
    local file_path="$2"
    
    if [ -f "$file_path" ]; then
        log "${GREEN}✓ $file_name exists${NC}"
        return 0
    else
        log "${RED}✗ $file_name does not exist${NC}"
        return 1
    fi
}

# Function to check if a directory exists
check_dir() {
    local dir_name="$1"
    local dir_path="$2"
    
    if [ -d "$dir_path" ]; then
        log "${GREEN}✓ $dir_name directory exists${NC}"
        return 0
    else
        log "${RED}✗ $dir_name directory does not exist${NC}"
        return 1
    fi
}

# Function to check API endpoint
check_api() {
    local api_name="$1"
    local api_url="$2"
    
    if command -v curl &> /dev/null; then
        if curl -s -f "$api_url" &> /dev/null; then
            log "${GREEN}✓ $api_name API is accessible${NC}"
            return 0
        else
            log "${RED}✗ $api_name API is not accessible${NC}"
            return 1
        fi
    else
        log "${YELLOW}⚠ Cannot check if $api_name API is accessible (curl not available)${NC}"
        return 2
    fi
}

# Title for verification sections
section_title() {
    log "\n${BLUE}=== $1 ===${NC}"
}

# Check required directories
section_title "Directory Structure"
DIRECTORIES=(
    "ai_agents/superagi:SuperAGI agents"
    "backend:Backend application"
    "backend/services:Backend services"
    "model_management:Model management"
    "web_ui:Web UI"
    "scripts:Scripts"
    "docs:Documentation"
    "logs:Logs"
    "monitoring:Monitoring"
)

DIR_ERRORS=0
for dir_entry in "${DIRECTORIES[@]}"; do
    dir_path="${dir_entry%%:*}"
    dir_name="${dir_entry#*:}"
    
    if ! check_dir "$dir_name" "${PROJECT_ROOT}/${dir_path}"; then
        DIR_ERRORS=$((DIR_ERRORS+1))
    fi
done

if [ $DIR_ERRORS -eq 0 ]; then
    log "${GREEN}All required directories are present.${NC}"
else
    log "${RED}$DIR_ERRORS directory errors found.${NC}"
fi

# Check required files
section_title "Configuration Files"
FILES=(
    ".env:Environment configuration"
    "requirements.txt:Python dependencies"
    "ai_agents/superagi/config.toml:SuperAGI configuration"
    "monitoring/prometheus.yml:Prometheus configuration"
)

FILE_ERRORS=0
for file_entry in "${FILES[@]}"; do
    file_path="${file_entry%%:*}"
    file_name="${file_entry#*:}"
    
    if ! check_file "$file_name" "${PROJECT_ROOT}/${file_path}"; then
        FILE_ERRORS=$((FILE_ERRORS+1))
    fi
done

if [ $FILE_ERRORS -eq 0 ]; then
    log "${GREEN}All required configuration files are present.${NC}"
else
    log "${RED}$FILE_ERRORS configuration file errors found.${NC}"
fi

# Check Python environment
section_title "Python Environment"
if [ -d "${PROJECT_ROOT}/venv" ]; then
    log "${GREEN}✓ Virtual environment exists${NC}"
    
    # Source the virtual environment to check installed packages
    source "${PROJECT_ROOT}/venv/bin/activate"
    
    # Check key packages
    PACKAGES=(
        "fastapi:FastAPI framework"
        "uvicorn:ASGI server"
        "pydantic:Data validation"
        "numpy:Numerical operations"
        "torch:PyTorch for ML"
        "transformers:Hugging Face Transformers"
        "langchain:LangChain for LLM applications"
    )
    
    PACKAGE_ERRORS=0
    for pkg_entry in "${PACKAGES[@]}"; do
        pkg_name="${pkg_entry%%:*}"
        pkg_desc="${pkg_entry#*:}"
        
        if python -c "import $pkg_name" &> /dev/null; then
            if [ "$pkg_name" = "torch" ]; then
                pkg_name="torch"
            fi
            if [ "$pkg_name" = "transformers" ]; then
                pkg_name="transformers"
            fi
            VERSION=$(python -c "import $pkg_name; print($pkg_name.__version__)" 2>/dev/null)
            log "${GREEN}✓ $pkg_desc ($pkg_name $VERSION) is installed${NC}"
        else
            log "${RED}✗ $pkg_desc ($pkg_name) is not installed${NC}"
            PACKAGE_ERRORS=$((PACKAGE_ERRORS+1))
        fi
    done
    
    if [ $PACKAGE_ERRORS -eq 0 ]; then
        log "${GREEN}All required Python packages are installed.${NC}"
    else
        log "${RED}$PACKAGE_ERRORS Python package errors found.${NC}"
    fi
    
    # Deactivate virtual environment
    deactivate
else
    log "${RED}✗ Virtual environment does not exist${NC}"
fi

# Check services
section_title "Services Status"
SERVICES=(
    "8000:Backend API"
    "8001:SuperAGI"
    "9090:Prometheus"
    "3000:Grafana"
    "9100:Node Exporter"
)

SERVICE_ERRORS=0
for service in "${SERVICES[@]}"; do
    port="${service%%:*}"
    name="${service#*:}"
    
    if ! check_port "$name" "$port"; then
        SERVICE_ERRORS=$((SERVICE_ERRORS+1))
    fi
done

if [ $SERVICE_ERRORS -eq 0 ]; then
    log "${GREEN}All services are running.${NC}"
else
    log "${RED}$SERVICE_ERRORS service errors found.${NC}"
fi

# Check API endpoints
section_title "API Health"
ENDPOINTS=(
    "http://localhost:8000/health:Backend API health"
    "http://localhost:8000/api/docs:Document processing API"
    "http://localhost:8000/api/code:Code generation API"
    "http://localhost:8000/api/diagrams:Diagram parsing API"
)

API_ERRORS=0
for endpoint in "${ENDPOINTS[@]}"; do
    url="${endpoint%%:*}"
    name="${endpoint#*:}"
    
    if ! check_api "$name" "$url"; then
        API_ERRORS=$((API_ERRORS+1))
    fi
done

if [ $API_ERRORS -eq 0 ]; then
    log "${GREEN}All API endpoints are accessible.${NC}"
else
    log "${RED}$API_ERRORS API endpoint errors found.${NC}"
fi

# Check model availability
section_title "Model Files"
MODELS=(
    "model_management/GPT4All:GPT4All model directory"
    "model_management/DeepSeek-Coder-33B:DeepSeek Coder model directory"
)

MODEL_ERRORS=0
for model_entry in "${MODELS[@]}"; do
    model_path="${model_entry%%:*}"
    model_name="${model_entry#*:}"
    
    if ! check_dir "$model_name" "${PROJECT_ROOT}/${model_path}"; then
        MODEL_ERRORS=$((MODEL_ERRORS+1))
        continue
    fi
    
    # Check if there are any model files in the directory
    if [ -d "${PROJECT_ROOT}/${model_path}" ]; then
        if [ "$(ls -A "${PROJECT_ROOT}/${model_path}" 2>/dev/null)" ]; then
            log "${GREEN}✓ $model_name contains files${NC}"
        else
            log "${YELLOW}⚠ $model_name directory is empty${NC}"
            MODEL_ERRORS=$((MODEL_ERRORS+1))
        fi
    fi
done

if [ $MODEL_ERRORS -eq 0 ]; then
    log "${GREEN}All model directories are present and contain files.${NC}"
else
    log "${YELLOW}$MODEL_ERRORS model issues found. You may need to download the model files.${NC}"
fi

# Summary
section_title "Verification Summary"
TOTAL_ERRORS=$((DIR_ERRORS + FILE_ERRORS + PACKAGE_ERRORS + SERVICE_ERRORS + API_ERRORS + MODEL_ERRORS))

if [ $TOTAL_ERRORS -eq 0 ]; then
    log "${GREEN}All checks passed! SutazAI is properly installed and running.${NC}"
    EXIT_CODE=0
elif [ $TOTAL_ERRORS -lt 3 ]; then
    log "${YELLOW}Verification completed with $TOTAL_ERRORS minor issues. SutazAI may work with limited functionality.${NC}"
    EXIT_CODE=1
else
    log "${RED}Verification completed with $TOTAL_ERRORS issues. SutazAI is not properly installed or running.${NC}"
    EXIT_CODE=2
fi

log "\nVerification completed at $(date)."
log "For detailed logs, see: $VERIFY_LOG"

exit $EXIT_CODE
