#!/bin/bash
# SutazAI Dependency Installation Script
# This enhanced script allows selective installation of dependencies based on what components are needed.
# Options:
#   --core-only       Install only core dependencies (fastest)
#   --with-ml         Include machine learning packages
#   --with-ai         Include AI and LLM components
#   --with-dev        Include development tools
#   --all             Install all dependencies (slowest)
#   --no-cache        Don't use pip cache (useful for resolving conflicts)

set -e  # Exit on error

# Color codes for pretty output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SYSTEM_DEPS=("docker" "docker-compose" "curl" "git" "postgresql" "redis-server")
BASE_DIR="/opt/sutazaiapp"
REQUIREMENTS_FILE="$BASE_DIR/requirements.txt"
TEMP_REQUIREMENTS="$BASE_DIR/temp_requirements.txt"

# Check if running with appropriate permissions
if [ "$(id -u)" -ne 0 ] && [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${RED}Please run as root or in a virtual environment${NC}"
    exit 1
fi

# Ensure Python 3.11 is available
if ! command -v python3.11 &>/dev/null; then
    echo -e "${YELLOW}Python 3.11 not found. Would you like to install it? (y/n)${NC}"
    read -p "> " install_python
    if [[ "$install_python" == "y" || "$install_python" == "Y" ]]; then
        bash "$BASE_DIR/scripts/install_python311.sh"
    else
        echo -e "${RED}Python 3.11 is required for SutazAI. Exiting.${NC}"
        exit 1
    fi
fi

# Function to install system dependencies
install_system_deps() {
    echo -e "${BLUE}Checking system dependencies...${NC}"
    for dep in "${SYSTEM_DEPS[@]}"; do
        if ! command -v "$dep" &>/dev/null; then
            echo -e "${YELLOW}Installing $dep...${NC}"
            apt-get update && apt-get install -y "$dep"
            echo -e "${GREEN}Installed $dep${NC}"
        else
            echo -e "${GREEN}✓ $dep already installed${NC}"
        fi
    done
}

# Function to create a temporary requirements file with specified packages
create_temp_requirements() {
    echo "# Generated temporary requirements file for SutazAI" > "$TEMP_REQUIREMENTS"
    echo "# Generated on $(date)" >> "$TEMP_REQUIREMENTS"
    echo "" >> "$TEMP_REQUIREMENTS"
    
    # Add core dependencies
    if [[ "$INSTALL_CORE" == "1" || "$INSTALL_ALL" == "1" ]]; then
        echo -e "${BLUE}Adding core dependencies...${NC}"
        cat << EOF >> "$TEMP_REQUIREMENTS"
# Core dependencies
pip>=21.0.0
setuptools>=42.0.0
wheel>=0.37.0
fastapi>=0.95.0
uvicorn[standard]>=0.20.0
python-multipart>=0.0.5
pydantic>=1.10.12,<2.0.0  # Avoid pydantic v2 conflicts
starlette>=0.27.0
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.5
asyncpg>=0.27.0
alembic>=1.10.0
redis>=4.5.0
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
python-dotenv>=1.0.0
loguru>=0.6.0
prometheus-client>=0.16.0
structlog>=23.1.0
requests>=2.28.0
aiohttp>=3.8.0
pyyaml>=6.0.0
python-dateutil>=2.8.0
jinja2>=3.1.0

EOF
    fi
    
    # Add ML dependencies if requested
    if [[ "$INSTALL_ML" == "1" || "$INSTALL_ALL" == "1" ]]; then
        echo -e "${BLUE}Adding machine learning dependencies...${NC}"
        cat << EOF >> "$TEMP_REQUIREMENTS"
# Machine learning dependencies
numpy>=1.23.0
pandas>=1.5.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
scipy>=1.10.0

EOF
    fi
    
    # Add AI dependencies if requested
    if [[ "$INSTALL_AI" == "1" || "$INSTALL_ALL" == "1" ]]; then
        echo -e "${BLUE}Adding AI and LLM dependencies...${NC}"
        cat << EOF >> "$TEMP_REQUIREMENTS"
# AI and LLM dependencies
torch>=2.0.0
transformers>=4.26.0
langchain>=0.0.200
chromadb>=0.4.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.0
pgvector>=0.2.0

EOF
    fi
    
    # Add development dependencies if requested
    if [[ "$INSTALL_DEV" == "1" || "$INSTALL_ALL" == "1" ]]; then
        echo -e "${BLUE}Adding development dependencies...${NC}"
        cat << EOF >> "$TEMP_REQUIREMENTS"
# Development tools
pytest>=7.0.0
black>=23.0.0
flake8>=6.0.0
mypy>=1.0.0
pytest-cov>=4.1.0

EOF
    fi
}

# Function to install Python dependencies
install_dependencies() {
    PYTHON_CMD=${PYTHON_CMD:-python3.11}
    PIP_CMD="${PYTHON_CMD} -m pip"
    
    if [[ "$INSTALL_ALL" == "1" ]]; then
        echo -e "${BLUE}Installing all dependencies from $REQUIREMENTS_FILE${NC}"
        echo -e "${YELLOW}This might take a while...${NC}"
        if [[ "$NO_CACHE" == "1" ]]; then
            $PIP_CMD install --no-cache-dir -r "$REQUIREMENTS_FILE"
        else
            $PIP_CMD install -r "$REQUIREMENTS_FILE"
        fi
    else
        # Create temporary requirements file with selected dependencies
        create_temp_requirements
        
        echo -e "${BLUE}Installing selected dependencies...${NC}"
        if [[ "$NO_CACHE" == "1" ]]; then
            $PIP_CMD install --no-cache-dir -r "$TEMP_REQUIREMENTS"
        else
            $PIP_CMD install -r "$TEMP_REQUIREMENTS"
        fi
        
        # Clean up
        rm -f "$TEMP_REQUIREMENTS"
    fi
}

# Parse command line arguments
INSTALL_CORE=0
INSTALL_ML=0
INSTALL_AI=0
INSTALL_DEV=0
INSTALL_ALL=0
NO_CACHE=0

# Check if no arguments are provided
if [ $# -eq 0 ]; then
    # Interactive mode
    echo -e "${BLUE}SutazAI Dependency Installer${NC}"
    echo -e "${YELLOW}What would you like to install?${NC}"
    echo "1. Core dependencies only (fastest)"
    echo "2. Core + Machine Learning"
    echo "3. Core + AI/LLM"
    echo "4. Core + Development tools"
    echo "5. All dependencies (slowest)"
    echo "6. Exit"
    read -p "> " choice
    
    case $choice in
        1)
            INSTALL_CORE=1
            ;;
        2)
            INSTALL_CORE=1
            INSTALL_ML=1
            ;;
        3)
            INSTALL_CORE=1
            INSTALL_AI=1
            ;;
        4)
            INSTALL_CORE=1
            INSTALL_DEV=1
            ;;
        5)
            INSTALL_ALL=1
            ;;
        6)
            echo -e "${GREEN}Exiting.${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid option.${NC}"
            exit 1
            ;;
    esac
    
    echo -e "${YELLOW}Use pip cache? (y/n)${NC}"
    read -p "> " use_cache
    if [[ "$use_cache" == "n" || "$use_cache" == "N" ]]; then
        NO_CACHE=1
    fi
else
    # Parse command-line arguments
    for arg in "$@"; do
        case $arg in
            --core-only)
                INSTALL_CORE=1
                ;;
            --with-ml)
                INSTALL_ML=1
                ;;
            --with-ai)
                INSTALL_AI=1
                ;;
            --with-dev)
                INSTALL_DEV=1
                ;;
            --all)
                INSTALL_ALL=1
                ;;
            --no-cache)
                NO_CACHE=1
                ;;
            *)
                echo -e "${RED}Unknown option: $arg${NC}"
                exit 1
                ;;
        esac
    done
    
    # If no specific group was chosen and not --all, default to core
    if [[ "$INSTALL_CORE" == "0" && "$INSTALL_ML" == "0" && "$INSTALL_AI" == "0" && "$INSTALL_DEV" == "0" && "$INSTALL_ALL" == "0" ]]; then
        INSTALL_CORE=1
    fi
fi

# Installation process
install_system_deps

echo -e "${BLUE}Installing Python dependencies...${NC}"
install_dependencies

echo -e "${GREEN}Installation completed successfully!${NC}"

# Summary
echo -e "\n${BLUE}Installation Summary:${NC}"
if [[ "$INSTALL_ALL" == "1" ]]; then
    echo -e "${GREEN}✓ All dependencies installed${NC}"
else
    [[ "$INSTALL_CORE" == "1" ]] && echo -e "${GREEN}✓ Core dependencies installed${NC}"
    [[ "$INSTALL_ML" == "1" ]] && echo -e "${GREEN}✓ Machine learning dependencies installed${NC}"
    [[ "$INSTALL_AI" == "1" ]] && echo -e "${GREEN}✓ AI/LLM dependencies installed${NC}" 
    [[ "$INSTALL_DEV" == "1" ]] && echo -e "${GREEN}✓ Development tools installed${NC}"
fi

echo -e "\n${GREEN}SutazAI is ready to use!${NC}"

# Exit successfully
exit 0 