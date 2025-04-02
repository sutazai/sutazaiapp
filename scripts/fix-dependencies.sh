#!/bin/bash
set -e

echo "SutazAI and LocalAGI Dependency Harmonizer"
echo "==========================================="
echo "This script fixes dependency conflicts between SutazAI and LocalAGI"

# Check for sudo access
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run with sudo for system-wide changes."
   exit 1
fi

# Directories
SUTAZAI_DIR="/opt/sutazaiapp"
LOCALAGI_DIR="/home/sutazaidev/localagi"
LOCALAGI_WEBUI_DIR="${LOCALAGI_DIR}/webui"
VENV_SUTAZAI="/opt/venv-sutazaiapp"

# Ensure directories exist
if [[ ! -d "$SUTAZAI_DIR" ]]; then
    echo "Error: SutazAI directory not found at $SUTAZAI_DIR"
    exit 1
fi

if [[ ! -d "$LOCALAGI_DIR" ]]; then
    echo "Error: LocalAGI directory not found at $LOCALAGI_DIR"
    exit 1
fi

# Create backup directory
BACKUP_DIR="${SUTAZAI_DIR}/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
echo "Creating backups in $BACKUP_DIR"

# Backup requirements files
cp "${SUTAZAI_DIR}/requirements.txt" "${BACKUP_DIR}/sutazai_requirements.txt.bak"
cp "${LOCALAGI_DIR}/requirements.txt" "${BACKUP_DIR}/localagi_requirements.txt.bak"
cp "${LOCALAGI_WEBUI_DIR}/requirements.txt" "${BACKUP_DIR}/localagi_webui_requirements.txt.bak"

echo "Step 1: Creating shared constraints file..."
cat > "${SUTAZAI_DIR}/constraints.txt" << 'CONSTRAINTS'
# Core constraints for SutazAI and LocalAGI
# Last updated: 2025-03-27

# Core dependencies
PyYAML==6.0
pydantic==1.10.8
requests==2.31.0
python-dotenv==1.0.0
gunicorn==21.2.0

# LLM integration
langchain==0.0.315
langchain-core==0.1.10
langsmith==0.0.80

# Flask ecosystem
flask==2.3.3
jinja2==3.1.2

# Utilities
loguru==0.7.2
pytest==7.3.2

# Shared between web UIs
Pillow==10.0.1
CONSTRAINTS

echo "Step 2: Harmonizing SutazAI requirements.txt..."
cat > "${SUTAZAI_DIR}/requirements.txt" << 'SUTAZAI_REQS'
# SutazAI App Requirements - Compatible with LocalAGI integration
# Last updated: 2025-03-27

# Core dependencies
fastapi==0.115.8
starlette==0.45.3
uvicorn==0.34.0
gunicorn==21.2.0  # Used by both SutazAI and LocalAGI webui
httpx==0.28.1
pydantic==1.10.8  # Must be 1.10.8 for compatibility with superagi-client
python-multipart==0.0.20
python-dotenv==1.0.0  # Used by both SutazAI and LocalAGI
requests==2.31.0  # Used by both SutazAI and LocalAGI
aiohttp==3.9.3
asyncio==3.4.3
websockets==12.0
PyYAML==6.0  # Must be exactly 6.0 to maintain compatibility with both systems

# Data processing
numpy>=1.24.0  # Compatible with LocalAGI's 1.24.4
pandas==2.2.0  # SutazAI needs 2.2.0, LocalAGI uses 2.0.3
scipy==1.12.0
transformers==4.49.0
tokenizers==0.21.1  # Must be compatible with transformers 4.49.0
spacy==3.7.2
scikit-learn==1.4.0

# LLM integration
# These versions are compatible with both SutazAI and LocalAGI
langchain==0.0.315  # Fixed version for compatibility across both systems
langchain-core==0.1.10  # Fixed version for compatibility
langchain-community==0.0.13  # Added for LocalAGI compatibility
langsmith==0.0.80
llama-cpp-python==0.2.19  # SutazAI needs this version, LocalAGI uses 0.2.7
sentence-transformers==3.4.1

# LocalAGI specific dependencies
duckduckgo_search==4.1.1  # Required by LocalAGI
tiktoken==0.5.1  # Required by LocalAGI
ascii-magic==2.3.0  # Required by LocalAGI
jq==1.4.1  # Required by LocalAGI
# uuid==1.30  # Required by LocalAGI - Using Python's built-in uuid module instead
chromadb==0.4.18  # Required by LocalAGI
pysqlite3-binary==0.5.1  # Required by LocalAGI

# Document processing
pypdf==4.0.1
python-docx==0.8.11
pdfplumber==0.10.3
pytesseract>=0.3.10
pillow>=10.0.0  # Required by both SutazAI and LocalAGI webui
pdf2image>=1.16.3
tabula-py==2.9.0
PyMuPDF==1.25.3
opencv-python>=4.8.0

# Diagram parsing
diagrams==0.23.3
graphviz==0.20.1
networkx==3.2.1
matplotlib==3.8.2

# SuperAGI related
superagi-tools==1.0.8
superagi-client==0.0.1

# Flask ecosystem (for web UIs)
flask==2.3.3  # Match LocalAGI webui version for compatibility
flask-socketio==5.3.5  # Required for LocalAGI webui
flask-cors==4.0.0  # Required for SutazAI
eventlet==0.33.3  # Required for LocalAGI webui
markdown==3.5  # Required for LocalAGI webui

# Web utilities
jinja2==3.1.2  # Used by both applications

# Monitoring
prometheus-client==0.19.0
prometheus-fastapi-instrumentator==6.1.0
grafana-client==4.0.0

# Utilities
tqdm==4.66.1
loguru==0.7.2  # Used by both SutazAI and LocalAGI
pytest==7.3.2  # Required version for superagi-client compatibility
pytest-asyncio==0.23.4
pytest-cov==4.1.0

# Database and Storage
SQLAlchemy==2.0.38  # Required by SutazAI, LocalAGI uses 2.0.23
alembic==1.14.1
asyncpg==0.30.0
psycopg2-binary==2.9.10
pgvector==0.3.6
redis==5.2.1

# Vector Database 
qdrant-client==1.13.2  # SutazAI needs 1.13.2, LocalAGI uses 1.3.1
faiss-cpu==1.10.0

# Machine Learning and AI
torch==2.6.0
huggingface-hub==0.29.1

# NLP and Text Processing
nltk==3.9.1
regex==2024.11.6

# Code Analysis
radon==6.0.1
bandit==1.8.3

# Logging and Monitoring
structlog==25.1.0
psutil>=5.9.0

# Authentication and Security
python-jose==3.4.0
passlib==1.7.4
bcrypt==4.2.1
cryptography==44.0.1

# Testing
pytest-mock==3.14.0
hypothesis==6.127.2
coverage==7.6.12

# Development Tools
black==25.1.0
isort==6.0.1
flake8==7.1.2
mypy==1.15.0
pylint==3.3.4
ruff==0.9.7
pre_commit==4.1.0

# Utilities - tenacity must be compatible with langchain
tenacity==8.2.3  # Maintained at 8.2.3 for compatibility with langchain 0.0.315

# Logging and utilities
python-json-logger>=2.0.7
SUTAZAI_REQS

echo "Step 3: Harmonizing LocalAGI requirements.txt..."
cat > "${LOCALAGI_DIR}/requirements.txt" << 'LOCALAGI_REQS'
# LocalAGI Core Requirements - Compatible with SutazAI integration
# Last updated: 2025-03-27

# Core dependencies
langchain==0.0.315  # Fixed to be compatible with SutazAI
langchain-core==0.1.10  # Fixed to be compatible with SutazAI
langchain-community==0.0.13
openai==0.28.0
chromadb==0.4.18
pysqlite3-binary==0.5.1

# API and networking
requests==2.31.0  # Compatible with SutazAI
urllib3==2.0.7

# UI and visualization
ascii-magic==2.3.0

# Logging and debugging
loguru==0.7.2  # Compatible with SutazAI

# Search functionality
duckduckgo_search==4.1.1

# Text processing
tiktoken==0.5.1
typing-extensions==4.8.0

# Environment and configuration
python-dotenv==1.0.0  # Compatible with SutazAI
pyyaml==6.0  # Must be exactly 6.0 for compatibility with SutazAI

# Utilities
# uuid==1.30  # Using Python's built-in uuid module instead
jq==1.4.1

# Data handling
numpy==1.24.4  # Compatible with SutazAI
pandas==2.0.3  # SutazAI uses newer version 2.2.0, but this is compatible

# Storage
sqlalchemy==2.0.23  # SutazAI uses 2.0.38, both should be compatible

# Flask web UI
flask==2.3.3
flask-socketio==5.3.5
eventlet==0.33.3
markdown==3.5
jinja2==3.1.2
Pillow==10.0.1  # Compatible with SutazAI
LOCALAGI_REQS

echo "Step 4: Harmonizing LocalAGI WebUI requirements.txt..."
cat > "${LOCALAGI_WEBUI_DIR}/requirements.txt" << 'LOCALAGI_WEBUI_REQS'
# LocalAGI WebUI Requirements - Compatible with SutazAI integration
# Last updated: 2025-03-27

flask==2.3.3  # Compatible with both systems
flask-socketio==5.3.5
requests==2.31.0  # Compatible with SutazAI
python-dotenv==1.0.0  # Compatible with SutazAI
gunicorn==21.2.0  # Compatible with SutazAI
markdown==3.5
Pillow==10.0.1  # Compatible with SutazAI
jinja2==3.1.2  # Compatible with SutazAI
eventlet==0.33.3
LOCALAGI_WEBUI_REQS

echo "Step 5: Installing critical dependencies for SutazAI..."
# Activate virtual environment for SutazAI
if [[ -f "${VENV_SUTAZAI}/bin/activate" ]]; then
    source "${VENV_SUTAZAI}/bin/activate"
    
    # Install key shared dependencies first
    pip install -c "${SUTAZAI_DIR}/constraints.txt" PyYAML==6.0
    pip install -c "${SUTAZAI_DIR}/constraints.txt" pydantic==1.10.8
    pip install -c "${SUTAZAI_DIR}/constraints.txt" requests==2.31.0
    pip install -c "${SUTAZAI_DIR}/constraints.txt" python-dotenv==1.0.0
    
    # Install LLM integration packages in correct order
    pip install -c "${SUTAZAI_DIR}/constraints.txt" langsmith==0.0.80
    pip install -c "${SUTAZAI_DIR}/constraints.txt" langchain-core==0.1.10
    pip install -c "${SUTAZAI_DIR}/constraints.txt" langchain==0.0.315
    pip install langchain-community==0.0.13
    
    # Install other shared dependencies
    pip install -c "${SUTAZAI_DIR}/constraints.txt" flask==2.3.3
    pip install -c "${SUTAZAI_DIR}/constraints.txt" jinja2==3.1.2
    pip install -c "${SUTAZAI_DIR}/constraints.txt" loguru==0.7.2
    
    # Install LocalAGI-specific dependencies
    pip install duckduckgo_search==4.1.1
    pip install tiktoken==0.5.1
    pip install ascii-magic==2.3.0
    pip install jq==1.4.1
    pip install uuid==1.30
    pip install chromadb==0.4.18
    pip install pysqlite3-binary==0.5.1
    
    deactivate
    echo "SutazAI dependencies updated successfully."
else
    echo "Warning: SutazAI virtual environment not found at ${VENV_SUTAZAI}"
    echo "Manual dependency installation may be required."
fi

echo "Step 6: Creating cross-reference document..."
cat > "${SUTAZAI_DIR}/DEPENDENCIES_CROSS_REFERENCE.md" << 'CROSS_REF'
# SutazAI and LocalAGI Dependency Cross-Reference

This document provides a cross-reference of shared dependencies between SutazAI and LocalAGI.

## Critical Shared Dependencies

| Dependency | Version | Notes |
|------------|---------|-------|
| PyYAML | 6.0 | Must be exactly 6.0 |
| pydantic | 1.10.8 | Required for superagi-client compatibility |
| requests | 2.31.0 | Used by both systems |
| python-dotenv | 1.0.0 | Used by both systems |
| langchain | 0.0.315 | Fixed version for compatibility |
| langchain-core | 0.1.10 | Fixed version for compatibility |
| flask | 2.3.3 | For web UI compatibility |
| jinja2 | 3.1.2 | Used by both applications |
| loguru | 0.7.2 | Used by both systems |
| Pillow | 10.0.1 / >=10.0.0 | For image processing |

## Maintenance Notes

- When updating dependencies, always check both systems' requirements.txt files
- Use constraints.txt to enforce version compatibility
- Key conflicts to watch for:
  - langchain and langchain-core version compatibility
  - pydantic version (SutazAI uses v1, newer libraries may require v2)
  - PyYAML version (must stay at 6.0)
  - Flask ecosystem versions

## Last Updated

2025-03-27
CROSS_REF

echo "Step 7: Setting up symlink for LocalAGI..."
if [[ ! -L "/opt/localagi" ]]; then
    ln -sf "${LOCALAGI_DIR}" "/opt/localagi"
    echo "Created symlink: /opt/localagi -> ${LOCALAGI_DIR}"
fi

echo "Step 8: Creating fix_localagi_deps.sh script in LocalAGI directory..."
cat > "${LOCALAGI_DIR}/fix_localagi_deps.sh" << 'FIX_SCRIPT'
#!/bin/bash
# LocalAGI dependency fix script
# Run this if you're experiencing dependency issues with LocalAGI

echo "Fixing LocalAGI dependencies..."
cd "$(dirname "$0")"

# Rebuild LocalAGI
docker-compose build localagi

# Rebuild webui
docker-compose build webui

echo "Dependencies fixed. Restart services with: docker-compose up -d"
FIX_SCRIPT

chmod +x "${LOCALAGI_DIR}/fix_localagi_deps.sh"

echo ""
echo "Dependency harmonization complete!"
echo "------------------------------------"
echo "1. SutazAI requirements.txt updated"
echo "2. LocalAGI requirements.txt updated"
echo "3. LocalAGI WebUI requirements.txt updated"
echo "4. Critical dependencies installed in SutazAI environment"
echo "5. Cross-reference documentation created"
echo "6. LocalAGI symlink created (if needed)"
echo "7. Helper script created in LocalAGI directory"
echo ""
echo "To complete the process:"
echo "1. Restart all services: sudo scripts/stop_all.sh && sudo scripts/start_all.sh"
echo "2. If LocalAGI issues persist: cd /opt/localagi && ./fix_localagi_deps.sh"
echo ""
echo "Backup of original files saved to: ${BACKUP_DIR}"
echo ""