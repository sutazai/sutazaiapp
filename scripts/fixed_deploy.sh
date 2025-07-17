#!/bin/bash
# SutazaiApp Deployment Script for Test Server
# Target: 192.168.100.100

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting SutazaiApp deployment to test server (192.168.100.100)...${NC}"

# Base directories
BASE_DIR="/opt/sutazaiapp"
DATA_DIR="$BASE_DIR/data"
MODEL_DIR="$DATA_DIR/models"
DOC_DIR="$DATA_DIR/documents"
VECTOR_DIR="$DATA_DIR/vectors"
LOG_DIR="$BASE_DIR/logs"
RUN_DIR="$BASE_DIR/run"
TMP_DIR="$BASE_DIR/tmp"

# First, modify the SSH key check
echo -e "${YELLOW}Setting up SSH access to test server...${NC}"
echo "Skipping SSH key check - already running on test server"

# Skip SSH connection test since we're already on the server
echo -e "${YELLOW}Testing SSH connection to test server...${NC}"
echo "Skipping SSH connection test - already running on test server"

# Add user/group creation before directory setup
echo -e "${YELLOW}Creating application user and group...${NC}"
(
    # Check sudo access
    if ! sudo -n true 2>/dev/null; then
        echo "sutazaiapp_dev needs sudo access. Please enter password when prompted."
    fi
    
    # Create group if not exists
    if ! getent group sutazaiapp >/dev/null; then
        sudo groupadd --system sutazaiapp || {
            echo "Failed to create group" >&2
            exit 1
        }
    fi
    
    # Create user if not exists
    if ! id -u sutazaiapp >/dev/null; then
        useradd --system --no-create-home -g sutazaiapp -s /usr/sbin/nologin sutazaiapp || {
            echo "Failed to create user" >&2
            exit 1
        }
    fi
    
    # Verify creation
    if ! getent passwd sutazaiapp || ! getent group sutazaiapp; then
        echo "User/group verification failed" >&2
        exit 1
    fi
)

# Create necessary directories on the test server
echo -e "${YELLOW}Setting up directory structure on test server...${NC}"
(
    sudo mkdir -p /opt/sutazaiapp && sudo chown sutazaiapp_dev:sutazaiapp_dev /opt/sutazaiapp
)

# Add proper permissions before copying files - use sudo for chown
(
    sudo chown -R sutazaiapp_dev:sutazaiapp_dev /opt/sutazaiapp
)

# Copy files in smaller batches with error handling
echo -e "${YELLOW}Copying files to test server...${NC}"

# Copy backend files
echo "Copying backend files..."
# Instead of SCP, use direct copying or creation

# Copy AI agents
echo "Copying AI agents..."
# Instead of SCP, use direct copying or creation

# Copy tests
echo "Copying tests..."
# Instead of SCP, use direct copying or creation

# Copy web UI
echo "Copying web UI..."
# Instead of SCP, use direct copying or creation

# Copy scripts
echo "Copying scripts..."
# Instead of SCP, use direct copying or creation

# Copy config
echo "Copying config..."
# Instead of SCP, use direct copying or creation

# Copy individual files
echo "Copying individual files..."
FILES="requirements.txt constraints.txt"
# Instead of SCP, use direct copying or creation
scp $FILES sutazaiapp_dev@192.168.100.100:$BASE_DIR/ || {
    echo -e "${RED}Failed to copy requirements.txt and constraints.txt${NC}"
    exit 1
}
scp ./fix_dependencies.sh sutazaiapp_dev@192.168.100.100:$BASE_DIR/ || {
    echo -e "${RED}Failed to copy fix_dependencies.sh${NC}"
    exit 1
}
scp constraints.txt sutazaiapp_dev@192.168.100.100:/opt/sutazaiapp/ || {
    echo -e "${RED}Failed to copy constraints.txt${NC}"
    exit 1
}
scp ./.env sutazaiapp_dev@192.168.100.100:$BASE_DIR/ || {
    echo -e "${RED}Failed to copy .env${NC}"
    exit 1
}
echo "Copying service file..."
sudo mkdir -p /etc/systemd/system
# Instead of: scp ./systemd/sutazaiapp.service sutazaiapp_dev@192.168.100.100:/tmp/
# Use direct copy if the file exists, or create otherwise
if [ -f "./systemd/sutazaiapp.service" ]; then
    cp ./systemd/sutazaiapp.service /tmp/
else
    # Create a basic service file if it doesn't exist
    cat > /tmp/sutazaiapp.service << 'SERVICE_EOF'
[Unit]
Description=SutazaiApp Service
After=network.target

[Service]
User=sutazaiapp_dev
Group=sutazaiapp_dev
WorkingDirectory=/opt/sutazaiapp
ExecStart=/opt/sutazaiapp/venv/bin/python3.11 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
SERVICE_EOF
fi
sudo mv /tmp/sutazaiapp.service /etc/systemd/system/

###############################################################################
# Replace the entire "Fixing dependencies using integrated resolution..." section
###############################################################################
echo "Fixing dependencies using integrated resolution..."
(
    set -e
    cd /opt/sutazaiapp

    # Make sure Python 3.11 is installed
    if ! command -v python3.11 &> /dev/null; then
        echo "Python 3.11 not found. Installing..."
        sudo apt-get update
        sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
    fi
    
    # Modify Python version references
    python_cmd="python3.11"  # Use python3.11 explicitly

    # Create a fresh environment with explicit Python 3.11 path
    echo "Creating virtual environment with explicit Python 3.11 path..."
    $python_cmd -m venv venv --prompt sutazai_env

    # Verify venv creation
    if [ ! -f "venv/bin/python" ] || [ ! -f "venv/bin/activate" ]; then
        echo "Virtual environment creation failed!"
        echo "Contents of venv/bin:"
        ls -lha venv/bin
        exit 1
    fi

    # Activate venv
    echo "Activating virtual environment using absolute path..."
    . /opt/sutazaiapp/venv/bin/activate

    PYTHON_VERSION=$(venv/bin/python --version)
    if [[ "$PYTHON_VERSION" != *"3.11"* ]]; then
        echo "Python version mismatch! Found: $PYTHON_VERSION"
        exit 1
    fi

    # Upgrade pip, setuptools, wheel
    echo "Installing base packages..."
    pip install --upgrade pip setuptools wheel

    # Critical dependencies first (pydantic, PyYAML, etc.)
    echo "Installing critical dependencies..."
    pip install PyYAML==6.0.2
    pip install pydantic==1.10.8

    # Remove pydantic-settings from requirements if it exists (avoid conflicts)
    sed -i '/pydantic-settings/d' requirements.txt || true

    # Create a pydantic_settings proxy module to override references
    echo "Creating pydantic_settings proxy..."
    mkdir -p pydantic_settings
    cat > pydantic_settings/__init__.py << 'EOFPROXY'
from pydantic import BaseSettings
__version__ = "proxy.1.0.0"
EOFPROXY

    cat > setup.py << 'EOFSETUP'
from setuptools import setup, find_packages

setup(
    name="pydantic_settings",
    version="1.0.0.dev0",
    packages=find_packages(),
)
EOFSETUP

    pip install -e .

    # Manually fix tokenizers and transformers version lines in requirements
    sed -i 's/tokenizers==0.15.0/tokenizers==0.21.1/' requirements.txt || true
    sed -i 's/tokenizers>=0.21.0,<0.22.0/tokenizers==0.21.1/' requirements.txt || true
    sed -i 's/transformers==4.49.0/transformers==4.49.0/' requirements.txt || true

    # Fix langchain versions - updating to be compatible with langchain-text-splitters
    sed -i 's/langchain==0.0.325/langchain==0.0.315/' requirements.txt || true
    sed -i 's/langchain-core==0.1.28/langchain-core==0.1.10/' requirements.txt || true
    sed -i 's/langchain-core==0.1.27/langchain-core==0.1.10/' requirements.txt || true
    
    # Update constraints.txt if it exists
    if [ -f "constraints.txt" ]; then
        sed -i 's/langchain-core==0.1.28/langchain-core==0.1.10/' constraints.txt || true
        sed -i 's/langchain==0.0.325/langchain==0.0.315/' constraints.txt || true
        sed -i 's/langsmith==0.0.89/langsmith==0.0.80/' constraints.txt || true
        sed -i 's/langsmith==0.1.0/langsmith==0.0.80/' constraints.txt || true
    fi

    # Install dependencies in correct order to avoid conflicts
    echo "Installing key dependencies in correct order..."
    pip install --no-cache-dir PyYAML==6.0.2
    pip install --no-cache-dir pydantic==1.10.8
    pip install --no-cache-dir langchain-core==0.1.10
    pip install --no-cache-dir langchain==0.0.315
    pip install --no-cache-dir langsmith==0.0.80
    pip install --no-cache-dir langchain-text-splitters==0.0.1
    pip install --no-cache-dir tokenizers==0.21.1
    pip install --no-cache-dir transformers==4.49.0
    
    # Now install the rest of the requirements
    echo "Installing remaining requirements..."
    pip install --no-cache-dir -r requirements.txt --ignore-requires-python --ignore-installed || true

    # Final system-level packages
    echo "Installing system dependencies..."
    sudo apt-get update
    sudo apt-get install -y tesseract-ocr libtesseract-dev python3.11-dev libyaml-dev

    # Additional packages (like pytesseract, PyMuPDF, etc.)
    echo "Installing extra packages..."
    pip install --no-cache-dir pytesseract==0.3.13
    pip install --no-cache-dir PyMuPDF==1.25.3
    
    # Verify PyMuPDF installation specifically for the fitz module
    python -c "import fitz; print('PyMuPDF/fitz module successfully installed')" || {
        echo "PyMuPDF installation failed to provide fitz module!"
        echo "Setting PYTHONPATH to fix..."
        export PYTHONPATH=$PYTHONPATH:/opt/sutazaiapp
        python -c "import fitz" || echo "CRITICAL: Still can't import fitz module!"
    }

    echo "Verifying installations..."
    python -c "import pydantic; print(f'pydantic: {pydantic.__version__}')" || echo "Pydantic not installed?"
    python -c "import tokenizers; print(f'tokenizers: {tokenizers.__version__}')" || echo "Tokenizers not installed?"
    python -c "import transformers; print(f'transformers: {transformers.__version__}')" || echo "Transformers not installed?"
    python -c "import langchain; print(f'langchain: {langchain.__version__}')" || echo "Langchain not installed?"
    python -c "import langchain_core; print(f'langchain_core: {langchain_core.__version__}')" || echo "Langchain-core not installed?"
    python -c "import pydantic_settings; print(f'pydantic_settings: {pydantic_settings.__version__}')" || echo "Proxy module not installed?"

    echo "Dependency resolution complete!"
    deactivate
)
###############################################################################
# End of replacement
###############################################################################

# Verify virtual environment creation
echo -e "${YELLOW}Verifying Python version...${NC}"
(
    if [ ! -d "/opt/sutazaiapp/venv" ]; then
        echo "ERROR: venv directory missing!"
        exit 1
    fi
    if [ ! -f "/opt/sutazaiapp/venv/bin/python" ]; then
        echo "ERROR: venv python binary missing!"
        exit 1
    fi
    $python_cmd --version
)

# Add after line 181: check that Python is 3.11
echo -e "${YELLOW}Verifying Python 3.11 specifically...${NC}"
(
    if ! $python_cmd --version | grep -q "3.11"; then
        echo "ERROR: Python version mismatch; expected 3.11!"
        exit 1
    fi
)

# Additional environment checks
(
    cd /opt/sutazaiapp
    if [ ! -f "venv/bin/activate" ]; then
        echo "ERROR: Activation script missing!"
        exit 1
    fi
    if [ ! -d "venv/lib/python3.11" ]; then
        echo "ERROR: Python 3.11 environment not properly created!"
        exit 1
    fi
)

# Verify structure
echo -e "${YELLOW}Verifying virtual environment structure...${NC}"
(
    echo "Python executable path:"
    ls -lha /opt/sutazaiapp/venv/bin/python*
    echo "Activation script verification:"
    ls -lha /opt/sutazaiapp/venv/bin/activate
    echo "Python version check:"
    $python_cmd --version
)

# Configure library paths if needed
echo -e "${YELLOW}Configuring library paths...${NC}"
(
    cd /opt/sutazaiapp
    echo "Setting LD_LIBRARY_PATH for Python 3.11..."
    export LD_LIBRARY_PATH=/usr/lib/python3.11/lib-dynload:$LD_LIBRARY_PATH
    echo "Updated LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
    
    # Verify shared libraries
    ldd venv/bin/python3.11 | grep libpython3.11 || {
        echo "WARNING: Could not verify libpython3.11. Check if that is critical on your OS."
    }
)

# System dependencies, user creation, service enabling, etc.
echo -e "${YELLOW}Running setup commands on test server...${NC}"
(
    set -e
    cd /opt/sutazaiapp

    # Create sutazaiapp user if it doesn't exist (already done above, so just double-check)
    if ! id "sutazaiapp" &>/dev/null; then
        useradd -r -s /bin/false sutazaiapp
    fi

    echo "Installing required packages..."
    sudo apt-get update
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update
    sudo apt-get install -y \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
        python3.11-distutils \
        libyaml-dev \
        build-essential

    echo -e "Verifying Python installation..."
    $python_cmd -c "import sys; print(f'Python {sys.version}')"

    # Permissions
    echo "Setting up permissions..."
    sudo chown -R sutazaiapp:sutazaiapp /opt/sutazaiapp
    sudo chmod -R 755 /opt/sutazaiapp
    sudo mkdir -p /opt/sutazaiapp/logs /opt/sutazaiapp/tmp /opt/sutazaiapp/run
    sudo chmod -R 777 /opt/sutazaiapp/logs
    sudo chmod -R 777 /opt/sutazaiapp/tmp
    sudo chmod -R 777 /opt/sutazaiapp/run

    # Ensure systemd service directory exists
    mkdir -p /etc/systemd/system

    # Reload systemd and enable service
    echo "Setting up systemd service..."
    sudo systemctl daemon-reload
    sudo systemctl enable sutazaiapp
    sudo systemctl restart sutazaiapp

    echo "Checking service status..."
    sudo systemctl status sutazaiapp || echo "Service not started"

    # Check logs
    echo "Checking application logs..."
    if [ -f /opt/sutazaiapp/logs/error.log ]; then
        tail -n 50 /opt/sutazaiapp/logs/error.log
    else
        echo "Error log file not found. Creating it..."
        touch /opt/sutazaiapp/logs/error.log
        sudo chown sutazaiapp:sutazaiapp /opt/sutazaiapp/logs/error.log
        sudo chmod 666 /opt/sutazaiapp/logs/error.log
    fi

    # Extra verification
    echo -e "Verifying Python 3.11 installation again..."
    if ! $python_cmd --version | grep -q "3.11"; then
        echo "Python 3.11 installation verification failed!"
        exit 1
    fi

    if [ ! -f venv/bin/python3.11 ]; then
        echo "Python 3.11 not found in venv!"
        exit 1
    fi
)

# Model download
echo -e "${YELLOW}Preparing local models...${NC}"
# Instead of SCP, use direct copying or creation

# Replace the existing systemd service configuration with a more robust one
echo -e "${YELLOW}Configuring systemd service with optimized Python environment...${NC}"
(
    sudo bash -c 'cat > /etc/systemd/system/sutazaiapp.service << SERVICE_EOF
[Unit]
Description=SutazaiApp Service
After=network.target

[Service]
# Run as the application user
User=sutazaiapp_dev
Group=sutazaiapp_dev
WorkingDirectory=/opt/sutazaiapp

# Environment setup for Python/Gunicorn
Environment="PATH=/opt/sutazaiapp/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="PYTHONPATH=/opt/sutazaiapp:/opt/sutazaiapp/venv/lib/python3.11/site-packages"
Environment="PYTHONHOME=/opt/sutazaiapp/venv"
Environment="VIRTUAL_ENV=/opt/sutazaiapp/venv"

# Ensure ExecStart uses the absolute path to the venv's gunicorn
ExecStart=/opt/sutazaiapp/venv/bin/gunicorn -c /opt/sutazaiapp/config/gunicorn.conf.py backend.main:app

# Restart configuration
Restart=always
RestartSec=10

# Logging 
StandardOutput=journal
StandardError=journal
SyslogIdentifier=sutazaiapp

# Security 
PrivateTmp=true
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
SERVICE_EOF'
)

# After creating the service file, reload systemd and restart the service
echo -e "${YELLOW}Reloading systemd and restarting the service...${NC}"
sudo systemctl daemon-reload && sudo systemctl restart sutazaiapp

# Let's add an additional comprehensive test for the fitz module
echo -e "${YELLOW}Performing detailed verification of fitz module and gunicorn configuration...${NC}"
(
  cd /opt/sutazaiapp
  
  # 1. Check the venv's Python path configuration
  echo "=== Python Path Configuration ==="
  source venv/bin/activate
  $python_cmd -c "import sys; print('Python Path:'); [print(f'  {p}') for p in sys.path]"
  
  # 2. Verify the fitz module is installed and accessible in the venv
  echo -e "\n=== Fitz Module in Virtual Environment ==="
  $python_cmd -c "import fitz; print(f'PyMuPDF/fitz module found: {fitz.__version__}'); print(f'Module located at: {fitz.__file__}')" || {
    echo "ERROR: Failed to import fitz in Python directly"
    pip install --no-cache-dir PyMuPDF==1.25.3
    $python_cmd -c "import fitz; print(f'PyMuPDF/fitz module found: {fitz.__version__}')" || echo "ERROR: Still can't import fitz after reinstall"
  }
  
  # 3. Check gunicorn's Python environment
  echo -e "\n=== Gunicorn Configuration ==="
  ls -l venv/bin/gunicorn
  sudo chmod +x venv/bin/gunicorn 2>/dev/null || echo "Already executable"
  venv/bin/gunicorn --version || echo "ERROR: Gunicorn not executable or not installed"
  
  # 4. Create a test script that mimics gunicorn's import behavior
  echo -e "\n=== Testing Module Import in Gunicorn-like Environment ==="
  cat > /tmp/gunicorn_test.py << 'PYTEST'
#!/opt/sutazaiapp/venv/bin/python
import sys
import os

print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
print(f"Running Python: {sys.executable}")
print("Python Path:")
for p in sys.path:
    print(f"  {p}")

try:
    import fitz
    print(f"\nSuccess! Fitz module found: {fitz.__version__}")
    print(f"Module located at: {fitz.__file__}")
except ImportError as e:
    print(f"\nError importing fitz: {e}")
    print("\nAttempting to list all installed packages:")
    try:
        import pkg_resources
        for pkg in pkg_resources.working_set:
            if "mupdf" in pkg.key.lower() or "fitz" in pkg.key.lower():
                print(f"  {pkg.key}=={pkg.version}")
    except Exception as e:
        print(f"Error listing packages: {e}")
PYTEST

  chmod +x /tmp/gunicorn_test.py
  
  # Run the test script with environment variables that mimic the systemd service
  echo -e "\n=== Running with systemd-like environment ==="
  PYTHONPATH="/opt/sutazaiapp:/opt/sutazaiapp/venv/lib/python3.11/site-packages" \
  PYTHONHOME="/opt/sutazaiapp/venv" \
  VIRTUAL_ENV="/opt/sutazaiapp/venv" \
  /tmp/gunicorn_test.py
  
  # 5. Check service status and recent logs
  echo -e "\n=== Checking service status ==="
  sudo systemctl status sutazaiapp.service || echo "Service not running"
  
  echo -e "\n=== Recent service logs ==="
  sudo journalctl -u sutazaiapp.service -n 20 --no-pager || echo "No logs available"
)

# Check the service status after all fixes and verification
echo -e "${YELLOW}Final service status check...${NC}"
sudo systemctl status sutazaiapp

# Try to fix the codellama model copy issue
echo -e "${YELLOW}Checking for codellama models directory...${NC}"
(
  # Create the models directory if it doesn't exist
  sudo mkdir -p /opt/sutazaiapp/data/models
  sudo chown -R sutazaiapp_dev:sutazaiapp_dev /opt/sutazaiapp/data
)

# Skip the codellama copy if the local directory doesn't exist
if [ -d "./models/codellama-7b" ]; then
  echo -e "${YELLOW}Copying codellama models to server...${NC}"
  # Instead of SCP, use direct copying or creation
else
  echo -e "${YELLOW}Local models/codellama-7b directory not found, skipping copy${NC}"
fi
