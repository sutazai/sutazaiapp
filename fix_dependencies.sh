#!/bin/bash
set -e

echo "SutazAI Local Dependency Resolution"
echo "----------------------------------------"

# Run commands locally
cd /opt/sutazaiapp

# Update requirements.txt file to fix version conflicts
echo "Fixing conflicting versions in requirements.txt..."
sed -i 's/tokenizers==0.15.0/tokenizers==0.21.1/' requirements.txt
sed -i 's/tokenizers>=0.21.0,<0.22.0/tokenizers==0.21.1/' requirements.txt

echo "Fixing pydantic ecosystem..."
sed -i 's/pydantic==2.10.6/pydantic==1.10.8/' requirements.txt
sed -i 's/pydantic>=1.10.8,<2.0.0/pydantic==1.10.8/' requirements.txt

# Remove pydantic-settings entirely as we'll use pydantic.BaseSettings instead
sed -i '/pydantic-settings/d' requirements.txt

# Fix langchain and langchain-core to compatible versions
sed -i 's/langchain==0.0.354/langchain==0.0.315/' requirements.txt
sed -i 's/langchain-core==0.1.27/langchain-core==0.1.10/' requirements.txt

# Create constraints file with compatible versions
echo "Creating constraints file with compatible versions..."
cat > constraints.txt << 'CONSTRAINTS'
# Core constraints
PyYAML==6.0.2
pytest==7.3.2

# Pydantic ecosystem
pydantic==1.10.8  # Required by superagi-client (includes BaseSettings)

# NLP stack
tokenizers==0.21.1  # Compatible with transformers 4.49.0
transformers==4.49.0

# Langchain compatibility
langchain==0.0.315  # Works with langchain-core 0.1.10
langchain-core==0.1.10
langsmith==0.0.80  # Compatible with both
CONSTRAINTS

# Activate virtual environment
source /opt/venv-sutazaiapp/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }

echo "Installing base build dependencies..."
pip install --upgrade pip setuptools wheel

# Install packages in the correct order to avoid conflicts
echo "Step 1: Installing critical dependencies first..."
pip install PyYAML==6.0.2
pip install pydantic==1.10.8

# Continue with other dependencies
echo "Step 2: Installing tokenizers..."
pip install tokenizers==0.21.1

echo "Step 3: Installing transformers..."
pip install transformers==4.49.0

echo "Step 4: Installing langchain ecosystem..."
pip install langsmith==0.0.80
pip install langchain-core==0.1.10
pip install langchain==0.0.315

# Create BaseSettings proxy module to replace pydantic-settings
echo "Step 5: Creating BaseSettings proxy module..."
mkdir -p pydantic_settings
cat > pydantic_settings/__init__.py << 'PROXYMODULE'
# Proxy module for pydantic_settings
# This reimports BaseSettings from pydantic v1 to maintain compatibility
from pydantic import BaseSettings
__version__ = "proxy.1.0.0"
PROXYMODULE

# Create setup.py for the proxy module
cat > setup.py << 'SETUPFILE'
from setuptools import setup, find_packages

setup(
    name="pydantic_settings",
    version="1.0.0.dev0",  # PEP 440 compliant version
    packages=find_packages(),
)
SETUPFILE

# Install the proxy module
pip install -e .

# Install remaining dependencies
echo "Step 6: Installing other required packages..."
# Base API packages
pip install -c constraints.txt fastapi==0.115.8 starlette==0.45.3 uvicorn==0.34.0 gunicorn==21.2.0
pip install -c constraints.txt requests==2.31.0 python-multipart==0.0.20 python-dotenv==1.0.0 

# Data packages
pip install -c constraints.txt numpy==1.26.3 pandas==2.2.0 scipy==1.12.0

# Tool packages
pip install -c constraints.txt pytest==7.3.2 pytest-asyncio==0.23.4 pytest-cov==4.1.0
pip install -c constraints.txt flask==3.0.0 flask-cors==4.0.0

# Install superagi-client separately (needs pydantic v1)
pip install superagi-client==0.0.1

# Verify critical installations
echo "Verifying installations:"
python -c "import pydantic; print(f'pydantic: {pydantic.__version__}')" || echo "Pydantic not installed correctly"
python -c "import tokenizers; print(f'tokenizers: {tokenizers.__version__}')" || echo "Tokenizers not installed correctly"
python -c "import transformers; print(f'transformers: {transformers.__version__}')" || echo "Transformers not installed correctly"

# Test pydantic_settings with our proxy module
python -c "import pydantic_settings; from pydantic_settings import BaseSettings; print(f'pydantic_settings: {pydantic_settings.__version__}')" || echo "Pydantic-settings proxy not installed correctly"

python -c "import langchain; print(f'langchain: {langchain.__version__}')" || echo "Langchain not installed correctly"
python -c "import langchain_core; print(f'langchain_core: {langchain_core.__version__}')" || echo "Langchain-core not installed correctly"

# Install Python packages in correct order
pip install PyYAML==6.0.2
pip install pydantic==1.10.8
pip install pytesseract==0.3.13
pip install PyMuPDF==1.25.3

echo "Dependency resolution complete!"

# Clean up variables
cd /opt/sutazaiapp
rm -rf __pycache__

echo "Fixing SutazAI dependency conflicts..."

# Update requirements.txt
sed -i 's/langchain-core==0.1.10/langchain-core==0.1.10/' requirements.txt
sed -i 's/langchain==0.0.315/langchain==0.0.315/' requirements.txt

# Update constraints.txt as well
sed -i 's/langchain-core==0.1.10/langchain-core==0.1.10/' constraints.txt
sed -i 's/langchain==0.0.315/langchain==0.0.315/' constraints.txt
sed -i 's/langsmith==0.0.80/langsmith==0.0.80/' constraints.txt

# Add explicit fix for PyMuPDF
echo "# Ensuring fitz module (part of PyMuPDF) is properly installed" >> requirements.txt
echo "PyMuPDF==1.25.3" >> requirements.txt

# Clean venv and reinstall
cd /opt/sutazaiapp

# Install packages with fixes for dependency conflicts
pip install langchain-core==0.1.10
pip install langchain==0.0.315
pip install langsmith==0.0.80
pip install langchain-text-splitters==0.0.1

# Verify installations
python -c "import langchain; print(f'langchain: {langchain.__version__}')"
python -c "import langchain_core; print(f'langchain_core: {langchain_core.__version__}')"
python -c "import fitz; print(f'PyMuPDF/fitz: {fitz.__version__}')"

echo "Dependency conflicts fixed!" 

# Make sure PyMuPDF is properly installed by ensuring this is included:
pip install --no-cache-dir PyMuPDF==1.25.3
python -c "import fitz; print('PyMuPDF/fitz module successfully installed')" || echo "PyMuPDF still not working!" 