#!/bin/bash

# Script to install Python dependencies for SutazAI

# Ensure script is run from the project root directory
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
cd "$PROJECT_ROOT" || exit 1

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt not found in project root."
    exit 1
fi

echo "Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# Check for installation errors
if [ $? -ne 0 ]; then
    echo "Error installing dependencies. Please check the output above."
    exit 1
fi

# Install system dependencies for Faiss, OCR, etc. if needed (example for Ubuntu)
# Check if running as root or using sudo
# if [[ $EUID -ne 0 ]]; then
#    echo "Some system dependencies might require sudo privileges to install."
#    # sudo apt-get update
#    # sudo apt-get install -y build-essential cmake libomp-dev # For FAISS
#    # sudo apt-get install -y tesseract-ocr # For OCR (pytesseract wrapper)
# fi

echo "Dependency installation completed successfully."

# Deactivate virtual environment if we activated it
if command -v deactivate &> /dev/null; then
    # Check if we are actually in a virtual env managed by this script
    # This check is basic, might need refinement
    if [[ "$VIRTUAL_ENV" == "$PROJECT_ROOT/venv" || "$VIRTUAL_ENV" == "$PROJECT_ROOT/.venv" ]]; then
      echo "Deactivating virtual environment."
      deactivate
    fi
fi

exit 0 