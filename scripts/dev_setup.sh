#!/bin/bash
# SutazAI Development Setup Script

set -e

echo "🚀 Setting up SutazAI development environment..."

# Check if we're in the right directory
if [ ! -f "requirements_comprehensive.txt" ]; then
    echo "❌ Please run this script from the SutazAI root directory"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements_comprehensive.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data logs cache models/ollama temp run backup

# Initialize database
echo "🗄️ Setting up database..."
python scripts/setup_database.py

# Install pre-commit hooks if available
if command -v pre-commit &> /dev/null; then
    echo "🔧 Installing pre-commit hooks..."
    pre-commit install
fi

echo "✅ Development environment setup complete!"
echo ""
echo "🚀 To start the application:"
echo "  source venv/bin/activate"
echo "  python scripts/deploy.py"
echo ""
echo "🐳 To use Docker:"
echo "  docker-compose up -d"
