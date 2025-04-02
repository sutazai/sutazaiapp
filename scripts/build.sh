#!/bin/bash
set -e

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source /opt/venv-sutazaiapp/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Install test dependencies
echo "Installing test dependencies..."
pip install -r requirements-test.txt

# Create necessary directories
mkdir -p data/{documents,models,vectors}
mkdir -p logs

# Download sentence transformer model
if [ ! -d "data/models/sentence-transformers" ]; then
    echo "Downloading sentence-transformers model..."
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
fi

# Initialize database (if PostgreSQL is available)
if command -v psql &> /dev/null; then
    echo "Setting up PostgreSQL database..."
    DB_NAME=$(grep POSTGRES_DB .env | cut -d= -f2)
    DB_USER=$(grep POSTGRES_USER .env | cut -d= -f2)
    
    # Check if database exists
    if ! psql -lqt | cut -d \| -f 1 | grep -qw "$DB_NAME"; then
        echo "Creating database $DB_NAME..."
        sudo -u postgres createdb "$DB_NAME"
        sudo -u postgres createuser "$DB_USER" -P
        sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;"
    else
        echo "Database $DB_NAME already exists."
    fi
    
    # Install pgvector extension
    echo "Ensuring pgvector extension is installed..."
    sudo -u postgres psql -d "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS vector;"
else
    echo "PostgreSQL not found. Skipping database setup."
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating default .env file..."
    cp .env.example .env || {
        echo "Creating .env file from scratch..."
        cat > .env << 'ENVFILE'
# Database settings
POSTGRES_USER=sutazaiapp
POSTGRES_PASSWORD=sutazaiapp
POSTGRES_DB=sutazaiapp
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# API settings
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true

# Redis settings
REDIS_HOST=localhost
REDIS_PORT=6379

# Vector DB settings
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Document processing settings
OCR_ENABLED=true
DOCUMENT_STORE_PATH=/opt/sutazaiapp/data/documents

# Model settings
MODEL_PATH=/opt/sutazaiapp/data/models
SENTENCE_TRANSFORMER_MODEL=all-MiniLM-L6-v2
CODE_GENERATION_MODEL=codellama/CodeLlama-7b-hf
ENVFILE
    }
fi

echo "Build completed successfully." 