# SutazaiApp Deployment Guide

This document provides comprehensive instructions for deploying SutazaiApp with Python 3.11.

## Prerequisites

- Python 3.11.x installed
- `sudo` privileges for system service setup
- Required system packages installed

## System Requirements

- OS: Ubuntu or compatible Linux distribution
- RAM: Minimum 4GB, recommended 8GB+
- Disk: Minimum 20GB free space
- CPU: 2+ cores recommended

## Quick Deployment

For a full automated deployment using all existing scripts:

```bash
# Go to the application directory
cd /opt/sutazaiapp

# Run the comprehensive deployment script
./scripts/deploy_all.sh
```

The deploy_all.sh script handles:
1. Environment checking
2. Stopping any running services
3. Setting up directories and permissions
4. Verifying the virtual environment
5. Starting all application components
6. Setting up systemd services
7. Starting the Web UI (if available)
8. Verifying the deployment
9. Running final health checks

## Manual Installation

### 1. Clone the Repository (if not already done)

```bash
git clone https://github.com/yourusername/sutazaiapp.git /opt/sutazaiapp
cd /opt/sutazaiapp
```

### 2. Set Up Python 3.11 Virtual Environment

```bash
# Create virtual environment with Python 3.11
python3.11 -m venv /opt/venv-sutazaiapp

# Create symbolic link in the app directory
ln -sf /opt/venv-sutazaiapp /opt/sutazaiapp/venv

# Activate the environment
source /opt/sutazaiapp/venv/bin/activate
```

### 3. Install Dependencies

```bash
# Upgrade pip and install base packages
pip install --upgrade pip setuptools wheel

# Install requirements
pip install -r /opt/sutazaiapp/requirements.txt

# Install extra packages if needed
pip install gunicorn uvicorn fastapi
```

### 4. Create Required Directories

```bash
# Create all necessary directories
sudo mkdir -p /opt/sutazaiapp/logs /opt/sutazaiapp/run /opt/sutazaiapp/tmp /opt/sutazaiapp/data/documents /opt/sutazaiapp/data/models /opt/sutazaiapp/data/vectors

# Set proper permissions
sudo chmod -R 755 /opt/sutazaiapp/logs /opt/sutazaiapp/run /opt/sutazaiapp/tmp /opt/sutazaiapp/data
```

### 5. Setup Systemd Services

Copy the service files to systemd:

```bash
sudo cp /opt/sutazaiapp/systemd/sutazaiapp.service /etc/systemd/system/
sudo cp /opt/sutazaiapp/systemd/sutazai-orchestrator.service /etc/systemd/system/
sudo systemctl daemon-reload
```

Enable and start the services:

```bash
sudo systemctl enable sutazaiapp sutazai-orchestrator
sudo systemctl start sutazaiapp
sudo systemctl start sutazai-orchestrator
```

## Configuration

The application configuration is stored in the `.env` file. Copy the example if needed:

```bash
cp .env.example .env
```

Edit the `.env` file to set your specific configuration:

- Database settings
- API settings
- Document processing settings
- Model paths

## Verifying the Deployment

### Using the Check Status Script

The app includes a status checking script:

```bash
/opt/sutazaiapp/scripts/check_status.sh
```

### Manual Verification

Check if API is running:

```bash
curl http://localhost:8000/health
```

Check service status:

```bash
sudo systemctl status sutazaiapp
sudo systemctl status sutazai-orchestrator
```

## Troubleshooting

### Service Fails to Start

Check logs:

```bash
sudo journalctl -u sutazaiapp -n 50
sudo journalctl -u sutazai-orchestrator -n 50
```

Verify Python version:

```bash
source /opt/sutazaiapp/venv/bin/activate
python --version  # Should be Python 3.11.x
```

### API Not Accessible

Check if the service is running:

```bash
sudo systemctl status sutazaiapp
```

Verify ports are not blocked:

```bash
sudo netstat -tulpn | grep 8000
```

### Python Version Issues

Ensure Python 3.11 is properly installed:

```bash
python3.11 --version
```

If needed, install Python 3.11:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev
```

## Starting and Stopping Services

### Start Services

```bash
sudo systemctl start sutazaiapp sutazai-orchestrator
```

### Stop Services

```bash
sudo systemctl stop sutazaiapp sutazai-orchestrator
```

### Restart Services

```bash
sudo systemctl restart sutazaiapp sutazai-orchestrator
```

## Alternative Startup Methods

### Using start.sh Script

You can start the application directly using the start.sh script:

```bash
/opt/sutazaiapp/scripts/start.sh
```

This script:
- Starts the Qdrant vector database (if Docker is available)
- Launches the AI orchestrator
- Starts the backend API server

## Log Files

- Backend API logs: `/opt/sutazaiapp/logs/backend.log`
- Orchestrator logs: `/opt/sutazaiapp/logs/orchestrator.log`
- Access logs: `/opt/sutazaiapp/logs/access.log`
- Error logs: `/opt/sutazaiapp/logs/error.log`

## Components

- Backend API (FastAPI)
- AI Orchestrator
- Web UI (optional)
- Qdrant Vector DB (optional)

## Directory Structure

```
/opt/sutazaiapp/
├── ai_agents/         # AI agent code
├── backend/           # Backend API
├── config/            # Configuration files
├── data/              # Data files
│   ├── documents/     # Document storage
│   ├── models/        # AI models
│   └── vectors/       # Vector database files
├── docs/              # Documentation
├── logs/              # Log files
├── model_management/  # Model management code
├── monitoring/        # Monitoring code
├── scripts/           # Deployment scripts
├── systemd/           # Systemd service files
├── tests/             # Test code
├── web_ui/            # Web UI code
├── venv               # Symbolic link to virtual environment
├── .env               # Environment configuration
└── requirements.txt   # Python dependencies
``` 