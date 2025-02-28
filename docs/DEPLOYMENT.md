# SutazAI Deployment Guide

## Prerequisites

### System Requirements
- Python 3.11 or higher
- Node.js 18.x or higher (for web_ui)
- Git
- PostgreSQL 14.x or higher
- Redis (optional, for caching)

### Server Requirements
- Code Server (192.168.100.28)
- Deployment Server (192.168.100.100)
- Minimum 4GB RAM
- 20GB available disk space

## Initial Setup

### 1. User Creation
```bash
# Create deployment user
sudo adduser sutazaiapp_dev
sudo usermod -aG sudo sutazaiapp_dev

# Switch to the user
su - sutazaiapp_dev
```

### 2. Repository Setup
```bash
# Navigate to opt directory
cd /opt

# Create and set permissions for sutazaiapp directory
sudo mkdir sutazaiapp
sudo chown -R sutazaiapp_dev:sutazaiapp_dev sutazaiapp

# Clone the repository
git clone https://sutazaiapp:github_token@github.com/sutazai/sutazaiapp.git
cd sutazaiapp

# Configure git pull strategy
git config pull.rebase false
```

### 3. Python Environment Setup
```bash
# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip and essential tools
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt

# Verify installations
pip freeze > installed_packages.txt
```

### 4. Frontend Setup (if applicable)
```bash
# Navigate to web_ui directory
cd web_ui

# Install Node.js dependencies
npm install

# Build frontend
npm run build
```

### 5. Directory Structure Setup
```bash
# Create required directories
mkdir -p ai_agents model_management backend web_ui scripts packages/wheels logs doc_data docs

# Set appropriate permissions
chmod -R 750 /opt/sutazaiapp
```

## Configuration

### 1. Environment Variables
```bash
# Copy example environment file
cp .env.example .env

# Edit environment variables
nano .env
```

### 2. Database Setup
```bash
# Create database
sudo -u postgres createdb sutazaidb

# Apply migrations
python scripts/db_migrate.py
```

### 3. Model Management
```bash
# Initialize model storage
python scripts/init_model_storage.py
```

## Deployment

### 1. Backend Deployment
```bash
# Start the backend server
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 2. Frontend Deployment (if applicable)
```bash
# Start the frontend server
cd web_ui
npm run serve
```

### 3. AI Agents Deployment
```bash
# Initialize AI agents
python scripts/init_ai_agents.py
```

## Health Checks

### 1. System Health
```bash
# Run system health check
python scripts/system_health_check.py
```

### 2. Model Health
```bash
# Verify model integrity
python scripts/verify_models.py
```

## Monitoring

### 1. Logs
- Application logs: `/opt/sutazaiapp/logs/app.log`
- Error logs: `/opt/sutazaiapp/logs/error.log`
- Access logs: `/opt/sutazaiapp/logs/access.log`

### 2. Metrics
```bash
# View system metrics
python scripts/view_metrics.py
```

## Backup and Recovery

### 1. Database Backup
```bash
# Create database backup
python scripts/backup_db.py
```

### 2. Model Backup
```bash
# Backup model files
python scripts/backup_models.py
```

## Troubleshooting

### Common Issues
1. Port conflicts
   ```bash
   # Check port usage
   sudo lsof -i :8000
   ```

2. Permission issues
   ```bash
   # Fix permissions
   sudo chown -R sutazaiapp_dev:sutazaiapp_dev /opt/sutazaiapp
   ```

3. Virtual environment issues
   ```bash
   # Recreate virtual environment
   rm -rf venv
   python3.11 -m venv venv
   ```

## Security Notes
- Keep `.env` file secure and never commit it to version control
- Regularly update dependencies
- Monitor system logs for suspicious activities
- Follow principle of least privilege for file permissions

## Maintenance

### Regular Tasks
1. Update dependencies monthly
2. Backup database weekly
3. Rotate logs daily
4. Monitor disk usage
5. Check system health daily

### Update Procedure
```bash
# Pull latest changes
git pull origin master

# Update dependencies
pip install -r requirements.txt

# Apply migrations
python scripts/db_migrate.py

# Restart services
sudo systemctl restart sutazai
``` 