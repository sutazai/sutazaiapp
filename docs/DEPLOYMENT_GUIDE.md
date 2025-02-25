# SutazAI Deployment Guide

## Prerequisites

### Hardware Requirements
- **Server**: Dell PowerEdge R720
- **CPU**: 12 × Intel® Xeon® E5-2640 @ 2.50GHz
- **RAM**: ~128GB
- **Storage**: ~14.31 TB

### Software Requirements
- **OS**: Ubuntu 20.04+ LTS
- **Python**: 3.8+ (Recommended 3.10+)
- **Node.js**: 16+ 

## Deployment Architecture

SutazAI uses a two-server architecture:
- **Code Server (192.168.100.28)**: Development and code management
- **Deployment Server (192.168.100.100)**: Production environment

## Deployment Steps

### 1. Environment Preparation

#### 1.1 System Update
```bash
sudo apt update && sudo apt upgrade -y
```

#### 1.2 Install Dependencies
```bash
sudo apt install -y python3-venv python3-pip nodejs npm
```

### 2. Repository Setup

#### 2.1 User Setup (if needed)
```bash
# Create dedicated user
sudo adduser sutazai_dev
sudo usermod -aG sudo sutazai_dev

# Set permissions
cd /opt
sudo mkdir -p sutazaiapp
sudo chown -R sutazai_dev:sutazai_dev sutazaiapp
```

#### 2.2 Clone Repository
```bash
mkdir -p /opt/sutazai_project
cd /opt/sutazai_project
git clone https://github.com/sutazai/sutazaiapp.git
cd sutazaiapp
```

#### 2.3 Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Dependency Installation

#### 3.1 Python Dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt || \
    pip install --no-index --find-links=packages/wheels -r requirements.txt
```

#### 3.2 Node.js Dependencies
```bash
cd web_ui
npm install
cd ..
```

### 4. Configuration

#### 4.1 Environment Variables
Create a `.env` file with necessary configurations:
```
# OTP Configuration
OTP_SECRET_KEY=your_secret_key
ROOT_USER_EMAIL=chrissuta01@gmail.com

# Network Access
DEFAULT_NET_ACCESS=disabled
```

#### 4.2 Server Synchronization (Two-Server Setup)
On Code Server as sutazai_dev:
```bash
ssh-keygen -t ed25519 -C "sutazai_deploy" -f ~/.ssh/sutazai_deploy -N ""
ssh-copy-id -i ~/.ssh/sutazai_deploy.pub root@192.168.100.100
```

Set up Git hook for automatic deployment:
```bash
# In /opt/sutazaiapp/.git/hooks/post-commit
#!/bin/bash
ssh root@192.168.100.100 "cd /opt/sutazaiapp && ./scripts/trigger_deploy.sh"

chmod +x .git/hooks/post-commit
```

### 5. Deployment Execution

#### 5.1 Run Deployment Script
```bash
./scripts/deploy.sh
```

The deployment script handles:
- Git pull from repository
- OTP validation for external operations
- Virtual environment setup
- Dependency installation (with offline fallback)
- System verification
- Service startup
- Health checks

#### 5.2 OTP-Based Security
External operations (network access, package installations) require OTP validation:
```bash
# Example of using OTP for deployment with external dependencies
OTP_TOKEN=123456 ./scripts/deploy.sh
```

### 6. Verification

#### 6.1 System Health Check
```bash
python scripts/test_pipeline.py
```

#### 6.2 Access Web Interface
- **URL**: http://[DEPLOYMENT_SERVER_IP]:8000
- **Initial Login**: OTP-based authentication

## Troubleshooting

### Common Issues
1. **Dependency Conflicts**
   - Ensure exact versions in `requirements.txt`
   - Use `pip install -r requirements.txt --no-deps`

2. **Network Access Restrictions**
   - Verify OTP mechanism
   - Check `online_calls.log` for details

3. **Model Loading Failures**
   - Confirm model files in `model_management/`
   - Verify model compatibility

4. **Git Branch Divergence**
   - Configure pull strategy: `git config pull.rebase false`
   - Or use: `git pull origin master --no-rebase`

### Rollback Procedure
If deployment fails or causes issues:
```bash
# Automatic rollback (triggered by deploy.sh on error)
# Manual rollback to specific commit
./scripts/deploy.sh --rollback <COMMIT_HASH>

# Complete repository reset
./scripts/setup_repos.sh
```

## Maintenance

### Regular Tasks
- Weekly dependency updates
- Monthly security scans
- Quarterly comprehensive system audit

### Monitoring
- Check logs in `/opt/sutazaiapp/logs/`
- Monitor system performance via Prometheus/Grafana (if configured)

## Contact Support
- **Primary Contact**: Florin Cristian Suta
- **Email**: chrissuta01@gmail.com
- **Phone**: +48517716005 

## Additional Resources
- [SutazAI Master Plan](/docs/SUTAZAI_MASTER_PLAN.md)
- [System Architecture](/docs/SYSTEM_ARCHITECTURE.md)
- [Security Policy](/docs/SECURITY_POLICY.md) 