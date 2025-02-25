# Installation Guide for SutazAI

This document provides comprehensive instructions for installing and setting up the SutazAI project.

## Prerequisites

### Hardware Requirements
- **CPU**: 8+ cores (Recommended: 12 × Intel® Xeon® E5-2640 @ 2.50GHz)
- **RAM**: 32GB+ (Recommended: 128GB)
- **Storage**: 256GB SSD (Recommended: 14+ TB for model storage)
- **Server**: Dell PowerEdge R720 (recommended)

### Software Requirements
- **OS**: Ubuntu 20.04+ LTS
- **Python**: 3.8+ (Recommended: 3.10+)
- **Pip**: 23.3+ (Python package installer)
- **Node.js**: 16+
- **Git**: Latest version

## Installation Steps

### 1. System Preparation

#### 1.1 System Updates
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-venv python3-pip nodejs npm git
```

#### 1.2 User Setup (Optional but Recommended)
```bash
# Create dedicated user
sudo adduser sutazai_dev
sudo usermod -aG sudo sutazai_dev

# Set up directory permissions
cd /opt
sudo mkdir -p sutazaiapp
sudo chown -R sutazai_dev:sutazai_dev sutazaiapp
```

### 2. Repository Setup

#### 2.1 Clone the Repository
```bash
git clone https://github.com/sutazai/sutazaiapp.git
cd sutazaiapp
```

#### 2.2 Git Configuration (if needed)
If you encounter branch divergence issues:
```bash
# Choose one strategy:
git config pull.rebase false  # Merge (traditional)
# OR
git config pull.ff only       # Fast-forward only
# OR
git config pull.rebase true   # Rebase
```

### 3. Environment Setup

#### 3.1 Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
```

#### 3.2 Install Dependencies
```bash
# Upgrade basic tools
pip install --upgrade pip setuptools wheel

# Standard installation
pip install -r requirements.txt

# OR Offline installation (if packages are available locally)
pip install --no-index --find-links=packages/wheels -r requirements.txt
```

#### 3.3 Node.js Dependencies (for Web UI)
```bash
cd web_ui
npm install
cd ..
```

### 4. Configuration

#### 4.1 Environment Variables
Create a `.env` file in the project root:
```
# OTP Configuration
OTP_SECRET_KEY=your_secret_key
ROOT_USER_EMAIL=chrissuta01@gmail.com

# Network Access
DEFAULT_NET_ACCESS=disabled
```

#### 4.2 OTP Setup (for External Operations)
The system uses OTP for security when accessing external resources:
```bash
# Generate a new OTP secret (if needed)
python -c "import pyotp; print(pyotp.random_base32())"

# Update the OTP_SECRET_KEY in .env with the generated value
```

### 5. System Initialization

#### 5.1 Initialize the System
```bash
python core_system/system_orchestrator.py
```

#### 5.2 Verify Installation
```bash
python scripts/test_pipeline.py
```

### 6. Running the Application

#### 6.1 Start the Backend
```bash
cd backend
python main.py
```

#### 6.2 Start the Web UI (in a separate terminal)
```bash
cd web_ui
npm run dev
```

#### 6.3 Access the Application
- Backend API: http://localhost:8000
- Web UI: http://localhost:3000

## Offline Installation

For environments without internet access:

### Pre-download Dependencies
On a machine with internet access:
```bash
# Python packages
pip download -r requirements.txt -d packages/wheels

# Node.js packages
cd web_ui
npm pack $(npm list --prod --parseable | sed 's/.*node_modules\///')
mv *.tgz ../packages/node/
cd ..
```

### Install from Local Packages
```bash
# Python packages
pip install --no-index --find-links=packages/wheels -r requirements.txt

# Node.js packages
cd web_ui
npm install --offline ../packages/node/*.tgz
cd ..
```

## Troubleshooting

### Common Issues

#### Python Version Conflicts
- Ensure you're using Python 3.8+ (preferably 3.10+)
- Check with: `python --version`

#### Dependency Installation Failures
- Try installing with `--no-deps` flag: `pip install -r requirements.txt --no-deps`
- Check for specific package errors in the output

#### Permission Issues
- Ensure proper ownership: `sudo chown -R sutazai_dev:sutazai_dev /opt/sutazaiapp`
- Set correct permissions: `chmod -R 750 /opt/sutazaiapp`

#### Git Issues
- If branches have diverged: `git config pull.rebase false && git pull origin master`
- For clean slate: `git fetch --all && git reset --hard origin/master`

## Additional Resources
- [SutazAI Master Plan](/docs/SUTAZAI_MASTER_PLAN.md)
- [Deployment Guide](/docs/DEPLOYMENT_GUIDE.md)
- [System Architecture](/docs/SYSTEM_ARCHITECTURE.md)
- [Security Policy](/docs/SECURITY_POLICY.md)