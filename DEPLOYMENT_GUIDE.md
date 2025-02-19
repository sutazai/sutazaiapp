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

#### 2.1 Clone Repository
```bash
mkdir -p /opt/sutazai_project
cd /opt/sutazai_project
git clone https://github.com/Readit2go/SutazAI.git
cd SutazAI
```

#### 2.2 Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Dependency Installation

#### 3.1 Python Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
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

### 5. Deployment Execution

#### 5.1 Run Deployment Script
```bash
./scripts/deploy.sh
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

## Maintenance

### Regular Tasks
- Weekly dependency updates
- Monthly security scans
- Quarterly comprehensive system audit

## Rollback Procedure
```bash
./scripts/setup_repos.sh  # Manual repository sync
```

## Contact Support
- **Primary Contact**: Florin Cristian Suta
- **Email**: chrissuta01@gmail.com
- **Phone**: +48517716005 