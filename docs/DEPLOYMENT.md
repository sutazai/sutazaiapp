# Sutazaiapp Deployment Guide

## Overview
Sutazaiapp is designed with an offline-first, security-first approach, utilizing OTP gating and comprehensive deployment strategies.

## Prerequisites
- Python 3.11+
- Node.js 18+
- Git
- Bash 4.0+

## Deployment Pipeline

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/your-org/sutazaiapp.git
cd sutazaiapp

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. OTP Configuration
- Generate OTP secret using `scripts/otp_override.py`
- Configure authenticator app with provisioning URI
- Secure OTP secret in encrypted storage

### 3. Deployment Stages
Each deployment stage requires OTP validation:
- `git_pull`: Synchronize codebase
- `pip_install`: Update dependencies
- `database_migration`: Apply database changes

### 4. Deployment Command
```bash
# Deploy with OTP validation
./scripts/deploy.sh <stage_name> <otp>
```

### 5. Offline Deployment
- Pregenerate wheel files in `wheels/` directory
- Use `--no-index` for offline installation

## Security Considerations
- All external calls require OTP validation
- Comprehensive logging of deployment attempts
- Automatic rollback on failure
- Encrypted OTP management

## Troubleshooting
Refer to `docs/TROUBLESHOOTING.md` for common issues. 