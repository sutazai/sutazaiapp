# SutazAI - Autonomous AI Development Platform

## Project Overview
SutazAI is an advanced, self-improving AI development platform designed to push the boundaries of artificial intelligence. This system is built with a focus on autonomy, security, and comprehensive AI capabilities.

## System Architecture
- **Two-Server Approach**:
  1. Code Server (sutazaicodeserver): 192.168.100.136
  2. Deployment Server (sutazaideploymenttestserver): 192.168.100.178

## Project Structure

```
/opt/sutazai/
├── ai_agents/                 # AI agent implementations
│   ├── supreme_ai/            # Supreme AI orchestrator
│   ├── auto_gpt/              # AutoGPT agent
│   └── ...
├── model_management/          # Local AI model storage
│   ├── gpt4all/
│   ├── deepseek/
│   └── ...
├── backend/                   # Core backend services
│   ├── main.py                # Application entry point
│   ├── api_routes.py          # API endpoints
│   └── services/              # Business logic
├── web_ui/                    # Frontend interface
│   ├── src/                   # Source code
│   └── package.json           # Node.js dependencies
├── scripts/                   # Deployment & utility scripts
│   ├── deploy.sh              # Main deployment script
│   ├── otp_manager.py         # OTP generation
│   └── test_pipeline.py       # Comprehensive testing
└── ...
```

## Key Features
- Autonomous Code Generation
- Self-Improvement Mechanisms
- OTP-Based Net Access Control
- Advanced Document & Diagram Processing
- Comprehensive Security Hardening

## System Requirements
- **Hardware**: Dell PowerEdge R720
  - 12 × Intel® Xeon® E5-2640 @ 2.50GHz
  - ~128GB RAM
  - ~14.31 TB Storage
- **OS**: Linux (Ubuntu 20.04+)
- **Python**: 3.8+
- **Dependencies**: See `requirements.txt`

## Deployment
1. Clone Repository

```bash
git clone https://github.com/Readit2go/SutazAI.git
cd SutazAI
```

2. Setup Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Run Deployment

```bash
./scripts/deploy.sh
```

## Security Model
- Root User: Florin Cristian Suta
- Net Access: OTP-Based Approval
- No External API Reliance
- Comprehensive Logging

## Contribution
1. Fork Repository
2. Create Feature Branch
3. Commit Changes
4. Create Pull Request

## Contact
- Creator: Florin Cristian Suta
- Email: <chrissuta01@gmail.com>
- Phone: +48517716005

## License
Proprietary - All Rights Reserved
