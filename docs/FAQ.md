# Sutazaiapp Frequently Asked Questions

## General
### Q: What is Sutazaiapp?
A: Sutazaiapp is an advanced, offline-first AI application with comprehensive security features, document parsing, and self-improvement capabilities.

### Q: What makes Sutazaiapp unique?
A: Key features include:
- OTP-gated external calls
- Local LLM code generation
- Self-improving AI orchestrator
- Comprehensive document and diagram parsing

## Technical Setup
### Q: What are the system requirements?
A: 
- Python 3.11+
- Node.js 18+
- 16GB RAM recommended
- 100GB SSD storage

### Q: How do I set up the development environment?
A: Follow the steps in `DEPLOYMENT.md`:
1. Clone the repository
2. Create a virtual environment
3. Install dependencies
4. Configure OTP

## Security
### Q: How does the OTP system work?
A: 
- Uses time-based one-time passwords
- Validates all critical external calls
- Encrypted secret storage
- Configurable validity window

### Q: What if I lose my OTP secret?
A: 
- Use emergency recovery procedures in `SECURITY.md`
- Regenerate OTP secret
- Update all authentication points

## AI Capabilities
### Q: What local LLMs are supported?
A: 
- GPT4All
- DeepSeek Coder
- Llama-cpp models

### Q: How does the self-improvement work?
A: The AI Orchestrator:
- Analyzes system logs
- Identifies improvement opportunities
- Applies code regeneration strategies
- Creates recovery snapshots

## Troubleshooting
### Q: Where are logs stored?
A: 
- Deployment logs: `/var/log/sutazaiapp/deployment.log`
- OTP logs: `/var/log/sutazaiapp/otp_attempts.log`
- System logs: `/var/log/sutazaiapp/orchestrator.log`

### Q: How do I report an issue?
A: 
1. Check `TROUBLESHOOTING.md`
2. Collect relevant log files
3. Open a GitHub issue with detailed information

## Training and Resources
### Q: Where can I learn more?
A: 
- Review documentation in `docs/`
- Check inline code comments
- Attend onboarding sessions
- Review training materials in `docs/guides/` 