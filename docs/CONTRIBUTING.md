# Contributing to SutazAI

## 🌟 Welcome Contributors!

Thank you for your interest in contributing to SutazAI, an ultra-comprehensive autonomous AI development platform. We value your contributions and want to make the process as smooth and transparent as possible.

## 📋 Table of Contents
1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Workflow](#development-workflow)
4. [Coding Standards](#coding-standards)
5. [Pull Request Process](#pull-request-process)
6. [Code Review Process](#code-review-process)
7. [Reporting Bugs](#reporting-bugs)
8. [Feature Requests](#feature-requests)

## 🤝 Code of Conduct

We are committed to providing a friendly, safe, and welcoming environment for all contributors. Our project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## 🚀 Getting Started

### Prerequisites
- Python 3.10 - 3.12
- Git
- Virtual environment tool (venv/conda)

### Setup Development Environment
1. Fork the repository
2. Clone your fork
```bash
git clone https://github.com/sutazai/sutazaiapp.git
cd sutazaiapp
```

3. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

4. Install development dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

5. Configure Git for the two-server architecture (if applicable)
```bash
# Set up Git pull strategy
git config pull.rebase false  # or choose another strategy

# If you're working on the Code Server (192.168.100.28)
# Set up SSH key for deployment server sync
ssh-keygen -t ed25519 -C "sutazai_deploy" -f ~/.ssh/sutazai_deploy -N ""
ssh-copy-id -i ~/.ssh/sutazai_deploy.pub root@192.168.100.100
```

## 🔧 Development Workflow

### Project Architecture
SutazAI follows a comprehensive development plan outlined in the [SutazAI Master Plan](/docs/SUTAZAI_MASTER_PLAN.md). Please familiarize yourself with this document to understand the project's phases and architecture.

### Two-Server Architecture
- **Code Server (192.168.100.28)**: Development and code management
- **Deployment Server (192.168.100.100)**: Production environment

### Branch Strategy
- `main`: Stable release branch
- `develop`: Integration branch for upcoming release
- Feature branches: `feature/your-feature-name`
- Bugfix branches: `bugfix/issue-description`

### Creating a Branch
```bash
git checkout -b feature/your-feature-name develop
```

External operations (network access, package installations) require OTP validation. When developing features that require external access:

1. Request an OTP token from the project maintainer
2. Use the token when running scripts that require external access:
```bash
OTP_TOKEN=123456 ./scripts/your_script.py
```

3. For testing, you can use the development OTP secret in your local environment

## 📝 Coding Standards

### Python Style Guide
- Follow PEP 8 guidelines
- Use type hints
- Write docstrings for all functions and classes
- Maximum line length: 120 characters

### Code Quality Tools
- Use `black` for code formatting
- Use `isort` for import sorting
- Use `flake8` for linting
- Use `mypy` for static type checking

### Pre-commit Hooks
Install pre-commit hooks to ensure code quality:
```bash
pre-commit install
```

## 🔍 Pull Request Process

1. Ensure your code follows all coding standards
2. Update documentation and docstrings
3. Add/update tests for new functionality
4. Ensure all tests pass
5. Create a pull request with a clear title and description

### Pull Request Template
```markdown
## Description
[Provide a clear and concise description of your changes]

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## How Tested
[Describe the tests performed to verify changes]

## Checklist
- [ ] I have performed a self-review of my code
- [ ] I have added tests
- [ ] Documentation updated
- [ ] Code follows project style guidelines
```

## 🕵️ Code Review Process

- All submissions require review
- At least two maintainers must approve the pull request
- Automated checks must pass
- Constructive feedback is encouraged

## 🐛 Reporting Bugs

### Bug Report Template
```markdown
**Describe the bug**
[A clear description of the bug]

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. See error

**Expected Behavior**
[What you expected to happen]

**Screenshots**
[If applicable, add screenshots]

**Environment**
- OS: [e.g. Ubuntu 20.04]
- Python Version: [e.g. 3.10.12]
- SutazAI Version: [e.g. 2.1.2]
```

## ✨ Feature Requests

1. Check existing issues to avoid duplicates
2. Provide a clear and detailed explanation
3. Include potential implementation details
4. Discuss potential impact on the project


### Reporting Process
- Include detailed information about the vulnerability
- Provide steps to reproduce (if possible)

## CI/CD Pipeline

SutazAI uses a comprehensive CI/CD pipeline for continuous testing and deployment:

### Pipeline Stages
1. **Static Analysis**: Code is analyzed using Semgrep, Bandit, and other tools
2. **Unit/Integration Tests**: Automated tests are run using Pytest
3. **Performance Tests**: Performance benchmarks are executed
5. **Deployment**: If all tests pass, code is deployed to the Deployment Server

### Monitoring Your PR
After submitting a PR, you can monitor its progress through the CI/CD pipeline. The pipeline will:
- Run automated tests
- Check code quality
- Test performance
- Deploy to staging (if applicable)

### Deployment
Deployment is handled automatically through the CI/CD pipeline. The `deploy.sh` script:
- Pulls the latest code
- Validates OTP for external operations
- Sets up the environment
- Installs dependencies
- Verifies system health
- Starts services

## 📊 Contribution Statistics

We track and appreciate all contributions. Top contributors may receive:
- Recognition in project documentation
- Potential collaboration opportunities
- Early access to experimental features

## 🏆 Contributor Levels

- 🥉 Bronze: 1-5 accepted PRs
- 🥈 Silver: 6-20 accepted PRs
- 🥇 Gold: 21+ accepted PRs
- 💎 Platinum: Significant architectural contributions

## 📞 Contact

Lead Developer: Florin Cristian Suta
- Email: chrissuta01@gmail.com
- Phone: +48517716005

---

*Together, we build the future of autonomous AI systems* 🚀 