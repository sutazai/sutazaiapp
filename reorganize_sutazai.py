#!/usr/bin/env python3
"""
SutazAI Application: Comprehensive Reorganization and Optimization Script

This script implements the complete reorganization, bug fixes, code optimization,
infrastructure improvements, and documentation processes for the SutazAI system.
"""

import os
import sys
import json
import shutil
import tempfile
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("reorganization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("reorganize")

# Base directory
BASE_DIR = Path("/opt/sutazaiapp")
CURRENT_DIR = Path(os.getcwd())


class SutazAIOrganizer:
    """Main class to organize and optimize the SutazAI system."""
    
    def __init__(self, base_dir: Path = BASE_DIR):
        """Initialize the organizer with the base directory."""
        self.base_dir = base_dir
        self.status = {
            "directory_setup": False,
            "virtual_env": False,
            "test_fixes": False,
            "code_analysis": False,
            "cicd_pipelines": False,
            "monitoring": False,
            "documentation": False
        }
        self.errors = []
        self.warnings = []
    
    def safe_run_command(self, cmd: List[str], cwd: Optional[Path] = None) -> Tuple[int, str, str]:
        """Safely run a command and return its exit code, stdout, and stderr."""
        try:
            cwd = cwd or self.base_dir
            logger.info(f"Running command: {' '.join(cmd)} in {cwd}")
            
            process = subprocess.run(
                cmd,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                check=False
            )
            
            if process.returncode != 0:
                logger.warning(f"Command failed with exit code {process.returncode}")
                logger.warning(f"stderr: {process.stderr}")
                self.warnings.append(f"Command failed: {' '.join(cmd)}")
                self.warnings.append(f"Error: {process.stderr}")
            
            return process.returncode, process.stdout, process.stderr
        except Exception as e:
            logger.error(f"Exception while running command: {e}")
            self.errors.append(f"Exception while running command: {e}")
            return 1, "", str(e)
    
    def setup_directories(self) -> bool:
        """Set up the directory structure based on the plan."""
        logger.info("Setting up directory structure...")
        
        # List of directories to create
        directories = [
            "ai_agents",
            "model_management",
            "backend",
            "web_ui",
            "scripts",
            "packages/wheels",
            "logs",
            "doc_data",
            "docs",
            "docs/audit"
        ]
        
        try:
            for directory in directories:
                dir_path = self.base_dir / directory
                if not dir_path.exists():
                    logger.info(f"Creating directory: {dir_path}")
                    dir_path.mkdir(parents=True, exist_ok=True)
                else:
                    logger.info(f"Directory already exists: {dir_path}")
            
            # Set permissions
            logger.info("Setting directory permissions...")
            self.safe_run_command(["chmod", "-R", "750", str(self.base_dir)])
            
            self.status["directory_setup"] = True
            return True
        except Exception as e:
            logger.error(f"Error setting up directories: {e}")
            self.errors.append(f"Directory setup failed: {e}")
            return False
    
    def setup_virtual_env(self) -> bool:
        """Set up the Python virtual environment and install dependencies."""
        logger.info("Setting up virtual environment...")
        
        venv_path = self.base_dir / "venv"
        if venv_path.exists():
            logger.info("Virtual environment already exists")
        else:
            logger.info("Creating new virtual environment...")
            exit_code, stdout, stderr = self.safe_run_command(
                ["python3.11", "-m", "venv", "venv"]
            )
            if exit_code != 0:
                return False
        
        # Install dependencies
        logger.info("Installing dependencies...")
        
        pip_path = venv_path / "bin" / "pip"
        if not pip_path.exists():
            logger.error(f"Pip not found at expected location: {pip_path}")
            self.errors.append("Pip not found in virtual environment")
            return False
        
        commands = [
            [str(pip_path), "install", "--upgrade", "pip", "setuptools", "wheel"],
            [str(pip_path), "install", "-r", str(self.base_dir / "requirements.txt")]
        ]
        
        for cmd in commands:
            exit_code, stdout, stderr = self.safe_run_command(cmd)
            if exit_code != 0:
                self.errors.append(f"Failed to install dependencies: {stderr}")
                return False
        
        # Set up Node.js dependencies if web_ui/package.json exists
        package_json = self.base_dir / "web_ui" / "package.json"
        if package_json.exists():
            logger.info("Installing Node.js dependencies...")
            exit_code, stdout, stderr = self.safe_run_command(
                ["npm", "install"],
                cwd=self.base_dir / "web_ui"
            )
            if exit_code != 0:
                self.warnings.append("Failed to install Node.js dependencies")
        
        self.status["virtual_env"] = True
        return True
    
    def fix_test_issues(self) -> bool:
        """Fix test issues using the Python scripts we've created."""
        logger.info("Fixing test issues...")
        
        # Run each of our test fix scripts
        try:
            # 1. Fix indentation and decorators
            run_module_path = str(CURRENT_DIR / "fix_indentation_and_decorators.py")
            if os.path.exists(run_module_path):
                logger.info("Running fix_indentation_and_decorators.py")
                exit_code, stdout, stderr = self.safe_run_command(
                    ["python", run_module_path]
                )
                if exit_code != 0:
                    self.warnings.append("Failed to fix indentation and decorators")
            
            # 2. Fix coroutine warnings
            run_module_path = str(CURRENT_DIR / "fix_coroutine_warnings.py")
            if os.path.exists(run_module_path):
                logger.info("Running fix_coroutine_warnings.py")
                exit_code, stdout, stderr = self.safe_run_command(
                    ["python", run_module_path]
                )
                if exit_code != 0:
                    self.warnings.append("Failed to fix coroutine warnings")
            
            # 3. Fix sync exception test
            run_module_path = str(CURRENT_DIR / "fix_sync_exception_test.py")
            if os.path.exists(run_module_path):
                logger.info("Running fix_sync_exception_test.py")
                exit_code, stdout, stderr = self.safe_run_command(
                    ["python", run_module_path]
                )
                if exit_code != 0:
                    self.warnings.append("Failed to fix sync exception test")
            
            # 4. Set up pytest configuration
            run_module_path = str(CURRENT_DIR / "setup_pytest_config.py")
            if os.path.exists(run_module_path):
                logger.info("Running setup_pytest_config.py")
                exit_code, stdout, stderr = self.safe_run_command(
                    ["python", run_module_path]
                )
                if exit_code != 0:
                    self.warnings.append("Failed to set up pytest configuration")
            
            # 5. Verify fixes
            run_module_path = str(CURRENT_DIR / "verify_fixes.py")
            if os.path.exists(run_module_path):
                logger.info("Running verify_fixes.py")
                exit_code, stdout, stderr = self.safe_run_command(
                    ["python", run_module_path]
                )
                if exit_code != 0:
                    self.warnings.append("Test verification reported issues")
            
            self.status["test_fixes"] = True
            return True
        except Exception as e:
            logger.error(f"Error fixing test issues: {e}")
            self.errors.append(f"Test fixes failed: {e}")
            return False
    
    def run_code_analysis(self) -> bool:
        """Run automated code analysis tools."""
        logger.info("Running code analysis...")
        
        # Activate the virtual environment for running tools
        venv_bin = self.base_dir / "venv" / "bin"
        if not venv_bin.exists():
            logger.error("Virtual environment bin directory not found")
            self.errors.append("Virtual environment not properly set up")
            return False
        
        # List of analysis tools to run
        analysis_tools = [
            [str(venv_bin / "semgrep"), "--config=auto", "."],
            [str(venv_bin / "pylint"), "backend/"],
            [str(venv_bin / "mypy"), "."],
            [str(venv_bin / "bandit"), "-r", "."]
        ]
        
        analysis_results = {}
        
        for tool_cmd in analysis_tools:
            tool_name = os.path.basename(tool_cmd[0])
            logger.info(f"Running {tool_name}...")
            
            exit_code, stdout, stderr = self.safe_run_command(tool_cmd)
            
            analysis_results[tool_name] = {
                "exit_code": exit_code,
                "output": stdout + stderr
            }
            
            # Log the summary
            if exit_code != 0:
                logger.warning(f"{tool_name} reported issues")
                self.warnings.append(f"{tool_name} reported issues")
        
        # Save analysis results to a file
        audit_dir = self.base_dir / "docs" / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        
        with open(audit_dir / "Audit_Report_v1.md", "w") as f:
            f.write("# Code Audit Report v1\n\n")
            f.write("## Automated Analysis Results\n\n")
            
            for tool_name, result in analysis_results.items():
                f.write(f"### {tool_name}\n\n")
                f.write(f"Exit code: {result['exit_code']}\n\n")
                f.write("```\n")
                f.write(result['output'][:10000])  # Truncate very long outputs
                if len(result['output']) > 10000:
                    f.write("\n... (output truncated) ...\n")
                f.write("```\n\n")
        
        self.status["code_analysis"] = True
        return True
    
    def setup_cicd_pipeline(self) -> bool:
        """Set up CI/CD pipeline and deployment scripts."""
        logger.info("Setting up CI/CD pipeline...")
        
        # Create deployment script
        deploy_script = self.base_dir / "scripts" / "deploy.sh"
        
        with open(deploy_script, "w") as f:
            f.write("""#!/bin/bash
set -euo pipefail

echo "Updating code..."
git pull origin master

source venv/bin/activate
pip install -r requirements.txt

cd web_ui && npm install

echo "Starting services..."
python3 backend/main.py &
npm start --prefix web_ui &
""")
        
        # Make the script executable
        os.chmod(deploy_script, 0o755)
        
        # Create sync script
        sync_script = self.base_dir / "scripts" / "sync_servers.sh"
        
        with open(sync_script, "w") as f:
            f.write("""#!/bin/bash
ssh root@192.168.100.100 "cd /opt/sutazaiapp && git pull origin master"
""")
        
        # Make the script executable
        os.chmod(sync_script, 0o755)
        
        # Document the CI/CD process
        deploy_doc = self.base_dir / "docs" / "DEPLOYMENT.md"
        
        with open(deploy_doc, "w") as f:
            f.write("""# Deployment Guide

This document outlines the deployment process for the SutazAI application.

## Prerequisites

1. Python 3.11 or higher
2. Node.js 18 or higher (for web UI)
3. PostgreSQL 14 or higher
4. Redis 6 or higher

## System Requirements

- CPU: 4+ cores
- RAM: 8GB minimum, 16GB recommended
- Storage: 50GB minimum
- OS: Ubuntu 20.04 LTS or higher

## Installation Steps

1. Clone the Repository:
   ```bash
   cd /opt
   sudo mkdir sutazaiapp
   sudo chown -R sutazaiapp_dev:sutazaiapp_dev sutazaiapp
   cd sutazaiapp
   git clone https://sutazaiapp:github_token@github.com/sutazai/sutazaiapp.git .
   ```

2. Set Up Python Virtual Environment:
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```

3. Install Node.js Dependencies:
   ```bash
   cd web_ui
   npm install
   ```

## Running the Application

1. Start Backend Server:
   ```bash
   cd /opt/sutazaiapp
   source venv/bin/activate
   python -m backend.backend_main
   ```

2. Start Web UI Development Server:
   ```bash
   cd web_ui
   npm run dev
   ```

## Deployment Automation

Use the deployment script for automated updates:

```bash
./scripts/deploy.sh
```

This script will:
1. Pull the latest code
2. Update dependencies
3. Start the application services
""")
        
        # Document repository sync
        sync_doc = self.base_dir / "docs" / "REPO_SYNC.md"
        
        with open(sync_doc, "w") as f:
            f.write("""# Repository Synchronization

This document describes the process for synchronizing code between development and production servers.

## Manual Synchronization

To manually sync the code:

```bash
./scripts/sync_servers.sh
```

## Automated Synchronization

Set up a cron job to automatically sync the repositories:

```bash
# Edit the crontab
crontab -e

# Add the following line to sync every hour
0 * * * * /opt/sutazaiapp/scripts/sync_servers.sh >> /opt/sutazaiapp/logs/sync.log 2>&1
```
""")
        
        self.status["cicd_pipelines"] = True
        return True
    
    def setup_monitoring(self) -> bool:
        """Set up monitoring and automated recovery."""
        logger.info("Setting up monitoring...")
        
        # Create monitoring document
        monitor_doc = self.base_dir / "docs" / "OBSERVABILITY.md"
        
        with open(monitor_doc, "w") as f:
            f.write("""# Monitoring and Observability

This document outlines the monitoring setup for the SutazAI application.

## Log Monitoring

Application logs are stored in the `logs` directory:

```bash
tail -f /opt/sutazaiapp/logs/*.log
```

## Log Rotation

Configure logrotate to manage log files:

```bash
# Create configuration file
sudo nano /etc/logrotate.d/sutazai

# Add the following content
/opt/sutazaiapp/logs/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 sutazaiapp_dev sutazaiapp_dev
}
```

## Health Checks

Regular health checks are performed by:

```bash
python scripts/monitor_health.py
```

## Error Detection and Recovery

Automated error detection and recovery is handled by:

```bash
python scripts/system_maintenance.py
```

## Setting Up Prometheus and Grafana

1. Install Prometheus:
   ```bash
   sudo apt-get update
   sudo apt-get install prometheus
   ```

2. Install Grafana:
   ```bash
   sudo apt-get install grafana
   ```

3. Configure Prometheus to monitor the SutazAI application by editing `/etc/prometheus/prometheus.yml`.

4. Start the services:
   ```bash
   sudo systemctl start prometheus
   sudo systemctl start grafana-server
   ```

5. Access Grafana at http://localhost:3000 and set up dashboards.
""")
        
        self.status["monitoring"] = True
        return True
    
    def update_documentation(self) -> bool:
        """Update and consolidate documentation."""
        logger.info("Updating documentation...")
        
        # Create security documentation
        security_doc = self.base_dir / "docs" / "SECURITY.md"
        
        with open(security_doc, "w") as f:
            f.write("""# Security Guidelines

This document outlines the security practices for the SutazAI application.

## User Management

1. Application users should have limited permissions:
   ```bash
   sudo adduser sutazaiapp_dev
   sudo usermod -aG sudo sutazaiapp_dev
   ```

2. File ownership should be maintained:
   ```bash
   sudo chown -R sutazaiapp_dev:sutazaiapp_dev /opt/sutazaiapp
   ```

3. File permissions should be restricted:
   ```bash
   chmod -R 750 /opt/sutazaiapp
   ```

## Secure Configuration

1. Sensitive information should not be stored in the repository
2. Use environment variables or secure configuration files
3. Use `.env` files for development, but not in production

## API Security

1. All API endpoints should require authentication
2. Rate limiting should be implemented
3. Input validation should be thorough
4. Use HTTPS for all connections

## Data Security

1. Encrypt sensitive data at rest
2. Use secure connections for data transfer
3. Implement proper access controls
4. Regular security audits
""")
        
        # Create code structure documentation
        code_doc = self.base_dir / "docs" / "CODE_STRUCTURE.md"
        
        with open(code_doc, "w") as f:
            f.write("""# Code Structure

This document outlines the code structure of the SutazAI application.

## Directory Structure

```
/opt/sutazaiapp/
├── ai_agents/           # AI agent modules and components
├── model_management/    # Model management and monitoring
├── backend/             # Backend API and services
├── web_ui/              # Frontend web interface
├── scripts/             # Utility and deployment scripts
├── packages/            # Local package dependencies
├── logs/                # Application logs
├── doc_data/            # Document processing data
└── docs/                # Documentation
```

## Key Components

### Backend

The backend is built with FastAPI and provides the API endpoints for the application.

Key files:
- `backend/main.py`: Main application entry point
- `backend/routes/`: API routes
- `backend/services/`: Business logic implementations
- `backend/models/`: Data models

### AI Agents

The AI agents are responsible for processing tasks and interacting with the AI models.

Key components:
- `ai_agents/superagi/`: SuperAGI integration
- `ai_agents/agent_factory.py`: Factory for creating agents
- `ai_agents/base_agent.py`: Base agent class

### Model Management

Model management handles the AI models used by the application.

Key components:
- Model loading and unloading
- Performance monitoring
- Model optimization

### Web UI

The web UI provides the user interface for the application.

Key components:
- React components
- State management
- API integration
""")
        
        # Create final README with instructions
        readme = self.base_dir / "README.md"
        
        with open(readme, "w") as f:
            f.write("""# SutazAI Application

## Overview

The SutazAI application is an advanced AI system that provides intelligent processing and analysis capabilities.

## Getting Started

### Prerequisites

1. Python 3.11 or higher
2. Node.js 18 or higher (for web UI)
3. PostgreSQL 14 or higher
4. Redis 6 or higher

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sutazai/sutazaiapp.git
   cd sutazaiapp
   ```

2. Set up the Python virtual environment:
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```

3. Install Node.js dependencies (for web UI):
   ```bash
   cd web_ui
   npm install
   ```

4. Configure the application:
   - Copy example configuration files
   - Update with your specific settings

5. Start the application:
   ```bash
   # Start backend
   python -m backend.backend_main
   
   # Start web UI (in a separate terminal)
   cd web_ui
   npm run dev
   ```

## Documentation

Detailed documentation is available in the `docs` directory:

- [Deployment Guide](docs/DEPLOYMENT.md)
- [Security Guidelines](docs/SECURITY.md)
- [Code Structure](docs/CODE_STRUCTURE.md)
- [Monitoring and Observability](docs/OBSERVABILITY.md)
- [Repository Synchronization](docs/REPO_SYNC.md)

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/)
- [React](https://reactjs.org/)
- [SuperAGI](https://superagi.com/)
- [PostgreSQL](https://www.postgresql.org/)
- [Redis](https://redis.io/)
""")
        
        self.status["documentation"] = True
        return True
    
    def run_full_optimization(self) -> bool:
        """Run full optimization process."""
        logger.info("Starting full optimization process...")
        
        # Step 1: Set up directories
        if not self.setup_directories():
            logger.error("Directory setup failed")
            return False
        
        # Step 2: Set up virtual environment
        if not self.setup_virtual_env():
            logger.error("Virtual environment setup failed")
            return False
        
        # Step 3: Fix test issues
        if not self.fix_test_issues():
            logger.warning("Test fixes had issues")
            # Continue anyway
        
        # Step 4: Run code analysis
        if not self.run_code_analysis():
            logger.warning("Code analysis had issues")
            # Continue anyway
        
        # Step 5: Set up CI/CD pipeline
        if not self.setup_cicd_pipeline():
            logger.error("CI/CD pipeline setup failed")
            return False
        
        # Step 6: Set up monitoring
        if not self.setup_monitoring():
            logger.error("Monitoring setup failed")
            return False
        
        # Step 7: Update documentation
        if not self.update_documentation():
            logger.error("Documentation update failed")
            return False
        
        # Final step: Tag the repository
        logger.info("Creating version tag...")
        exit_code, stdout, stderr = self.safe_run_command(
            ["git", "tag", "-a", "v1.0.0", "-m", "Completed Restructuring and Optimization"]
        )
        if exit_code != 0:
            logger.warning("Failed to create git tag")
            self.warnings.append("Failed to create git tag")
        
        exit_code, stdout, stderr = self.safe_run_command(
            ["git", "push", "origin", "v1.0.0"]
        )
        if exit_code != 0:
            logger.warning("Failed to push git tag")
            self.warnings.append("Failed to push git tag")
        
        logger.info("Optimization process completed!")
        return True
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a report of the optimization process."""
        report = {
            "status": self.status,
            "errors": self.errors,
            "warnings": self.warnings,
            "completion_percentage": sum(1 for v in self.status.values() if v) / len(self.status) * 100
        }
        
        # Save report to file
        with open("optimization_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        return report


def main() -> int:
    """Main function to run the optimization process."""
    logger.info("Starting SutazAI reorganization and optimization...")
    
    # Check if running as root or with sudo
    if os.geteuid() == 0:
        logger.info("Running with elevated privileges")
    else:
        logger.warning("Not running with elevated privileges. Some operations may fail.")
    
    organizer = SutazAIOrganizer()
    success = organizer.run_full_optimization()
    
    report = organizer.generate_report()
    logger.info(f"Optimization process {'succeeded' if success else 'failed'}")
    logger.info(f"Completion percentage: {report['completion_percentage']:.2f}%")
    
    if report['errors']:
        logger.error("Errors encountered:")
        for error in report['errors']:
            logger.error(f"  - {error}")
    
    if report['warnings']:
        logger.warning("Warnings encountered:")
        for warning in report['warnings']:
            logger.warning(f"  - {warning}")
    
    logger.info("See optimization_report.json for details")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 