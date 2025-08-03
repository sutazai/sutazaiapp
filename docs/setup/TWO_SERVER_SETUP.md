# SutazAI Two-Server Setup & Supreme AI Orchestrator

This document provides a comprehensive guide to setting up and maintaining the SutazAI application across two servers: a Code Server and a Deployment Server. The Supreme AI Orchestrator coordinates operations between these servers to ensure seamless deployment, monitoring, and management.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Server Setup](#server-setup)
3. [SSH Key Configuration](#ssh-key-configuration)
4. [Synchronization](#synchronization)
5. [Supreme AI Orchestrator](#supreme-ai-orchestrator)
6. [Deployment Workflow](#deployment-workflow)
7. [Monitoring](#monitoring)
8. [Troubleshooting](#troubleshooting)

## Architecture Overview

The SutazAI system uses a two-server architecture:

- **Code Server**: Used for development, testing, and orchestration. This server contains the most recent code and manages deployments to the production server.
- **Deployment Server**: The production environment where the application runs and serves user requests.

The **Supreme AI Orchestrator** runs on the Code Server and manages communication, synchronization, and monitoring between the two servers.

```
┌─────────────────────┐                 ┌─────────────────────┐
│                     │                 │                     │
│    Code Server      │                 │  Deployment Server  │
│                     │                 │                     │
│  ┌───────────────┐  │     Secure      │  ┌───────────────┐  │
│  │ LocalAGI AI    │  │      SSH        │  │ SutazAI       │  │
│  │ Orchestrator  │──┼─────────────────┼─▶│ Application   │  │
│  └───────────────┘  │   Connection    │  └───────────────┘  │
│                     │                 │                     │
│  ┌───────────────┐  │                 │  ┌───────────────┐  │
│  │ Development   │  │                 │  │ Production    │  │
│  │ Environment   │  │                 │  │ Environment   │  │
│  └───────────────┘  │                 │  └───────────────┘  │
│                     │                 │                     │
└─────────────────────┘                 └─────────────────────┘
```

## Server Setup

### Code Server Requirements

- Modern Linux distribution (Ubuntu 20.04+ recommended)
- Python 3.8+
- Git
- SSH client
- rsync

### Deployment Server Requirements

- Modern Linux distribution (Ubuntu 20.04+ recommended)
- Python 3.8+
- SSH server
- Sufficient resources to run the SutazAI application

## SSH Key Configuration

Secure communication between servers is established using SSH keys:

1. **Generate SSH Key** on the Code Server:
   ```bash
   ssh-keygen -t ed25519 -C "sutazaiapp_deploy" -f ~/.ssh/sutazaiapp_deploy -N ""
   ```

2. **Copy Public Key** to the Deployment Server:
   ```bash
   ssh-copy-id -i ~/.ssh/sutazaiapp_deploy.pub root@<deployment-server-ip>
   ```

3. **Set Proper Permissions** for the private key:
   ```bash
   chmod 600 ~/.ssh/sutazaiapp_deploy
   ```

4. **Test Connection**:
   ```bash
   ssh -i ~/.ssh/sutazaiapp_deploy root@<deployment-server-ip> "echo Connection successful"
   ```

## Synchronization

The `sync_servers.sh` script handles file synchronization between the Code Server and Deployment Server:

### Usage

```bash
bash scripts/sync_servers.sh [--dry-run] [--fast | --full]
```

### Options

- `--dry-run`: Shows what would be synchronized without making changes
- `--fast`: Performs a quick sync, skipping large files like models
- `--full`: Performs a complete sync including all files

### Excluding Files

Create a `.syncignore` file in the project root to specify files and directories to exclude from synchronization. By default, this includes:

- `.git/` and Git-related files
- `__pycache__/` and `.pyc` files
- `venv/` and `node_modules/` directories
- Log files and large model files
- `.env` file (to keep environment-specific configurations)

## Supreme AI Orchestrator

The Supreme AI Orchestrator is a Python application that manages the communication and coordination between the two servers.

### Key Features

- Server connectivity monitoring
- File synchronization
- Remote command execution
- Deployment automation
- Health monitoring
- Service management

### Usage

The Orchestrator can be invoked using the provided CLI wrapper:

```bash
bash scripts/orchestrator.sh [ACTION] [OPTIONS]
```

### Actions

- `monitor`: Start monitoring of servers and services (default)
- `sync`: Synchronize code to the deployment server
- `deploy`: Deploy the application to production
- `restart`: Restart services on a server
- `status`: Show status of all servers and services

### Options

- `--help`, `-h`: Show help message
- `--config=PATH`: Specify custom config file path
- `--interval=SECS`: Set monitoring interval in seconds (default: 300)
- `--server=TYPE`: Specify server type (code or deployment, default: deployment)
- `--sync-mode=MODE`: Specify sync mode (normal, fast, full, default: normal)

### Examples

```bash
# Start monitoring with 1-minute interval
scripts/orchestrator.sh monitor --interval=60

# Perform fast sync (skipping large files)
scripts/orchestrator.sh sync --sync-mode=fast

# Deploy to production
scripts/orchestrator.sh deploy

# Restart services on code server
scripts/orchestrator.sh restart --server=code

# Check status of all servers
scripts/orchestrator.sh status
```

## Deployment Workflow

The typical deployment workflow involves:

1. **Development**: Make changes on the Code Server
2. **Testing**: Test changes locally on the Code Server
3. **Synchronization**: Sync changes to the Deployment Server
4. **Deployment**: Deploy the application on the Deployment Server
5. **Verification**: Verify the deployment was successful

To execute this workflow:

```bash
# 1. Sync changes to deployment server
scripts/orchestrator.sh sync

# 2. Deploy to production
scripts/orchestrator.sh deploy

# 3. Check status to verify
scripts/orchestrator.sh status
```

## Monitoring

The Orchestrator provides continuous monitoring of both servers and the application:

```bash
# Start monitoring with default settings
scripts/orchestrator.sh monitor
```

This will:
- Check server connectivity at regular intervals
- Run health checks on the Deployment Server
- Log all findings to `logs/orchestrator.log`
- Take action if critical issues are detected

You can also manually check the current status:

```bash
scripts/orchestrator.sh status
```

## Troubleshooting

### Common Issues

#### SSH Connectivity Problems

If you encounter SSH connectivity issues:

1. Verify the SSH key exists and has correct permissions:
   ```bash
   ls -la ~/.ssh/sutazaiapp_deploy*
   ```

2. Check if the public key is properly installed on the Deployment Server:
   ```bash
   ssh -i ~/.ssh/sutazaiapp_deploy root@<deployment-server-ip> "cat ~/.ssh/authorized_keys"
   ```

3. Ensure the SSH service is running on the Deployment Server:
   ```bash
   ssh -i ~/.ssh/sutazaiapp_deploy root@<deployment-server-ip> "systemctl status sshd"
   ```

#### Synchronization Failures

If file synchronization fails:

1. Check network connectivity between servers
2. Verify that `rsync` is installed on both servers
3. Ensure the destination directories have appropriate permissions
4. Run with `--dry-run` to diagnose what might be failing

#### Deployment Issues

If deployment fails:

1. Check the logs:
   ```bash
   cat logs/orchestrator.log
   ssh -i ~/.ssh/sutazaiapp_deploy root@<deployment-server-ip> "cat /opt/sutazaiapp/logs/deploy.log"
   ``` 