# SutazAI Toolset Guide

This document provides an overview of all the scripts and tools available in the SutazAI project, along with instructions on how to use them effectively.

## Table of Contents

1. [Installation Scripts](#installation-scripts)
2. [Start/Stop Scripts](#startstop-scripts)
3. [Monitoring Scripts](#monitoring-scripts)
4. [Deployment Scripts](#deployment-scripts)
5. [Maintenance Scripts](#maintenance-scripts)
6. [Troubleshooting Scripts](#troubleshooting-scripts)

---

## Installation Scripts

### `scripts/check_environment.sh`

**Purpose**: Verifies that your system meets all the requirements for running SutazAI.

**Usage**:
```bash
bash scripts/check_environment.sh
```

**Options**:
- `--fix`: Attempts to fix any issues found (requires sudo privileges)
- `--verbose`: Shows detailed output

**Expected Output**: A summary of checks with either PASS or FAIL status, and recommendations for any failed checks.

### `scripts/setup_repos.sh`

**Purpose**: Sets up required repositories and downloads model files.

**Usage**:
```bash
bash scripts/setup_repos.sh
```

**Expected Behavior**: Clones required repositories, downloads AI models, and creates necessary directories.

### `scripts/verify_installation.sh`

**Purpose**: Verifies that the installation was successful and all components are properly installed.

**Usage**:
```bash
bash scripts/verify_installation.sh
```

**Expected Output**: A comprehensive verification report showing the status of all components.

---

## Start/Stop Scripts

### `scripts/start_backend.sh`

**Purpose**: Starts the SutazAI backend server.

**Usage**:
```bash
bash scripts/start_backend.sh
```

**Expected Behavior**: The backend server will start and run in the background, with logs directed to `logs/backend.log`.

### `scripts/stop_backend.sh`

**Purpose**: Stops the running backend server.

**Usage**:
```bash
bash scripts/stop_backend.sh
```

**Expected Behavior**: Gracefully stops the backend server, with a forced termination after 30 seconds if needed.

### `scripts/start_superagi.sh`

**Purpose**: Starts the SuperAGI agent.

**Usage**:
```bash
bash scripts/start_superagi.sh
```

**Options**:
- `--service`: Runs the SuperAGI agent as a background service
- `--debug`: Enables debug-level logging

**Expected Behavior**: The SuperAGI agent will start and run either in the foreground or as a background service.

### `scripts/stop_superagi.sh`

**Purpose**: Stops the running SuperAGI agent.

**Usage**:
```bash
bash scripts/stop_superagi.sh
```

**Expected Behavior**: Gracefully stops the SuperAGI agent, with a forced termination after 30 seconds if needed.

### `scripts/start_monitoring.sh`

**Purpose**: Starts the Prometheus monitoring service.

**Usage**:
```bash
bash scripts/start_monitoring.sh
```

**Expected Behavior**: The Prometheus monitoring service will start and run in the background.

---

## Monitoring Scripts

### `scripts/health_check.sh`

**Purpose**: Performs a comprehensive health check of all SutazAI components.

**Usage**:
```bash
bash scripts/health_check.sh
```

**Expected Output**: A detailed health report showing the status of processes, API endpoints, directories, files, system resources, logs, database, virtual environment, and model files.

**Exit Codes**:
- `0`: All healthy
- `1`: Warnings found
- `2`: Critical issues found

### `scripts/setup_monitoring.sh`

**Purpose**: Sets up the monitoring infrastructure for SutazAI.

**Usage**:
```bash
bash scripts/setup_monitoring.sh
```

**Expected Behavior**: Installs and configures Prometheus for monitoring the SutazAI services.

---

## Deployment Scripts

### `scripts/trigger_deploy.sh`

**Purpose**: Provides a secure way to trigger deployments, typically used in CI/CD pipelines or by authorized users.

**Usage**:
```bash
bash scripts/trigger_deploy.sh [options]
```

**Options**:
- `--force`: Bypasses user permission check
- `--skip-checks`: Skips pre-deployment checks
- `--quiet`: Reduces verbosity and runs non-interactively
- `--env=<environment>`: Specifies the target environment (default: production)

**Expected Behavior**: Triggers the deployment process with appropriate security checks, including OTP verification if configured.

### `scripts/deploy.sh`

**Purpose**: Handles the actual deployment of the SutazAI application.

**Usage**:
```bash
bash scripts/deploy.sh [--env=<environment>]
```

**Options**:
- `--env=<environment>`: Specifies the target environment (default: production)

**Expected Behavior**: Deploys the application to the specified environment, updating code, configurations, and restarting services as needed.

---

## Maintenance Scripts

### `scripts/create_archive.sh`

**Purpose**: Creates a backup archive of the SutazAI application.

**Usage**:
```bash
bash scripts/create_archive.sh [options]
```

**Options**:
- `--output=<path>`: Specifies the output directory for the archive
- `--exclude-models`: Excludes model files from the archive to reduce size

**Expected Behavior**: Creates a timestamped tar.gz archive of the application.

---

## Troubleshooting Scripts

### `scripts/clear_logs.sh`

**Purpose**: Clears or rotates log files to free up disk space.

**Usage**:
```bash
bash scripts/clear_logs.sh [options]
```

**Options**:
- `--rotate`: Rotates logs instead of clearing them
- `--all`: Clears all logs, including archived ones
- `--days=<number>`: Clears logs older than the specified number of days

**Expected Behavior**: Clears or rotates log files based on the specified options.

### `scripts/reset_environment.sh`

**Purpose**: Resets the environment to a clean state, typically used for troubleshooting.

**Usage**:
```bash
bash scripts/reset_environment.sh [options]
```

**Options**:
- `--soft`: Performs a soft reset (preserving data and configurations)
- `--hard`: Performs a hard reset (WARNING: This will delete all data)
- `--backup`: Creates a backup before performing the reset

**Expected Behavior**: Resets the environment to a clean state based on the specified options.

---

## Best Practices

1. **Always use the provided scripts** instead of manually starting, stopping, or deploying services.
2. **Run health checks regularly** to ensure the system is operating correctly.
3. **Create backups before major changes** using the `create_archive.sh` script.
4. **Monitor logs** for any errors or warnings that might indicate issues.
5. **Use the deployment scripts** for controlled and reproducible deployments.

## Troubleshooting Common Issues 