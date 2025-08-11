# SutazAI Scripts Inventory

This document provides a complete inventory of all scripts organized in the `/scripts` directory.

## Directory Structure

```
/scripts/
├── deployment/           # Deployment and infrastructure scripts
│   ├── production/      # Production deployment scripts
│   ├── development/     # Development environment scripts  
│   ├── staging/         # Staging environment scripts
│   └── deployment-master.sh  # Master deployment orchestrator
├── maintenance/         # System maintenance scripts
│   ├── cleanup/        # Cleanup and housekeeping scripts
│   ├── optimization/   # Performance optimization scripts
│   ├── validation/     # System validation scripts
│   └── maintenance-master.sh  # Master maintenance orchestrator
├── monitoring/          # Monitoring and observability scripts
│   ├── health-checks/  # Health check scripts
│   ├── alerts/         # Alert management scripts
│   ├── logging/        # Log processing scripts
│   └── monitoring-master.py  # Master monitoring orchestrator
├── testing/            # Testing and QA scripts
│   ├── integration/    # Integration test scripts
│   ├── unit/           # Unit test scripts
│   ├── load/           # Load testing scripts
│   └── security/       # Security testing scripts
├── security/           # Security and compliance scripts
│   ├── hardening/      # System hardening scripts
│   ├── audit/          # Security audit scripts
│   └── remediation/    # Security remediation scripts
├── utils/              # Utility and helper scripts
│   ├── analysis/       # Analysis and reporting scripts
│   ├── reporting/      # Report generation scripts
│   └── helpers/        # General utility scripts
├── automation/         # Build and CI/CD automation
│   ├── build/          # Build automation scripts
│   ├── deploy/         # Deployment automation
│   └── ci-cd/          # CI/CD pipeline scripts
├── database/           # Database management scripts
│   ├── migration/      # Database migration scripts
│   ├── backup/         # Database backup scripts
│   └── maintenance/    # Database maintenance scripts
└── backup/             # System backup scripts
```

## Usage

### Master Scripts
- `deployment/deployment-master.sh [production|staging|development]`
- `maintenance/maintenance-master.sh [cleanup|optimization|validation|all]`  
- `monitoring/monitoring-master.py [health|alerts|logs|all]`

### Script Categories
Each script is categorized by its primary function and placed in the appropriate directory.
All scripts follow naming conventions and include proper documentation headers.

## Script Standards
- All scripts are executable with proper permissions
- Headers include purpose, author, date, and usage instructions
- Error handling with proper exit codes
- Logging with timestamps
- One clear purpose per script
- No duplicate functionality across scripts
