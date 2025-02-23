# SutazAI System Architecture Documentation

## Overview

SutazAI is an advanced, autonomous AI development platform deployed on dedicated hardware. It features a robust two-server architecture consisting of a Code Server for development and a Deployment Server for production. The system is built to operate autonomously with self-auditing, auto-fix, and file organization scripts, ensuring a comprehensive and resilient environment.

## Two-Server Synergy

- **Code Server (sutazaicodeserver)**: Hosts the code repository, handles code modifications, and runs the Supreme AI orchestrator under non-root privileges.
- **Deployment Server (sutazaideploymenttestserver)**: Receives code updates, runs the deploy.sh pipeline, and hosts the final production environment.

## Directory Structure

The project root should maintain a well-organized structure, including the following directories and key files:

- **ai_agents/**: Contains AI agent implementations (Supreme AI, AutoGPT, SuperAGI, etc.).
- **model_management/**: Hosts local AI models (GPT4All, DeepSeek, Molmo, etc.).
- **backend/**: Core backend services and API endpoints (e.g., main.py, api_routes.py).
- **web_ui/**: Frontend source files, package.json, and build output.
- **scripts/**: Deployment, audit, organization, auto-fix scripts, and supporting utilities.
- **packages/**: Stores pinned Python wheels and Node packages.
- **logs/**: Contains logs (deploy.log, audit.log, auto_fix.log, etc.) crucial for cross-referencing and debugging.
- **doc_data/**: Contains test documents and diagrams for processing tasks.
- **venv/**: The Python virtual environment.
- **docs/**: This documentation and additional design references.
- **misc/**: Automatically collected files that were not part of the expected structure.
- **README.md**: High level project overview and instructions.
- **requirements.txt**: Lists all Python dependencies.

## Deployment Pipeline

1. **Audit & Organization**: 
   - `scripts/audit_system.py`: Conducts a comprehensive audit checking directory structure, critical files, dependencies, and code syntax.
   - `scripts/organize_project.py`: Organizes the project root by moving unrecognized files into the `misc/` folder.
2. **Auto-Fix**: 
   - `scripts/auto_fix.py`: Automatically creates missing directories and initializes the virtual environment if needed.
3. **Deployment**:
   - A deployment script (e.g., deploy.sh) handles dependency installation, model verification, and starts all necessary services.

## Security and Integrity

- **OTP Net Override**: External network access is strictly controlled via an OTP-based system, ensuring privileged actions only after proper authentication.
- **Non-root Supreme AI**: Supreme AI runs without root privileges, with all critical OS-level tasks reserved for the root user (Florin Cristian Suta).
- **Comprehensive Logging**: All process outputs, changes, and audits are logged in the `logs/` directory, allowing cross-referencing and detailed traceability.

## Dependency and Module Management

- **Python Dependencies**: Managed via `requirements.txt` and verified against wheels stored in `packages/wheels/`.
- **Node Modules**: Dependency management for the web UI is handled via `web_ui/package.json` and local caches in `packages/node/`.
- **Version Control and Integrity Checks**: Integrated audit scripts ensure that any discrepancy in structure or dependencies is promptly identified and addressed.

## Audit and Auto-Fix Tools

- **Audit System**: `scripts/audit_system.py` checks complete system integrity, including syntax, structure, and sensitive data scanning.
- **File Organization**: `scripts/organize_project.py` ensures the project root remains organized by moving miscellaneous files to the `misc/` directory.
- **Auto-Fix**: `scripts/auto_fix.py` attempts to automatically resolve basic configuration issues.

## Conclusion

This documentation provides a detailed insight into the SutazAI system architecture. All components are designed to work autonomously, ensuring that the application remains well-organized, secure, and performs at peak efficiency. All changes are tracked via comprehensive logs and are subject to automated audit processes, allowing cross-referencing and rapid troubleshooting. 