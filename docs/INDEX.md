# SutazAI Documentation Index

> **Last Updated:** August 8, 2025  
> **Purpose:** Central entry point for all SutazAI documentation  
> **Primary Source of Truth:** [CLAUDE.md](/opt/sutazaiapp/CLAUDE.md)

---

## For New Developers

Start here if you're new to the SutazAI project:

### Getting Started
1. **[System Reality Check](/opt/sutazaiapp/CLAUDE.md)** - **READ THIS FIRST** - The truth about what actually works
2. **[Quick Start Guide](/opt/sutazaiapp/docs/training/QUICK_START.md)** - Get up and running quickly
3. **[Developer Guide](/opt/sutazaiapp/docs/training/DEVELOPER_GUIDE.md)** - Development environment setup and workflows
4. **[System Reality Report](/opt/sutazaiapp/docs/SYSTEM_REALITY_REPORT.md)** - Current state assessment

### System Overview
- **[System Overview](/opt/sutazaiapp/docs/architecture/01-system-overview.md)** - High-level architecture
- **[Component Architecture](/opt/sutazaiapp/docs/architecture/02-component-architecture.md)** - System components and interactions
- **[Technology Stack](/opt/sutazaiapp/docs/architecture/04-technology-stack.md)** - Technologies and frameworks used
- **[Port Registry](/opt/sutazaiapp/CLAUDE.md#accurate-port-registry)** - Service port mappings

### Development Resources
- **[Working Endpoints](/opt/sutazaiapp/docs/PHASE_0_WORKING_ENDPOINTS.md)** - Currently functional API endpoints
- **[Backend Endpoints](/opt/sutazaiapp/docs/backend_endpoints.md)** - Backend API documentation
- **[API Reference](/opt/sutazaiapp/docs/api/JARVIS_API_REFERENCE.md)** - Complete API documentation

---

## Architecture Documentation

Comprehensive architectural documentation for system design and implementation:

### Core Architecture
- **[Master System Blueprint](/opt/sutazaiapp/docs/architecture/MASTER_SYSTEM_BLUEPRINT_v2.2.md)** - **Primary architectural reference**
- **[System Architecture Blueprint](/opt/sutazaiapp/IMPORTANT/10_canonical/additional_docs/system_architecture_blueprint.md)** - Canonical architecture documentation
- **[Technical Architecture](/opt/sutazaiapp/IMPORTANT/10_canonical/additional_docs/technical_architecture_documentation.md)** - Detailed technical specifications

### System Design
- **[Data Flow Architecture](/opt/sutazaiapp/docs/architecture/03-data-flow.md)** - Data movement and processing
- **[Scalability Design](/opt/sutazaiapp/docs/architecture/05-scalability-design.md)** - Scaling strategies
- **[Architecture Redesign v59](/opt/sutazaiapp/docs/architecture-redesign-v59.md)** - Latest architectural updates

### Agent Architecture
- **[Agent Implementation Guide](/opt/sutazaiapp/docs/AGENT_IMPLEMENTATION_GUIDE.md)** - How to implement agents
- **[Agent Implementation Reality](/opt/sutazaiapp/docs/architecture/agents/AGENT_IMPLEMENTATION_REALITY.md)** - Current agent status
- **[AI Agent Orchestrator](/opt/sutazaiapp/docs/architecture/agents/01_AI_AGENT_ORCHESTRATOR.md)** - Core orchestration agent
- **[Hardware Resource Optimizer](/opt/sutazaiapp/docs/architecture/agents/02_HARDWARE_RESOURCE_OPTIMIZER.md)** - Resource management
- **[Task Assignment Coordinator](/opt/sutazaiapp/docs/architecture/agents/03_TASK_ASSIGNMENT_COORDINATOR.md)** - Task distribution

---

## Architecture Decision Records (ADRs)

Key architectural decisions and their rationale:

- **[ADR Template](/opt/sutazaiapp/docs/architecture/adrs/adr-template.md)** - Template for new ADRs
- **[ADR-0001: Architecture Principles](/opt/sutazaiapp/docs/architecture/adrs/0001-architecture-principles.md)** - Core architectural principles
- **[ADR-0002: Technology Choices](/opt/sutazaiapp/docs/architecture/adrs/0002-technology-choices.md)** - Technology selection rationale
- **[ADR-0001: Canonical Docs](/opt/sutazaiapp/IMPORTANT/10_canonical/standards/ADR-0001.md)** - Documentation structure
- **[ADR-0002: Issue Tracking](/opt/sutazaiapp/IMPORTANT/10_canonical/standards/ADR-0002.md)** - Issue management approach
- **[ADR-0003: Phased Migration](/opt/sutazaiapp/IMPORTANT/10_canonical/standards/ADR-0003.md)** - Migration strategy
- **[ADR-0004: MVP Architecture](/opt/sutazaiapp/IMPORTANT/10_canonical/standards/ADR-0004.md)** - MVP design decisions
- **[ADR-0005: Observability](/opt/sutazaiapp/IMPORTANT/10_canonical/standards/ADR-0005.md)** - Monitoring strategy
- **[Remove Service Mesh Decision](/opt/sutazaiapp/docs/decisions/2025-08-07-remove-service-mesh.md)** - Service mesh removal rationale

---

## Canonical Documentation (Source of Truth)

The authoritative documentation that overrides all other sources:

### Main Index
- **[Canonical Documentation Index](/opt/sutazaiapp/IMPORTANT/10_canonical/INDEX.md)** - **Start here for canonical docs**

### Current State
- **[System Reality](/opt/sutazaiapp/IMPORTANT/10_canonical/current_state/system_reality.md)** - What actually exists
- **[Deployment Topology](/opt/sutazaiapp/IMPORTANT/10_canonical/current_state/deployment_topology.md)** - Current deployment architecture
- **[Environments](/opt/sutazaiapp/IMPORTANT/10_canonical/current_state/environments.md)** - Environment configurations

### Target State & Planning
- **[MVP Architecture](/opt/sutazaiapp/IMPORTANT/10_canonical/target_state/mvp_architecture.md)** - Target MVP design
- **[Migration Plan](/opt/sutazaiapp/IMPORTANT/20_plan/migration_plan.md)** - System migration strategy
- **[Phased Migration Plan](/opt/sutazaiapp/IMPORTANT/20_plan/phased_migration_plan.md)** - Detailed migration phases
- **[Gap Analysis](/opt/sutazaiapp/IMPORTANT/20_plan/gap_analysis.md)** - Current vs target state analysis

### Standards & Governance
- **[Engineering Standards](/opt/sutazaiapp/IMPORTANT/10_canonical/standards/engineering_standards.md)** - Development standards
- **[Codebase Rules](/opt/sutazaiapp/IMPORTANT/10_canonical/standards/codebase_rules.md)** - Mandatory coding rules
- **[Comprehensive Engineering Standards](/opt/sutazaiapp/IMPORTANT/COMPREHENSIVE_ENGINEERING_STANDARDS.md)** - Full standards document

### API & Data
- **[API Contracts](/opt/sutazaiapp/IMPORTANT/10_canonical/api_contracts/contracts.md)** - Service API definitions
- **[Data Management](/opt/sutazaiapp/IMPORTANT/10_canonical/data/data_management.md)** - Data handling strategies
- **[Domain Glossary](/opt/sutazaiapp/IMPORTANT/10_canonical/domain_model/domain_glossary.md)** - Business domain terms

---

## Issue Tracking

Current system issues and remediation efforts:

### Issue Registry
- **[Issues Directory](/opt/sutazaiapp/IMPORTANT/02_issues/)** - All tracked issues
- **[Risk Register](/opt/sutazaiapp/IMPORTANT/01_findings/risk_register.md)** - System risks and mitigation
- **[Conflicts Report](/opt/sutazaiapp/IMPORTANT/01_findings/conflicts.md)** - Documentation conflicts
- **[Quick Wins](/opt/sutazaiapp/IMPORTANT/20_plan/quick_wins.md)** - Immediate improvement opportunities
- **[Remediation Backlog](/opt/sutazaiapp/IMPORTANT/20_plan/remediation_backlog.csv)** - Prioritized fix list

### Critical Issues (P0)
- **[ISSUE-0001](/opt/sutazaiapp/IMPORTANT/02_issues/ISSUE-0001.md)** - Documentation scattered across 500+ files
- **[ISSUE-0002](/opt/sutazaiapp/IMPORTANT/02_issues/ISSUE-0002.md)** - Multiple conflicting architecture blueprints
- **[ISSUE-0003](/opt/sutazaiapp/IMPORTANT/02_issues/ISSUE-0003.md)** - 200+ agents defined but only 7 running

---

## Reports and Analysis

System analysis and audit reports:

### Executive Reports
- **[Phase 1 Executive Summary](/opt/sutazaiapp/IMPORTANT/PHASE1_EXECUTIVE_SUMMARY.md)** - Initial assessment summary
- **[Phase 2 Reconciliation Report](/opt/sutazaiapp/IMPORTANT/PHASE2_RECONCILIATION_REPORT.md)** - Documentation reconciliation
- **[Audit Status Report](/opt/sutazaiapp/IMPORTANT/AUDIT_STATUS_REPORT.md)** - System audit findings

### Product Documentation
- **[Product Requirements](/opt/sutazaiapp/IMPORTANT/SUTAZAI_PRD.md)** - Product requirements document
- **[MVP Documentation](/opt/sutazaiapp/IMPORTANT/SUTAZAI_MVP.md)** - MVP scope and features
- **[POC Documentation](/opt/sutazaiapp/IMPORTANT/SUTAZAI_POC.md)** - Proof of concept details
- **[Strategy Plan](/opt/sutazaiapp/IMPORTANT/Strategy _Plan.md)** - Strategic planning document

### System Inventory
- **[System Inventory](/opt/sutazaiapp/IMPORTANT/00_inventory/inventory.md)** - Complete documentation inventory
- **[Document Review Matrix](/opt/sutazaiapp/IMPORTANT/00_inventory/doc_review_matrix.csv)** - Documentation review status

---

## Operations & Deployment

### Deployment Guides
- **[Deployment Runbook](/opt/sutazaiapp/docs/runbooks/DEPLOYMENT_RUNBOOK.md)** - Step-by-step deployment
- **[Blue-Green Deployment](/opt/sutazaiapp/docs/BLUE_GREEN_DEPLOYMENT_GUIDE.md)** - Zero-downtime deployment
- **[Tiered Deployment Guide](/opt/sutazaiapp/docs/tiered-deployment-guide.md)** - Environment promotion strategy
- **[DevOps README](/opt/sutazaiapp/docs/DEVOPS_README.md)** - DevOps practices and tools

### Operations
- **[Operations Runbook](/opt/sutazaiapp/docs/runbooks/OPERATIONS_RUNBOOK.md)** - Daily operations procedures
- **[Maintenance Procedures](/opt/sutazaiapp/docs/runbooks/MAINTENANCE_PROCEDURES.md)** - System maintenance
- **[Incident Response](/opt/sutazaiapp/docs/runbooks/INCIDENT_RESPONSE.md)** - Incident handling procedures
- **[Disaster Recovery](/opt/sutazaiapp/docs/runbooks/DISASTER_RECOVERY.md)** - Recovery procedures

### Monitoring & Observability
- **[Monitoring Guide](/opt/sutazaiapp/IMPORTANT/10_canonical/additional_docs/monitoring_observability_guide.md)** - Monitoring setup
- **[Agent Observability](/opt/sutazaiapp/docs/monitoring/AGENT_OBSERVABILITY_SUMMARY.md)** - Agent monitoring
- **[Agent Dashboards](/opt/sutazaiapp/docs/monitoring/agent-dashboards.md)** - Grafana dashboard setup
- **[Post Go-Live Monitoring](/opt/sutazaiapp/docs/POST_GOLIVE_MONITORING.md)** - Production monitoring

---

## Quick Reference

Essential information for daily development:

### Port Registry
```yaml
# Core Services
10000: PostgreSQL database
10001: Redis cache
10002: Neo4j browser
10003: Neo4j bolt
10005: Kong API Gateway
10006: Consul
10007: RabbitMQ AMQP
10008: RabbitMQ UI
10010: Backend FastAPI
10011: Frontend Streamlit
10104: Ollama LLM

# Monitoring
10200: Prometheus
10201: Grafana
10202: Loki
10203: AlertManager

# Agent Services (Stubs)
8002: Hardware Optimizer
8551: Task Assignment
8587: Multi-Agent Coordinator
8588: Resource Arbitration
8589: AI Agent Orchestrator
```

### Common Commands
```bash
# Start system
docker-compose up -d

# Check status
docker ps --format "table {{.Names}}\t{{.Ports}}\t{{.Status}}"

# View logs
docker-compose logs -f [service-name]

# Test backend
curl http://127.0.0.1:10010/health

# Access UI
open http://localhost:10011
```

### Environment Variables
- See [Environment Template](/opt/sutazaiapp/IMPORTANT/10_canonical/operations/env_template.md)
- Configuration in `.env` file at project root

### Troubleshooting
- **[System Reality Check](/opt/sutazaiapp/CLAUDE.md#common-issues--real-solutions)** - Common issues and solutions
- **[Rules Compliance Checklist](/opt/sutazaiapp/docs/RULES_COMPLIANCE_CHECKLIST.md)** - Development checklist
- **[Security Runbook](/opt/sutazaiapp/docs/runbooks/SECURITY_RUNBOOK.md)** - Security procedures

---

## Training & Documentation

### User Documentation
- **[User Manual](/opt/sutazaiapp/docs/training/USER_MANUAL.md)** - End user guide
- **[Administrator Guide](/opt/sutazaiapp/docs/training/ADMINISTRATOR_GUIDE.md)** - System administration

### Workshop Materials
- **[Day 1: System Overview](/opt/sutazaiapp/docs/training/workshop/DAY_1_SYSTEM_OVERVIEW.md)** - Introduction workshop
- **[Day 2: Advanced Features](/opt/sutazaiapp/docs/training/workshop/DAY_2_ADVANCED_FEATURES.md)** - Advanced features
- **[Day 3: Administration](/opt/sutazaiapp/docs/training/workshop/DAY_3_ADMINISTRATION_MAINTENANCE.md)** - Admin training

---

## Reading Order Recommendations

### For Developers (New to Project)
1. [CLAUDE.md](/opt/sutazaiapp/CLAUDE.md) - System reality check
2. [Quick Start Guide](/opt/sutazaiapp/docs/training/QUICK_START.md)
3. [System Overview](/opt/sutazaiapp/docs/architecture/01-system-overview.md)
4. [Working Endpoints](/opt/sutazaiapp/docs/PHASE_0_WORKING_ENDPOINTS.md)
5. [Developer Guide](/opt/sutazaiapp/docs/training/DEVELOPER_GUIDE.md)

### For System Architects
1. [Master System Blueprint](/opt/sutazaiapp/docs/architecture/MASTER_SYSTEM_BLUEPRINT_v2.2.md)
2. [Canonical Documentation Index](/opt/sutazaiapp/IMPORTANT/10_canonical/INDEX.md)
3. [Architecture Principles ADR](/opt/sutazaiapp/docs/architecture/adrs/0001-architecture-principles.md)
4. [MVP Architecture](/opt/sutazaiapp/IMPORTANT/10_canonical/target_state/mvp_architecture.md)
5. [Gap Analysis](/opt/sutazaiapp/IMPORTANT/20_plan/gap_analysis.md)

### For DevOps Engineers
1. [DevOps README](/opt/sutazaiapp/docs/DEVOPS_README.md)
2. [Deployment Runbook](/opt/sutazaiapp/docs/runbooks/DEPLOYMENT_RUNBOOK.md)
3. [Operations Runbook](/opt/sutazaiapp/docs/runbooks/OPERATIONS_RUNBOOK.md)
4. [Monitoring Guide](/opt/sutazaiapp/IMPORTANT/10_canonical/additional_docs/monitoring_observability_guide.md)
5. [CI/CD Multi-arch](/opt/sutazaiapp/IMPORTANT/10_canonical/operations/ci_cd_multiarch.md)

### For Product Managers
1. [Product Requirements](/opt/sutazaiapp/IMPORTANT/SUTAZAI_PRD.md)
2. [MVP Documentation](/opt/sutazaiapp/IMPORTANT/SUTAZAI_MVP.md)
3. [Strategy Plan](/opt/sutazaiapp/IMPORTANT/Strategy _Plan.md)
4. [Phase 1 Executive Summary](/opt/sutazaiapp/IMPORTANT/PHASE1_EXECUTIVE_SUMMARY.md)
5. [Quick Wins](/opt/sutazaiapp/IMPORTANT/20_plan/quick_wins.md)

### For Security Teams
1. [Security & Privacy](/opt/sutazaiapp/IMPORTANT/10_canonical/security/security_privacy.md)
2. [Security Runbook](/opt/sutazaiapp/docs/runbooks/SECURITY_RUNBOOK.md)
3. [Security Compliance](/opt/sutazaiapp/IMPORTANT/10_canonical/additional_docs/security_compliance_documentation.md)
4. [Image Scanning](/opt/sutazaiapp/IMPORTANT/10_canonical/operations/image_scanning.md)

---

## Important Notes

1. **Primary Source of Truth**: [CLAUDE.md](/opt/sutazaiapp/CLAUDE.md) contains the reality check of what actually works
2. **Canonical Documentation**: Documents in `/opt/sutazaiapp/IMPORTANT/10_canonical/` override all other sources
3. **System Status**: Many documented features are stubs or not implemented - verify with actual testing
4. **Model Reality**: System uses TinyLlama (637MB), not gpt-oss as documentation may claim
5. **Agent Reality**: Only 7 Flask stub agents are running, returning hardcoded responses

---

## Change Log

- **[System Changelog](/opt/sutazaiapp/docs/CHANGELOG.md)** - Recent system changes
- **[IMPORTANT Changelog](/opt/sutazaiapp/IMPORTANT/CHANGELOG.md)** - Critical system updates

---

*For questions or issues with documentation, refer to the [Issue Tracking](#issue-tracking) section above.*