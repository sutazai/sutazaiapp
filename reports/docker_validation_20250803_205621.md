# Docker Structure Validation Report

Generated: 2025-08-03T20:56:21.754889

## Compliance Summary

**Overall Score: 16.67%**
**Status: NON_COMPLIANT**

### Component Scores
- Structure Compliance: 50%
- Dockerfile Standards: 0%
- Compose Configuration: 0%

## Issues Summary

- Structure Issues: 115
- Dockerfile Issues: 338
- Compose Issues: 118
- Auto-fixes Applied: 0

## Critical Issues

### ⚠️ curl_pipe_sh
- **Location**: `/opt/sutazaiapp/docker/tabbyml/Dockerfile`
- **Type**: security
- **Severity**: critical

### ⚠️ latest_tag
- **Location**: `/opt/sutazaiapp/docker/tabbyml-cpu/Dockerfile`
- **Type**: security
- **Severity**: critical

### ⚠️ latest_tag
- **Location**: `/opt/sutazaiapp/docker/agents/Dockerfile.tabbyml`
- **Type**: security
- **Severity**: critical

### ⚠️ latest_tag
- **Location**: `/opt/sutazaiapp/docker/services/infrastructure/ollama/Dockerfile`
- **Type**: security
- **Severity**: critical

### ⚠️ latest_tag
- **Location**: `/opt/sutazaiapp/docker/services/monitoring/prometheus/Dockerfile`
- **Type**: security
- **Severity**: critical

### ⚠️ latest_tag
- **Location**: `/opt/sutazaiapp/docker/services/agents/deployment-automation-master/Dockerfile`
- **Type**: security
- **Severity**: critical

### ⚠️ latest_tag
- **Location**: `/opt/sutazaiapp/docker/services/agents/infrastructure-devops-manager/Dockerfile`
- **Type**: security
- **Severity**: critical

### ⚠️ latest_tag
- **Location**: `/opt/sutazaiapp/docker/services/agents/senior-ai-engineer/Dockerfile`
- **Type**: security
- **Severity**: critical

### ⚠️ curl_pipe_sh
- **Location**: `/opt/sutazaiapp/self-healing/Dockerfile`
- **Type**: security
- **Severity**: critical

### ⚠️ latest_tag
- **Location**: `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.localagi.agi_backup`
- **Type**: security
- **Severity**: critical

### ⚠️ latest_tag
- **Location**: `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.localagi`
- **Type**: security
- **Severity**: critical

### ⚠️ latest_tag
- **Location**: `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.tabbyml`
- **Type**: security
- **Severity**: critical

### ⚠️ privileged_container
- **Location**: `/opt/sutazaiapp/docker-compose.monitoring.yml`
- **Type**: security
- **Severity**: critical

### ⚠️ privileged_container
- **Location**: `/opt/sutazaiapp/docker-compose.yml`
- **Type**: security
- **Severity**: critical

### ⚠️ privileged_container
- **Location**: `/opt/sutazaiapp/deployment/monitoring/docker-compose-monitoring.yml`
- **Type**: security
- **Severity**: critical

### ⚠️ privileged_container
- **Location**: `/opt/sutazaiapp/archive/20250803_193506_pre_cleanup/docker-compose.monitoring.yml`
- **Type**: security
- **Severity**: critical

### ⚠️ privileged_container
- **Location**: `/opt/sutazaiapp/archive/20250803_193506_pre_cleanup/docker-compose.yml`
- **Type**: security
- **Severity**: critical

## Recommendations

### HIGH: Structure
- Reorganize Docker files into proper docker/ directory structure

### MEDIUM: Optimization
- Consider multi-stage builds for 17 large Dockerfiles

### CRITICAL: Security
- Address critical security issues immediately

## Detailed Findings

### Structure Issues
- **missing_subdirectory**: `docker/backend/` - Create docker/backend/ directory
- **missing_subdirectory**: `docker/frontend/` - Create docker/frontend/ directory
- **missing_dockerignore**: `docker/.dockerignore` - Create .dockerignore file
- **misplaced_dockerfile**: `/opt/sutazaiapp/backend/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/frontend/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/self-healing/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/mcp_server/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/workflows/deployments/Dockerfile.task_router` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/workflows/deployments/Dockerfile.monitor` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/workflows/deployments/Dockerfile.self_healer` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/workflows/deployments/Dockerfile.workflow_manager` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/monitoring/ai-metrics-exporter/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/deployment/monitoring/Dockerfile.agent-monitor` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/services/llamaindex/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/services/faiss/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/services/chainlit/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/distributed-computing-architect/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/autogpt/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/autogen/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/agentzero-coordinator/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/cognitive-architecture-designer/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/product-strategy-architect/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/health-monitor/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/multi-modal-fusion-coordinator/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/senior-frontend-developer/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.crewai` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.langflow` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.privategpt` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.localagi.agi_backup` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.shellgpt` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.gpt-engineer` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.localagi` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.langchain` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.agentgpt` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.documind` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.llamaindex` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.bigagi.agi_backup` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.agentzero` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.finrobot` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.autogen` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.tabbyml` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.browseruse` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.flowise` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.skyvern` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.bigagi` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.semgrep` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.autogpt` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.aider` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.dify` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/private-data-analyst/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/senior-backend-developer/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/synthetic-data-generator/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/reinforcement-learning-trainer/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/flowiseai-flow-manager/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/ollama-integration/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/crewai/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/data-analysis-engineer/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/agent-message-bus/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/meta-learning-specialist/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/mcp-server/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/ai-product-manager/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/opendevin-code-generator/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/context-framework/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/task-assignment-coordinator/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/letta/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/gpt-engineer/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/agent-registry/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/infrastructure-devops/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/awesome-code-ai/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/devika/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/model-training-specialist/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/bigagi-system-manager/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/episodic-memory-engineer/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/agentgpt-autonomous-executor/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/langflow-workflow-designer/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/knowledge-graph-builder/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/browser-automation-orchestrator/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/neuromorphic-computing-expert/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/dify-automation-specialist/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/quantum-computing-optimizer/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/transformers-migration-specialist/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/semgrep-security-analyzer/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/ai-scrum-master/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/context-optimizer/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/privategpt/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/knowledge-distillation-expert/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/memory-persistence-manager/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/explainable-ai-specialist/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/jarvis-voice-interface/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/garbage-collector-coordinator/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/fsdp/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/agentgpt/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/shellgpt/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/babyagi/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/observability-monitoring-engineer/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/service-hub/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/self-healing-orchestrator/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/symbolic-reasoning-engine/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/intelligence-optimization-monitor/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/attention-optimizer/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/data-pipeline-engineer/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/code-improver/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/code-generation-improver/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/ai-agent-debugger/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/causal-inference-expert/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/federated-learning-coordinator/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/document-knowledge-manager/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/aider/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/senior-ai-engineer/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/hardware-optimizer/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/gradient-compression-specialist/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/edge-computing-optimizer/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/localagi-orchestration-manager/Dockerfile` - Move to appropriate docker/ subdirectory
- **misplaced_dockerfile**: `/opt/sutazaiapp/agents/finrobot/Dockerfile` - Move to appropriate docker/ subdirectory

### Dockerfile Analysis

#### `/opt/sutazaiapp/docker/Dockerfile.healthcheck` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/autogpt/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/autogpt/Dockerfile.simple` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/autogen/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/opendevin/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/jax/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/health-monitor/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/flowise/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/agentzero/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/context-engineering/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/llamaindex/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/crewai/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/tensorflow/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/faiss/Dockerfile` - Status: needs_fixes
- no_version_pinning: Image 'application AS development

CMD ["python", "faiss_service.py"]
' not version-pinned

#### `/opt/sutazaiapp/docker/context-framework/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/enhanced-model-manager/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/letta/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/gpt-engineer/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/health-check/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/autogpt-real/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/awesome-code-ai/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/task-scheduler/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/tabbyml/Dockerfile` - Status: critical
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/api-gateway/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/model-optimizer/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/localagi/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/localagi/Dockerfile.agi_backup` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/semgrep/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/pytorch/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/privategpt/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/fsdp/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/dify/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/agentgpt/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/shellgpt/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/knowledge-manager/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/documind/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/langflow/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/service-hub/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/bigagi/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/tabbyml-cpu/Dockerfile` - Status: critical
- no_version_pinning: Image 'tabbyml/tabby' not version-pinned
- no_user_context: Dockerfile runs as root user
- no_workdir: No WORKDIR specified
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/pentestgpt/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/agi-lightweight/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/fms-fsdp/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/code-improver/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/skyvern/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/jarvis-ai/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/jarvis-ai/Dockerfile.agi_backup` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/agents/Dockerfile.pentestgpt` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/agents/Dockerfile.crewai` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/agents/Dockerfile.langflow` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/agents/Dockerfile.privategpt` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/agents/Dockerfile.shellgpt` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/agents/Dockerfile.gpt-engineer` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/agents/Dockerfile.langchain` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/agents/Dockerfile.agentgpt` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/agents/Dockerfile.documind` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/agents/Dockerfile.llamaindex` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/agents/Dockerfile.agentzero` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/agents/Dockerfile.finrobot` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/agents/Dockerfile.autogen` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/agents/Dockerfile.tabbyml` - Status: critical
- no_version_pinning: Image 'tabbyml/tabby' not version-pinned
- no_user_context: Dockerfile runs as root user
- no_workdir: No WORKDIR specified

#### `/opt/sutazaiapp/docker/agents/Dockerfile.browseruse` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/agents/Dockerfile.flowise` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/agents/Dockerfile.skyvern` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/agents/Dockerfile.semgrep` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/agents/Dockerfile.autogpt` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/agents/Dockerfile.aider` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/agents/Dockerfile.dify` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/jarvis-agi/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/aider/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/langchain-agents/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/browser-use/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/finrobot/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/services/infrastructure/ollama/Dockerfile` - Status: critical
- no_version_pinning: Image 'ollama/ollama' not version-pinned
- no_workdir: No WORKDIR specified
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/services/monitoring/prometheus/Dockerfile` - Status: critical
- no_version_pinning: Image 'prom/prometheus' not version-pinned
- no_workdir: No WORKDIR specified
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/services/agents/deployment-automation-master/Dockerfile` - Status: critical
- no_version_pinning: Image 'sutazai/agent-base' not version-pinned
- no_workdir: No WORKDIR specified
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/services/agents/infrastructure-devops-manager/Dockerfile` - Status: critical
- no_version_pinning: Image 'sutazai/agent-base' not version-pinned
- no_workdir: No WORKDIR specified
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/services/agents/senior-ai-engineer/Dockerfile` - Status: critical
- no_version_pinning: Image 'sutazai/agent-base' not version-pinned
- no_workdir: No WORKDIR specified
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/frontend/Dockerfile` - Status: needs_fixes
- no_version_pinning: Image 'application AS development

# Enable auto-reload for development
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.runOnSave=true", "--browser.gatherUsageStats=false"]' not version-pinned
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/self-healing/Dockerfile` - Status: critical
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/workflows/deployments/Dockerfile.task_router` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/workflows/deployments/Dockerfile.monitor` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/workflows/deployments/Dockerfile.self_healer` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/workflows/deployments/Dockerfile.workflow_manager` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/deployment/monitoring/Dockerfile.agent-monitor` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/services/llamaindex/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/services/faiss/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/services/chainlit/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/distributed-computing-architect/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/autogpt/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/autogen/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/agentzero-coordinator/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/cognitive-architecture-designer/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/product-strategy-architect/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/health-monitor/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/multi-modal-fusion-coordinator/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/senior-frontend-developer/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.crewai` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.langflow` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.privategpt` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.localagi.agi_backup` - Status: critical
- no_version_pinning: Image 'alpine' not version-pinned
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.shellgpt` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.gpt-engineer` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.localagi` - Status: critical
- no_version_pinning: Image 'alpine' not version-pinned
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.langchain` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.agentgpt` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.documind` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.llamaindex` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.bigagi.agi_backup` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.agentzero` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.finrobot` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.autogen` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.tabbyml` - Status: critical
- no_version_pinning: Image 'tabbyml/tabby' not version-pinned
- no_user_context: Dockerfile runs as root user
- no_workdir: No WORKDIR specified
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.browseruse` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.flowise` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.skyvern` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.bigagi` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.semgrep` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.autogpt` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.aider` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/dockerfiles/Dockerfile.dify` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/private-data-analyst/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/senior-backend-developer/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/synthetic-data-generator/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/reinforcement-learning-trainer/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/flowiseai-flow-manager/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/ollama-integration/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/crewai/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/data-analysis-engineer/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/agent-message-bus/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/meta-learning-specialist/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/mcp-server/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/ai-product-manager/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/opendevin-code-generator/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/context-framework/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/task-assignment-coordinator/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/letta/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/gpt-engineer/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/agent-registry/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/infrastructure-devops/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/awesome-code-ai/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/devika/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/model-training-specialist/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/bigagi-system-manager/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/episodic-memory-engineer/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/agentgpt-autonomous-executor/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/langflow-workflow-designer/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/knowledge-graph-builder/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/browser-automation-orchestrator/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/neuromorphic-computing-expert/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/dify-automation-specialist/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/quantum-computing-optimizer/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/transformers-migration-specialist/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/semgrep-security-analyzer/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/ai-scrum-master/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/privategpt/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/knowledge-distillation-expert/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/memory-persistence-manager/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/explainable-ai-specialist/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/jarvis-voice-interface/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/garbage-collector-coordinator/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/fsdp/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/agentgpt/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/shellgpt/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/babyagi/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/observability-monitoring-engineer/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/service-hub/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/self-healing-orchestrator/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/symbolic-reasoning-engine/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/intelligence-optimization-monitor/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/attention-optimizer/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/data-pipeline-engineer/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/code-improver/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/code-generation-improver/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/ai-agent-debugger/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/causal-inference-expert/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/federated-learning-coordinator/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/document-knowledge-manager/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/aider/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/senior-ai-engineer/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/hardware-optimizer/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/gradient-compression-specialist/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/edge-computing-optimizer/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/localagi-orchestration-manager/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/agents/finrobot/Dockerfile` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found
