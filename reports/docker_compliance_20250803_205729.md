# Docker Compliance Validation Report

Generated: Sun Aug  3 20:57:29 CEST 2025

## Validation Summary

### Docker Structure Validation (Rule 11)
Latest report: /opt/sutazaiapp/reports/docker_validation_20250803_205729.md
Generated: 2025-08-03T20:57:29.467455

## Compliance Summary

**Overall Score: 33.33%**
**Status: NON_COMPLIANT**

### Component Scores
- Structure Compliance: 100%
- Dockerfile Standards: 0%
- Compose Configuration: 0%

## Issues Summary

- Structure Issues: 0
- Dockerfile Issues: 340
- Compose Issues: 134
- Auto-fixes Applied: 148

## Critical Issues

### ⚠️ latest_tag
- **Location**: `/opt/sutazaiapp/docker/dockerfiles/Dockerfile.localagi.agi_backup`
- **Type**: security
- **Severity**: critical

### ⚠️ latest_tag
- **Location**: `/opt/sutazaiapp/docker/dockerfiles/Dockerfile.localagi.bak`
- **Type**: security
- **Severity**: critical

### ⚠️ latest_tag
- **Location**: `/opt/sutazaiapp/docker/dockerfiles/Dockerfile.localagi`
- **Type**: security
- **Severity**: critical

### ⚠️ latest_tag
- **Location**: `/opt/sutazaiapp/docker/dockerfiles/Dockerfile.tabbyml`
- **Type**: security
- **Severity**: critical

### ⚠️ curl_pipe_sh
- **Location**: `/opt/sutazaiapp/docker/tabbyml/Dockerfile`
- **Type**: security
- **Severity**: critical

### ⚠️ curl_pipe_sh
- **Location**: `/opt/sutazaiapp/docker/tabbyml/Dockerfile.bak`
- **Type**: security
- **Severity**: critical

### ⚠️ curl_pipe_sh
- **Location**: `/opt/sutazaiapp/docker/self-healing/Dockerfile`
- **Type**: security
- **Severity**: critical

### ⚠️ curl_pipe_sh
- **Location**: `/opt/sutazaiapp/docker/self-healing/Dockerfile.bak`
- **Type**: security
- **Severity**: critical

### ⚠️ latest_tag
- **Location**: `/opt/sutazaiapp/docker/tabbyml-cpu/Dockerfile`
- **Type**: security
- **Severity**: critical

### ⚠️ latest_tag
- **Location**: `/opt/sutazaiapp/docker/tabbyml-cpu/Dockerfile.bak`
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
- **Location**: `/opt/sutazaiapp/docker/services/infrastructure/ollama/Dockerfile.bak`
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
- **Location**: `/opt/sutazaiapp/docker/services/agents/deployment-automation-master/Dockerfile.bak`
- **Type**: security
- **Severity**: critical

### ⚠️ latest_tag
- **Location**: `/opt/sutazaiapp/docker/services/agents/infrastructure-devops-manager/Dockerfile`
- **Type**: security
- **Severity**: critical

### ⚠️ latest_tag
- **Location**: `/opt/sutazaiapp/docker/services/agents/infrastructure-devops-manager/Dockerfile.bak`
- **Type**: security
- **Severity**: critical

### ⚠️ latest_tag
- **Location**: `/opt/sutazaiapp/docker/services/agents/senior-ai-engineer/Dockerfile`
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

### MEDIUM: Optimization
- Consider multi-stage builds for 22 large Dockerfiles

### CRITICAL: Security
- Address critical security issues immediately

## Detailed Findings

### Dockerfile Analysis

#### `/opt/sutazaiapp/docker/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/distributed-computing-architect/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/autogpt/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/autogen/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/agentzero-coordinator/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/deployments/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/cognitive-architecture-designer/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/opendevin/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/product-strategy-architect/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/jax/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/health-monitor/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/multi-modal-fusion-coordinator/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/flowise/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/agentzero/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/dockerfiles/Dockerfile.localagi.agi_backup` - Status: critical
- no_version_pinning: Image 'alpine' not version-pinned
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/dockerfiles/Dockerfile.bigagi.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/dockerfiles/Dockerfile.localagi.bak` - Status: critical
- no_version_pinning: Image 'alpine' not version-pinned
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/dockerfiles/Dockerfile.localagi` - Status: critical
- no_version_pinning: Image 'alpine' not version-pinned
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/dockerfiles/Dockerfile.tabbyml` - Status: critical
- no_version_pinning: Image 'tabbyml/tabby' not version-pinned
- no_workdir: No WORKDIR specified
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/dockerfiles/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/private-data-analyst/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/context-engineering/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/synthetic-data-generator/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/reinforcement-learning-trainer/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/flowiseai-flow-manager/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/llamaindex/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/ollama-integration/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/crewai/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/data-analysis-engineer/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/agent-message-bus/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/meta-learning-specialist/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/tensorflow/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/mcp-server/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/faiss/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/ai-product-manager/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/opendevin-code-generator/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/context-framework/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/task-assignment-coordinator/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/enhanced-model-manager/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/letta/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/gpt-engineer/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/agent-registry/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/health-check/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/infrastructure-devops/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/autogpt-real/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/awesome-code-ai/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/task-scheduler/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/devika/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/model-training-specialist/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/bigagi-system-manager/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/episodic-memory-engineer/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/agentgpt-autonomous-executor/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/tabbyml/Dockerfile` - Status: critical

#### `/opt/sutazaiapp/docker/tabbyml/Dockerfile.bak` - Status: critical
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/langflow-workflow-designer/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/knowledge-graph-builder/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/backend/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/api-gateway/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/browser-automation-orchestrator/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/model-optimizer/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/localagi/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/neuromorphic-computing-expert/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/dify-automation-specialist/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/semgrep/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/quantum-computing-optimizer/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/pytorch/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/transformers-migration-specialist/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/semgrep-security-analyzer/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/ai-scrum-master/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/privategpt/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/knowledge-distillation-expert/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/memory-persistence-manager/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/explainable-ai-specialist/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/jarvis-voice-interface/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/garbage-collector-coordinator/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/frontend/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/fsdp/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/monitoring/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/dify/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/agentgpt/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/self-healing/Dockerfile` - Status: critical
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/self-healing/Dockerfile.bak` - Status: critical
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/shellgpt/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/babyagi/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/knowledge-manager/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/observability-monitoring-engineer/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/documind/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/langflow/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/service-hub/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/bigagi/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/tabbyml-cpu/Dockerfile` - Status: critical
- no_version_pinning: Image 'tabbyml/tabby' not version-pinned
- no_workdir: No WORKDIR specified
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/tabbyml-cpu/Dockerfile.bak` - Status: critical
- no_version_pinning: Image 'tabbyml/tabby' not version-pinned
- no_user_context: Dockerfile runs as root user
- no_workdir: No WORKDIR specified
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/self-healing-orchestrator/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/chainlit/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/symbolic-reasoning-engine/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/pentestgpt/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/intelligence-optimization-monitor/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/attention-optimizer/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/agi-lightweight/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/fms-fsdp/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/data-pipeline-engineer/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/code-improver/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/code-generation-improver/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/skyvern/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/ai-agent-debugger/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/causal-inference-expert/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/federated-learning-coordinator/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/jarvis-ai/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/agents/Dockerfile.tabbyml` - Status: critical
- no_version_pinning: Image 'tabbyml/tabby' not version-pinned
- no_workdir: No WORKDIR specified

#### `/opt/sutazaiapp/docker/agents/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/document-knowledge-manager/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/jarvis-agi/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/aider/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/senior-ai-engineer/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/hardware-optimizer/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/gradient-compression-specialist/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/edge-computing-optimizer/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/langchain-agents/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/localagi-orchestration-manager/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/browser-use/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/finrobot/Dockerfile.bak` - Status: needs_fixes
- no_user_context: Dockerfile runs as root user

#### `/opt/sutazaiapp/docker/services/infrastructure/ollama/Dockerfile` - Status: critical
- no_version_pinning: Image 'ollama/ollama' not version-pinned
- no_workdir: No WORKDIR specified
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/services/infrastructure/ollama/Dockerfile.bak` - Status: critical
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

#### `/opt/sutazaiapp/docker/services/agents/deployment-automation-master/Dockerfile.bak` - Status: critical
- no_version_pinning: Image 'sutazai/agent-base' not version-pinned
- no_workdir: No WORKDIR specified
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/services/agents/infrastructure-devops-manager/Dockerfile` - Status: critical
- no_version_pinning: Image 'sutazai/agent-base' not version-pinned
- no_workdir: No WORKDIR specified
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/services/agents/infrastructure-devops-manager/Dockerfile.bak` - Status: critical
- no_version_pinning: Image 'sutazai/agent-base' not version-pinned
- no_workdir: No WORKDIR specified
- missing_dockerignore: No .dockerignore file found

#### `/opt/sutazaiapp/docker/services/agents/senior-ai-engineer/Dockerfile` - Status: critical
- no_version_pinning: Image 'sutazai/agent-base' not version-pinned
- no_workdir: No WORKDIR specified
- missing_dockerignore: No .dockerignore file found
