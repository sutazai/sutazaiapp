#!/bin/bash
# Rollback script for requirements cleanup
# Generated: 2025-08-03T19:28:21.266577

set -e

echo "🔄 Rolling back requirements files..."

PROJECT_ROOT="/opt/sutazaiapp"
BACKUP_ROOT="/opt/sutazaiapp/archive/requirements_cleanup_20250803_192821"

echo "Restoring requirements.txt"
cp "$BACKUP_ROOT/requirements.txt" "$PROJECT_ROOT/requirements.txt"

echo "Restoring pyproject.toml"
cp "$BACKUP_ROOT/pyproject.toml" "$PROJECT_ROOT/pyproject.toml"

echo "Restoring backend/requirements-test.txt"
cp "$BACKUP_ROOT/backend/requirements-test.txt" "$PROJECT_ROOT/backend/requirements-test.txt"

echo "Restoring docs/requirements/backend/requirements-test.txt"
cp "$BACKUP_ROOT/docs/requirements/backend/requirements-test.txt" "$PROJECT_ROOT/docs/requirements/backend/requirements-test.txt"

echo "Restoring backend/requirements.secure.txt"
cp "$BACKUP_ROOT/backend/requirements.secure.txt" "$PROJECT_ROOT/backend/requirements.secure.txt"

echo "Restoring frontend/requirements.secure.txt"
cp "$BACKUP_ROOT/frontend/requirements.secure.txt" "$PROJECT_ROOT/frontend/requirements.secure.txt"

echo "Restoring backend/requirements.minimal.txt"
cp "$BACKUP_ROOT/backend/requirements.minimal.txt" "$PROJECT_ROOT/backend/requirements.minimal.txt"

echo "Restoring backend/requirements-minimal.txt"
cp "$BACKUP_ROOT/backend/requirements-minimal.txt" "$PROJECT_ROOT/backend/requirements-minimal.txt"

echo "Restoring backend/requirements.txt"
cp "$BACKUP_ROOT/backend/requirements.txt" "$PROJECT_ROOT/backend/requirements.txt"

echo "Restoring frontend/requirements.txt"
cp "$BACKUP_ROOT/frontend/requirements.txt" "$PROJECT_ROOT/frontend/requirements.txt"

echo "Restoring agents/requirements.txt"
cp "$BACKUP_ROOT/agents/requirements.txt" "$PROJECT_ROOT/agents/requirements.txt"

echo "Restoring docs/requirements/backend/requirements.txt"
cp "$BACKUP_ROOT/docs/requirements/backend/requirements.txt" "$PROJECT_ROOT/docs/requirements/backend/requirements.txt"

echo "Restoring docs/requirements/frontend/requirements.txt"
cp "$BACKUP_ROOT/docs/requirements/frontend/requirements.txt" "$PROJECT_ROOT/docs/requirements/frontend/requirements.txt"

echo "Restoring backend/requirements-fast.txt"
cp "$BACKUP_ROOT/backend/requirements-fast.txt" "$PROJECT_ROOT/backend/requirements-fast.txt"

echo "Restoring system-validator/requirements.txt"
cp "$BACKUP_ROOT/system-validator/requirements.txt" "$PROJECT_ROOT/system-validator/requirements.txt"

echo "Restoring tests/requirements-test.txt"
cp "$BACKUP_ROOT/tests/requirements-test.txt" "$PROJECT_ROOT/tests/requirements-test.txt"

echo "Restoring docs/requirements/tests/requirements-test.txt"
cp "$BACKUP_ROOT/docs/requirements/tests/requirements-test.txt" "$PROJECT_ROOT/docs/requirements/tests/requirements-test.txt"

echo "Restoring self-healing/requirements.txt"
cp "$BACKUP_ROOT/self-healing/requirements.txt" "$PROJECT_ROOT/self-healing/requirements.txt"

echo "Restoring docker/autogpt/requirements.txt"
cp "$BACKUP_ROOT/docker/autogpt/requirements.txt" "$PROJECT_ROOT/docker/autogpt/requirements.txt"

echo "Restoring docs/requirements/autogpt/requirements.txt"
cp "$BACKUP_ROOT/docs/requirements/autogpt/requirements.txt" "$PROJECT_ROOT/docs/requirements/autogpt/requirements.txt"

echo "Restoring agents/autogpt/requirements.txt"
cp "$BACKUP_ROOT/agents/autogpt/requirements.txt" "$PROJECT_ROOT/agents/autogpt/requirements.txt"

echo "Restoring docker/health-monitor/requirements.txt"
cp "$BACKUP_ROOT/docker/health-monitor/requirements.txt" "$PROJECT_ROOT/docker/health-monitor/requirements.txt"

echo "Restoring docs/requirements/health-monitor/requirements.txt"
cp "$BACKUP_ROOT/docs/requirements/health-monitor/requirements.txt" "$PROJECT_ROOT/docs/requirements/health-monitor/requirements.txt"

echo "Restoring agents/health-monitor/requirements.txt"
cp "$BACKUP_ROOT/agents/health-monitor/requirements.txt" "$PROJECT_ROOT/agents/health-monitor/requirements.txt"

echo "Restoring docker/crewai/requirements.txt"
cp "$BACKUP_ROOT/docker/crewai/requirements.txt" "$PROJECT_ROOT/docker/crewai/requirements.txt"

echo "Restoring docs/requirements/crewai/requirements.txt"
cp "$BACKUP_ROOT/docs/requirements/crewai/requirements.txt" "$PROJECT_ROOT/docs/requirements/crewai/requirements.txt"

echo "Restoring agents/crewai/requirements.txt"
cp "$BACKUP_ROOT/agents/crewai/requirements.txt" "$PROJECT_ROOT/agents/crewai/requirements.txt"

echo "Restoring docker/requirements/requirements.secure.txt"
cp "$BACKUP_ROOT/docker/requirements/requirements.secure.txt" "$PROJECT_ROOT/docker/requirements/requirements.secure.txt"

echo "Restoring docker/requirements/requirements.txt"
cp "$BACKUP_ROOT/docker/requirements/requirements.txt" "$PROJECT_ROOT/docker/requirements/requirements.txt"

echo "Restoring docs/requirements/requirements/requirements.txt"
cp "$BACKUP_ROOT/docs/requirements/requirements/requirements.txt" "$PROJECT_ROOT/docs/requirements/requirements/requirements.txt"

echo "Restoring docker/letta/requirements.txt"
cp "$BACKUP_ROOT/docker/letta/requirements.txt" "$PROJECT_ROOT/docker/letta/requirements.txt"

echo "Restoring docs/requirements/letta/requirements.txt"
cp "$BACKUP_ROOT/docs/requirements/letta/requirements.txt" "$PROJECT_ROOT/docs/requirements/letta/requirements.txt"

echo "Restoring agents/letta/requirements.txt"
cp "$BACKUP_ROOT/agents/letta/requirements.txt" "$PROJECT_ROOT/agents/letta/requirements.txt"

echo "Restoring docker/gpt-engineer/requirements.txt"
cp "$BACKUP_ROOT/docker/gpt-engineer/requirements.txt" "$PROJECT_ROOT/docker/gpt-engineer/requirements.txt"

echo "Restoring docs/requirements/gpt-engineer/requirements.txt"
cp "$BACKUP_ROOT/docs/requirements/gpt-engineer/requirements.txt" "$PROJECT_ROOT/docs/requirements/gpt-engineer/requirements.txt"

echo "Restoring agents/gpt-engineer/requirements.txt"
cp "$BACKUP_ROOT/agents/gpt-engineer/requirements.txt" "$PROJECT_ROOT/agents/gpt-engineer/requirements.txt"

echo "Restoring docker/awesome-code-ai/requirements.txt"
cp "$BACKUP_ROOT/docker/awesome-code-ai/requirements.txt" "$PROJECT_ROOT/docker/awesome-code-ai/requirements.txt"

echo "Restoring docs/requirements/awesome-code-ai/requirements.txt"
cp "$BACKUP_ROOT/docs/requirements/awesome-code-ai/requirements.txt" "$PROJECT_ROOT/docs/requirements/awesome-code-ai/requirements.txt"

echo "Restoring agents/awesome-code-ai/requirements.txt"
cp "$BACKUP_ROOT/agents/awesome-code-ai/requirements.txt" "$PROJECT_ROOT/agents/awesome-code-ai/requirements.txt"

echo "Restoring docker/base/requirements-base.txt"
cp "$BACKUP_ROOT/docker/base/requirements-base.txt" "$PROJECT_ROOT/docker/base/requirements-base.txt"

echo "Restoring docker/base/requirements-agent.txt"
cp "$BACKUP_ROOT/docker/base/requirements-agent.txt" "$PROJECT_ROOT/docker/base/requirements-agent.txt"

echo "Restoring docker/base/requirements-security.txt"
cp "$BACKUP_ROOT/docker/base/requirements-security.txt" "$PROJECT_ROOT/docker/base/requirements-security.txt"

echo "Restoring docker/api-gateway/requirements.txt"
cp "$BACKUP_ROOT/docker/api-gateway/requirements.txt" "$PROJECT_ROOT/docker/api-gateway/requirements.txt"

echo "Restoring docker/semgrep/requirements.txt"
cp "$BACKUP_ROOT/docker/semgrep/requirements.txt" "$PROJECT_ROOT/docker/semgrep/requirements.txt"

echo "Restoring docs/requirements/semgrep/requirements.txt"
cp "$BACKUP_ROOT/docs/requirements/semgrep/requirements.txt" "$PROJECT_ROOT/docs/requirements/semgrep/requirements.txt"

echo "Restoring docker/fsdp/requirements.txt"
cp "$BACKUP_ROOT/docker/fsdp/requirements.txt" "$PROJECT_ROOT/docker/fsdp/requirements.txt"

echo "Restoring agents/fsdp/requirements.txt"
cp "$BACKUP_ROOT/agents/fsdp/requirements.txt" "$PROJECT_ROOT/agents/fsdp/requirements.txt"

echo "Restoring docker/pentestgpt/requirements.txt"
cp "$BACKUP_ROOT/docker/pentestgpt/requirements.txt" "$PROJECT_ROOT/docker/pentestgpt/requirements.txt"

echo "Restoring docs/requirements/pentestgpt/requirements.txt"
cp "$BACKUP_ROOT/docs/requirements/pentestgpt/requirements.txt" "$PROJECT_ROOT/docs/requirements/pentestgpt/requirements.txt"

echo "Restoring agents/pentestgpt/requirements.txt"
cp "$BACKUP_ROOT/agents/pentestgpt/requirements.txt" "$PROJECT_ROOT/agents/pentestgpt/requirements.txt"

echo "Restoring docker/aider/requirements.txt"
cp "$BACKUP_ROOT/docker/aider/requirements.txt" "$PROJECT_ROOT/docker/aider/requirements.txt"

echo "Restoring docs/requirements/aider/requirements.txt"
cp "$BACKUP_ROOT/docs/requirements/aider/requirements.txt" "$PROJECT_ROOT/docs/requirements/aider/requirements.txt"

echo "Restoring agents/aider/requirements.txt"
cp "$BACKUP_ROOT/agents/aider/requirements.txt" "$PROJECT_ROOT/agents/aider/requirements.txt"

echo "Restoring monitoring/ai-metrics-exporter/requirements.txt"
cp "$BACKUP_ROOT/monitoring/ai-metrics-exporter/requirements.txt" "$PROJECT_ROOT/monitoring/ai-metrics-exporter/requirements.txt"

echo "Restoring docs/requirements/deployments/requirements.txt"
cp "$BACKUP_ROOT/docs/requirements/deployments/requirements.txt" "$PROJECT_ROOT/docs/requirements/deployments/requirements.txt"

echo "Restoring docs/requirements/system/requirements.txt"
cp "$BACKUP_ROOT/docs/requirements/system/requirements.txt" "$PROJECT_ROOT/docs/requirements/system/requirements.txt"

echo "Restoring docs/requirements/ollama-integration/requirements.txt"
cp "$BACKUP_ROOT/docs/requirements/ollama-integration/requirements.txt" "$PROJECT_ROOT/docs/requirements/ollama-integration/requirements.txt"

echo "Restoring docs/requirements/agent-message-bus/requirements.txt"
cp "$BACKUP_ROOT/docs/requirements/agent-message-bus/requirements.txt" "$PROJECT_ROOT/docs/requirements/agent-message-bus/requirements.txt"

echo "Restoring docs/requirements/agent-registry/requirements.txt"
cp "$BACKUP_ROOT/docs/requirements/agent-registry/requirements.txt" "$PROJECT_ROOT/docs/requirements/agent-registry/requirements.txt"

echo "Restoring docs/requirements/infrastructure-devops/requirements.txt"
cp "$BACKUP_ROOT/docs/requirements/infrastructure-devops/requirements.txt" "$PROJECT_ROOT/docs/requirements/infrastructure-devops/requirements.txt"

echo "Restoring agents/infrastructure-devops/requirements.txt"
cp "$BACKUP_ROOT/agents/infrastructure-devops/requirements.txt" "$PROJECT_ROOT/agents/infrastructure-devops/requirements.txt"

echo "Restoring docs/requirements/localagi/requirements.txt"
cp "$BACKUP_ROOT/docs/requirements/localagi/requirements.txt" "$PROJECT_ROOT/docs/requirements/localagi/requirements.txt"

echo "Restoring docs/requirements/brain/requirements_minimal.txt"
cp "$BACKUP_ROOT/docs/requirements/brain/requirements_minimal.txt" "$PROJECT_ROOT/docs/requirements/brain/requirements_minimal.txt"

echo "Restoring docs/requirements/web_learning/requirements.txt"
cp "$BACKUP_ROOT/docs/requirements/web_learning/requirements.txt" "$PROJECT_ROOT/docs/requirements/web_learning/requirements.txt"

echo "Restoring docs/requirements/context-optimizer/requirements.txt"
cp "$BACKUP_ROOT/docs/requirements/context-optimizer/requirements.txt" "$PROJECT_ROOT/docs/requirements/context-optimizer/requirements.txt"

echo "Restoring agents/context-optimizer/requirements.txt"
cp "$BACKUP_ROOT/agents/context-optimizer/requirements.txt" "$PROJECT_ROOT/agents/context-optimizer/requirements.txt"

echo "Restoring docs/requirements/universal-agent/requirements.txt"
cp "$BACKUP_ROOT/docs/requirements/universal-agent/requirements.txt" "$PROJECT_ROOT/docs/requirements/universal-agent/requirements.txt"

echo "Restoring docs/requirements/archive/requirements-test.txt"
cp "$BACKUP_ROOT/docs/requirements/archive/requirements-test.txt" "$PROJECT_ROOT/docs/requirements/archive/requirements-test.txt"

echo "Restoring docs/requirements/archive/requirements-optimized.txt"
cp "$BACKUP_ROOT/docs/requirements/archive/requirements-optimized.txt" "$PROJECT_ROOT/docs/requirements/archive/requirements-optimized.txt"

echo "Restoring docs/requirements/archive/requirements-minimal.txt"
cp "$BACKUP_ROOT/docs/requirements/archive/requirements-minimal.txt" "$PROJECT_ROOT/docs/requirements/archive/requirements-minimal.txt"

echo "Restoring docs/requirements/archive/requirements-agi.txt"
cp "$BACKUP_ROOT/docs/requirements/archive/requirements-agi.txt" "$PROJECT_ROOT/docs/requirements/archive/requirements-agi.txt"

echo "Restoring docs/requirements/jarvis-agi/requirements_super.txt"
cp "$BACKUP_ROOT/docs/requirements/jarvis-agi/requirements_super.txt" "$PROJECT_ROOT/docs/requirements/jarvis-agi/requirements_super.txt"

echo "Restoring docs/requirements/hardware-optimizer/requirements.txt"
cp "$BACKUP_ROOT/docs/requirements/hardware-optimizer/requirements.txt" "$PROJECT_ROOT/docs/requirements/hardware-optimizer/requirements.txt"

echo "Restoring agents/hardware-optimizer/requirements.txt"
cp "$BACKUP_ROOT/agents/hardware-optimizer/requirements.txt" "$PROJECT_ROOT/agents/hardware-optimizer/requirements.txt"

echo "Restoring deployment/monitoring/requirements-monitor.txt"
cp "$BACKUP_ROOT/deployment/monitoring/requirements-monitor.txt" "$PROJECT_ROOT/deployment/monitoring/requirements-monitor.txt"

echo "Restoring services/llamaindex/requirements.txt"
cp "$BACKUP_ROOT/services/llamaindex/requirements.txt" "$PROJECT_ROOT/services/llamaindex/requirements.txt"

echo "Restoring services/chainlit/requirements.txt"
cp "$BACKUP_ROOT/services/chainlit/requirements.txt" "$PROJECT_ROOT/services/chainlit/requirements.txt"

echo "Restoring agents/distributed-computing-architect/requirements.txt"
cp "$BACKUP_ROOT/agents/distributed-computing-architect/requirements.txt" "$PROJECT_ROOT/agents/distributed-computing-architect/requirements.txt"

echo "Restoring agents/autogen/requirements.txt"
cp "$BACKUP_ROOT/agents/autogen/requirements.txt" "$PROJECT_ROOT/agents/autogen/requirements.txt"

echo "Restoring agents/agentzero-coordinator/requirements.txt"
cp "$BACKUP_ROOT/agents/agentzero-coordinator/requirements.txt" "$PROJECT_ROOT/agents/agentzero-coordinator/requirements.txt"

echo "Restoring agents/cognitive-architecture-designer/requirements.txt"
cp "$BACKUP_ROOT/agents/cognitive-architecture-designer/requirements.txt" "$PROJECT_ROOT/agents/cognitive-architecture-designer/requirements.txt"

echo "Restoring agents/product-strategy-architect/requirements.txt"
cp "$BACKUP_ROOT/agents/product-strategy-architect/requirements.txt" "$PROJECT_ROOT/agents/product-strategy-architect/requirements.txt"

echo "Restoring agents/multi-modal-fusion-coordinator/requirements.txt"
cp "$BACKUP_ROOT/agents/multi-modal-fusion-coordinator/requirements.txt" "$PROJECT_ROOT/agents/multi-modal-fusion-coordinator/requirements.txt"

echo "Restoring agents/senior-frontend-developer/requirements.txt"
cp "$BACKUP_ROOT/agents/senior-frontend-developer/requirements.txt" "$PROJECT_ROOT/agents/senior-frontend-developer/requirements.txt"

echo "Restoring agents/private-data-analyst/requirements.txt"
cp "$BACKUP_ROOT/agents/private-data-analyst/requirements.txt" "$PROJECT_ROOT/agents/private-data-analyst/requirements.txt"

echo "Restoring agents/senior-backend-developer/requirements.txt"
cp "$BACKUP_ROOT/agents/senior-backend-developer/requirements.txt" "$PROJECT_ROOT/agents/senior-backend-developer/requirements.txt"

echo "Restoring agents/synthetic-data-generator/requirements.txt"
cp "$BACKUP_ROOT/agents/synthetic-data-generator/requirements.txt" "$PROJECT_ROOT/agents/synthetic-data-generator/requirements.txt"

echo "Restoring agents/reinforcement-learning-trainer/requirements.txt"
cp "$BACKUP_ROOT/agents/reinforcement-learning-trainer/requirements.txt" "$PROJECT_ROOT/agents/reinforcement-learning-trainer/requirements.txt"

echo "Restoring agents/flowiseai-flow-manager/requirements.txt"
cp "$BACKUP_ROOT/agents/flowiseai-flow-manager/requirements.txt" "$PROJECT_ROOT/agents/flowiseai-flow-manager/requirements.txt"

echo "Restoring agents/data-analysis-engineer/requirements.txt"
cp "$BACKUP_ROOT/agents/data-analysis-engineer/requirements.txt" "$PROJECT_ROOT/agents/data-analysis-engineer/requirements.txt"

echo "Restoring agents/meta-learning-specialist/requirements.txt"
cp "$BACKUP_ROOT/agents/meta-learning-specialist/requirements.txt" "$PROJECT_ROOT/agents/meta-learning-specialist/requirements.txt"

echo "Restoring agents/mcp-server/requirements.txt"
cp "$BACKUP_ROOT/agents/mcp-server/requirements.txt" "$PROJECT_ROOT/agents/mcp-server/requirements.txt"

echo "Restoring agents/ai-product-manager/requirements.txt"
cp "$BACKUP_ROOT/agents/ai-product-manager/requirements.txt" "$PROJECT_ROOT/agents/ai-product-manager/requirements.txt"

echo "Restoring agents/opendevin-code-generator/requirements.txt"
cp "$BACKUP_ROOT/agents/opendevin-code-generator/requirements.txt" "$PROJECT_ROOT/agents/opendevin-code-generator/requirements.txt"

echo "Restoring agents/context-framework/requirements.txt"
cp "$BACKUP_ROOT/agents/context-framework/requirements.txt" "$PROJECT_ROOT/agents/context-framework/requirements.txt"

echo "Restoring agents/task-assignment-coordinator/requirements.txt"
cp "$BACKUP_ROOT/agents/task-assignment-coordinator/requirements.txt" "$PROJECT_ROOT/agents/task-assignment-coordinator/requirements.txt"

echo "Restoring agents/devika/requirements.txt"
cp "$BACKUP_ROOT/agents/devika/requirements.txt" "$PROJECT_ROOT/agents/devika/requirements.txt"

echo "Restoring agents/model-training-specialist/requirements.txt"
cp "$BACKUP_ROOT/agents/model-training-specialist/requirements.txt" "$PROJECT_ROOT/agents/model-training-specialist/requirements.txt"

echo "Restoring agents/bigagi-system-manager/requirements.txt"
cp "$BACKUP_ROOT/agents/bigagi-system-manager/requirements.txt" "$PROJECT_ROOT/agents/bigagi-system-manager/requirements.txt"

echo "Restoring agents/episodic-memory-engineer/requirements.txt"
cp "$BACKUP_ROOT/agents/episodic-memory-engineer/requirements.txt" "$PROJECT_ROOT/agents/episodic-memory-engineer/requirements.txt"

echo "Restoring agents/agentgpt-autonomous-executor/requirements.txt"
cp "$BACKUP_ROOT/agents/agentgpt-autonomous-executor/requirements.txt" "$PROJECT_ROOT/agents/agentgpt-autonomous-executor/requirements.txt"

echo "Restoring agents/langflow-workflow-designer/requirements.txt"
cp "$BACKUP_ROOT/agents/langflow-workflow-designer/requirements.txt" "$PROJECT_ROOT/agents/langflow-workflow-designer/requirements.txt"

echo "Restoring agents/knowledge-graph-builder/requirements.txt"
cp "$BACKUP_ROOT/agents/knowledge-graph-builder/requirements.txt" "$PROJECT_ROOT/agents/knowledge-graph-builder/requirements.txt"

echo "Restoring agents/browser-automation-orchestrator/requirements.txt"
cp "$BACKUP_ROOT/agents/browser-automation-orchestrator/requirements.txt" "$PROJECT_ROOT/agents/browser-automation-orchestrator/requirements.txt"

echo "Restoring agents/neuromorphic-computing-expert/requirements.txt"
cp "$BACKUP_ROOT/agents/neuromorphic-computing-expert/requirements.txt" "$PROJECT_ROOT/agents/neuromorphic-computing-expert/requirements.txt"

echo "Restoring agents/dify-automation-specialist/requirements.txt"
cp "$BACKUP_ROOT/agents/dify-automation-specialist/requirements.txt" "$PROJECT_ROOT/agents/dify-automation-specialist/requirements.txt"

echo "Restoring agents/quantum-computing-optimizer/requirements.txt"
cp "$BACKUP_ROOT/agents/quantum-computing-optimizer/requirements.txt" "$PROJECT_ROOT/agents/quantum-computing-optimizer/requirements.txt"

echo "Restoring agents/transformers-migration-specialist/requirements.txt"
cp "$BACKUP_ROOT/agents/transformers-migration-specialist/requirements.txt" "$PROJECT_ROOT/agents/transformers-migration-specialist/requirements.txt"

echo "Restoring agents/semgrep-security-analyzer/requirements.txt"
cp "$BACKUP_ROOT/agents/semgrep-security-analyzer/requirements.txt" "$PROJECT_ROOT/agents/semgrep-security-analyzer/requirements.txt"

echo "Restoring agents/ai-scrum-master/requirements.txt"
cp "$BACKUP_ROOT/agents/ai-scrum-master/requirements.txt" "$PROJECT_ROOT/agents/ai-scrum-master/requirements.txt"

echo "Restoring agents/privategpt/requirements.txt"
cp "$BACKUP_ROOT/agents/privategpt/requirements.txt" "$PROJECT_ROOT/agents/privategpt/requirements.txt"

echo "Restoring agents/knowledge-distillation-expert/requirements.txt"
cp "$BACKUP_ROOT/agents/knowledge-distillation-expert/requirements.txt" "$PROJECT_ROOT/agents/knowledge-distillation-expert/requirements.txt"

echo "Restoring agents/memory-persistence-manager/requirements.txt"
cp "$BACKUP_ROOT/agents/memory-persistence-manager/requirements.txt" "$PROJECT_ROOT/agents/memory-persistence-manager/requirements.txt"

echo "Restoring agents/explainable-ai-specialist/requirements.txt"
cp "$BACKUP_ROOT/agents/explainable-ai-specialist/requirements.txt" "$PROJECT_ROOT/agents/explainable-ai-specialist/requirements.txt"

echo "Restoring agents/jarvis-voice-interface/requirements.txt"
cp "$BACKUP_ROOT/agents/jarvis-voice-interface/requirements.txt" "$PROJECT_ROOT/agents/jarvis-voice-interface/requirements.txt"

echo "Restoring agents/garbage-collector-coordinator/requirements.txt"
cp "$BACKUP_ROOT/agents/garbage-collector-coordinator/requirements.txt" "$PROJECT_ROOT/agents/garbage-collector-coordinator/requirements.txt"

echo "Restoring agents/agentgpt/requirements.txt"
cp "$BACKUP_ROOT/agents/agentgpt/requirements.txt" "$PROJECT_ROOT/agents/agentgpt/requirements.txt"

echo "Restoring docker/agentgpt/package.json"
cp "$BACKUP_ROOT/docker/agentgpt/package.json" "$PROJECT_ROOT/docker/agentgpt/package.json"

echo "Restoring agents/shellgpt/requirements.txt"
cp "$BACKUP_ROOT/agents/shellgpt/requirements.txt" "$PROJECT_ROOT/agents/shellgpt/requirements.txt"

echo "Restoring agents/babyagi/requirements.txt"
cp "$BACKUP_ROOT/agents/babyagi/requirements.txt" "$PROJECT_ROOT/agents/babyagi/requirements.txt"

echo "Restoring agents/observability-monitoring-engineer/requirements.txt"
cp "$BACKUP_ROOT/agents/observability-monitoring-engineer/requirements.txt" "$PROJECT_ROOT/agents/observability-monitoring-engineer/requirements.txt"

echo "Restoring agents/service-hub/requirements.txt"
cp "$BACKUP_ROOT/agents/service-hub/requirements.txt" "$PROJECT_ROOT/agents/service-hub/requirements.txt"

echo "Restoring agents/self-healing-orchestrator/requirements.txt"
cp "$BACKUP_ROOT/agents/self-healing-orchestrator/requirements.txt" "$PROJECT_ROOT/agents/self-healing-orchestrator/requirements.txt"

echo "Restoring agents/symbolic-reasoning-engine/requirements.txt"
cp "$BACKUP_ROOT/agents/symbolic-reasoning-engine/requirements.txt" "$PROJECT_ROOT/agents/symbolic-reasoning-engine/requirements.txt"

echo "Restoring agents/intelligence-optimization-monitor/requirements.txt"
cp "$BACKUP_ROOT/agents/intelligence-optimization-monitor/requirements.txt" "$PROJECT_ROOT/agents/intelligence-optimization-monitor/requirements.txt"

echo "Restoring agents/attention-optimizer/requirements.txt"
cp "$BACKUP_ROOT/agents/attention-optimizer/requirements.txt" "$PROJECT_ROOT/agents/attention-optimizer/requirements.txt"

echo "Restoring agents/data-pipeline-engineer/requirements.txt"
cp "$BACKUP_ROOT/agents/data-pipeline-engineer/requirements.txt" "$PROJECT_ROOT/agents/data-pipeline-engineer/requirements.txt"

echo "Restoring agents/code-improver/requirements.txt"
cp "$BACKUP_ROOT/agents/code-improver/requirements.txt" "$PROJECT_ROOT/agents/code-improver/requirements.txt"

echo "Restoring agents/ai-agent-debugger/requirements.txt"
cp "$BACKUP_ROOT/agents/ai-agent-debugger/requirements.txt" "$PROJECT_ROOT/agents/ai-agent-debugger/requirements.txt"

echo "Restoring agents/causal-inference-expert/requirements.txt"
cp "$BACKUP_ROOT/agents/causal-inference-expert/requirements.txt" "$PROJECT_ROOT/agents/causal-inference-expert/requirements.txt"

echo "Restoring agents/federated-learning-coordinator/requirements.txt"
cp "$BACKUP_ROOT/agents/federated-learning-coordinator/requirements.txt" "$PROJECT_ROOT/agents/federated-learning-coordinator/requirements.txt"

echo "Restoring agents/document-knowledge-manager/requirements.txt"
cp "$BACKUP_ROOT/agents/document-knowledge-manager/requirements.txt" "$PROJECT_ROOT/agents/document-knowledge-manager/requirements.txt"

echo "Restoring agents/gradient-compression-specialist/requirements.txt"
cp "$BACKUP_ROOT/agents/gradient-compression-specialist/requirements.txt" "$PROJECT_ROOT/agents/gradient-compression-specialist/requirements.txt"

echo "Restoring agents/edge-computing-optimizer/requirements.txt"
cp "$BACKUP_ROOT/agents/edge-computing-optimizer/requirements.txt" "$PROJECT_ROOT/agents/edge-computing-optimizer/requirements.txt"

echo "Restoring agents/localagi-orchestration-manager/requirements.txt"
cp "$BACKUP_ROOT/agents/localagi-orchestration-manager/requirements.txt" "$PROJECT_ROOT/agents/localagi-orchestration-manager/requirements.txt"

echo "Restoring agents/finrobot/requirements.txt"
cp "$BACKUP_ROOT/agents/finrobot/requirements.txt" "$PROJECT_ROOT/agents/finrobot/requirements.txt"

echo "Restoring mcp_server/package.json"
cp "$BACKUP_ROOT/mcp_server/package.json" "$PROJECT_ROOT/mcp_server/package.json"

echo "Restoring docker/flowise/package.json"
cp "$BACKUP_ROOT/docker/flowise/package.json" "$PROJECT_ROOT/docker/flowise/package.json"

echo "Restoring docker/dify/web/package.json"
cp "$BACKUP_ROOT/docker/dify/web/package.json" "$PROJECT_ROOT/docker/dify/web/package.json"

echo "Restoring config/project/package.json"
cp "$BACKUP_ROOT/config/project/package.json" "$PROJECT_ROOT/config/project/package.json"

echo "Restoring config/project/pyproject.toml"
cp "$BACKUP_ROOT/config/project/pyproject.toml" "$PROJECT_ROOT/config/project/pyproject.toml"

echo "Restoring data/n8n/nodes/package.json"
cp "$BACKUP_ROOT/data/n8n/nodes/package.json" "$PROJECT_ROOT/data/n8n/nodes/package.json"


echo "✅ Rollback complete!"
echo "Verify with: python scripts/validate-container-infrastructure.py --critical-only"
