# Requirements Cleanup Report

Generated: 2025-08-03T19:28:21.439468
Mode: DRY RUN
Backup Directory: `/opt/sutazaiapp/archive/requirements_cleanup_20250803_192821`

## Summary

- **Requirements Files Discovered**: 104
- **Exact Duplicates Found**: 7
- **Redundant Files Found**: 32
- **Critical Files Protected**: 4
- **Planned Actions**: 106

## Exact Duplicates

### Hash: e0f0162a...
- **Keep**: `/opt/sutazaiapp/frontend/requirements.secure.txt`
- **Remove**: 1 files
  - `/opt/sutazaiapp/docs/requirements/frontend/requirements.txt`

### Hash: b06254bb...
- **Keep**: `/opt/sutazaiapp/docker/health-monitor/requirements.txt`
- **Remove**: 2 files
  - `/opt/sutazaiapp/docker/semgrep/requirements.txt`
  - `/opt/sutazaiapp/docker/pentestgpt/requirements.txt`

### Hash: 130713ee...
- **Keep**: `/opt/sutazaiapp/docs/requirements/health-monitor/requirements.txt`
- **Remove**: 2 files
  - `/opt/sutazaiapp/docs/requirements/semgrep/requirements.txt`
  - `/opt/sutazaiapp/docs/requirements/pentestgpt/requirements.txt`

### Hash: c704e3d3...
- **Keep**: `/opt/sutazaiapp/agents/health-monitor/requirements.txt`
- **Remove**: 54 files
  - `/opt/sutazaiapp/agents/awesome-code-ai/requirements.txt`
  - `/opt/sutazaiapp/agents/fsdp/requirements.txt`
  - `/opt/sutazaiapp/agents/distributed-computing-architect/requirements.txt`
  - `/opt/sutazaiapp/agents/autogen/requirements.txt`
  - `/opt/sutazaiapp/agents/agentzero-coordinator/requirements.txt`
  - `/opt/sutazaiapp/agents/cognitive-architecture-designer/requirements.txt`
  - `/opt/sutazaiapp/agents/product-strategy-architect/requirements.txt`
  - `/opt/sutazaiapp/agents/multi-modal-fusion-coordinator/requirements.txt`
  - `/opt/sutazaiapp/agents/senior-frontend-developer/requirements.txt`
  - `/opt/sutazaiapp/agents/private-data-analyst/requirements.txt`
  - `/opt/sutazaiapp/agents/senior-backend-developer/requirements.txt`
  - `/opt/sutazaiapp/agents/synthetic-data-generator/requirements.txt`
  - `/opt/sutazaiapp/agents/reinforcement-learning-trainer/requirements.txt`
  - `/opt/sutazaiapp/agents/flowiseai-flow-manager/requirements.txt`
  - `/opt/sutazaiapp/agents/data-analysis-engineer/requirements.txt`
  - `/opt/sutazaiapp/agents/meta-learning-specialist/requirements.txt`
  - `/opt/sutazaiapp/agents/mcp-server/requirements.txt`
  - `/opt/sutazaiapp/agents/ai-product-manager/requirements.txt`
  - `/opt/sutazaiapp/agents/opendevin-code-generator/requirements.txt`
  - `/opt/sutazaiapp/agents/context-framework/requirements.txt`
  - `/opt/sutazaiapp/agents/task-assignment-coordinator/requirements.txt`
  - `/opt/sutazaiapp/agents/model-training-specialist/requirements.txt`
  - `/opt/sutazaiapp/agents/bigagi-system-manager/requirements.txt`
  - `/opt/sutazaiapp/agents/episodic-memory-engineer/requirements.txt`
  - `/opt/sutazaiapp/agents/langflow-workflow-designer/requirements.txt`
  - `/opt/sutazaiapp/agents/knowledge-graph-builder/requirements.txt`
  - `/opt/sutazaiapp/agents/browser-automation-orchestrator/requirements.txt`
  - `/opt/sutazaiapp/agents/neuromorphic-computing-expert/requirements.txt`
  - `/opt/sutazaiapp/agents/dify-automation-specialist/requirements.txt`
  - `/opt/sutazaiapp/agents/quantum-computing-optimizer/requirements.txt`
  - `/opt/sutazaiapp/agents/transformers-migration-specialist/requirements.txt`
  - `/opt/sutazaiapp/agents/semgrep-security-analyzer/requirements.txt`
  - `/opt/sutazaiapp/agents/ai-scrum-master/requirements.txt`
  - `/opt/sutazaiapp/agents/knowledge-distillation-expert/requirements.txt`
  - `/opt/sutazaiapp/agents/memory-persistence-manager/requirements.txt`
  - `/opt/sutazaiapp/agents/explainable-ai-specialist/requirements.txt`
  - `/opt/sutazaiapp/agents/jarvis-voice-interface/requirements.txt`
  - `/opt/sutazaiapp/agents/garbage-collector-coordinator/requirements.txt`
  - `/opt/sutazaiapp/agents/observability-monitoring-engineer/requirements.txt`
  - `/opt/sutazaiapp/agents/service-hub/requirements.txt`
  - `/opt/sutazaiapp/agents/self-healing-orchestrator/requirements.txt`
  - `/opt/sutazaiapp/agents/symbolic-reasoning-engine/requirements.txt`
  - `/opt/sutazaiapp/agents/intelligence-optimization-monitor/requirements.txt`
  - `/opt/sutazaiapp/agents/attention-optimizer/requirements.txt`
  - `/opt/sutazaiapp/agents/data-pipeline-engineer/requirements.txt`
  - `/opt/sutazaiapp/agents/code-improver/requirements.txt`
  - `/opt/sutazaiapp/agents/ai-agent-debugger/requirements.txt`
  - `/opt/sutazaiapp/agents/causal-inference-expert/requirements.txt`
  - `/opt/sutazaiapp/agents/federated-learning-coordinator/requirements.txt`
  - `/opt/sutazaiapp/agents/document-knowledge-manager/requirements.txt`
  - `/opt/sutazaiapp/agents/gradient-compression-specialist/requirements.txt`
  - `/opt/sutazaiapp/agents/edge-computing-optimizer/requirements.txt`
  - `/opt/sutazaiapp/agents/localagi-orchestration-manager/requirements.txt`
  - `/opt/sutazaiapp/agents/finrobot/requirements.txt`

### Hash: 6622fe7a...
- **Keep**: `/opt/sutazaiapp/docker/requirements/requirements.secure.txt`
- **Remove**: 1 files
  - `/opt/sutazaiapp/docker/requirements/requirements.txt`

### Hash: fd897041...
- **Keep**: `/opt/sutazaiapp/docs/requirements/gpt-engineer/requirements.txt`
- **Remove**: 1 files
  - `/opt/sutazaiapp/docs/requirements/aider/requirements.txt`

### Hash: c7d073c7...
- **Keep**: `/opt/sutazaiapp/agents/agentgpt-autonomous-executor/requirements.txt`
- **Remove**: 1 files
  - `/opt/sutazaiapp/agents/agentgpt/requirements.txt`


## Planned Actions

| Action | Path | Reason | Risk Level |
|--------|------|--------|------------|
| Remove | `/opt/sutazaiapp/docs/requirements/frontend/requirements.txt` | Exact duplicate of /opt/sutazaiapp/frontend/requirements.secure.txt | low |
| Remove | `/opt/sutazaiapp/docker/semgrep/requirements.txt` | Exact duplicate of /opt/sutazaiapp/docker/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/docker/pentestgpt/requirements.txt` | Exact duplicate of /opt/sutazaiapp/docker/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/docs/requirements/semgrep/requirements.txt` | Exact duplicate of /opt/sutazaiapp/docs/requirements/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/docs/requirements/pentestgpt/requirements.txt` | Exact duplicate of /opt/sutazaiapp/docs/requirements/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/awesome-code-ai/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/fsdp/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/distributed-computing-architect/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/autogen/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/agentzero-coordinator/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/cognitive-architecture-designer/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/product-strategy-architect/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/multi-modal-fusion-coordinator/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/senior-frontend-developer/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/private-data-analyst/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/senior-backend-developer/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/synthetic-data-generator/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/reinforcement-learning-trainer/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/flowiseai-flow-manager/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/data-analysis-engineer/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/meta-learning-specialist/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/mcp-server/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/ai-product-manager/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/opendevin-code-generator/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/context-framework/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/task-assignment-coordinator/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/model-training-specialist/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/bigagi-system-manager/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/episodic-memory-engineer/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/langflow-workflow-designer/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/knowledge-graph-builder/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/browser-automation-orchestrator/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/neuromorphic-computing-expert/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/dify-automation-specialist/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/quantum-computing-optimizer/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/transformers-migration-specialist/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/semgrep-security-analyzer/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/ai-scrum-master/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/knowledge-distillation-expert/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/memory-persistence-manager/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/explainable-ai-specialist/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/jarvis-voice-interface/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/garbage-collector-coordinator/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/observability-monitoring-engineer/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/service-hub/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/self-healing-orchestrator/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/symbolic-reasoning-engine/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/intelligence-optimization-monitor/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/attention-optimizer/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/data-pipeline-engineer/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/code-improver/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/ai-agent-debugger/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/causal-inference-expert/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/federated-learning-coordinator/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/document-knowledge-manager/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/gradient-compression-specialist/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/edge-computing-optimizer/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/localagi-orchestration-manager/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/finrobot/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/health-monitor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/docker/requirements/requirements.txt` | Exact duplicate of /opt/sutazaiapp/docker/requirements/requirements.secure.txt | low |
| Remove | `/opt/sutazaiapp/docs/requirements/aider/requirements.txt` | Exact duplicate of /opt/sutazaiapp/docs/requirements/gpt-engineer/requirements.txt | low |
| Remove | `/opt/sutazaiapp/agents/agentgpt/requirements.txt` | Exact duplicate of /opt/sutazaiapp/agents/agentgpt-autonomous-executor/requirements.txt | low |
| Remove | `/opt/sutazaiapp/docs/requirements/backend/requirements-test.txt` | Superseded by requirements-test.txt | medium |
| Remove | `/opt/sutazaiapp/backend/requirements.secure.txt` | Superseded by requirements.secure.txt | medium |
| Remove | `/opt/sutazaiapp/frontend/requirements.txt` | Superseded by requirements.txt | medium |
| Remove | `/opt/sutazaiapp/agents/requirements.txt` | Superseded by requirements.txt | medium |
| Remove | `/opt/sutazaiapp/docs/requirements/backend/requirements.txt` | Superseded by requirements.txt | medium |
| Remove | `/opt/sutazaiapp/docs/requirements/frontend/requirements.txt` | Superseded by requirements.txt | medium |
| Remove | `/opt/sutazaiapp/tests/requirements-test.txt` | Superseded by requirements-test.txt | medium |
| Remove | `/opt/sutazaiapp/docs/requirements/autogpt/requirements.txt` | Superseded by requirements.txt | medium |
| Remove | `/opt/sutazaiapp/agents/autogpt/requirements.txt` | Superseded by requirements.txt | medium |
| Remove | `/opt/sutazaiapp/docs/requirements/crewai/requirements.txt` | Superseded by requirements.txt | medium |
| Remove | `/opt/sutazaiapp/agents/crewai/requirements.txt` | Superseded by requirements.txt | medium |
| Remove | `/opt/sutazaiapp/docker/requirements/requirements.secure.txt` | Superseded by requirements.txt | medium |
| Remove | `/opt/sutazaiapp/docker/requirements/requirements.txt` | Superseded by requirements.txt | medium |
| Remove | `/opt/sutazaiapp/docs/requirements/letta/requirements.txt` | Superseded by requirements.txt | medium |
| Remove | `/opt/sutazaiapp/agents/letta/requirements.txt` | Superseded by requirements.txt | medium |
| Remove | `/opt/sutazaiapp/docs/requirements/gpt-engineer/requirements.txt` | Superseded by requirements.txt | medium |
| Remove | `/opt/sutazaiapp/agents/gpt-engineer/requirements.txt` | Superseded by requirements.txt | medium |
| Remove | `/opt/sutazaiapp/docs/requirements/awesome-code-ai/requirements.txt` | Superseded by requirements.txt | medium |
| Remove | `/opt/sutazaiapp/agents/awesome-code-ai/requirements.txt` | Superseded by requirements.txt | medium |
| Remove | `/opt/sutazaiapp/docker/base/requirements-base.txt` | Superseded by requirements-security.txt | medium |
| Remove | `/opt/sutazaiapp/docker/base/requirements-agent.txt` | Superseded by requirements-security.txt | medium |
| Remove | `/opt/sutazaiapp/docker/semgrep/requirements.txt` | Superseded by requirements.txt | medium |
| Remove | `/opt/sutazaiapp/agents/fsdp/requirements.txt` | Superseded by requirements.txt | medium |
| Remove | `/opt/sutazaiapp/docker/pentestgpt/requirements.txt` | Superseded by requirements.txt | medium |
| Remove | `/opt/sutazaiapp/docs/requirements/pentestgpt/requirements.txt` | Superseded by requirements.txt | medium |
| Remove | `/opt/sutazaiapp/docs/requirements/aider/requirements.txt` | Superseded by requirements.txt | medium |
| Remove | `/opt/sutazaiapp/agents/aider/requirements.txt` | Superseded by requirements.txt | medium |
| Remove | `/opt/sutazaiapp/agents/infrastructure-devops/requirements.txt` | Superseded by requirements.txt | medium |
| Remove | `/opt/sutazaiapp/docs/requirements/context-optimizer/requirements.txt` | Superseded by requirements.txt | medium |
| Remove | `/opt/sutazaiapp/docs/requirements/archive/requirements-minimal.txt` | Superseded by requirements-optimized.txt | medium |
| Remove | `/opt/sutazaiapp/docs/requirements/archive/requirements-agi.txt` | Superseded by requirements-optimized.txt | medium |
| Remove | `/opt/sutazaiapp/docs/requirements/hardware-optimizer/requirements.txt` | Superseded by requirements.txt | medium |

## Rollback Instructions

If you need to rollback changes:

```bash
cd /opt/sutazaiapp/archive/requirements_cleanup_20250803_192821
./rollback.sh
```

Then verify with:
```bash
python scripts/validate-container-infrastructure.py --critical-only
```
