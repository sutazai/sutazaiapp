# Port Registry System for SUTAZAIAPP

This registry assigns stable, non-overlapping ports to each Dockerized service according to the defined ranges. Existing ports noted in the diagram are preserved. New assignments fill gaps to ensure complete coverage.

Legend of ranges:
- 10000-10199: Infrastructure Services
- 10200-10299: Monitoring Stack
- 10300-10499: External Integrations
- 10500-10599: AGI System
- 11000-11148: AI Agents (STANDARD)
- Ollama LLM: 10104 (reserved)

## Infrastructure Services (10000-10199)

- 10000: core/postgresql (database)
- 10001: core/redis (cache)
- 10002-10003: core/neo4j (bolt/http)
- 10005: core/kong-gateway (API gateway)
- 10006: ai-tier-2/service-mesh/consul (service discovery)
- 10007-10008: core/rabbitmq (amqp/mgmt)
- 10010: application/backend-api (FastAPI)
- 10011: application/modern-ui/jarvis-interface (UI)
- 10012: application/specialized-processing/document-processing
- 10013: application/specialized-processing/code-processing
- 10014: application/specialized-processing/research-processing
- 10100: ai-tier-2/vector-intelligence/chromadb
- 10101-10102: ai-tier-2/vector-intelligence/qdrant
- 10103: ai-tier-2/vector-intelligence/faiss
- 10104: ai-tier-2/model-management/ollama-engine (reserved)
- 10105: ai-tier-2/model-management/model-registry
- 10106: ai-tier-2/vector-intelligence/embedding-service
- 10120: ai-tier-2/ml-frameworks/pytorch-service
- 10121: ai-tier-2/ml-frameworks/tensorflow-service
- 10122: ai-tier-2/ml-frameworks/jax-service
- 10123: ai-tier-2/ml-frameworks/fsdp-service
- 10130: ai-tier-2/voice-services/speech-to-text
- 10131: ai-tier-2/voice-services/text-to-speech
- 10132: ai-tier-2/voice-services/voice-processing
- 10140: ai-tier-2/service-mesh/load-balancing

## Monitoring Stack (10200-10299)

- 10200: monitoring/metrics-collection/prometheus
- 10201: monitoring/visualization/grafana
- 10202: monitoring/logging/loki
- 10203: monitoring/alerting/alertmanager
- 10210: monitoring/custom-exporters/jarvis-exporter
- 10211: monitoring/custom-exporters/ai-comprehensive-exporter
- 10212: monitoring/custom-exporters/training-exporter (enhanced)
- 10220: monitoring/system-exporters/node-exporter
- 10221: monitoring/system-exporters/cadvisor

## External Integrations (10300-10499)

- 10300: application/api-gateway/nginx-proxy
- 10310: deployment-orchestration/automation/webhook-integration (reserved)
- 10311: monitoring/alerting/integrations/webhook-integration
- 10312: monitoring/alerting/integrations/slack-integration
- 10313: monitoring/alerting/integrations/email-integration

## AGI System (10500-10599)

- 10500: agent-tier-3/jarvis-core/jarvis-brain
- 10501: agent-tier-3/jarvis-core/jarvis-memory
- 10502: agent-tier-3/jarvis-core/jarvis-skills
- 10503: agent-tier-3/jarvis-ecosystem/jarvis-synthesis-engine
- 10504: agent-tier-3/jarvis-ecosystem/agent-coordination

## AI Agents (STANDARD) (11000-11148)

# Task Automation Agents
- 11000: agent-tier-3/task-automation-agents/letta-agent
- 11001: agent-tier-3/task-automation-agents/autogpt-agent
- 11002: agent-tier-3/task-automation-agents/localagi-agent
- 11003: agent-tier-3/task-automation-agents/agent-zero

# Code Intelligence Agents
- 11010: agent-tier-3/code-intelligence-agents/tabbyml-agent
- 11011: agent-tier-3/code-intelligence-agents/semgrep-agent
- 11012: agent-tier-3/code-intelligence-agents/gpt-engineer-agent
- 11013: agent-tier-3/code-intelligence-agents/opendevin-agent
- 11014: agent-tier-3/code-intelligence-agents/aider-agent

# Research & Analysis Agents
- 11020: agent-tier-3/research-analysis-agents/deep-researcher-agent
- 11021: agent-tier-3/research-analysis-agents/deep-agent
- 11022: agent-tier-3/research-analysis-agents/finrobot-agent

# Orchestration Agents
- 11030: agent-tier-3/orchestration-agents/langchain-agent
- 11031: agent-tier-3/orchestration-agents/autogen-agent
- 11032: agent-tier-3/orchestration-agents/crewai-agent
- 11033: agent-tier-3/orchestration-agents/bigagi-agent

# Browser Automation Agents
- 11040: agent-tier-3/browser-automation-agents/browser-use-agent
- 11041: agent-tier-3/browser-automation-agents/skyvern-agent
- 11042: agent-tier-3/browser-automation-agents/agentgpt-agent

# Workflow Platforms
- 11050: agent-tier-3/workflow-platforms/langflow-agent
- 11051: agent-tier-3/workflow-platforms/dify-agent
- 11052: agent-tier-3/workflow-platforms/flowise-agent

# Specialized Agents
- 11060: agent-tier-3/specialized-agents/privateGPT-agent
- 11061: agent-tier-3/specialized-agents/llamaindex-agent
- 11062: agent-tier-3/specialized-agents/shellgpt-agent
- 11063: agent-tier-3/specialized-agents/pentestgpt-agent

## Notes

- Where a component exposes multiple ports (e.g., Neo4j, RabbitMQ), a small consecutive block is reserved.
- “Enhanced” and “Self-Coding/UltraThink” variants reuse the same base port unless they are distinct concurrently running services; assign a suffix offset (+50) if both variants must run at the same time (e.g., langflow-agent standard 11050, enhanced 11100).
- Ollama remains bound to 10104 as already established in the diagram.

