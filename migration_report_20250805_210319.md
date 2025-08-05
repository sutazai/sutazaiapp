# Agent Port Migration Report

**Date**: 2025-08-05 21:03:19
**Mode**: Dry Run

## Migrations Performed

| Service | Container | Old Port | New Port | File |
|---------|-----------|----------|----------|------|
| ai-runtime | sutazai-ai-runtime | 10046 | 11000 | docker-compose-optimized.yml |
| hardware-resource-optimizer | sutazai-hardware-resource-optimizer | 8002 | 11001 | docker-compose.agents-20.yml |
| ai-system-architect | sutazai-ai-system-architect | 8003 | 11083 | docker-compose.agents-20.yml |
| ai-agent-orchestrator | sutazai-ai-agent-orchestrator | 8004 | 11084 | docker-compose.agents-20.yml |
| infrastructure-devops-manager | sutazai-infrastructure-devops-manager | 8005 | 11085 | docker-compose.agents-20.yml |
| senior-ai-engineer | sutazai-senior-ai-engineer | 8007 | 11087 | docker-compose.agents-20.yml |
| agent-orchestrator | sutazai-agent-orchestrator | 8008 | 11088 | docker-compose.agents-20.yml |
| agent-creator | sutazai-agent-creator | 8010 | 11090 | docker-compose.agents-20.yml |
| docker-specialist | sutazai-docker-specialist | 8011 | 11091 | docker-compose.agents-20.yml |
| testing-qa-validator | sutazai-testing-qa-validator | 8012 | 11092 | docker-compose.agents-20.yml |
| ai-testing-qa-validator | sutazai-ai-testing-qa-validator | 8013 | 11093 | docker-compose.agents-20.yml |
| ai-senior-backend-developer | sutazai-ai-senior-backend-developer | 8014 | 11094 | docker-compose.agents-20.yml |
| ai-senior-frontend-developer | sutazai-ai-senior-frontend-developer | 8015 | 11095 | docker-compose.agents-20.yml |
| ai-senior-full-stack-developer | sutazai-ai-senior-full-stack-developer | 8016 | 11096 | docker-compose.agents-20.yml |
| ai-system-validator | sutazai-ai-system-validator | 8017 | 11097 | docker-compose.agents-20.yml |
| container-orchestrator-k3s | sutazai-container-orchestrator-k3s | 8018 | 11098 | docker-compose.agents-20.yml |
| mega-code-auditor | sutazai-mega-code-auditor | 8019 | 11099 | docker-compose.agents-20.yml |
| ollama-integration-specialist | sutazai-ollama-integration-specialist | 8020 | 11002 | docker-compose.agents-20.yml |
| agent-debugger | sutazai-agent-debugger | 8021 | 11003 | docker-compose.agents-20.yml |
| ai-system-architect | sutazai-ai-system-architect | 8001 | 11004 | docker-compose.agents-deploy.yml |
| hardware-resource-optimizer | sutazai-hardware-resource-optimizer | 8002 | 11005 | docker-compose.agents-deploy.yml |
| ai-agent-orchestrator | sutazai-ai-agent-orchestrator | 8003 | 11083 | docker-compose.agents-deploy.yml |
| infrastructure-devops-manager | sutazai-infrastructure-devops-manager | 8004 | 11084 | docker-compose.agents-deploy.yml |
| ai-system-architect | sutazai-ai-system-architect | 8001 | 11006 | docker-compose.agents-final.yml |
| hardware-resource-optimizer | sutazai-hardware-resource-optimizer | 8002 | 11007 | docker-compose.agents-final.yml |
| ai-agent-orchestrator | sutazai-ai-agent-orchestrator | 8003 | 11083 | docker-compose.agents-final.yml |
| infrastructure-devops-manager | sutazai-infrastructure-devops-manager | 8004 | 11084 | docker-compose.agents-final.yml |
| edge-computing-optimizer | sutazai-edge-computing-optimizer | 8365 | 11100 | docker-compose.agents-fix.yml |
| data-analysis-engineer | sutazai-data-analysis-engineer | 8388 | 11101 | docker-compose.agents-fix.yml |
| deep-local-brain-builder | sutazai-deep-local-brain-builder | 8726 | 11107 | docker-compose.agents-fix.yml |
| document-knowledge-manager | sutazai-document-knowledge-manager | 8729 | 11008 | docker-compose.agents-fix.yml |
| ai-system-validator | sutazai-ai-system-validator | 10321 | 11009 | docker-compose.agents-fixed.yml |
| ai-testing-qa-validator | sutazai-ai-testing-qa-validator | 10322 | 11010 | docker-compose.agents-fixed.yml |
| container-orchestrator-k3s | sutazai-container-orchestrator-k3s | 10323 | 11011 | docker-compose.agents-fixed.yml |
| ai-system-architect | sutazai-ai-system-architect | 8001 | 11012 | docker-compose.agents.yml |
| hardware-resource-optimizer | sutazai-hardware-resource-optimizer | 8002 | 11013 | docker-compose.agents.yml |
| ai-agent-orchestrator | sutazai-ai-agent-orchestrator | 8003 | 11083 | docker-compose.agents.yml |
| infrastructure-devops-manager | sutazai-infrastructure-devops-manager | 8004 | 11084 | docker-compose.agents.yml |
| ollama-integration-specialist | sutazai-ollama-integration-specialist | 8006 | 11086 | docker-compose.agents.yml |
| senior-ai-engineer | sutazai-senior-ai-engineer | 8007 | 11087 | docker-compose.agents.yml |
| testing-qa-validator | sutazai-testing-qa-validator | 8008 | 11088 | docker-compose.agents.yml |
| agent-creator | sutazai-agent-creator | 8009 | 11089 | docker-compose.agents.yml |
| agent-debugger | sutazai-agent-debugger | 8010 | 11090 | docker-compose.agents.yml |
| agent-orchestrator | sutazai-agent-orchestrator | 8011 | 11091 | docker-compose.agents.yml |
| ai-senior-backend-developer | sutazai-ai-senior-backend-developer | 8012 | 11092 | docker-compose.agents.yml |
| ai-senior-engineer | sutazai-ai-senior-engineer | 8013 | 11093 | docker-compose.agents.yml |
| ai-senior-frontend-developer | sutazai-ai-senior-frontend-developer | 8014 | 11094 | docker-compose.agents.yml |
| ai-senior-full-stack-developer | sutazai-ai-senior-full-stack-developer | 8015 | 11095 | docker-compose.agents.yml |
| ai-system-validator | sutazai-ai-system-validator | 8016 | 11096 | docker-compose.agents.yml |
| ai-testing-qa-validator | sutazai-ai-testing-qa-validator | 8017 | 11097 | docker-compose.agents.yml |
| container-orchestrator-k3s | sutazai-container-orchestrator-k3s | 8018 | 11098 | docker-compose.agents.yml |
| mega-code-auditor | sutazai-mega-code-auditor | 8020 | 11014 | docker-compose.agents.yml |
| service-account-manager | sutazai-service-account-manager | 10055 | 11015 | docker-compose.auth.yml |
| alertmanager | alertmanager | 9093 | 11016 | docker-compose.distributed.yml |
| sutazai-service-discovery | sutazai-service-discovery | 10000 | 11017 | docker-compose.external-integration.yml |
| sutazai-api-gateway | sutazai-api-gateway | 10001 | 11018 | docker-compose.external-integration.yml |
| sutazai-api-gateway | sutazai-api-gateway | 10002 | 11019 | docker-compose.external-integration.yml |
| sutazai-api-gateway | sutazai-api-gateway | 10003 | 11020 | docker-compose.external-integration.yml |
| sutazai-metrics-aggregator | sutazai-metrics-aggregator | 10010 | 11021 | docker-compose.external-integration.yml |
| sutazai-log-collector | sutazai-log-collector | 10020 | 11022 | docker-compose.external-integration.yml |
| sutazai-service-mesh | sutazai-service-mesh | 10030 | 11023 | docker-compose.external-integration.yml |
| sutazai-service-mesh | sutazai-service-mesh | 10031 | 11024 | docker-compose.external-integration.yml |
| sutazai-postgres-adapter | sutazai-postgres-adapter | 10100 | 11025 | docker-compose.external-integration.yml |
| sutazai-redis-adapter | sutazai-redis-adapter | 10110 | 11026 | docker-compose.external-integration.yml |
| sutazai-config-service | sutazai-config-service | 10040 | 11027 | docker-compose.external-integration.yml |
| sutazai-config-service | sutazai-config-service | 10041 | 11028 | docker-compose.external-integration.yml |
| sutazai-integration-dashboard | sutazai-integration-dashboard | 10050 | 11029 | docker-compose.external-integration.yml |
| fusion-coordinator | sutazai-fusion-coordinator | 8766 | 11030 | docker-compose.fusion.yml |
| agentzero-coordinator | sutazai-agentzero-coordinator | 10300 | 11031 | docker-compose.missing-agents-optimized.yml |
| agent-orchestrator | sutazai-agent-orchestrator | 10301 | 11032 | docker-compose.missing-agents-optimized.yml |
| task-assignment-coordinator | sutazai-task-assignment-coordinator | 10302 | 11033 | docker-compose.missing-agents-optimized.yml |
| ai-system-architect | sutazai-ai-system-architect | 10320 | 11034 | docker-compose.missing-agents-optimized.yml |
| ai-system-validator | sutazai-ai-system-validator | 10321 | 11035 | docker-compose.missing-agents-optimized.yml |
| ai-testing-qa-validator | sutazai-ai-testing-qa-validator | 10322 | 11036 | docker-compose.missing-agents-optimized.yml |
| ai-product-manager | sutazai-ai-product-manager | 10340 | 11037 | docker-compose.missing-agents-optimized.yml |
| ai-scrum-master | sutazai-ai-scrum-master | 10341 | 11038 | docker-compose.missing-agents-optimized.yml |
| ai-senior-engineer | sutazai-ai-senior-engineer | 10360 | 11039 | docker-compose.missing-agents-optimized.yml |
| ai-senior-backend-developer | sutazai-ai-senior-backend-developer | 10361 | 11040 | docker-compose.missing-agents-optimized.yml |
| ai-senior-frontend-developer | sutazai-ai-senior-frontend-developer | 10362 | 11041 | docker-compose.missing-agents-optimized.yml |
| ai-senior-full-stack-developer | sutazai-ai-senior-full-stack-developer | 10363 | 11042 | docker-compose.missing-agents-optimized.yml |
| ai-qa-team-lead | sutazai-ai-qa-team-lead | 10380 | 11043 | docker-compose.missing-agents-optimized.yml |
| agentzero-coordinator | sutazai-agentzero-coordinator | 10300 | 11044 | docker-compose.missing-agents.yml |
| agent-orchestrator | sutazai-agent-orchestrator | 10301 | 11045 | docker-compose.missing-agents.yml |
| task-assignment-coordinator | sutazai-task-assignment-coordinator | 10302 | 11046 | docker-compose.missing-agents.yml |
| bigagi-system-manager | sutazai-bigagi-system-manager | 10304 | 11047 | docker-compose.missing-agents.yml |
| deep-learning-brain-architect | sutazai-deep-learning-brain-architect | 10320 | 11048 | docker-compose.missing-agents.yml |
| neural-architecture-search | sutazai-neural-architecture-search | 10321 | 11049 | docker-compose.missing-agents.yml |
| model-training-specialist | sutazai-model-training-specialist | 10322 | 11050 | docker-compose.missing-agents.yml |
| transformers-migration-specialist | sutazai-transformers-migration-specialist | 10323 | 11051 | docker-compose.missing-agents.yml |
| gradient-compression-specialist | sutazai-gradient-compression-specialist | 10324 | 11052 | docker-compose.missing-agents.yml |
| senior-ai-engineer | sutazai-senior-ai-engineer | 10350 | 11053 | docker-compose.missing-agents.yml |
| senior-backend-developer | sutazai-senior-backend-developer | 10351 | 11054 | docker-compose.missing-agents.yml |
| senior-frontend-developer | sutazai-senior-frontend-developer | 10352 | 11055 | docker-compose.missing-agents.yml |
| senior-full-stack-developer | sutazai-senior-full-stack-developer | 10353 | 11056 | docker-compose.missing-agents.yml |
| gpt-engineer | sutazai-gpt-engineer | 10354 | 11057 | docker-compose.missing-agents.yml |
| infrastructure-devops-manager | sutazai-infrastructure-devops-manager | 10380 | 11058 | docker-compose.missing-agents.yml |
| container-orchestrator-k3s | sutazai-container-orchestrator-k3s | 10382 | 11059 | docker-compose.missing-agents.yml |
| cicd-pipeline-orchestrator | sutazai-cicd-pipeline-orchestrator | 10383 | 11060 | docker-compose.missing-agents.yml |
| resource-manager | sutazai-resource-manager | 10009 | 11061 | docker-compose.missing-services.yml |
| alertmanager | sutazai-alertmanager | 10203 | 11062 | docker-compose.missing-services.yml |
| ai-metrics-exporter | sutazai-ai-metrics | 10204 | 11063 | docker-compose.missing-services.yml |
| alertmanager | sutazai-alertmanager | 10203 | 11064 | docker-compose.monitoring.yml |
| pool-manager | ollama-pool-manager | 8081 | 11065 | docker-compose.ollama-cluster-optimized.yml |
| agentzero-coordinator | sutazai-agentzero-coordinator | 8586 | 11103 | docker-compose.orchestration-agents.yml |
| multi-agent-coordinator | sutazai-multi-agent-coordinator | 8587 | 11104 | docker-compose.orchestration-agents.yml |
| ai-agent-orchestrator | sutazai-ai-agent-orchestrator | 8589 | 11106 | docker-compose.orchestration-agents.yml |
| task-assignment-coordinator | sutazai-task-assignment-coordinator | 8551 | 11102 | docker-compose.orchestration-agents.yml |
| resource-arbitration-agent | sutazai-resource-arbitration-agent | 8588 | 11105 | docker-compose.orchestration-agents.yml |
| ai-system-architect | sutazai-ai-system-architect | 8200 | 11069 | docker-compose.phase1-critical-activation.yml |
| mega-code-auditor | sutazai-mega-code-auditor | 8202 | 11071 | docker-compose.phase1-critical-activation.yml |
| system-optimizer-reorganizer | sutazai-system-optimizer-reorganizer | 8203 | 11072 | docker-compose.phase1-critical-activation.yml |
| hardware-resource-optimizer | sutazai-hardware-resource-optimizer-critical | 8204 | 11073 | docker-compose.phase1-critical-activation.yml |
| ollama-integration-specialist | sutazai-ollama-integration-specialist | 8205 | 11074 | docker-compose.phase1-critical-activation.yml |
| infrastructure-devops-manager | sutazai-infrastructure-devops-manager | 8206 | 11075 | docker-compose.phase1-critical-activation.yml |
| ai-agent-orchestrator | sutazai-ai-agent-orchestrator | 8207 | 11076 | docker-compose.phase1-critical-activation.yml |
| ai-senior-backend-developer | sutazai-ai-senior-backend-developer | 8208 | 11077 | docker-compose.phase1-critical-activation.yml |
| ai-senior-frontend-developer | sutazai-ai-senior-frontend-developer | 8209 | 11078 | docker-compose.phase1-critical-activation.yml |
| testing-qa-validator | sutazai-testing-qa-validator-critical | 8210 | 11079 | docker-compose.phase1-critical-activation.yml |
| document-knowledge-manager | sutazai-document-knowledge-manager | 8211 | 11080 | docker-compose.phase1-critical-activation.yml |
| security-pentesting-specialist | sutazai-security-pentesting-specialist | 8212 | 11081 | docker-compose.phase1-critical-activation.yml |
| cicd-pipeline-orchestrator | sutazai-cicd-pipeline-orchestrator | 8213 | 11082 | docker-compose.phase1-critical-activation.yml |
| agentgpt | sutazai-agentgpt | 10305 | 11066 | docker-compose.yml |
| agentzero | sutazai-agentzero | 10411 | 11067 | docker-compose.yml |
| ai-metrics-exporter | sutazai-ai-metrics-exporter | 10209 | 11068 | docker-compose.yml |
| alertmanager | sutazai-alertmanager | 10203 | 11108 | docker-compose.yml |
| gpt-engineer | sutazai-gpt-engineer | 10302 | 11109 | docker-compose.yml |
| hardware-resource-optimizer | sutazai-hardware-resource-optimizer | 10413 | 11110 | docker-compose.yml |

**Total Migrations**: 127

## Next Steps

1. Update the port registry at `/opt/sutazaiapp/config/port-registry.yaml`
2. Restart affected services:
   ```bash
   docker-compose -f docker-compose.yml down
   docker-compose -f docker-compose.yml up -d
   docker-compose -f docker-compose.fusion.yml down
   docker-compose -f docker-compose.fusion.yml up -d
   docker-compose -f docker-compose.external-integration.yml down
   docker-compose -f docker-compose.external-integration.yml up -d
   docker-compose -f docker-compose.phase1-critical-activation.yml down
   docker-compose -f docker-compose.phase1-critical-activation.yml up -d
   docker-compose -f docker-compose.agents-final.yml down
   docker-compose -f docker-compose.agents-final.yml up -d
   docker-compose -f docker-compose.ollama-cluster-optimized.yml down
   docker-compose -f docker-compose.ollama-cluster-optimized.yml up -d
   docker-compose -f docker-compose.auth.yml down
   docker-compose -f docker-compose.auth.yml up -d
   docker-compose -f docker-compose.agents.yml down
   docker-compose -f docker-compose.agents.yml up -d
   docker-compose -f docker-compose.agents-fix.yml down
   docker-compose -f docker-compose.agents-fix.yml up -d
   docker-compose -f docker-compose.missing-agents.yml down
   docker-compose -f docker-compose.missing-agents.yml up -d
   docker-compose -f docker-compose.distributed.yml down
   docker-compose -f docker-compose.distributed.yml up -d
   docker-compose -f docker-compose.agents-deploy.yml down
   docker-compose -f docker-compose.agents-deploy.yml up -d
   docker-compose -f docker-compose.missing-services.yml down
   docker-compose -f docker-compose.missing-services.yml up -d
   docker-compose -f docker-compose.agents-20.yml down
   docker-compose -f docker-compose.agents-20.yml up -d
   docker-compose -f docker-compose.orchestration-agents.yml down
   docker-compose -f docker-compose.orchestration-agents.yml up -d
   docker-compose -f docker-compose.agents-fixed.yml down
   docker-compose -f docker-compose.agents-fixed.yml up -d
   docker-compose -f docker-compose-optimized.yml down
   docker-compose -f docker-compose-optimized.yml up -d
   docker-compose -f docker-compose.missing-agents-optimized.yml down
   docker-compose -f docker-compose.missing-agents-optimized.yml up -d
   docker-compose -f docker-compose.monitoring.yml down
   docker-compose -f docker-compose.monitoring.yml up -d
   ```
3. Validate port allocations:
   ```bash
   python3 scripts/validate_ports.py
   ```
