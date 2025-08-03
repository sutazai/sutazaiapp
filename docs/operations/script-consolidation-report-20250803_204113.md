
# Script Consolidation and Python Sanity Enforcement Report
Generated: 2025-08-03T20:41:11.497205

## Summary
- Total violations found: 2628
- Duplicate groups found: 52
- Actions taken: 0
- Fixes applied: 0

## Violations by Type

### Underscores (482)
- /opt/sutazaiapp/comprehensive_agent_qa_validator.py: Naming violation: Uses underscores instead of hyphens
- /opt/sutazaiapp/analyze_service_deps.py: Naming violation: Uses underscores instead of hyphens
- /opt/sutazaiapp/analyze_shared_dependencies.py: Naming violation: Uses underscores instead of hyphens
- /opt/sutazaiapp/service_dependency_graph.py: Naming violation: Uses underscores instead of hyphens
- /opt/sutazaiapp/scripts/check_secrets.py: Naming violation: Uses underscores instead of hyphens
- /opt/sutazaiapp/scripts/enforce_claude_md_rules.py: Naming violation: Uses underscores instead of hyphens
- /opt/sutazaiapp/scripts/verify_claude_rules.py: Naming violation: Uses underscores instead of hyphens
- /opt/sutazaiapp/scripts/validate_agents.py: Naming violation: Uses underscores instead of hyphens
- /opt/sutazaiapp/scripts/check_naming.py: Naming violation: Uses underscores instead of hyphens
- /opt/sutazaiapp/scripts/enforce_claude_md_simple.py: Naming violation: Uses underscores instead of hyphens
... and 472 more

### Missing Docstring (511)
- /opt/sutazaiapp/comprehensive_agent_qa_validator.py: Missing proper module docstring with Purpose/Usage/Requirements
- /opt/sutazaiapp/analyze_service_deps.py: Missing proper module docstring with Purpose/Usage/Requirements
- /opt/sutazaiapp/analyze_shared_dependencies.py: Missing proper module docstring with Purpose/Usage/Requirements
- /opt/sutazaiapp/service_dependency_graph.py: Missing proper module docstring with Purpose/Usage/Requirements
- /opt/sutazaiapp/scripts/check_secrets.py: Missing proper module docstring with Purpose/Usage/Requirements
- /opt/sutazaiapp/scripts/hygiene-monitor.py: Missing proper module docstring with Purpose/Usage/Requirements
- /opt/sutazaiapp/scripts/enforce_claude_md_rules.py: Missing proper module docstring with Purpose/Usage/Requirements
- /opt/sutazaiapp/scripts/verify_claude_rules.py: Missing proper module docstring with Purpose/Usage/Requirements
- /opt/sutazaiapp/scripts/validate_agents.py: Missing proper module docstring with Purpose/Usage/Requirements
- /opt/sutazaiapp/scripts/check_naming.py: Missing proper module docstring with Purpose/Usage/Requirements
... and 501 more

### Hardcoded Path (420)
- /opt/sutazaiapp/comprehensive_agent_qa_validator.py: Hardcoded value found at line 24: self.agents_dir = Path("/opt/sutazaiapp/.claude/agents")
- /opt/sutazaiapp/comprehensive_agent_qa_validator.py: Hardcoded value found at line 612: json_file = f"/opt/sutazaiapp/comprehensive_agent_qa_report_{timestamp}.json"
- /opt/sutazaiapp/comprehensive_agent_qa_validator.py: Hardcoded value found at line 617: md_file = f"/opt/sutazaiapp/COMPREHENSIVE_AGENT_QA_VALIDATION_REPORT_{timestamp}.md"
- /opt/sutazaiapp/analyze_service_deps.py: Hardcoded value found at line 17: docker_compose_path = Path("/opt/sutazaiapp/docker-compose.yml")
- /opt/sutazaiapp/scripts/create-container-requirements-map.py: Hardcoded value found at line 225: script.append("BACKUP_DIR='/opt/sutazaiapp/requirements_backup_$(date +%Y%m%d_%H%M%S)'")
- /opt/sutazaiapp/scripts/create-container-requirements-map.py: Hardcoded value found at line 232: script.append("    rel_path=${f#/opt/sutazaiapp/}")
- /opt/sutazaiapp/scripts/create-container-requirements-map.py: Hardcoded value found at line 243: script.append(f"rm -f '/opt/sutazaiapp/{file_path}'")
- /opt/sutazaiapp/scripts/create-container-requirements-map.py: Hardcoded value found at line 254: with open('/opt/sutazaiapp/container-requirements-map.json', 'w') as f:
- /opt/sutazaiapp/scripts/create-container-requirements-map.py: Hardcoded value found at line 258: with open('/opt/sutazaiapp/scripts/validate-containers.sh', 'w') as f:
- /opt/sutazaiapp/scripts/create-container-requirements-map.py: Hardcoded value found at line 260: os.chmod('/opt/sutazaiapp/scripts/validate-containers.sh', 0o755)
... and 410 more

### Print Usage (164)
- /opt/sutazaiapp/comprehensive_agent_qa_validator.py: Using print() at lines: [68, 69, 75, 80, 87, 90, 598, 599, 600, 621, 622, 623, 624, 625, 626, 629, 631, 633, 635, 637]. Should use logging.
- /opt/sutazaiapp/analyze_service_deps.py: Using print() at lines: [144]. Should use logging.
- /opt/sutazaiapp/analyze_shared_dependencies.py: Using print() at lines: [20, 50, 67, 69, 70, 71, 75, 83, 84, 85, 86, 90, 92, 97, 99, 102, 104, 106, 107]. Should use logging.
- /opt/sutazaiapp/service_dependency_graph.py: Using print() at lines: [169, 170, 171, 172]. Should use logging.
- /opt/sutazaiapp/scripts/check_secrets.py: Using print() at lines: [84, 85, 87, 88, 89, 90]. Should use logging.
- /opt/sutazaiapp/scripts/create-container-requirements-map.py: Using print() at lines: [268, 269, 270, 271, 272, 273, 274, 275, 276, 278, 279, 280, 288, 292, 294, 296, 297, 298, 302, 303, 305, 307, 308, 309, 314, 315, 317, 318, 319, 320, 321, 322, 324, 325, 326, 327, 328, 329, 330, 332]. Should use logging.
- /opt/sutazaiapp/scripts/enforce_claude_md_rules.py: Using print() at lines: [82, 138, 180, 183, 286, 290, 303, 307, 313, 317, 320, 340, 343, 345, 347, 349, 351, 369, 371, 372, 373, 374, 375]. Should use logging.
- /opt/sutazaiapp/scripts/quick-container-analysis.py: Using print() at lines: [136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 147, 152, 154]. Should use logging.
- /opt/sutazaiapp/scripts/safe-requirements-cleanup.py: Using print() at lines: [578, 581, 594, 595, 598, 600, 601]. Should use logging.
- /opt/sutazaiapp/scripts/verify_claude_rules.py: Using print() at lines: [18, 19, 28, 35, 36, 42, 43, 46, 47, 54, 56]. Should use logging.
... and 154 more

### No Cli Args (287)
- /opt/sutazaiapp/comprehensive_agent_qa_validator.py: Script appears executable but lacks CLI argument handling
- /opt/sutazaiapp/scripts/check_secrets.py: Script appears executable but lacks CLI argument handling
- /opt/sutazaiapp/scripts/create-container-requirements-map.py: Script appears executable but lacks CLI argument handling
- /opt/sutazaiapp/scripts/hygiene-monitor.py: Script appears executable but lacks CLI argument handling
- /opt/sutazaiapp/scripts/enforce_claude_md_rules.py: Script appears executable but lacks CLI argument handling
- /opt/sutazaiapp/scripts/quick-container-analysis.py: Script appears executable but lacks CLI argument handling
- /opt/sutazaiapp/scripts/verify_claude_rules.py: Script appears executable but lacks CLI argument handling
- /opt/sutazaiapp/scripts/validate_agents.py: Script appears executable but lacks CLI argument handling
- /opt/sutazaiapp/scripts/check_naming.py: Script appears executable but lacks CLI argument handling
- /opt/sutazaiapp/scripts/analyze-docker-requirements.py: Script appears executable but lacks CLI argument handling
... and 277 more

### No Main Guard (123)
- /opt/sutazaiapp/analyze_service_deps.py: Missing __name__ == '__main__' guard
- /opt/sutazaiapp/analyze_shared_dependencies.py: Missing __name__ == '__main__' guard
- /opt/sutazaiapp/service_dependency_graph.py: Missing __name__ == '__main__' guard
- /opt/sutazaiapp/tests/test_smoke.py: Missing __name__ == '__main__' guard
- /opt/sutazaiapp/agents/semgrep_service.py: Missing __name__ == '__main__' guard
- /opt/sutazaiapp/scripts/test/test_transformer_environment.py: Missing __name__ == '__main__' guard
- /opt/sutazaiapp/scripts/utils/remove_litellm_from_registry.py: Missing __name__ == '__main__' guard
- /opt/sutazaiapp/docker/autogen/autogen_service.py: Missing __name__ == '__main__' guard
- /opt/sutazaiapp/docker/llamaindex/llamaindex_service.py: Missing __name__ == '__main__' guard
- /opt/sutazaiapp/docker/enhanced-model-manager/model_manager.py: Missing __name__ == '__main__' guard
... and 113 more

### Hardcoded Localhost (441)
- /opt/sutazaiapp/scripts/container-health-monitor.py: Hardcoded value found at line 128: 'backend': 'http://localhost:8000/health',
- /opt/sutazaiapp/scripts/container-health-monitor.py: Hardcoded value found at line 129: 'frontend': 'http://localhost:3000',
- /opt/sutazaiapp/scripts/container-health-monitor.py: Hardcoded value found at line 130: 'grafana': 'http://localhost:3001',
- /opt/sutazaiapp/scripts/container-health-monitor.py: Hardcoded value found at line 131: 'prometheus': 'http://localhost:9090/-/healthy'
- /opt/sutazaiapp/workflows/security_scan_workflow.py: Hardcoded value found at line 16: API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
- /opt/sutazaiapp/workflows/simple_code_review.py: Hardcoded value found at line 15: API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
- /opt/sutazaiapp/workflows/practical_examples.py: Hardcoded value found at line 16: def __init__(self, api_url: str = "http://localhost:8000"):
- /opt/sutazaiapp/workflows/deployment_automation.py: Hardcoded value found at line 16: API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
- /opt/sutazaiapp/workflows/deployment_automation.py: Hardcoded value found at line 227: {"name": "Health check", "command": "curl http://localhost:8000/health"}
- /opt/sutazaiapp/workflows/test_code_improvement.py: Hardcoded value found at line 17: base_url = "http://localhost:8000"
... and 431 more

### Hardcoded Ip (11)
- /opt/sutazaiapp/tests/security_test_suite.py: Hardcoded value found at line 361: {"url": f"{self.base_url}/api/v1/admin", "headers": {"X-Forwarded-For": "127.0.0.1"}},
- /opt/sutazaiapp/tests/security_test_suite.py: Hardcoded value found at line 362: {"url": f"{self.base_url}/api/v1/admin", "headers": {"X-Real-IP": "127.0.0.1"}},
- /opt/sutazaiapp/tests/ai_powered_test_suite.py: Hardcoded value found at line 1133: headers = {"X-Forwarded-For": "127.0.0.1", "User-Agent": "<script>alert('xss')</script>"}
- /opt/sutazaiapp/scripts/test/tests_security_test_security_hardening.py: Hardcoded value found at line 127: mock_request.client.host = "127.0.0.1"
- /opt/sutazaiapp/scripts/test/implement_security_fixes.py: Hardcoded value found at line 246: "http://127.0.0.1:8501",
- /opt/sutazaiapp/backend/ai_agents/orchestration/agent_registry_service.py: Hardcoded value found at line 791: endpoint.host in ["localhost", "127.0.0.1"] or
- /opt/sutazaiapp/backend/app/core/config.py: Hardcoded value found at line 24: LOCAL_IP: str = Field("127.0.0.1", env="LOCAL_IP")
- /opt/sutazaiapp/backend/app/api/v1/endpoints/network_recon.py: Hardcoded value found at line 914: target=request.targets[0] if request.targets else "127.0.0.1",
- /opt/sutazaiapp/tests/security/test_security_comprehensive.py: Hardcoded value found at line 299: "|| ping -c 1 127.0.0.1"
- /opt/sutazaiapp/tests/security/test_security_comprehensive.py: Hardcoded value found at line 913: localhost = "127.0.0.1"
... and 1 more

### Version Suffix (4)
- /opt/sutazaiapp/tests/comprehensive_test_report_final.py: Naming violation: Has version suffix
- /opt/sutazaiapp/scripts/data/fix_yaml_indentation_final.py: Naming violation: Has version suffix
- /opt/sutazaiapp/scripts/data/fix_yaml_frontmatter_final.py: Naming violation: Has version suffix
- /opt/sutazaiapp/scripts/data/fix_yaml_structure_final.py: Naming violation: Has version suffix

### Hardcoded Password (17)
- /opt/sutazaiapp/scripts/utils/docs_fix_all_issues.py: Hardcoded value found at line 168: password="sutazai123"
- /opt/sutazaiapp/scripts/agents/script-consolidation-enforcer.py: Hardcoded value found at line 225: (r'password="[^"]*"', "hardcoded_password"),
- /opt/sutazaiapp/scripts/agents/script-consolidation-enforcer.py: Hardcoded value found at line 226: (r"password='[^']*'", "hardcoded_password"),
- /opt/sutazaiapp/workflows/scripts/workflow_manager.py: Hardcoded value found at line 89: password='redis_password',
- /opt/sutazaiapp/workflows/scripts/deploy_dify_workflows.py: Hardcoded value found at line 379: self.redis_client = redis.Redis(host='redis', port=6379, password='redis_password')
- /opt/sutazaiapp/tests/docker/test_containers.py: Hardcoded value found at line 382: password="test_pass"
- /opt/sutazaiapp/tests/health/test_service_health.py: Hardcoded value found at line 188: password="postgres",
- /opt/sutazaiapp/tests/health/test_service_health.py: Hardcoded value found at line 415: user="postgres", password="postgres", connect_timeout=5
- /opt/sutazaiapp/tests/unit/test_security.py: Hardcoded value found at line 102: password = "my_secure_password"
- /opt/sutazaiapp/deploy.sh: Hardcoded value found at line 949: postgres_password="${POSTGRES_PASSWORD:-$(cat "$secrets_dir/postgres_password.txt" 2>/dev/null || echo "")}"
... and 7 more

### Syntax Error (5)
- /opt/sutazaiapp/docker/health-monitor/main.py: Python syntax error in script
- /opt/sutazaiapp/docker/documind/documind_service.py: Python syntax error in script
- /opt/sutazaiapp/backend/ai_agents/universal_agent_adapter.py: Python syntax error in script
- /opt/sutazaiapp/archive/20250803_193506_pre_cleanup/docker/health-monitor/main.py: Python syntax error in script
- /opt/sutazaiapp/archive/20250803_193506_pre_cleanup/docker/documind/documind_service.py: Python syntax error in script

### Missing Header (123)
- /opt/sutazaiapp/health_check.sh: Missing proper header comment with Purpose/Usage/Requires
- /opt/sutazaiapp/validate_deployment.sh: Missing proper header comment with Purpose/Usage/Requires
- /opt/sutazaiapp/deploy_optimized.sh: Missing proper header comment with Purpose/Usage/Requires
- /opt/sutazaiapp/validate-deployment-hygiene.sh: Missing proper header comment with Purpose/Usage/Requires
- /opt/sutazaiapp/deploy.sh: Missing proper header comment with Purpose/Usage/Requires
- /opt/sutazaiapp/scripts/validate-containers.sh: Missing proper header comment with Purpose/Usage/Requires
- /opt/sutazaiapp/scripts/start-hygiene-monitor.sh: Missing proper header comment with Purpose/Usage/Requires
- /opt/sutazaiapp/scripts/build_optimized.sh: Missing proper header comment with Purpose/Usage/Requires
- /opt/sutazaiapp/scripts/setup-hygiene-cron.sh: Missing proper header comment with Purpose/Usage/Requires
- /opt/sutazaiapp/scripts/update-dockerfiles.sh: Missing proper header comment with Purpose/Usage/Requires
... and 113 more

### No Error Handling (40)
- /opt/sutazaiapp/scripts/setup-hygiene-cron.sh: Script lacks proper error handling with exit codes
- /opt/sutazaiapp/scripts/update-dockerfiles.sh: Script lacks proper error handling with exit codes
- /opt/sutazaiapp/scripts/cleanup-requirements.sh: Script lacks proper error handling with exit codes
- /opt/sutazaiapp/scripts/build-base-images.sh: Script lacks proper error handling with exit codes
- /opt/sutazaiapp/agents/startup.sh: Script lacks proper error handling with exit codes
- /opt/sutazaiapp/scripts/test/test_automation.sh: Script lacks proper error handling with exit codes
- /opt/sutazaiapp/scripts/test/run_tests.sh: Script lacks proper error handling with exit codes
- /opt/sutazaiapp/scripts/test/test_cleanup.sh: Script lacks proper error handling with exit codes
- /opt/sutazaiapp/scripts/test/test_performance.sh: Script lacks proper error handling with exit codes
- /opt/sutazaiapp/scripts/test/verify_complete_system.sh: Script lacks proper error handling with exit codes
... and 30 more

## Duplicate Script Groups

### Group 1 (101 scripts)
- /opt/sutazaiapp/analyze_service_deps.py
- /opt/sutazaiapp/docker/agentzero/agentzero_service.py
- /opt/sutazaiapp/backend/processing_engine/__init__.py
- /opt/sutazaiapp/backend/utils/__init__.py
- /opt/sutazaiapp/backend/monitoring/__init__.py
- /opt/sutazaiapp/backend/agent_orchestration/__init__.py
- /opt/sutazaiapp/backend/ai_agents/memory/__init__.py
- /opt/sutazaiapp/backend/ai_agents/orchestrator/__init__.py
- /opt/sutazaiapp/backend/ai_agents/interaction/__init__.py
- /opt/sutazaiapp/backend/ai_agents/protocols/__init__.py
- /opt/sutazaiapp/backend/app/core/__init__.py
- /opt/sutazaiapp/backend/app/orchestration/__init__.py
- /opt/sutazaiapp/backend/app/services/__init__.py
- /opt/sutazaiapp/backend/app/api/__init__.py
- /opt/sutazaiapp/backend/app/api/v1/__init__.py
- /opt/sutazaiapp/backend/app/api/v1/endpoints/__init__.py
- /opt/sutazaiapp/backend/app/api/v1/endpoints/documents.py
- /opt/sutazaiapp/backend/app/api/v1/endpoints/system.py
- /opt/sutazaiapp/tests/docker/__init__.py
- /opt/sutazaiapp/tests/security/__init__.py
- /opt/sutazaiapp/tests/load/__init__.py
- /opt/sutazaiapp/tests/health/__init__.py
- /opt/sutazaiapp/frontend/components/__init__.py
- /opt/sutazaiapp/archive/20250803_193506_pre_cleanup/docker/agentzero/agentzero_service.py
- /opt/sutazaiapp/agents/distributed-computing-architect/__init__.py
- /opt/sutazaiapp/agents/autogpt/__init__.py
- /opt/sutazaiapp/agents/autogen/__init__.py
- /opt/sutazaiapp/agents/agentzero-coordinator/__init__.py
- /opt/sutazaiapp/agents/cognitive-architecture-designer/__init__.py
- /opt/sutazaiapp/agents/product-strategy-architect/__init__.py
- /opt/sutazaiapp/agents/health-monitor/__init__.py
- /opt/sutazaiapp/agents/multi-modal-fusion-coordinator/__init__.py
- /opt/sutazaiapp/agents/senior-frontend-developer/__init__.py
- /opt/sutazaiapp/agents/private-data-analyst/__init__.py
- /opt/sutazaiapp/agents/senior-backend-developer/__init__.py
- /opt/sutazaiapp/agents/synthetic-data-generator/__init__.py
- /opt/sutazaiapp/agents/reinforcement-learning-trainer/__init__.py
- /opt/sutazaiapp/agents/flowiseai-flow-manager/__init__.py
- /opt/sutazaiapp/agents/crewai/__init__.py
- /opt/sutazaiapp/agents/data-analysis-engineer/__init__.py
- /opt/sutazaiapp/agents/meta-learning-specialist/__init__.py
- /opt/sutazaiapp/agents/mcp-server/__init__.py
- /opt/sutazaiapp/agents/ai-product-manager/__init__.py
- /opt/sutazaiapp/agents/opendevin-code-generator/__init__.py
- /opt/sutazaiapp/agents/context-framework/__init__.py
- /opt/sutazaiapp/agents/task-assignment-coordinator/__init__.py
- /opt/sutazaiapp/agents/letta/__init__.py
- /opt/sutazaiapp/agents/gpt-engineer/__init__.py
- /opt/sutazaiapp/agents/awesome-code-ai/__init__.py
- /opt/sutazaiapp/agents/devika/__init__.py
- /opt/sutazaiapp/agents/model-training-specialist/__init__.py
- /opt/sutazaiapp/agents/bigagi-system-manager/__init__.py
- /opt/sutazaiapp/agents/episodic-memory-engineer/__init__.py
- /opt/sutazaiapp/agents/agentgpt-autonomous-executor/__init__.py
- /opt/sutazaiapp/agents/langflow-workflow-designer/__init__.py
- /opt/sutazaiapp/agents/knowledge-graph-builder/__init__.py
- /opt/sutazaiapp/agents/browser-automation-orchestrator/__init__.py
- /opt/sutazaiapp/agents/neuromorphic-computing-expert/__init__.py
- /opt/sutazaiapp/agents/dify-automation-specialist/__init__.py
- /opt/sutazaiapp/agents/quantum-computing-optimizer/__init__.py
- /opt/sutazaiapp/agents/transformers-migration-specialist/__init__.py
- /opt/sutazaiapp/agents/semgrep-security-analyzer/__init__.py
- /opt/sutazaiapp/agents/ai-scrum-master/__init__.py
- /opt/sutazaiapp/agents/privategpt/__init__.py
- /opt/sutazaiapp/agents/knowledge-distillation-expert/__init__.py
- /opt/sutazaiapp/agents/memory-persistence-manager/__init__.py
- /opt/sutazaiapp/agents/explainable-ai-specialist/__init__.py
- /opt/sutazaiapp/agents/jarvis-voice-interface/__init__.py
- /opt/sutazaiapp/agents/garbage-collector-coordinator/__init__.py
- /opt/sutazaiapp/agents/fsdp/__init__.py
- /opt/sutazaiapp/agents/agentgpt/__init__.py
- /opt/sutazaiapp/agents/shellgpt/__init__.py
- /opt/sutazaiapp/agents/babyagi/__init__.py
- /opt/sutazaiapp/agents/observability-monitoring-engineer/__init__.py
- /opt/sutazaiapp/agents/service-hub/__init__.py
- /opt/sutazaiapp/agents/self-healing-orchestrator/__init__.py
- /opt/sutazaiapp/agents/symbolic-reasoning-engine/__init__.py
- /opt/sutazaiapp/agents/pentestgpt/__init__.py
- /opt/sutazaiapp/agents/intelligence-optimization-monitor/__init__.py
- /opt/sutazaiapp/agents/attention-optimizer/__init__.py
- /opt/sutazaiapp/agents/data-pipeline-engineer/__init__.py
- /opt/sutazaiapp/agents/code-improver/__init__.py
- /opt/sutazaiapp/agents/ai-agent-debugger/__init__.py
- /opt/sutazaiapp/agents/causal-inference-expert/__init__.py
- /opt/sutazaiapp/agents/federated-learning-coordinator/__init__.py
- /opt/sutazaiapp/agents/document-knowledge-manager/__init__.py
- /opt/sutazaiapp/agents/aider/__init__.py
- /opt/sutazaiapp/agents/gradient-compression-specialist/__init__.py
- /opt/sutazaiapp/agents/edge-computing-optimizer/__init__.py
- /opt/sutazaiapp/agents/localagi-orchestration-manager/__init__.py
- /opt/sutazaiapp/agents/finrobot/__init__.py
- /opt/sutazaiapp/agents/deployment-automation-master/shared/agent_base.py
- /opt/sutazaiapp/agents/ollama-integration-specialist/shared/agent_base.py
- /opt/sutazaiapp/agents/testing-qa-validator/shared/agent_base.py
- /opt/sutazaiapp/agents/infrastructure-devops-manager/shared/agent_base.py
- /opt/sutazaiapp/agents/senior-ai-engineer/shared/agent_base.py
- /opt/sutazaiapp/agents/deployment-automation-master/startup.sh
- /opt/sutazaiapp/agents/ollama-integration-specialist/startup.sh
- /opt/sutazaiapp/agents/testing-qa-validator/startup.sh
- /opt/sutazaiapp/agents/infrastructure-devops-manager/startup.sh
- /opt/sutazaiapp/agents/senior-ai-engineer/startup.sh

### Group 2 (6 scripts)
- /opt/sutazaiapp/agents/agent_base.py
- /opt/sutazaiapp/agents/deployment-automation-master/agent_base.py
- /opt/sutazaiapp/agents/ollama-integration-specialist/agent_base.py
- /opt/sutazaiapp/agents/testing-qa-validator/agent_base.py
- /opt/sutazaiapp/agents/infrastructure-devops-manager/agent_base.py
- /opt/sutazaiapp/agents/senior-ai-engineer/agent_base.py

### Group 3 (2 scripts)
- /opt/sutazaiapp/agents/crewai_config.py
- /opt/sutazaiapp/config/agents/crewai_config.py

### Group 4 (2 scripts)
- /opt/sutazaiapp/scripts/test/security_audit_env_bin_pwiz.py
- /opt/sutazaiapp/security_audit_env/bin/pwiz.py

### Group 5 (2 scripts)
- /opt/sutazaiapp/scripts/test/tests_security_test_security_hardening.py
- /opt/sutazaiapp/tests/security/test_security_hardening.py

### Group 6 (2 scripts)
- /opt/sutazaiapp/scripts/utils/compact_monitor.py
- /opt/sutazaiapp/scripts/monitoring/static_monitor.py

### Group 7 (2 scripts)
- /opt/sutazaiapp/scripts/agents/services_codegen_aider_service.py
- /opt/sutazaiapp/services/codegen/aider_service.py

### Group 8 (2 scripts)
- /opt/sutazaiapp/scripts/data/services_ml_pytorch_service.py
- /opt/sutazaiapp/services/ml/pytorch_service.py

### Group 9 (2 scripts)
- /opt/sutazaiapp/scripts/data/services_ml_tensorflow_service.py
- /opt/sutazaiapp/services/ml/tensorflow_service.py

### Group 10 (2 scripts)
- /opt/sutazaiapp/scripts/data/fix_yaml_indentation_final.py
- /opt/sutazaiapp/scripts/data/fix_yaml_structure_final.py

### Group 11 (2 scripts)
- /opt/sutazaiapp/scripts/data/services_ml_jax_service.py
- /opt/sutazaiapp/services/ml/jax_service.py

### Group 12 (2 scripts)
- /opt/sutazaiapp/docker/autogpt/autogpt_service.py
- /opt/sutazaiapp/archive/20250803_193506_pre_cleanup/docker/autogpt/autogpt_service.py

### Group 13 (2 scripts)
- /opt/sutazaiapp/docker/autogen/autogen_agent_server.py
- /opt/sutazaiapp/archive/20250803_193506_pre_cleanup/docker/autogen/autogen_agent_server.py

### Group 14 (2 scripts)
- /opt/sutazaiapp/docker/autogen/autogen_service.py
- /opt/sutazaiapp/archive/20250803_193506_pre_cleanup/docker/autogen/autogen_service.py

### Group 15 (2 scripts)
- /opt/sutazaiapp/docker/jax/web_interface.py
- /opt/sutazaiapp/archive/20250803_193506_pre_cleanup/docker/jax/web_interface.py

### Group 16 (2 scripts)
- /opt/sutazaiapp/docker/health-monitor/main.py
- /opt/sutazaiapp/archive/20250803_193506_pre_cleanup/docker/health-monitor/main.py

### Group 17 (4 scripts)
- /opt/sutazaiapp/docker/context-engineering/health_check.py
- /opt/sutazaiapp/docker/fms-fsdp/health_check.py
- /opt/sutazaiapp/archive/20250803_193506_pre_cleanup/docker/context-engineering/health_check.py
- /opt/sutazaiapp/archive/20250803_193506_pre_cleanup/docker/fms-fsdp/health_check.py

### Group 18 (2 scripts)
- /opt/sutazaiapp/docker/context-engineering/context_engine.py
- /opt/sutazaiapp/archive/20250803_193506_pre_cleanup/docker/context-engineering/context_engine.py

### Group 19 (2 scripts)
- /opt/sutazaiapp/docker/llamaindex/llamaindex_service.py
- /opt/sutazaiapp/archive/20250803_193506_pre_cleanup/docker/llamaindex/llamaindex_service.py

### Group 20 (2 scripts)
- /opt/sutazaiapp/docker/crewai/crewai_service.py
- /opt/sutazaiapp/archive/20250803_193506_pre_cleanup/docker/crewai/crewai_service.py

### Group 21 (2 scripts)
- /opt/sutazaiapp/docker/faiss/health_check.py
- /opt/sutazaiapp/archive/20250803_193506_pre_cleanup/docker/faiss/health_check.py

### Group 22 (2 scripts)
- /opt/sutazaiapp/docker/faiss/faiss_service.py
- /opt/sutazaiapp/archive/20250803_193506_pre_cleanup/docker/faiss/faiss_service.py

### Group 23 (2 scripts)
- /opt/sutazaiapp/docker/enhanced-model-manager/deepseek_integration.py
- /opt/sutazaiapp/archive/20250803_193506_pre_cleanup/docker/enhanced-model-manager/deepseek_integration.py

### Group 24 (2 scripts)
- /opt/sutazaiapp/docker/enhanced-model-manager/enhanced_model_service.py
- /opt/sutazaiapp/archive/20250803_193506_pre_cleanup/docker/enhanced-model-manager/enhanced_model_service.py

### Group 25 (2 scripts)
- /opt/sutazaiapp/docker/enhanced-model-manager/model_manager.py
- /opt/sutazaiapp/archive/20250803_193506_pre_cleanup/docker/enhanced-model-manager/model_manager.py

### Group 26 (2 scripts)
- /opt/sutazaiapp/docker/letta/letta_service.py
- /opt/sutazaiapp/archive/20250803_193506_pre_cleanup/docker/letta/letta_service.py

### Group 27 (2 scripts)
- /opt/sutazaiapp/docker/gpt-engineer/gpt_engineer_service.py
- /opt/sutazaiapp/archive/20250803_193506_pre_cleanup/docker/gpt-engineer/gpt_engineer_service.py

### Group 28 (2 scripts)
- /opt/sutazaiapp/docker/health-check/health_check.py
- /opt/sutazaiapp/archive/20250803_193506_pre_cleanup/docker/health-check/health_check.py

### Group 29 (2 scripts)
- /opt/sutazaiapp/docker/autogpt-real/agent.py
- /opt/sutazaiapp/archive/20250803_193506_pre_cleanup/docker/autogpt-real/agent.py

### Group 30 (2 scripts)
- /opt/sutazaiapp/docker/awesome-code-ai/awesome_code_service.py
- /opt/sutazaiapp/archive/20250803_193506_pre_cleanup/docker/awesome-code-ai/awesome_code_service.py

### Group 31 (2 scripts)
- /opt/sutazaiapp/docker/awesome-code-ai/code_ai_manager.py
- /opt/sutazaiapp/archive/20250803_193506_pre_cleanup/docker/awesome-code-ai/code_ai_manager.py

### Group 32 (2 scripts)
- /opt/sutazaiapp/docker/localagi/localagi_service.py
- /opt/sutazaiapp/archive/20250803_193506_pre_cleanup/docker/localagi/localagi_service.py

### Group 33 (4 scripts)
- /opt/sutazaiapp/docker/semgrep/main.py
- /opt/sutazaiapp/docker/pentestgpt/main.py
- /opt/sutazaiapp/archive/20250803_193506_pre_cleanup/docker/semgrep/main.py
- /opt/sutazaiapp/archive/20250803_193506_pre_cleanup/docker/pentestgpt/main.py

### Group 34 (2 scripts)
- /opt/sutazaiapp/docker/privategpt/privategpt_service.py
- /opt/sutazaiapp/archive/20250803_193506_pre_cleanup/docker/privategpt/privategpt_service.py

### Group 35 (2 scripts)
- /opt/sutazaiapp/docker/fsdp/fsdp_service.py
- /opt/sutazaiapp/archive/20250803_193506_pre_cleanup/docker/fsdp/fsdp_service.py

### Group 36 (2 scripts)
- /opt/sutazaiapp/docker/agentgpt/agentgpt_service.py
- /opt/sutazaiapp/archive/20250803_193506_pre_cleanup/docker/agentgpt/agentgpt_service.py

### Group 37 (2 scripts)
- /opt/sutazaiapp/docker/knowledge-manager/knowledge_manager.py
- /opt/sutazaiapp/archive/20250803_193506_pre_cleanup/docker/knowledge-manager/knowledge_manager.py

### Group 38 (2 scripts)
- /opt/sutazaiapp/docker/documind/documind_service.py
- /opt/sutazaiapp/archive/20250803_193506_pre_cleanup/docker/documind/documind_service.py

### Group 39 (2 scripts)
- /opt/sutazaiapp/docker/service-hub/service_hub.py
- /opt/sutazaiapp/archive/20250803_193506_pre_cleanup/docker/service-hub/service_hub.py

### Group 40 (2 scripts)
- /opt/sutazaiapp/docker/fms-fsdp/fsdp_service.py
- /opt/sutazaiapp/archive/20250803_193506_pre_cleanup/docker/fms-fsdp/fsdp_service.py

### Group 41 (2 scripts)
- /opt/sutazaiapp/docker/code-improver/code_improver.py
- /opt/sutazaiapp/archive/20250803_193506_pre_cleanup/docker/code-improver/code_improver.py

### Group 42 (2 scripts)
- /opt/sutazaiapp/docker/aider/aider_service.py
- /opt/sutazaiapp/archive/20250803_193506_pre_cleanup/docker/aider/aider_service.py

### Group 43 (2 scripts)
- /opt/sutazaiapp/docker/langchain-agents/langchain_agent_server.py
- /opt/sutazaiapp/archive/20250803_193506_pre_cleanup/docker/langchain-agents/langchain_agent_server.py

### Group 44 (2 scripts)
- /opt/sutazaiapp/docker/browser-use/browser_use_server.py
- /opt/sutazaiapp/archive/20250803_193506_pre_cleanup/docker/browser-use/browser_use_server.py

### Group 45 (2 scripts)
- /opt/sutazaiapp/docker/finrobot/finrobot_service.py
- /opt/sutazaiapp/archive/20250803_193506_pre_cleanup/docker/finrobot/finrobot_service.py

### Group 46 (2 scripts)
- /opt/sutazaiapp/backend/app/models/__init__.py
- /opt/sutazaiapp/backend/app/agents/__init__.py

### Group 47 (67 scripts)
- /opt/sutazaiapp/agents/distributed-computing-architect/app.py
- /opt/sutazaiapp/agents/autogpt/app.py
- /opt/sutazaiapp/agents/autogen/app.py
- /opt/sutazaiapp/agents/agentzero-coordinator/app.py
- /opt/sutazaiapp/agents/cognitive-architecture-designer/app.py
- /opt/sutazaiapp/agents/product-strategy-architect/app.py
- /opt/sutazaiapp/agents/health-monitor/app.py
- /opt/sutazaiapp/agents/multi-modal-fusion-coordinator/app.py
- /opt/sutazaiapp/agents/senior-frontend-developer/app.py
- /opt/sutazaiapp/agents/private-data-analyst/app.py
- /opt/sutazaiapp/agents/senior-backend-developer/app.py
- /opt/sutazaiapp/agents/synthetic-data-generator/app.py
- /opt/sutazaiapp/agents/reinforcement-learning-trainer/app.py
- /opt/sutazaiapp/agents/flowiseai-flow-manager/app.py
- /opt/sutazaiapp/agents/crewai/app.py
- /opt/sutazaiapp/agents/data-analysis-engineer/app.py
- /opt/sutazaiapp/agents/meta-learning-specialist/app.py
- /opt/sutazaiapp/agents/mcp-server/app.py
- /opt/sutazaiapp/agents/ai-product-manager/app.py
- /opt/sutazaiapp/agents/opendevin-code-generator/app.py
- /opt/sutazaiapp/agents/context-framework/app.py
- /opt/sutazaiapp/agents/task-assignment-coordinator/app.py
- /opt/sutazaiapp/agents/letta/app.py
- /opt/sutazaiapp/agents/gpt-engineer/app.py
- /opt/sutazaiapp/agents/awesome-code-ai/app.py
- /opt/sutazaiapp/agents/devika/app.py
- /opt/sutazaiapp/agents/model-training-specialist/app.py
- /opt/sutazaiapp/agents/bigagi-system-manager/app.py
- /opt/sutazaiapp/agents/episodic-memory-engineer/app.py
- /opt/sutazaiapp/agents/agentgpt-autonomous-executor/app.py
- /opt/sutazaiapp/agents/langflow-workflow-designer/app.py
- /opt/sutazaiapp/agents/knowledge-graph-builder/app.py
- /opt/sutazaiapp/agents/browser-automation-orchestrator/app.py
- /opt/sutazaiapp/agents/neuromorphic-computing-expert/app.py
- /opt/sutazaiapp/agents/dify-automation-specialist/app.py
- /opt/sutazaiapp/agents/quantum-computing-optimizer/app.py
- /opt/sutazaiapp/agents/transformers-migration-specialist/app.py
- /opt/sutazaiapp/agents/semgrep-security-analyzer/app.py
- /opt/sutazaiapp/agents/ai-scrum-master/app.py
- /opt/sutazaiapp/agents/privategpt/app.py
- /opt/sutazaiapp/agents/knowledge-distillation-expert/app.py
- /opt/sutazaiapp/agents/memory-persistence-manager/app.py
- /opt/sutazaiapp/agents/explainable-ai-specialist/app.py
- /opt/sutazaiapp/agents/jarvis-voice-interface/app.py
- /opt/sutazaiapp/agents/garbage-collector-coordinator/app.py
- /opt/sutazaiapp/agents/fsdp/app.py
- /opt/sutazaiapp/agents/agentgpt/app.py
- /opt/sutazaiapp/agents/shellgpt/app.py
- /opt/sutazaiapp/agents/babyagi/app.py
- /opt/sutazaiapp/agents/observability-monitoring-engineer/app.py
- /opt/sutazaiapp/agents/service-hub/app.py
- /opt/sutazaiapp/agents/self-healing-orchestrator/app.py
- /opt/sutazaiapp/agents/symbolic-reasoning-engine/app.py
- /opt/sutazaiapp/agents/pentestgpt/app.py
- /opt/sutazaiapp/agents/intelligence-optimization-monitor/app.py
- /opt/sutazaiapp/agents/attention-optimizer/app.py
- /opt/sutazaiapp/agents/data-pipeline-engineer/app.py
- /opt/sutazaiapp/agents/code-improver/app.py
- /opt/sutazaiapp/agents/ai-agent-debugger/app.py
- /opt/sutazaiapp/agents/causal-inference-expert/app.py
- /opt/sutazaiapp/agents/federated-learning-coordinator/app.py
- /opt/sutazaiapp/agents/document-knowledge-manager/app.py
- /opt/sutazaiapp/agents/aider/app.py
- /opt/sutazaiapp/agents/gradient-compression-specialist/app.py
- /opt/sutazaiapp/agents/edge-computing-optimizer/app.py
- /opt/sutazaiapp/agents/localagi-orchestration-manager/app.py
- /opt/sutazaiapp/agents/finrobot/app.py

### Group 48 (2 scripts)
- /opt/sutazaiapp/docker/validate.sh
- /opt/sutazaiapp/archive/20250803_193506_pre_cleanup/docker/validate.sh

### Group 49 (2 scripts)
- /opt/sutazaiapp/docker/build.sh
- /opt/sutazaiapp/archive/20250803_193506_pre_cleanup/docker/build.sh

### Group 50 (2 scripts)
- /opt/sutazaiapp/scripts/utils/ollama_cleanup.sh
- /opt/sutazaiapp/scripts/models/ollama/ollama_cleanup.sh

### Group 51 (2 scripts)
- /opt/sutazaiapp/scripts/utils/ollama_health_check.sh
- /opt/sutazaiapp/scripts/models/ollama/ollama_health_check.sh

### Group 52 (2 scripts)
- /opt/sutazaiapp/docker/code-improver/improve_cron.sh
- /opt/sutazaiapp/archive/20250803_193506_pre_cleanup/docker/code-improver/improve_cron.sh
