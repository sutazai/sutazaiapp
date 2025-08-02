# Reusable Scripts Analysis & Requirements

## Comprehensive Script Inventory

### ✅ Core Infrastructure Scripts (Reusable)
1. **deploy_complete_system.sh** - Main deployment script (already unified)
2. **common.sh** - Shared utilities (logging, error handling)
3. **health_check.sh** - System health monitoring
4. **ollama_health_check.sh** - Ollama-specific health checks
5. **setup_monitoring.sh** - Prometheus/Grafana setup
6. **verify_deployment.sh** - Deployment verification

### ✅ Model Management Scripts (Reusable)
1. **download_models.sh** - Model downloading
2. **setup_models.sh** - Model configuration
3. **preload_models.sh** - Model preloading
4. **ollama-startup.sh** - Ollama service startup
5. **optimize_ollama.sh** - Ollama optimization

### ✅ System Management Scripts (Reusable)
1. **start_services.sh** - Service startup
2. **stop_services.sh** - Service shutdown
3. **check_system_status.sh** - System status checks
4. **optimize_system.sh** - System optimization
5. **memory_optimizer.sh** - Memory management
6. **cleanup_cache.sh** - Cache cleanup

### ✅ Agent Management Scripts (Enhance/Reuse)
1. **check_agent_services.sh** - Agent health monitoring
2. **deploy_all_agents.sh** - Agent deployment
3. **orchestrator.sh** - Agent orchestration

### ✅ Development Scripts (Reusable)
1. **build.sh** - Build processes
2. **run_tests.sh** - Testing
3. **check_code_quality.sh** - Code quality checks

### ⚠️ Scripts Needing Enhancement
1. **setup_all_models.sh** - Needs Ollama integration updates
2. **install_services.sh** - Needs new AI agent installations
3. **check_environment.sh** - Needs new dependencies

## New Scripts Required

### 1. AI Agent Integration Scripts
```bash
# scripts/install_ai_agents.sh - NEW
#!/bin/bash
# Install and configure all 40+ AI agents with Ollama integration

# scripts/configure_ollama_agents.sh - NEW  
#!/bin/bash
# Configure all agents to use Ollama backend

# scripts/batch_agent_deployment.sh - NEW
#!/bin/bash
# Deploy agents in batches of 10-15 for system stability
```

### 2. Autonomous Code Generation Scripts
```bash
# scripts/autonomous_codegen.sh - NEW
#!/bin/bash
# Enable autonomous code generation capabilities

# scripts/self_improvement_runner.sh - NEW
#!/bin/bash  
# Run self-improvement analysis and apply changes

# scripts/neural_processing_setup.sh - NEW
#!/bin/bash
# Setup neural processing and consciousness systems
```

### 3. Container Management Scripts
```bash
# scripts/fix_container_issues.sh - NEW
#!/bin/bash
# Fix known container startup issues (Loki, N8N, backend-agi, frontend-agi)

# scripts/create_missing_dockerfiles.sh - NEW
#!/bin/bash
# Generate missing Dockerfiles for agents

# scripts/container_health_monitor.sh - NEW
#!/bin/bash
# Monitor and restart unhealthy containers
```

### 4. Integration & Configuration Scripts
```bash
# scripts/litellm_proxy_setup.sh - NEW
#!/bin/bash
# Setup LiteLLM proxy for OpenAI API compatibility

# scripts/enterprise_features_setup.sh - NEW
#!/bin/bash
# Setup enterprise AGI features

# scripts/vector_db_integration.sh - NEW
#!/bash
# Integrate and optimize vector databases
```

## Script Reusability Strategy

### Phase 1: Fix & Enhance Existing Scripts
1. **Enhance deploy_complete_system.sh**
   - Add new AI agents
   - Fix container issues
   - Add autonomous features

2. **Update setup_models.sh**
   - Add Deepseek-R1, Qwen3
   - Configure LiteLLM proxy
   - Optimize model loading

3. **Enhance check_agent_services.sh**
   - Add 40+ new agents
   - Improve health checks
   - Add agent capability detection

### Phase 2: Create New Required Scripts
1. **AI Agent Batch Installer**
   - Process 10-15 agents at a time
   - Handle dependencies
   - Configure Ollama integration

2. **Container Issue Resolver**
   - Fix Loki configuration
   - Fix N8N environment
   - Create missing Dockerfiles

3. **Autonomous System Setup**
   - Neural processing engine
   - Self-improvement system
   - Code generation capabilities

### Phase 3: Integration & Testing Scripts
1. **Comprehensive Integration Tester**
   - Test all 40+ agents
   - Verify Ollama connectivity
   - Check autonomous features

2. **Performance Optimizer**
   - Optimize resource usage
   - Balance agent loads
   - Tune system parameters

## Batch Processing Plan (50 Files at a Time)

### Batch 1: Core Infrastructure (15 files)
1. Fix container issues (Dockerfiles)
2. Update docker-compose.yml
3. Enhance deployment scripts
4. Fix monitoring configurations
5. Update environment variables

### Batch 2: AI Agent Dockerfiles (25 files)
1. Create missing agent Dockerfiles
2. Update existing agent configurations
3. Add Ollama integration patterns
4. Configure health checks
5. Set up networking

### Batch 3: Backend Integration (20 files)
1. Enhance backend AGI system
2. Add agent orchestration
3. Implement neural processing
4. Create API endpoints
5. Add autonomous features

### Batch 4: Frontend & UI (15 files)
1. Fix frontend container
2. Update Streamlit interface
3. Add agent management UI
4. Create monitoring dashboards
5. Implement user controls

### Batch 5: Configuration & Optimization (25 files)
1. Configuration files
2. Environment setups
3. Monitoring configs
4. Security settings
5. Performance optimizations

## Implementation Priority

### Critical (Week 1)
1. Fix existing container issues
2. Create missing Dockerfiles
3. Update deployment script
4. Setup LiteLLM proxy

### Important (Week 2)
1. Deploy first 20 AI agents
2. Configure Ollama integration
3. Implement basic orchestration
4. Setup monitoring

### Enhancement (Week 3)
1. Deploy remaining agents
2. Implement autonomous features
3. Add neural processing
4. Create comprehensive testing

### Advanced (Week 4)
1. Performance optimization
2. Advanced enterprise features
3. Security enhancements
4. Documentation completion

## Script Templates

### Generic Agent Installation Template
```bash
#!/bin/bash
source /opt/sutazaiapp/scripts/common.sh

AGENT_NAME="$1"
AGENT_PORT="$2"
AGENT_TYPE="$3"

install_agent() {
    log_info "Installing $AGENT_NAME..."
    
    # Create Dockerfile if needed
    create_dockerfile_if_missing "$AGENT_NAME"
    
    # Update docker-compose.yml
    add_agent_to_compose "$AGENT_NAME" "$AGENT_PORT" "$AGENT_TYPE"
    
    # Configure Ollama integration
    configure_ollama_backend "$AGENT_NAME"
    
    # Deploy agent
    docker-compose up -d "$AGENT_NAME"
    
    # Verify deployment
    verify_agent_health "$AGENT_NAME" "$AGENT_PORT"
    
    log_success "$AGENT_NAME installed successfully"
}
```

### Ollama Integration Template
```bash
#!/bin/bash
configure_ollama_integration() {
    local agent_name="$1"
    local config_file="/opt/sutazaiapp/docker/$agent_name/config.yaml"
    
    cat > "$config_file" <<EOF
model_backend: ollama
ollama_url: http://ollama:11434
default_model: llama3.2:3b
models:
  - tinyllama
  - qwen3:8b
  - codellama:7b
EOF

    log_success "Ollama integration configured for $agent_name"
}
```

## Success Metrics

1. **Script Reuse Rate**: >80% of existing scripts enhanced rather than replaced
2. **Deployment Success**: All 40+ agents deployable via scripts
3. **Automation Level**: 100% automated installation and configuration
4. **Error Recovery**: Automatic fixing of common container issues
5. **Performance**: Batch processing completes within expected timeframes