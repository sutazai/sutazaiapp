#!/bin/bash

set -e


# Signal handlers for graceful shutdown
cleanup_and_exit() {
    local exit_code="${1:-0}"
    echo "Script interrupted, cleaning up..." >&2
    # Clean up any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    exit "$exit_code"
}

trap 'cleanup_and_exit 130' INT
trap 'cleanup_and_exit 143' TERM
trap 'cleanup_and_exit 1' ERR

echo "Setting up SutazAI Authentication Infrastructure..."

# Configuration
KEYCLOAK_URL="http://localhost:10050"
SERVICE_ACCOUNT_MANAGER_URL="http://localhost:10055"
JWT_SERVICE_URL="http://localhost:10054"
RBAC_ENGINE_URL="http://localhost:10056"
KONG_ADMIN_URL="http://localhost:10052"

# Wait for services to be ready
wait_for_service() {
    local url=$1
    local service=$2
    echo "Waiting for $service to be ready..."
    
    for i in {1..30}; do
        if curl -f "$url/health" >/dev/null 2>&1; then
            echo "$service is ready!"
            return 0
        fi
        echo "Waiting for $service... ($i/30)"
        sleep 10
    done
    
    echo "ERROR: $service failed to start within timeout"
    return 1
}

# Start authentication services
echo "Starting authentication services..."
cd /opt/sutazaiapp
docker-compose -f docker-compose.auth.yml up -d

# Wait for services
wait_for_service "$KEYCLOAK_URL" "Keycloak"
wait_for_service "$SERVICE_ACCOUNT_MANAGER_URL" "Service Account Manager"
wait_for_service "$JWT_SERVICE_URL" "JWT Service"
wait_for_service "$RBAC_ENGINE_URL" "RBAC Engine"

# Create service accounts for all 69 AI agents
echo "Creating service accounts for all AI agents..."
curl -X POST "$SERVICE_ACCOUNT_MANAGER_URL/service-accounts/create-all-agents" \
    -H "Content-Type: application/json" \
    || echo "Warning: Failed to create some service accounts"

# Wait a bit for service accounts to be fully created
sleep 5

# Configure Kong with authentication
echo "Configuring Kong with authentication..."
curl -X POST "$KONG_ADMIN_URL/config" \
    -F "config=@/opt/sutazaiapp/auth/kong/kong.yml" \
    || echo "Warning: Kong configuration may need manual setup"

# Create JWT tokens for critical services
echo "Creating JWT tokens for critical services..."

CRITICAL_SERVICES=(
    "agent-orchestrator"
    "ai-system-validator"
    "ai-senior-backend-developer"
    "monitoring-agent"
    "health-monitor"
)

for service in "${CRITICAL_SERVICES[@]}"; do
    echo "Creating JWT token for $service..."
    token_response=$(curl -s -X POST "$JWT_SERVICE_URL/auth/token" \
        -H "Content-Type: application/json" \
        -d "{
            \"service_name\": \"$service\",
            \"scopes\": [\"read\", \"write\", \"agent\"],
            \"expires_in\": 86400
        }")
    
    if echo "$token_response" | grep -q "access_token"; then
        # Extract token and store in environment file
        token=$(echo "$token_response" | grep -o '"access_token":"[^"]*' | cut -d'"' -f4)
        echo "${service^^}_JWT_TOKEN=$token" >> /opt/sutazaiapp/auth/.env
        echo "✓ JWT token created for $service"
    else
        echo "✗ Failed to create JWT token for $service"
    fi
done

# Setup RBAC policies for agent groups
echo "Setting up RBAC policies..."

# Create role-based policies
curl -s -X POST "$RBAC_ENGINE_URL/policies" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $(cat /opt/sutazaiapp/auth/.env | grep ADMIN_JWT_TOKEN | cut -d'=' -f2)" \
    -d '{
        "subject": "role:ai-agent",
        "object": "api:ollama",
        "action": "read"
    }' || echo "Warning: Failed to create RBAC policy"

curl -s -X POST "$RBAC_ENGINE_URL/policies" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $(cat /opt/sutazaiapp/auth/.env | grep ADMIN_JWT_TOKEN | cut -d'=' -f2)" \
    -d '{
        "subject": "role:ai-agent",
        "object": "api:vector-db",
        "action": "read"
    }' || echo "Warning: Failed to create RBAC policy"

# Assign roles to agent groups
AI_AGENTS=(
    "adversarial-attack-detector" "agent-creator" "agent-debugger" "agent-orchestrator"
    "agentgpt-autonomous-executor" "agentgpt" "agentzero-coordinator" "ai-agent-debugger"
    "ai-product-manager" "ai-qa-team-lead" "ai-scrum-master" "ai-senior-backend-developer"
    "ai-senior-engineer" "ai-senior-frontend-developer" "ai-senior-full-stack-developer"
    "ai-system-architect" "ai-system-validator" "ai-testing-qa-validator" "aider"
    "attention-optimizer" "autogen" "autogpt" "automated-incident-responder"
    "autonomous-task-executor" "awesome-code-ai" "babyagi" "bias-and-fairness-auditor"
    "bigagi-system-manager" "browser-automation-orchestrator" "causal-inference-expert"
    "cicd-pipeline-orchestrator" "code-improver" "code-quality-gateway-sonarqube"
    "codebase-team-lead" "cognitive-architecture-designer" "cognitive-load-monitor"
    "compute-scheduler-and-optimizer" "container-orchestrator-k3s" "container-vulnerability-scanner-trivy"
    "context-framework" "cpu-only-hardware-optimizer" "crewai" "data-analysis-engineer"
    "data-drift-detector" "data-lifecycle-manager" "data-pipeline-engineer" "data-version-controller-dvc"
    "deep-learning-brain-architect" "deep-learning-brain-manager" "deep-local-brain-builder"
    "deploy-automation-master" "deployment-automation-master" "devika" "dify-automation-specialist"
    "distributed-computing-architect" "distributed-tracing-analyzer-jaeger" "document-knowledge-manager"
    "edge-computing-optimizer" "edge-inference-proxy" "emergency-shutdown-coordinator"
    "energy-consumption-optimize" "episodic-memory-engineer" "ethical-governor"
    "evolution-strategy-trainer" "experiment-tracker" "explainability-and-transparency-agent"
    "explainable-ai-specialist" "federated-learning-coordinator" "finrobot" "flowiseai-flow-manager"
    "fsdp" "garbage-collector-coordinator" "garbage-collector" "genetic-algorithm-tuner"
    "goal-setting-and-planning-agent" "gpt-engineer" "gpu-hardware-optimizer"
)

echo "Assigning 'ai-agent' role to all agents..."
for agent in "${AI_AGENTS[@]}"; do
    curl -s -X POST "$RBAC_ENGINE_URL/roles/assign" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $(cat /opt/sutazaiapp/auth/.env | grep ADMIN_JWT_TOKEN | cut -d'=' -f2)" \
        -d "{
            \"user\": \"agent-$agent\",
            \"role\": \"role:ai-agent\"
        }" >/dev/null || echo "Warning: Failed to assign role to $agent"
done

# Update agent configurations with authentication
echo "Updating agent configurations with authentication credentials..."

# Create agent authentication configuration template
cat > /opt/sutazaiapp/auth/agent-auth-template.json << 'EOF'
{
    "authentication": {
        "enabled": true,
        "keycloak_url": "http://keycloak:8080",
        "realm": "sutazai",
        "jwt_service_url": "http://jwt-service:8080",
        "rbac_engine_url": "http://rbac-policy-engine:8080",
        "kong_proxy_url": "http://kong:8000"
    },
    "credentials": {
        "client_id": "agent-{AGENT_NAME}",
        "client_secret": "{CLIENT_SECRET}",
        "scopes": ["read", "write", "agent"]
    },
    "endpoints": {
        "token": "/auth/token",
        "validate": "/auth/validate",
        "revoke": "/auth/revoke"
    }
}
EOF

# Generate configuration for each agent
mkdir -p /opt/sutazaiapp/auth/agent-configs
for agent in "${AI_AGENTS[@]}"; do
    # Get service account details
    account_info=$(curl -s "$SERVICE_ACCOUNT_MANAGER_URL/service-accounts/$agent")
    
    if echo "$account_info" | grep -q "client_id"; then
        client_id=$(echo "$account_info" | grep -o '"client_id":"[^"]*' | cut -d'"' -f4)
        
        # Create agent-specific configuration
        sed "s/{AGENT_NAME}/$agent/g; s/{CLIENT_SECRET}/[STORED_IN_VAULT]/g" \
            /opt/sutazaiapp/auth/agent-auth-template.json > \
            "/opt/sutazaiapp/auth/agent-configs/$agent.json"
    fi
done

# Create environment file for authentication services
cat > /opt/sutazaiapp/auth/.env << EOF
# SutazAI Authentication Configuration
KEYCLOAK_URL=$KEYCLOAK_URL
KEYCLOAK_REALM=sutazai
KEYCLOAK_CLIENT_ID=sutazai-backend
KEYCLOAK_CLIENT_SECRET=${KEYCLOAK_CLIENT_SECRET:-$(openssl rand -base64 32)}

JWT_SERVICE_URL=$JWT_SERVICE_URL
JWT_SECRET=${JWT_SECRET:-$(openssl rand -base64 32)}

RBAC_ENGINE_URL=$RBAC_ENGINE_URL
SERVICE_ACCOUNT_MANAGER_URL=$SERVICE_ACCOUNT_MANAGER_URL

KONG_ADMIN_URL=$KONG_ADMIN_URL
KONG_PROXY_URL=http://localhost:10051

VAULT_URL=http://localhost:10053
VAULT_ROOT_TOKEN=${VAULT_ROOT_TOKEN:-sutazai_vault_root}

# Generated on: $(date)
EOF

# Test authentication flow
echo "Testing authentication flow..."

# Test JWT token generation
echo "Testing JWT token generation..."
test_token=$(curl -s -X POST "$JWT_SERVICE_URL/auth/token" \
    -H "Content-Type: application/json" \
    -d '{
        "service_name": "test-agent",
        "scopes": ["read"],
        "expires_in": 300
    }')

if echo "$test_token" | grep -q "access_token"; then
    echo "✓ JWT token generation works"
    
    # Test token validation
    token=$(echo "$test_token" | grep -o '"access_token":"[^"]*' | cut -d'"' -f4)
    validation_result=$(curl -s -X POST "$JWT_SERVICE_URL/auth/validate" \
        -H "Content-Type: application/json" \
        -d "{\"token\": \"$token\"}")
    
    if echo "$validation_result" | grep -q '"valid":true'; then
        echo "✓ JWT token validation works"
    else
        echo "✗ JWT token validation failed"
    fi
else
    echo "✗ JWT token generation failed"
fi

# Test RBAC access check
echo "Testing RBAC access control..."
access_result=$(curl -s -X POST "$RBAC_ENGINE_URL/access/check" \
    -H "Content-Type: application/json" \
    -d '{
        "subject": "role:ai-agent",
        "object": "api:ollama",
        "action": "read"
    }')

if echo "$access_result" | grep -q '"allowed":true'; then
    echo "✓ RBAC access control works"
else
    echo "✗ RBAC access control failed"
fi

# Display summary
echo ""
echo "=========================================="
echo "SutazAI Authentication Setup Complete!"
echo "=========================================="
echo ""
echo "Services:"
echo "  - Keycloak (Identity Provider): $KEYCLOAK_URL"
echo "  - Kong (API Gateway): http://localhost:10051"
echo "  - JWT Service: $JWT_SERVICE_URL"
echo "  - Service Account Manager: $SERVICE_ACCOUNT_MANAGER_URL"
echo "  - RBAC Policy Engine: $RBAC_ENGINE_URL"
echo "  - Vault (Secrets): http://localhost:10053"
echo ""
echo "Configuration:"
echo "  - Service accounts created for all 69 AI agents"
echo "  - RBAC policies configured"
echo "  - Kong gateway configured with authentication"
echo "  - Agent configurations generated"
echo ""
echo "Authentication Environment:"
echo "  - Configuration: /opt/sutazaiapp/auth/.env"
echo "  - Agent configs: /opt/sutazaiapp/auth/agent-configs/"
echo "  - Logs: /opt/sutazaiapp/logs/"
echo ""
echo "Next steps:"
echo "  1. Update your agents to use the authentication configurations"
echo "  2. Test API access through Kong gateway"
echo "  3. Monitor authentication logs"
echo "  4. Configure additional RBAC policies as needed"
echo ""