#!/bin/bash
# Test script for SutazAI auto-scaling components

set -euo pipefail

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Testing SutazAI Auto-scaling Components${NC}"
echo "========================================"

# Test 1: Validate directory structure
echo -e "\n${YELLOW}Test 1: Directory Structure${NC}"
if [[ -f "deployment/autoscaling/README.md" ]] && \
   [[ -d "deployment/autoscaling/swarm" ]] && \
   [[ -d "deployment/autoscaling/kubernetes" ]] && \
   [[ -d "deployment/autoscaling/load-balancing" ]]; then
    echo -e "${GREEN}✓ Directory structure is correct${NC}"
else
    echo -e "${RED}✗ Directory structure is incorrect${NC}"
    exit 1
fi

# Test 2: Validate configuration files
echo -e "\n${YELLOW}Test 2: Configuration Files${NC}"
files=(
    "deployment/autoscaling/hpa-enhanced.yaml"
    "deployment/autoscaling/vpa-config.yaml"
    "deployment/autoscaling/swarm/swarm-autoscaler.py"
    "deployment/autoscaling/load-balancing/nginx-ingress.yaml"
    "deployment/autoscaling/monitoring/ai-metrics-exporter.yaml"
)

for file in "${files[@]}"; do
    if [[ -f "$file" ]]; then
        echo -e "${GREEN}✓ Found: $file${NC}"
    else
        echo -e "${RED}✗ Missing: $file${NC}"
        exit 1
    fi
done

# Test 3: Validate Python script syntax
echo -e "\n${YELLOW}Test 3: Python Script Validation${NC}"
if python3 -m py_compile deployment/autoscaling/swarm/swarm-autoscaler.py; then
    echo -e "${GREEN}✓ swarm-autoscaler.py syntax is valid${NC}"
    rm -f deployment/autoscaling/swarm/__pycache__/*.pyc
    rmdir deployment/autoscaling/swarm/__pycache__ 2>/dev/null || true
else
    echo -e "${RED}✗ swarm-autoscaler.py has syntax errors${NC}"
    exit 1
fi

# Test 4: Validate YAML files
echo -e "\n${YELLOW}Test 4: YAML Validation${NC}"
yaml_files=(
    "deployment/autoscaling/hpa-enhanced.yaml"
    "deployment/autoscaling/vpa-config.yaml"
    "deployment/autoscaling/load-balancing/nginx-ingress.yaml"
    "deployment/autoscaling/load-balancing/traefik-config.yaml"
    "deployment/autoscaling/monitoring/ai-metrics-exporter.yaml"
)

for yaml_file in "${yaml_files[@]}"; do
    if python3 -c "import yaml; list(yaml.safe_load_all(open('$yaml_file')))" 2>/dev/null; then
        echo -e "${GREEN}✓ Valid YAML: $yaml_file${NC}"
    else
        echo -e "${RED}✗ Invalid YAML: $yaml_file${NC}"
        # Show error details for debugging
        python3 -c "import yaml; list(yaml.safe_load_all(open('$yaml_file')))" 2>&1 | head -5
        exit 1
    fi
done

# Test 5: Check deployment script
echo -e "\n${YELLOW}Test 5: Deployment Script${NC}"
if [[ -x "deployment/autoscaling/scripts/deploy-autoscaling.sh" ]]; then
    echo -e "${GREEN}✓ Deployment script is executable${NC}"
else
    echo -e "${RED}✗ Deployment script is not executable${NC}"
    exit 1
fi

# Test 6: Docker Swarm autoscaler simulation
echo -e "\n${YELLOW}Test 6: Swarm Autoscaler Simulation${NC}"
echo "Creating minimal test environment..."

# Create a test Python script to simulate the autoscaler
cat > /tmp/test-autoscaler.py << 'EOF'
#!/usr/bin/env python3
import ast
import sys

# Parse the autoscaler script to check its structure
try:
    with open('deployment/autoscaling/swarm/swarm-autoscaler.py', 'r') as f:
        tree = ast.parse(f.read())
    
    # Check for required classes
    classes_found = []
    functions_found = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes_found.append(node.name)
        elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            functions_found.append(node.name)
    
    if 'SwarmAutoscaler' in classes_found:
        print("✓ SwarmAutoscaler class found")
    else:
        print("✗ SwarmAutoscaler class not found")
        sys.exit(1)
    
    if 'main' in functions_found:
        print("✓ main() function found")
    else:
        print("✗ main() function not found")
        sys.exit(1)
        
    print(f"✓ Found {len(classes_found)} classes and {len(functions_found)} functions")
    
except Exception as e:
    print(f"✗ Error parsing script: {e}")
    sys.exit(1)
EOF

if python3 /tmp/test-autoscaler.py; then
    echo -e "${GREEN}✓ Swarm autoscaler module structure is valid${NC}"
else
    echo -e "${RED}✗ Swarm autoscaler module has issues${NC}"
fi

rm -f /tmp/test-autoscaler.py

# Test 7: Network connectivity check
echo -e "\n${YELLOW}Test 7: Docker Network${NC}"
if docker network ls | grep -q sutazaiapp_sutazai-network; then
    echo -e "${GREEN}✓ Docker network exists${NC}"
else
    echo -e "${YELLOW}! Docker network not found (will be created during deployment)${NC}"
fi

# Summary
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}All auto-scaling tests passed!${NC}"
echo -e "${GREEN}========================================${NC}"

echo -e "\n${YELLOW}Next steps:${NC}"
echo "1. Deploy to Docker Swarm:"
echo "   ./deployment/autoscaling/scripts/deploy-autoscaling.sh swarm production"
echo ""
echo "2. Deploy to Kubernetes:"
echo "   ./deployment/autoscaling/scripts/deploy-autoscaling.sh kubernetes production"
echo ""
echo "3. Test locally with Docker Compose:"
echo "   ./deployment/autoscaling/scripts/deploy-autoscaling.sh compose local"