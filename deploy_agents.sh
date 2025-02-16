#!/bin/bash
# SutazAI Hypervisor 7.0 Deployment Script

set -eo pipefail
shopt -s inherit_errexit

# Enhanced logging
exec > >(tee -a "${CONFIG[LOG_DIR]}/deployment.log") 2>&1

# Initialize deployment engine
ENGINE="/opt/sutazai/engine/deploy.py"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --insecure) INSECURE=1; shift ;;
        --validate) VALIDATE=1; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

# Main execution with error handling
main() {
    try {
        setup_logging
        validate_config
        optimize_hardware
        deploy_services
        validate_deployment
    } catch {
        echo "❌ Deployment failed"
        exit 1
    }
}

# Execute deployment
python3 $ENGINE \
    --root-dir "/opt/sutazai" \
    --log-dir "/var/log/sutazai" \
    --model-registry "/opt/sutazai/models" \
    ${INSECURE:+--insecure} \
    ${VALIDATE:+--validate}

echo "✅ Deployment completed successfully" 