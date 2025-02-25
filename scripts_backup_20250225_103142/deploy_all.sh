#!/bin/bash
set -euo pipefail

# Load environment specific config
source ./configs/environments/${ENVIRONMENT}.yaml

# Pre-deployment checks
./scripts/pre_deploy_check.sh

# Deployment steps
echo "Starting full deployment..."
./scripts/deploy_ai.sh
./dags/etl_pipeline.py --environment ${ENVIRONMENT}

# Post-deployment verification
./scripts/post_deploy_verify.sh

# Notify deployment status
python3 ./utilities/slack_notifier.py "Deployment to ${ENVIRONMENT} completed successfully"

echo "Deployment completed successfully" 