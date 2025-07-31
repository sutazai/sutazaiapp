#!/bin/bash

echo "Starting Universal Agent: $AGENT_NAME"
echo "Model Provider: $MODEL_PROVIDER"
echo "Model: $MODEL_NAME"
echo "Capabilities: $AGENT_CAPABILITIES"

# Start the agent runtime
exec python agent_runtime.py