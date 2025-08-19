#!/bin/bash
# Setup Claude Code Router for temperature control

echo "Installing Claude Code Router..."

# Install claude-code-router
npm install -g @musistudio/claude-code-router

# Create configuration directory
mkdir -p ~/.claude-code-router

# Create config with low temperature for accuracy
cat > ~/.claude-code-router/config.json << 'EOF'
{
  "LOG": true,
  "API_TIMEOUT_MS": 600000,
  "Providers": [
    {
      "name": "anthropic-accurate",
      "api_base_url": "https://api.anthropic.com/v1/messages",
      "models": ["claude-3-opus-20240229"],
      "transformer": {
        "use": ["sampling"],
        "options": {
          "temperature": 0.2,
          "top_p": 0.6
        }
      }
    }
  ]
}
EOF

echo "Claude Code Router configured with temperature 0.2 for accuracy"
echo "To use: export ANTHROPIC_API_BASE=http://localhost:8080"
echo "Then run: claude-code-router"