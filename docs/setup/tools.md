# Development Tools and IDEs

**Last Updated**: 2025-01-03  
**Version**: 1.0.0  
**Maintainer**: Development Team

## Table of Contents

1. [Required Development Tools](#required-development-tools)
2. [Recommended IDEs](#recommended-ides)
3. [Code Editors Setup](#code-editors-setup)
4. [Terminal Tools](#terminal-tools)
5. [Database Management Tools](#database-management-tools)
6. [API Development Tools](#api-development-tools)
7. [Container Management](#container-management)
8. [Monitoring and Debugging](#monitoring-and-debugging)
9. [Version Control](#version-control)
10. [AI/ML Development Tools](#aiml-development-tools)

## Required Development Tools

### Core Development Tools

```bash
# Essential command-line tools
git: 2.40+          # Version control
make: 4.3+          # Build automation  
curl: 7.88+         # HTTP client
wget: 1.21+         # File downloader
jq: 1.6+            # JSON processor
yq: 4.30+           # YAML processor
htop: 3.2+          # Process viewer
tmux: 3.3+          # Terminal multiplexer
```

### Installation Commands

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y git make curl wget jq htop tmux

# Install yq
wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /usr/local/bin/yq
chmod +x /usr/local/bin/yq

# macOS
brew install git make curl wget jq yq htop tmux

# Windows (WSL2)
# First install WSL2, then follow Ubuntu instructions
```

## Recommended IDEs

### Visual Studio Code

**Version**: Latest stable (1.85+)

**Essential Extensions**:
```json
{
  "recommendations": [
    // Python Development
    "ms-python.python",
    "ms-python.vscode-pylance",
    "ms-python.black-formatter",
    "ms-python.mypy-type-checker",
    
    // TypeScript/JavaScript
    "dbaeumer.vscode-eslint",
    "esbenp.prettier-vscode",
    "ms-vscode.vscode-typescript-next",
    
    // Docker & DevOps
    "ms-azuretools.vscode-docker",
    "ms-vscode-remote.remote-containers",
    "ms-kubernetes-tools.vscode-kubernetes-tools",
    
    // Database
    "mtxr.sqltools",
    "mtxr.sqltools-driver-pg",
    "cweijan.vscode-database-client2",
    
    // AI/ML
    "ms-toolsai.jupyter",
    "ms-toolsai.vscode-jupyter-cell-tags",
    
    // Utilities
    "streetsidesoftware.code-spell-checker",
    "wayou.vscode-todo-highlight",
    "gruntfuggly.todo-tree",
    "eamodio.gitlens",
    "redhat.vscode-yaml",
    "tamasfe.even-better-toml"
  ]
}
```

**Settings Configuration** (`.vscode/settings.json`):
```json
{
  "python.defaultInterpreterPath": "/opt/sutazaiapp/backend/venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.flake8Enabled": true,
  "python.linting.mypyEnabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "python.testing.unittestEnabled": false,
  
  "editor.formatOnSave": true,
  "editor.rulers": [88, 120],
  "editor.tabSize": 4,
  
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    "**/node_modules": true,
    "**/.git": true
  },
  
  "docker.defaultRegistryPath": "sutazai",
  "docker.dockerComposeBuild": false,
  
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter"
  },
  "[typescript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[json]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  }
}
```

### JetBrains PyCharm Professional

**Version**: 2023.3+

**Configuration**:
```xml
<!-- .idea/sutazai.iml -->
<module type="PYTHON_MODULE" version="4">
  <component name="NewModuleRootManager">
    <content url="file://$MODULE_DIR$">
      <sourceFolder url="file://$MODULE_DIR$/backend/app" isTestSource="false" />
      <sourceFolder url="file://$MODULE_DIR$/backend/tests" isTestSource="true" />
      <excludeFolder url="file://$MODULE_DIR$/backend/venv" />
      <excludeFolder url="file://$MODULE_DIR$/frontend/venv" />
      <excludeFolder url="file://$MODULE_DIR$/node_modules" />
    </content>
    <orderEntry type="jdk" jdkName="Python 3.12 (sutazai)" jdkType="Python SDK" />
  </component>
</module>
```

**Run Configurations**:
```xml
<!-- Backend FastAPI -->
<configuration name="Backend API" type="Python.FastAPI">
  <module name="sutazai" />
  <option name="SCRIPT_NAME" value="$PROJECT_DIR$/backend/app/main.py" />
  <option name="PARAMETERS" value="--host 0.0.0.0 --port 10200 --reload" />
  <option name="WORKING_DIRECTORY" value="$PROJECT_DIR$/backend" />
</configuration>

<!-- Frontend Streamlit -->
<configuration name="Frontend UI" type="PythonConfigurationType">
  <module name="sutazai" />
  <option name="SCRIPT_NAME" value="streamlit" />
  <option name="PARAMETERS" value="run app.py" />
  <option name="WORKING_DIRECTORY" value="$PROJECT_DIR$/frontend" />
</configuration>
```

### Cursor AI

**Configuration for AI-Assisted Development**:
```json
{
  "cursor.aiProvider": "anthropic",
  "cursor.apiKey": "${ANTHROPIC_API_KEY}",
  "cursor.model": "claude-3-opus",
  "cursor.contextWindow": 200000,
  "cursor.includeProjectContext": true,
  "cursor.autoSuggestions": true,
  "cursor.codebaseIndexing": {
    "enabled": true,
    "includePaths": [
      "backend/app",
      "frontend/src",
      "mcp-servers"
    ],
    "excludePaths": [
      "node_modules",
      "venv",
      "__pycache__"
    ]
  }
}
```

## Code Editors Setup

### Neovim Configuration

```lua
-- init.lua for Neovim 0.9+
require('packer').startup(function(use)
  -- Core plugins
  use 'neovim/nvim-lspconfig'
  use 'nvim-treesitter/nvim-treesitter'
  use 'nvim-telescope/telescope.nvim'
  
  -- Python development
  use 'mfussenegger/nvim-dap-python'
  use 'psf/black'
  
  -- TypeScript/JavaScript
  use 'jose-elias-alvarez/typescript.nvim'
  
  -- Docker & YAML
  use 'cuducos/yaml.nvim'
end)

-- LSP configurations
local lspconfig = require('lspconfig')
lspconfig.pyright.setup{}
lspconfig.tsserver.setup{}
lspconfig.dockerls.setup{}
lspconfig.yamlls.setup{}
```

### Vim Configuration

```vim
" .vimrc for SutazAI development
set nocompatible
filetype plugin indent on
syntax enable

" Python settings
au FileType python setlocal
    \ tabstop=4
    \ softtabstop=4
    \ shiftwidth=4
    \ expandtab
    \ autoindent
    \ fileformat=unix

" TypeScript/JavaScript settings
au FileType javascript,typescript,typescriptreact setlocal
    \ tabstop=2
    \ softtabstop=2
    \ shiftwidth=2
    \ expandtab

" Plugins (using vim-plug)
call plug#begin()
Plug 'dense-analysis/ale'           " Linting
Plug 'psf/black', { 'branch': 'stable' }  " Python formatting
Plug 'prettier/vim-prettier'        " JS/TS formatting
Plug 'tpope/vim-fugitive'          " Git integration
Plug 'preservim/nerdtree'          " File explorer
call plug#end()
```

## Terminal Tools

### Zsh with Oh-My-Zsh

```bash
# Install Zsh and Oh-My-Zsh
sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# .zshrc configuration
plugins=(
  git
  docker
  docker-compose
  python
  pip
  virtualenv
  kubectl
  npm
  node
  postgres
  redis-cli
)

# Custom aliases for SutazAI
alias sai='cd /opt/sutazaiapp'
alias sai-up='cd /opt/sutazaiapp && ./deploy.sh'
alias sai-down='cd /opt/sutazaiapp && docker compose down'
alias sai-logs='cd /opt/sutazaiapp && ./scripts/monitoring/live_logs.sh live'
alias sai-test='cd /opt/sutazaiapp/backend && ./venv/bin/pytest'
alias sai-shell='docker exec -it sutazai-backend bash'
```

### Tmux Configuration

```bash
# .tmux.conf
# Enable mouse support
set -g mouse on

# Better prefix key
unbind C-b
set -g prefix C-a

# Split panes
bind | split-window -h -c "#{pane_current_path}"
bind - split-window -v -c "#{pane_current_path}"

# SutazAI workspace
bind-key S source-file ~/.tmux/sutazai.session

# ~/.tmux/sutazai.session
new-session -s sutazai -n backend -c /opt/sutazaiapp/backend
send-keys 'source venv/bin/activate' C-m
split-window -h -c /opt/sutazaiapp/frontend
send-keys 'source venv/bin/activate' C-m
new-window -n docker -c /opt/sutazaiapp
send-keys 'docker compose ps' C-m
new-window -n logs -c /opt/sutazaiapp
send-keys './scripts/monitoring/live_logs.sh live' C-m
```

## Database Management Tools

### DBeaver Community

```bash
# Installation
wget https://dbeaver.io/files/dbeaver-ce_latest_amd64.deb
sudo dpkg -i dbeaver-ce_latest_amd64.deb

# Connection profiles
# PostgreSQL
Host: localhost
Port: 10000
Database: jarvis_ai
User: jarvis
Password: sutazai_secure_2024

# Neo4j
URL: bolt://localhost:10003
User: neo4j
Password: sutazai_secure_2024
```

### pgAdmin 4

```bash
# Docker installation
docker run -d \
  --name pgadmin \
  --network sutazai-network \
  -e PGADMIN_DEFAULT_EMAIL=admin@sutazai.com \
  -e PGADMIN_DEFAULT_PASSWORD=admin \
  -p 5050:80 \
  dpage/pgadmin4

# Access at http://localhost:5050
```

### Redis Commander

```bash
# NPM installation
npm install -g redis-commander

# Run with configuration
redis-commander \
  --redis-host localhost \
  --redis-port 10001 \
  --redis-password sutazai_secure_2024 \
  --port 8081

# Access at http://localhost:8081
```

## API Development Tools

### Postman Configuration

```json
{
  "name": "SutazAI API",
  "environments": [
    {
      "name": "Local",
      "variables": {
        "base_url": "http://localhost:10200",
        "ws_url": "ws://localhost:10200",
        "auth_token": "{{auth_token}}"
      }
    }
  ],
  "auth": {
    "type": "bearer",
    "bearer": {
      "token": "{{auth_token}}"
    }
  },
  "pre-request": {
    "script": "pm.request.headers.add({key: 'X-Request-ID', value: pm.variables.replaceIn('{{$guid}}')});"
  }
}
```

### Insomnia Configuration

```yaml
# insomnia.workspace.yaml
_type: workspace
name: SutazAI API
environments:
  - name: Local
    data:
      base_url: http://localhost:10200
      ws_url: ws://localhost:10200
      auth_token: "{{ _.auth_token }}"
resources:
  - name: Auth
    method: POST
    url: "{{ _.base_url }}/api/v1/auth/login"
    body:
      mimeType: application/json
      text: |
        {
          "email": "admin@sutazai.com",
          "password": "admin123"
        }
```

### Bruno API Client

```javascript
// bruno.collection.bru
meta {
  name: SutazAI API
  type: collection
}

vars {
  base_url: http://localhost:10200
  auth_token: 
}

auth {
  mode: bearer
  bearer {
    token: {{auth_token}}
  }
}

headers {
  X-Request-ID: {{$guid}}
  Content-Type: application/json
}
```

## Container Management

### Portainer CE

```bash
# Deploy Portainer
docker volume create portainer_data
docker run -d \
  --name portainer \
  --restart=always \
  -p 9000:9000 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v portainer_data:/data \
  portainer/portainer-ce

# Access at http://localhost:9000
```

### Lazydocker

```bash
# Installation
curl https://raw.githubusercontent.com/jesseduffield/lazydocker/master/scripts/install_update_linux.sh | bash

# Configuration (~/.config/lazydocker/config.yml)
gui:
  scrollHeight: 2
  theme:
    selectedLineBgColor:
      - reverse
    selectedRangeBgColor:
      - reverse
logs:
  timestamps: false
  since: '60m'
```

### Dive (Docker Image Analysis)

```bash
# Installation
wget https://github.com/wagoodman/dive/releases/latest/download/dive_0.11.0_linux_amd64.deb
sudo dpkg -i dive_0.11.0_linux_amd64.deb

# Analyze image
dive sutazai-backend:latest
```

## Monitoring and Debugging

### K9s (Kubernetes TUI)

```bash
# Installation
curl -sS https://webinstall.dev/k9s | bash

# Configuration (~/.k9s/config.yml)
k9s:
  refreshRate: 2
  enableMouse: true
  headless: false
  logoless: false
  crumbsless: false
  noIcons: false
```

### Grafana & Prometheus Setup

```yaml
# docker-compose-monitoring.yml
version: '3.8'
services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    networks:
      - sutazai-network

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    networks:
      - sutazai-network
```

### Debug Tools

```bash
# Python debugger (pdb++)
pip install pdbpp

# Node.js debugger
npm install -g node-inspect

# HTTP debugging proxy
docker run -d --name mitmproxy \
  -p 8080:8080 \
  -p 8081:8081 \
  mitmproxy/mitmproxy mitmweb --web-host 0.0.0.0
```

## Version Control

### Git Configuration

```bash
# Global git config
git config --global user.name "Your Name"
git config --global user.email "you@sutazai.com"
git config --global core.editor "vim"
git config --global merge.tool "vimdiff"
git config --global pull.rebase true
git config --global init.defaultBranch main

# Git aliases
git config --global alias.st status
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.cm commit
git config --global alias.lg "log --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit"
```

### GitHub CLI

```bash
# Installation
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo gpg --dearmor -o /usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update && sudo apt install gh

# Authentication
gh auth login

# Useful commands
gh repo clone sutazai/sutazaiapp
gh pr create --title "Feature: Add new API endpoint"
gh issue list --assignee @me
```

## AI/ML Development Tools

### Jupyter Lab

```bash
# Installation
pip install jupyterlab ipywidgets

# Configuration (~/.jupyter/jupyter_lab_config.py)
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.root_dir = '/opt/sutazaiapp/notebooks'
```

### MLflow

```bash
# Installation and setup
pip install mlflow

# Start MLflow server
mlflow server \
  --backend-store-uri postgresql://jarvis:sutazai_secure_2024@localhost:10000/mlflow \
  --default-artifact-root s3://sutazai-mlflow \
  --host 0.0.0.0 \
  --port 5000
```

### TensorBoard

```bash
# Installation
pip install tensorboard

# Run TensorBoard
tensorboard --logdir=/opt/sutazaiapp/logs/tensorboard --host 0.0.0.0 --port 6006
```

## IDE Launch Scripts

### all_tools.sh

```bash
#!/bin/bash
# scripts/ide/launch_all_tools.sh

# Start databases
docker compose -f docker-compose-core.yml up -d

# Wait for services
sleep 10

# Start API documentation
docker run -d --name swagger-ui \
  -e SWAGGER_JSON=/app/openapi.json \
  -v /opt/sutazaiapp/backend/openapi.json:/app/openapi.json \
  -p 8082:8080 \
  swaggerapi/swagger-ui

# Start database tools
redis-commander &
pgadmin4 &

# Start monitoring
docker compose -f docker-compose-monitoring.yml up -d

# Launch IDE
code /opt/sutazaiapp

echo "All development tools started!"
echo "VS Code: Opening..."
echo "Swagger UI: http://localhost:8082"
echo "Redis Commander: http://localhost:8081"
echo "pgAdmin: http://localhost:5050"
echo "Grafana: http://localhost:3000"
echo "Prometheus: http://localhost:9090"
```

## Tool Integration Matrix

| Tool Category | Primary | Alternative | Integration |
|---------------|---------|-------------|-------------|
| IDE | VS Code | PyCharm | Git, Docker, Python |
| API Testing | Postman | Insomnia | Import OpenAPI spec |
| Database GUI | DBeaver | pgAdmin | All databases |
| Container UI | Portainer | Lazydocker | Docker management |
| Monitoring | Grafana | Prometheus | Metrics collection |
| Terminal | Zsh+Tmux | Bash+Screen | Custom aliases |
| Debugging | VS Code | pdb++ | Breakpoints |
| Version Control | Git | GitHub CLI | PR management |

## Related Documentation

- [Dependencies Guide](./dependencies.md)
- [Troubleshooting Guide](./troubleshooting.md)
- [Coding Standards](../development/coding_standards.md)
- [Git Workflow](../development/git_workflow.md)

## Support

- **IDE Setup Help**: ide-support@sutazai.com
- **Tool Issues**: Create ticket in #tools-support
- **License Questions**: Contact legal@sutazai.com