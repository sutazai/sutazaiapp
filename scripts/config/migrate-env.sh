#!/bin/bash
# ============================================================================
# Environment Configuration Migration Script
# Purpose: Migrate from multiple .env files to consolidated .env.master
# Created: 2025-08-15
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "============================================================================"
echo "SutazAI Environment Configuration Migration"
echo "============================================================================"

# Backup existing env files
backup_env_files() {
    echo "Creating backup of existing .env files..."
    BACKUP_DIR="$PROJECT_ROOT/backups/env_backup_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    for file in "$PROJECT_ROOT"/.env* "$PROJECT_ROOT"/config/environments/*.env; do
        if [ -f "$file" ]; then
            cp "$file" "$BACKUP_DIR/" 2>/dev/null || true
            echo "  Backed up: $(basename "$file")"
        fi
    done
    
    echo "Backups created in: $BACKUP_DIR"
}

# Create symlinks for backward compatibility
create_symlinks() {
    echo ""
    echo "Creating symlinks for backward compatibility..."
    
    cd "$PROJECT_ROOT"
    
    # Main .env symlink
    if [ -f ".env" ] && [ ! -L ".env" ]; then
        mv .env .env.old
    fi
    ln -sf .env.master .env
    echo "  Created symlink: .env -> .env.master"
    
    # .env.secure symlink
    if [ -f ".env.secure" ] && [ ! -L ".env.secure" ]; then
        mv .env.secure .env.secure.old
    fi
    ln -sf .env.master .env.secure
    echo "  Created symlink: .env.secure -> .env.master"
}

# Validate master env file
validate_master() {
    echo ""
    echo "Validating .env.master..."
    
    if [ ! -f "$PROJECT_ROOT/.env.master" ]; then
        echo "ERROR: .env.master not found!"
        exit 1
    fi
    
    # Check for required variables
    required_vars=(
        "SUTAZAI_ENV"
        "POSTGRES_USER"
        "REDIS_HOST"
        "BACKEND_PORT"
        "FRONTEND_PORT"
    )
    
    for var in "${required_vars[@]}"; do
        if ! grep -q "^$var=" "$PROJECT_ROOT/.env.master"; then
            echo "WARNING: Required variable $var not found in .env.master"
        fi
    done
    
    echo "  Validation complete"
}

# Generate secrets template
generate_secrets_template() {
    echo ""
    echo "Generating secrets template..."
    
    cat > "$PROJECT_ROOT/.env.secrets.template" << 'EOF'
# ============================================================================
# SutazAI Secrets Template
# Copy this file to .env.secrets and fill in the values
# NEVER commit .env.secrets to version control
# ============================================================================

# Database Passwords
POSTGRES_PASSWORD=
REDIS_PASSWORD=
NEO4J_PASSWORD=
RABBITMQ_PASSWORD=

# Security Keys
SECRET_KEY=
JWT_SECRET=
SECURITY_SALT=
ENCRYPTION_KEY=
SESSION_SECRET=
COOKIE_SECRET=
API_KEY=
WEBHOOK_SECRET=

# Monitoring Passwords
GRAFANA_PASSWORD=

# Generate secure passwords:
# openssl rand -base64 32
EOF
    
    echo "  Created: .env.secrets.template"
    echo "  Copy to .env.secrets and fill in secure values"
}

# Main execution
main() {
    echo ""
    echo "Starting migration process..."
    echo ""
    
    # Step 1: Backup
    backup_env_files
    
    # Step 2: Validate
    validate_master
    
    # Step 3: Create symlinks
    create_symlinks
    
    # Step 4: Generate secrets template
    generate_secrets_template
    
    echo ""
    echo "============================================================================"
    echo "Migration complete!"
    echo ""
    echo "Next steps:"
    echo "1. Copy .env.secrets.template to .env.secrets"
    echo "2. Fill in all secret values in .env.secrets"
    echo "3. Source both files in your deployment:"
    echo "   source .env.master"
    echo "   source .env.secrets"
    echo ""
    echo "Old env files have been backed up and symlinks created for compatibility"
    echo "============================================================================"
}

# Run main function
main "$@"