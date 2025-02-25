#!/bin/bash
set -eo pipefail

# Permanent logo display
echo -e "\033[1;35m"
cat << "EOF"
  _________         __                      _____   .__ 
 /   _____/ __ __ _/  |_ _____   ________  /  _  \  |__|
 \_____  \ |  |  \\   __\\__  \  \___   / /  /_\  \ |  |
 /        \|  |  / |  |   / __ \_ /    / /    |    \|  |
/_______  /|____/  |__|  (____  //_____ \\____|__  /|__|
        \/                    \/       \/        \/      
EOF
echo -e "\033[0m"

# Configuration
BASE_DIR="/root/sutazai/v1"
REQUIRED_PYTHON="3.9"

verify_selinux() {
    if command -v sestatus >/dev/null; then
        if [ "$(sestatus | grep 'Current mode' | awk '{print $3}')" != "permissive" ]; then
            echo "❌ SELinux is not in permissive mode during deployment"
            exit 1
        fi
    fi
}

    
    # Verify SELinux state
    if command -v sestatus >/dev/null; then
        local current_mode=$(sestatus | grep 'Current mode' | awk '{print $3}')
        if [ "$current_mode" != "permissive" ]; then
            echo "❌ SELinux should be in permissive mode during deployment"
            exit 1
        fi
    fi
    
    # Verify directory ownership
    if [ "$(stat -c %U /root/sutazai/v1)" != "root" ]; then
        echo "❌ Incorrect directory ownership"
        exit 1
    fi
    
}

verify_selinux_context() {
    echo "🔍 Checking SELinux context..."
    
    if command -v matchpathcon >/dev/null; then
        context=$(matchpathcon -V "/root/sutazai/v1" | grep "verified")
        if [ $? -ne 0 ]; then
            echo "❌ Invalid SELinux context:"
            ls -dZ "/root/sutazai/v1"
            exit 1
        fi
    fi
    
    echo "✅ SELinux context verified"
}

verify_deployment() {
    verify_selinux_context
    echo "=== SYSTEM VERIFICATION ==="
    verify_selinux
    
    # Check Python version
    py_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if [ "$py_version" != "3.9" ]; then
        echo "❌ Python 3.9 required, found $py_version"
        exit 1
    fi

    # Check directory structure
    declare -a dirs=(
        "/root/sutazai/v1/agents/architect"
        "/root/sutazai/v1/services"
    )
    
    for dir in "${dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            echo "❌ Missing directory: $dir"
            exit 1
        fi
    done

    # Check file permissions
    declare -A perms=(
        ["/root/sutazai/v1/deploy_sutazai.sh"]="755"
    )
    
    for file in "${!perms[@]}"; do
        if [ $(stat -c "%a" "$file") -ne "${perms[$file]}" ]; then
            echo "❌ Incorrect permissions for $file (should be ${perms[$file]})"
            exit 1
        fi
    done
    
    echo "✅ System verification passed"
}

verify_permissions() {
    echo "Verifying directory permissions..."
    
    # Check directory permissions
    find "$BASE_DIR" -type d -not -perm 755 && {
        echo "❌ Found directories without 755 permissions"
        return 1
    }
    
    # Check script permissions
    find "$BASE_DIR/scripts" -name "*.sh" -not -perm 755 && {
        echo "❌ Found scripts without execute permissions"
        return 1
    }
    
    echo "✅ Directory permissions verified"
    return 0
}

# Add permission verification to main checks
verify_deployment
verify_permissions

# 1. Directory Structure Verification
echo "=== DIRECTORY STRUCTURE VERIFICATION ==="
required_dirs=(
    "backend"
    "frontend"
    "scripts"
    "agents"
    "data"
    "config"
    "models"
    "monitoring"
)
for dir in "${required_dirs[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "❌ Missing directory: $dir"
        exit 1
    else
        echo "✅ Verified directory: $dir"
    fi
done

# 2. File Existence Check
echo -e "\n=== ESSENTIAL FILE VERIFICATION ==="
critical_files=(
    "deploy_sutazai.sh"
    "backend/main.py"
    "frontend/app.py"
    "config/settings.py"
    "scripts/database.sh"
)
for file in "${critical_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Missing critical file: $file"
        exit 1
    else
        echo "✅ Verified file: $file"
    fi
done

# 3. File Permissions Check
echo -e "\n=== FILE PERMISSIONS VERIFICATION ==="
check_perms() {
    perm=$(stat -c "%a" "$1")
    if [ "$perm" != "$2" ]; then
        echo "❌ Incorrect permissions for $1 (has $perm, should be $2)"
        exit 1
    else
        echo "✅ Correct permissions for $1"
    fi
}

check_perms "deploy_sutazai.sh" "750"
check_perms "scripts/*.sh" "750"

# 4. Docker Credentials Validation
echo -e "\n=== DOCKER CREDENTIALS VERIFICATION ==="
temp_creds=$(mktemp)
if ! gpg --decrypt --cipher-algo AES256 --digest-algo SHA512 \
    --passphrase "1988" --batch --no-tty \
    echo "❌ GPG Decryption Failed - Critical System Error"
    echo "Technical Details:"
    cat gpg-errors.log
    echo -e "\nPossible Solutions:"
    echo "1. Run hardware diagnostics: memtester 4M && smartctl -a /dev/sda"
    echo "2. Verify system clock: timedatectl status"
    exit 1
fi

# Validate decrypted file structure
if ! file "$temp_creds" | grep -q "ASCII text"; then
    echo "❌ Decrypted File Corruption Detected"
    echo "Hex Dump:"
    hexdump -C "$temp_creds" | head -n 5
    exit 1
fi

required_vars=(
    "DOCKER_REGISTRY"
    "DOCKER_USER"
    "DOCKER_PASS"
)
for var in "${required_vars[@]}"; do
    if ! grep -q "^$var=" "$temp_creds"; then
        echo "❌ Missing $var in credentials"
        echo "File content:"
        cat "$temp_creds"
        exit 1
    fi
done
echo "✅ Docker credentials format valid"
rm -f "$temp_creds" gpg-errors.log

# Add checksum file check
    echo "❌ Missing credentials checksum file"
    echo "Run: make generate-checksum"
    exit 1
fi

# Update GPG version check
required_gpg_version="2.4.4"
current_gpg_version=$(gpg --version | head -n1 | awk '{print $3}')
if [ "$(printf '%s\n' "$required_gpg_version" "$current_gpg_version" | sort -V | head -n1)" != "$required_gpg_version" ]; then
    echo "❌ GPG version too old (need $required_gpg_version+)"
    exit 1
fi

# 5. Dependency Verification
echo -e "\n=== DEPENDENCY VERIFICATION ==="
check_dependency() {
    if ! command -v "$1" &> /dev/null; then
        echo "❌ Missing dependency: $1"
        exit 1
    else
        echo "✅ Verified dependency: $1"
    fi
}

check_dependency "docker"
check_dependency "gpg"
check_dependency "python3"
check_dependency "pip"

# 6. Configuration Validation
echo -e "\n=== CONFIGURATION VALIDATION ==="
python3 - <<EOF
from config import settings

required_settings = [
    'DATABASE_URL', 'JWT_SECRET', 'MODEL_PATH', 
    'GPU_ENABLED', 'LOG_LEVEL', 'API_PORT'
]

for setting in required_settings:
    if not hasattr(settings, setting):
        print(f"❌ Missing configuration: {setting}")
        exit(1)
print("✅ All required configurations present")
EOF

# 7. Service Port Availability
echo -e "\n=== PORT AVAILABILITY CHECK ==="
check_port() {
    if lsof -i :"$1" > /dev/null; then
        echo "❌ Port $1 is already in use"
        exit 1
    else
        echo "✅ Port $1 available"
    fi
}

check_port 8000  # API Port
check_port 8501  # Frontend Port
check_port 5432  # Database Port

echo -e "\n=== SECURITY AUDIT ==="
if grep -r "password" . | grep -v "docker_creds.gpg"; then
    echo "❌ Clear-text credentials found in codebase!"
    exit 1
else
    echo "✅ No clear-text credentials detected"
fi

# Add to verification checks
echo -e "\n=== GPG ENVIRONMENT VERIFICATION ==="
gpg --version | head -n 1
gpgconf --list-dirs | grep '^homedir:'

echo -e "\n=== CREDENTIALS INTEGRITY CHECK ==="
    echo "❌ Credentials file validation failed"
    exit 1
}

# Add GPG environment check
echo -e "\n=== GPG ENVIRONMENT TEST ==="
test_file=$(mktemp)
echo "test" > "$test_file"
gpg --batch --passphrase "test" -c "$test_file" 2>/dev/null || {
    echo "❌ GPG basic functionality broken"
    exit 1
}
rm -f "$test_file" "${test_file}.gpg"

# Add credentials lifecycle test
echo -e "\n=== CREDENTIALS LIFECYCLE TEST ==="
./scripts/generate_credentials.sh
    echo "❌ Credentials regeneration failed"
    exit 1
fi

echo -e "\n=== SYSTEM VERIFICATION COMPLETE ==="
echo "All critical checks passed successfully!"

# Change from critical error to warning
lscpu | grep -q 'avx512' || log "WARN" "AVX-512 optional for basic operation"

# Update hardware checks
check_hardware() {
    # NVIDIA driver validation
    if lspci | grep -qi 'nvidia'; then
        if ! command -v nvidia-smi &>/dev/null; then
            echo "❌ NVIDIA drivers missing - installing..."
            sudo apt-get install -y nvidia-driver-535 nvidia-container-toolkit
            sudo systemctl restart docker
        fi
        echo "✅ NVIDIA drivers: $(nvidia-smi --query | grep 'Driver Version')"
    fi
} 