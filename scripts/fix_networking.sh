#!/bin/bash

# SutazAI Network Fix Script - Comprehensive WSL2 Docker Networking Solution
# This script fixes all DNS and networking issues for WSL2 Docker containers

set -e

echo "🔧 SutazAI Network Fix - Comprehensive WSL2 Docker Solution"
echo "════════════════════════════════════════════════════════════"

# 1. Fix WSL2 DNS Configuration
echo "📋 Step 1: Fixing WSL2 DNS Configuration..."
cat > /etc/wsl.conf << EOF
[network]
generateResolvConf = false

[boot]
systemd = true

[interop]
enabled = true
appendWindowsPath = true
EOF

# 2. Set reliable DNS servers
echo "📋 Step 2: Configuring reliable DNS servers..."
cat > /etc/resolv.conf << EOF
nameserver 8.8.8.8
nameserver 8.8.4.4
nameserver 1.1.1.1
EOF

# 3. Optimize Docker daemon configuration for WSL2
echo "📋 Step 3: Optimizing Docker daemon for WSL2..."
cat > /etc/docker/daemon.json << EOF
{
    "log-level": "warn",
    "storage-driver": "overlay2",
    "exec-opts": ["native.cgroupdriver=systemd"],
    "live-restore": true,
    "max-concurrent-downloads": 10,
    "max-concurrent-uploads": 10,
    "dns": ["8.8.8.8", "8.8.4.4", "1.1.1.1"],
    "dns-search": [],
    "dns-opts": ["ndots:0"],
    "mtu": 1500,
    "bip": "172.17.0.1/16",
    "default-address-pools": [
        {
            "base": "172.80.0.0/12",
            "size": 24
        }
    ],
    "default-ulimits": {
        "memlock": {
            "Hard": -1,
            "Name": "memlock",
            "Soft": -1
        },
        "nofile": {
            "Hard": 65536,
            "Name": "nofile", 
            "Soft": 65536
        }
    },
    "experimental": false,
    "features": {
        "buildkit": true
    },
    "iptables": true,
    "userland-proxy": false,
    "ip-forward": true,
    "ip-masq": true
}
EOF

# 4. Restart Docker with new configuration
echo "📋 Step 4: Restarting Docker with optimized configuration..."
systemctl restart docker
sleep 5

# 5. Test Docker networking
echo "📋 Step 5: Testing Docker networking..."
if docker run --rm alpine:latest ping -c 2 8.8.8.8 > /dev/null 2>&1; then
    echo "✅ Docker container external connectivity: WORKING"
else
    echo "❌ Docker container external connectivity: FAILED"
fi

# 6. Test Docker DNS
if docker run --rm alpine:latest nslookup google.com > /dev/null 2>&1; then
    echo "✅ Docker container DNS resolution: WORKING"
else
    echo "❌ Docker container DNS resolution: FAILED"
fi

echo "✅ Network configuration completed!"
echo "🔄 Ready for clean Docker Compose deployment"