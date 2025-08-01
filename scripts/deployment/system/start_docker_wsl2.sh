#!/bin/bash
# Start Docker in WSL2 without systemd issues

echo "🐋 Starting Docker for WSL2..."

# Kill any existing Docker processes
sudo pkill -f dockerd 2>/dev/null || true
sudo pkill -f containerd 2>/dev/null || true
sudo rm -f /var/run/docker.pid /var/run/docker.sock 2>/dev/null || true
sleep 2

# Create minimal daemon.json
sudo tee /etc/docker/daemon.json > /dev/null << 'EOF'
{
  "storage-driver": "overlay2",
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "dns": ["8.8.8.8", "1.1.1.1"]
}
EOF

# Start dockerd with minimal options
echo "Starting dockerd..."
sudo dockerd --iptables=false >/tmp/dockerd.log 2>&1 &

# Wait for Docker to start
echo -n "Waiting for Docker to start..."
for i in {1..30}; do
    if docker version >/dev/null 2>&1; then
        echo " ✅"
        echo "Docker is running!"
        docker version
        exit 0
    fi
    echo -n "."
    sleep 1
done

echo " ❌"
echo "Failed to start Docker. Check /tmp/dockerd.log for errors."
tail -20 /tmp/dockerd.log
exit 1