#!/bin/bash
# WSL2 Performance Optimization Script

echo "=== WSL2 Performance Optimization ==="

# 1. Move Docker data to WSL2 filesystem
echo "Moving Docker data to WSL2 filesystem..."
if [ ! -d /opt/docker-wsl2 ]; then
    sudo systemctl stop docker
    sudo cp -r /var/lib/docker /opt/docker-wsl2
    sudo rm -rf /var/lib/docker
    sudo ln -s /opt/docker-wsl2 /var/lib/docker
    sudo systemctl start docker
    echo "Docker data moved to WSL2 filesystem"
else
    echo "Docker data already on WSL2 filesystem"
fi

# 2. Configure Docker daemon for WSL2
echo "Configuring Docker daemon..."
sudo tee /etc/docker/daemon.json << DAEMON
{
  "storage-driver": "overlay2",
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "experimental": true,
  "features": {
    "buildkit": true
  },
  "default-ulimits": {
    "nofile": {
      "Name": "nofile",
      "Hard": 64000,
      "Soft": 64000
    }
  }
}
DAEMON

# 3. Optimize system parameters
echo "Optimizing system parameters..."
sudo sysctl -w vm.max_map_count=262144
sudo sysctl -w fs.file-max=65536
sudo sysctl -w net.core.somaxconn=1024
sudo sysctl -w net.ipv4.tcp_syncookies=1

# 4. Make optimizations persistent
echo "Making optimizations persistent..."
sudo tee -a /etc/sysctl.conf << EOF
vm.max_map_count=262144
fs.file-max=65536
net.core.somaxconn=1024
net.ipv4.tcp_syncookies=1
EOF

# 5. Restart Docker
echo "Restarting Docker..."
sudo systemctl restart docker

echo "WSL2 optimization complete!"
echo "Note: For best performance, ensure your project files are in the WSL2 filesystem (not /mnt/c)"