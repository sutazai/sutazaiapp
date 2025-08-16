#!/bin/bash
# Hardware optimization rollback script
# Generated automatically - use with caution

sudo sysctl -w vm.swappiness=60
sudo sysctl -w vm.dirty_ratio=20
sudo sysctl -w vm.dirty_background_ratio=10
sudo sysctl -w vm.overcommit_memory=0
sudo sysctl -w vm.overcommit_ratio=50
sudo sysctl -w vm.vfs_cache_pressure=100
sudo sysctl -w net.core.somaxconn=4096
sudo sysctl -w net.ipv4.tcp_fin_timeout=60
sudo sysctl -w net.ipv4.tcp_tw_reuse=2
sudo sysctl -w net.ipv4.tcp_keepalive_time=7200
sudo sysctl -w net.ipv4.tcp_keepalive_probes=9
sudo sysctl -w net.ipv4.tcp_keepalive_intvl=75
sudo sysctl -w net.core.rmem_max=212992
sudo sysctl -w net.core.wmem_max=212992
sudo sysctl -w net.ipv4.tcp_rmem=4096	131072	6291456
sudo sysctl -w net.ipv4.tcp_wmem=4096	16384	4194304
sudo sysctl -w vm.dirty_expire_centisecs=3000
sudo sysctl -w vm.dirty_writeback_centisecs=500

echo 'Rollback completed. Consider restarting Docker daemon.'
