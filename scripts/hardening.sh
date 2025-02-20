configure_kernel_params() {
    echo "kernel.kptr_restrict=2" >> /etc/sysctl.d/99-sutazai.conf
    echo "kernel.yama.ptrace_scope=1" >> /etc/sysctl.d/99-sutazai.conf
    sysctl -p /etc/sysctl.d/99-sutazai.conf
}

harden_containers() {
    containerd config default > /etc/containerd/config.toml
    sed -i 's/SystemdCgroup = false/SystemdCgroup = true/' /etc/containerd/config.toml
    systemctl restart containerd
} 