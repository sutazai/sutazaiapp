configure_service_mesh() {
  linkerd install | kubectl apply -f -
  linkerd viz install | kubectl apply -f -
  annotate_services "config.linkerd.io/inject=enabled"
}

configure_ai_network() {
    # Create dedicated network namespace
    ip netns add ai-net
    ip link add veth-ai type veth peer name veth-host
    ip link set veth-ai netns ai-net
    
    # Configure DNS and limits
    echo "nameserver 1.1.1.1" > /etc/netns/ai-net/resolv.conf
    iptables -A OUTPUT -m owner --uid-owner ai-user -j ACCEPT
} 