# Consul Configuration for SutazAI Service Discovery
datacenter = "sutazai-dc"
data_dir = "/consul/data"
log_level = "INFO"
node_name = "sutazai-consul-node"

# Server configuration
server = true
bootstrap_expect = 1

# Network configuration
bind_addr = "0.0.0.0"
client_addr = "0.0.0.0"

# UI configuration
ui_config {
  enabled = true
}

# Performance settings
performance {
  raft_multiplier = 1
}

# Logging
enable_syslog = false
log_rotate_duration = "24h"
log_rotate_bytes = 0
log_rotate_max_files = 0

# Ports
ports {
  grpc = 8502
  http = 8500
  https = -1
  serf_lan = 8301
  serf_wan = 8302
  server = 8300
}

# Connect configuration
connect {
  enabled = true
}

# Health checks
check_update_interval = "5m"