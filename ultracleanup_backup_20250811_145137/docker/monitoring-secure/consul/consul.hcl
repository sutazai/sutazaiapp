# Secure Consul Configuration
# Optimized for non-root user operation

datacenter = "sutazai-dc1"
data_dir = "/consul/data"
log_level = "INFO"
node_name = "sutazai-consul"
server = true
bootstrap_expect = 1

# UI Configuration
ui_config {
  enabled = true
}

# Client Configuration
client_addr = "0.0.0.0"

# Bind addresses
bind_addr = "0.0.0.0"

# Disable script checks for security
enable_script_checks = false
disable_remote_exec = true

# ACL Configuration (disabled for local development)
acl = {
  enabled = false
  default_policy = "allow"
}

# Service mesh configuration
connect {
  enabled = true
}

# Performance tuning
performance {
  raft_multiplier = 1
}

# Telemetry
telemetry {
  prometheus_retention_time = "60s"
  disable_hostname = true
}

# DNS Configuration
dns_config {
  enable_truncate = true
  only_passing = true
}

# Limits
limits {
  http_max_conns_per_client = 100
  rpc_max_conns_per_client = 100
}