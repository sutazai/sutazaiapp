datacenter = "dc1"
data_dir  = "/consul/data"
log_level = "INFO"

addresses = {
  http = "0.0.0.0"
}

ui_config = {
  enabled = true
}

bind_addr    = "0.0.0.0"
server       = true
bootstrap_expect = 1
client_addr  = "0.0.0.0"

# Prevent trying to join non-existent nodes
retry_join = []
retry_max = 0

# Disable rejoin after leave
rejoin_after_leave = false

ports = {
  http = 8500
}

enable_script_checks = true

# Performance tuning for single-node
performance = {
  raft_multiplier = 1
}
