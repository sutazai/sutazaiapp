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
bootstrap    = true
client_addr  = "0.0.0.0"

ports = {
  http = 8500
}

enable_script_checks = true
