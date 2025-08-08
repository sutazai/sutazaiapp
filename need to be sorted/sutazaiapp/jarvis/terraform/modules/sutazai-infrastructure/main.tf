# Terraform Infrastructure as Code for Sutazai 69-Agent System
# Implements resource-constrained deployment with proper limits and monitoring

terraform {
  required_version = ">= 1.0"
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
  }
}

# Variables
variable "environment" {
  description = "Deployment environment (development, staging, production)"
  type        = string
  default     = "production"
}

variable "project_root" {
  description = "Project root directory"
  type        = string
  default     = "/opt/sutazaiapp"
}

variable "max_cpu_cores" {
  description = "Maximum CPU cores available for agents"
  type        = number
  default     = 10.0
}

variable "max_memory_gb" {
  description = "Maximum memory in GB available for agents"
  type        = number
  default     = 25
}

variable "enable_monitoring" {
  description = "Enable monitoring stack deployment"
  type        = bool
  default     = true
}

variable "enable_security_scanning" {
  description = "Enable security scanning for containers"
  type        = bool
  default     = true
}

# Local values for resource calculations
locals {
  # Environment-specific resource multipliers
  resource_multipliers = {
    development = 0.5
    staging     = 0.8
    production  = 1.0
  }
  
  resource_multiplier = local.resource_multipliers[var.environment]
  
  # Adjusted resource limits
  adjusted_max_cpu    = var.max_cpu_cores * local.resource_multiplier
  adjusted_max_memory = var.max_memory_gb * local.resource_multiplier
  
  # Tier configurations
  tier_configs = {
    critical = {
      priority        = "high"
      max_agents      = 15
      cpu_per_agent   = 2.0
      memory_per_agent = 4
      storage_per_agent = 10
    }
    performance = {
      priority        = "medium"
      max_agents      = 25
      cpu_per_agent   = 1.0
      memory_per_agent = 2
      storage_per_agent = 5
    }
    specialized = {
      priority        = "low"
      max_agents      = 29
      cpu_per_agent   = 0.5
      memory_per_agent = 1
      storage_per_agent = 2
    }
  }
}

# Docker provider configuration
provider "docker" {
  host = "unix:///var/run/docker.sock"
}

# Network resources
resource "docker_network" "sutazai_network" {
  name   = "sutazai-network"
  driver = "bridge"
  
  ipam_config {
    subnet  = "172.20.0.0/16"
    gateway = "172.20.0.1"
  }
  
  options = {
    "com.docker.network.bridge.enable_icc"           = "true"
    "com.docker.network.bridge.enable_ip_masquerade" = "true"
    "com.docker.network.driver.mtu"                  = "1500"
  }
  
  labels {
    label = "sutazai.component"
    value = "network"
  }
}

# Secure internal network for databases
resource "docker_network" "sutazai_internal" {
  name     = "sutazai-internal"
  driver   = "bridge"
  internal = true
  
  ipam_config {
    subnet = "172.21.0.0/24"
  }
  
  labels {
    label = "sutazai.component"
    value = "database-network"
  }
}

# Volumes for persistent data
resource "docker_volume" "postgres_data" {
  name = "sutazai-postgres-data"
  
  labels {
    label = "sutazai.component"
    value = "database"
  }
}

resource "docker_volume" "redis_data" {
  name = "sutazai-redis-data"
  
  labels {
    label = "sutazai.component"
    value = "cache"
  }
}

resource "docker_volume" "neo4j_data" {
  name = "sutazai-neo4j-data"
  
  labels {
    label = "sutazai.component"
    value = "graph-database"
  }
}

resource "docker_volume" "agent_workspaces" {
  name = "sutazai-agent-workspaces"
  
  labels {
    label = "sutazai.component"
    value = "agent-storage"
  }
}

resource "docker_volume" "monitoring_data" {
  name = "sutazai-monitoring-data"
  
  labels {
    label = "sutazai.component"
    value = "monitoring"
  }
}

# PostgreSQL Database
resource "docker_container" "postgres" {
  name  = "sutazai-postgres"
  image = "postgres:15-alpine"
  
  # Resource constraints
  memory = 4096 # 4GB
  cpu_set = "0-1" # 2 CPU cores
  
  # Environment variables
  env = [
    "POSTGRES_DB=sutazai",
    "POSTGRES_USER=sutazai",
    "POSTGRES_PASSWORD_FILE=/run/secrets/postgres_password",
    "POSTGRES_INITDB_ARGS=--auth-host=scram-sha-256 --auth-local=scram-sha-256"
  ]
  
  # Health check
  healthcheck {
    test     = ["CMD-SHELL", "pg_isready -U sutazai"]
    interval = "10s"
    timeout  = "5s"
    retries  = 5
  }
  
  # Networks
  networks_advanced {
    name = docker_network.sutazai_network.name
  }
  networks_advanced {
    name = docker_network.sutazai_internal.name
  }
  
  # Volumes
  volumes {
    volume_name    = docker_volume.postgres_data.name
    container_path = "/var/lib/postgresql/data"
  }
  
  # Mount secrets
  volumes {
    host_path      = "${var.project_root}/secrets/postgres_password.txt"
    container_path = "/run/secrets/postgres_password"
    read_only      = true
  }
  
  # Security options
  user = "postgres"
  
  # Labels
  labels {
    label = "sutazai.component"
    value = "database"
  }
  labels {
    label = "sutazai.tier"
    value = "infrastructure"
  }
  
  restart = "unless-stopped"
}

# Redis Cache
resource "docker_container" "redis" {
  name  = "sutazai-redis"
  image = "redis:7-alpine"
  
  # Resource constraints
  memory = 2048 # 2GB
  cpu_set = "2" # 1 CPU core
  
  # Command with authentication
  command = [
    "redis-server",
    "--requirepass", file("${var.project_root}/secrets/redis_password.txt"),
    "--maxmemory", "1.5gb",
    "--maxmemory-policy", "allkeys-lru",
    "--save", "900", "1",
    "--save", "300", "10",
    "--save", "60", "10000"
  ]
  
  # Health check
  healthcheck {
    test     = ["CMD", "redis-cli", "--no-auth-warning", "-a", file("${var.project_root}/secrets/redis_password.txt"), "ping"]
    interval = "10s"
    timeout  = "3s"
    retries  = 3
  }
  
  # Networks
  networks_advanced {
    name = docker_network.sutazai_network.name
  }
  networks_advanced {
    name = docker_network.sutazai_internal.name
  }
  
  # Volumes
  volumes {
    volume_name    = docker_volume.redis_data.name
    container_path = "/data"
  }
  
  # Labels
  labels {
    label = "sutazai.component"
    value = "cache"
  }
  labels {
    label = "sutazai.tier"
    value = "infrastructure"
  }
  
  restart = "unless-stopped"
}

# Neo4j Graph Database
resource "docker_container" "neo4j" {
  name  = "sutazai-neo4j"
  image = "neo4j:5-community"
  
  # Resource constraints
  memory = 3072 # 3GB
  cpu_set = "3-4" # 2 CPU cores
  
  # Environment variables
  env = [
    "NEO4J_AUTH=neo4j/${file("${var.project_root}/secrets/neo4j_password.txt")}",
    "NEO4J_server_memory_heap_initial__size=1G",
    "NEO4J_server_memory_heap_max__size=2G",
    "NEO4J_server_memory_pagecache_size=512M",
    "NEO4J_server_jvm_additional=-XX:+UseG1GC",
    "NEO4J_server_default__listen__address=0.0.0.0"
  ]
  
  # Health check
  healthcheck {
    test     = ["CMD", "cypher-shell", "-u", "neo4j", "-p", file("${var.project_root}/secrets/neo4j_password.txt"), "RETURN 1"]
    interval = "30s"
    timeout  = "10s"
    retries  = 5
  }
  
  # Networks
  networks_advanced {
    name = docker_network.sutazai_network.name
  }
  networks_advanced {
    name = docker_network.sutazai_internal.name
  }
  
  # Volumes
  volumes {
    volume_name    = docker_volume.neo4j_data.name
    container_path = "/data"
  }
  
  # Ports (only for development)
  dynamic "ports" {
    for_each = var.environment == "development" ? [1] : []
    content {
      internal = 7474
      external = 7474
    }
  }
  
  # Labels
  labels {
    label = "sutazai.component"
    value = "graph-database"
  }
  labels {
    label = "sutazai.tier"
    value = "infrastructure"
  }
  
  restart = "unless-stopped"
}

# Service Discovery - Consul
resource "docker_container" "consul" {
  name  = "sutazai-consul"
  image = "consul:1.16"
  
  # Resource constraints
  memory = 1024 # 1GB
  cpu_set = "5" # 1 CPU core
  
  # Command
  command = [
    "consul", "agent",
    "-dev",
    "-client", "0.0.0.0",
    "-datacenter", "sutazai-dc1",
    "-node", "consul-server",
    "-log-level", "INFO"
  ]
  
  # Health check
  healthcheck {
    test     = ["CMD", "consul", "members"]
    interval = "10s"
    timeout  = "5s"
    retries  = 3
  }
  
  # Networks
  networks_advanced {
    name = docker_network.sutazai_network.name
  }
  
  # Ports
  ports {
    internal = 8500
    external = 8500
  }
  
  # Labels
  labels {
    label = "sutazai.component"
    value = "service-discovery"
  }
  labels {
    label = "sutazai.tier"
    value = "infrastructure"
  }
  
  restart = "unless-stopped"
}

# API Gateway - Kong
resource "docker_container" "kong" {
  name  = "sutazai-kong"
  image = "kong:3.4-alpine"
  
  # Resource constraints
  memory = 1024 # 1GB
  cpu_set = "6" # 1 CPU core
  
  # Environment variables
  env = [
    "KONG_DATABASE=off",
    "KONG_DECLARATIVE_CONFIG=/kong/declarative/kong.yml",
    "KONG_PROXY_ACCESS_LOG=/dev/stdout",
    "KONG_ADMIN_ACCESS_LOG=/dev/stdout",
    "KONG_PROXY_ERROR_LOG=/dev/stderr",
    "KONG_ADMIN_ERROR_LOG=/dev/stderr",
    "KONG_ADMIN_LISTEN=0.0.0.0:8001"
  ]
  
  # Health check
  healthcheck {
    test     = ["CMD", "kong", "health"]
    interval = "10s"
    timeout  = "5s"
    retries  = 5
  }
  
  # Networks
  networks_advanced {
    name = docker_network.sutazai_network.name
  }
  
  # Ports
  ports {
    internal = 8000
    external = 8000
  }
  ports {
    internal = 8001
    external = 8001
  }
  
  # Mount Kong configuration
  volumes {
    host_path      = "${var.project_root}/configs/kong.yml"
    container_path = "/kong/declarative/kong.yml"
    read_only      = true
  }
  
  # Labels
  labels {
    label = "sutazai.component"
    value = "api-gateway"
  }
  labels {
    label = "sutazai.tier"
    value = "infrastructure"
  }
  
  restart = "unless-stopped"
  
  depends_on = [docker_container.consul]
}

# Message Queue - RabbitMQ
resource "docker_container" "rabbitmq" {
  name  = "sutazai-rabbitmq"
  image = "rabbitmq:3.12-management-alpine"
  
  # Resource constraints
  memory = 1024 # 1GB
  cpu_set = "7" # 1 CPU core
  
  # Environment variables
  env = [
    "RABBITMQ_DEFAULT_USER=sutazai",
    "RABBITMQ_DEFAULT_PASS=${file("${var.project_root}/secrets/redis_password.txt")}",
    "RABBITMQ_VM_MEMORY_HIGH_WATERMARK=0.8"
  ]
  
  # Health check
  healthcheck {
    test     = ["CMD", "rabbitmq-diagnostics", "ping"]
    interval = "30s"
    timeout  = "10s"
    retries  = 5
  }
  
  # Networks
  networks_advanced {
    name = docker_network.sutazai_network.name
  }
  
  # Ports (management interface only in development)
  dynamic "ports" {
    for_each = var.environment == "development" ? [1] : []
    content {
      internal = 15672
      external = 15672
    }
  }
  
  # Labels
  labels {
    label = "sutazai.component"
    value = "message-queue"
  }
  labels {
    label = "sutazai.tier"
    value = "infrastructure"
  }
  
  restart = "unless-stopped"
}

# Monitoring Stack - Prometheus
resource "docker_container" "prometheus" {
  count = var.enable_monitoring ? 1 : 0
  
  name  = "sutazai-prometheus"
  image = "prom/prometheus:v2.45.0"
  
  # Resource constraints
  memory = 2048 # 2GB
  cpu_set = "8" # 1 CPU core
  
  # Command
  command = [
    "--config.file=/etc/prometheus/prometheus.yml",
    "--storage.tsdb.path=/prometheus",
    "--web.console.libraries=/etc/prometheus/console_libraries",
    "--web.console.templates=/etc/prometheus/consoles",
    "--storage.tsdb.retention.time=30d",
    "--storage.tsdb.retention.size=15GB",
    "--web.enable-lifecycle"
  ]
  
  # Networks
  networks_advanced {
    name = docker_network.sutazai_network.name
  }
  
  # Ports
  ports {
    internal = 9090
    external = 9090
  }
  
  # Volumes
  volumes {
    volume_name    = docker_volume.monitoring_data.name
    container_path = "/prometheus"
  }
  
  volumes {
    host_path      = "${var.project_root}/monitoring/prometheus.yml"
    container_path = "/etc/prometheus/prometheus.yml"
    read_only      = true
  }
  
  # Labels
  labels {
    label = "sutazai.component"
    value = "monitoring"
  }
  labels {
    label = "sutazai.tier"
    value = "infrastructure"
  }
  
  restart = "unless-stopped"
}

# Monitoring Stack - Grafana
resource "docker_container" "grafana" {
  count = var.enable_monitoring ? 1 : 0
  
  name  = "sutazai-grafana"
  image = "grafana/grafana:10.0.0"
  
  # Resource constraints
  memory = 1024 # 1GB
  cpu_set = "9" # 1 CPU core
  
  # Environment variables
  env = [
    "GF_SECURITY_ADMIN_PASSWORD=${file("${var.project_root}/secrets/grafana_password.txt")}",
    "GF_USERS_ALLOW_SIGN_UP=false",
    "GF_SECURITY_SECRET_KEY=${file("${var.project_root}/secrets/jwt_secret.txt")}",
    "GF_INSTALL_PLUGINS=grafana-piechart-panel"
  ]
  
  # Networks
  networks_advanced {
    name = docker_network.sutazai_network.name
  }
  
  # Ports
  ports {
    internal = 3000
    external = 3000
  }
  
  # Volumes
  volumes {
    host_path      = "${var.project_root}/monitoring/grafana-datasources.yml"
    container_path = "/etc/grafana/provisioning/datasources/datasources.yml"
    read_only      = true
  }
  
  # Labels
  labels {
    label = "sutazai.component"
    value = "monitoring"
  }
  labels {
    label = "sutazai.tier"
    value = "infrastructure"
  }
  
  user    = "grafana"
  restart = "unless-stopped"
  
  depends_on = [docker_container.prometheus]
}

# Resource monitoring container
resource "docker_container" "resource_monitor" {
  name  = "sutazai-resource-monitor"
  image = "prom/node-exporter:v1.6.0"
  
  # Resource constraints
  memory = 128 # 128MB
  cpu_set = "10" # Shared CPU
  
  # Command
  command = [
    "--path.rootfs=/host",
    "--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)"
  ]
  
  # Networks
  networks_advanced {
    name = docker_network.sutazai_network.name
  }
  
  # Volumes for system monitoring
  volumes {
    host_path      = "/"
    container_path = "/host"
    read_only      = true
  }
  
  volumes {
    host_path      = "/proc"
    container_path = "/host/proc"
    read_only      = true
  }
  
  volumes {
    host_path      = "/sys"
    container_path = "/host/sys"
    read_only      = true
  }
  
  # Privileged mode for system access
  privileged = true
  
  # Labels
  labels {
    label = "sutazai.component"
    value = "monitoring"
  }
  labels {
    label = "sutazai.tier"
    value = "infrastructure"
  }
  
  restart = "unless-stopped"
}

# Outputs
output "infrastructure_status" {
  value = {
    postgres_name   = docker_container.postgres.name
    redis_name      = docker_container.redis.name
    neo4j_name      = docker_container.neo4j.name
    consul_name     = docker_container.consul.name
    kong_name       = docker_container.kong.name
    rabbitmq_name   = docker_container.rabbitmq.name
    network_name    = docker_network.sutazai_network.name
    monitoring_enabled = var.enable_monitoring
  }
}

output "resource_allocation" {
  value = {
    total_cpu_allocated    = local.adjusted_max_cpu
    total_memory_allocated = local.adjusted_max_memory
    environment           = var.environment
    resource_multiplier   = local.resource_multiplier
  }
}

output "service_endpoints" {
  value = {
    postgres_url  = "postgresql://sutazai@postgres:5432/sutazai"
    redis_url     = "redis://redis:6379/0"
    neo4j_url     = "bolt://neo4j:7687"
    consul_url    = "http://consul:8500"
    kong_admin    = "http://kong:8001"
    kong_proxy    = "http://kong:8000"
    rabbitmq_url  = "amqp://sutazai@rabbitmq:5672"
    prometheus_url = var.enable_monitoring ? "http://prometheus:9090" : null
    grafana_url   = var.enable_monitoring ? "http://grafana:3000" : null
  }
}