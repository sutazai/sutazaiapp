# Port Registry vs Compose Reconciliation
Generated: 20250819_214428_UTC

## Source of Truth
- /opt/sutazaiapp/IMPORTANT/diagrams/PortRegistry.md

## Declared Ports in PortRegistry.md
  15:- 10000: PostgreSQL database (sutazai-postgres)
  16:- 10001: Redis cache (sutazai-redis)
  17:- 10002: Neo4j HTTP interface (sutazai-neo4j)
  18:- 10003: Neo4j Bolt protocol (sutazai-neo4j)
  19:- 10005: Kong API Gateway proxy (sutazai-kong)
  20:- 10006: Consul service discovery (sutazai-consul)
  21:- 10007: RabbitMQ AMQP (sutazai-rabbitmq)
  22:- 10008: RabbitMQ Management UI (sutazai-rabbitmq)
  23:- 10010: FastAPI backend (sutazai-backend)
  24:- 10011: Streamlit frontend (sutazai-frontend)
  25:- 10015: Kong Admin API (sutazai-kong)
  29:- 10100: ChromaDB vector database (sutazai-chromadb)
  30:- 10101: Qdrant HTTP API (sutazai-qdrant)
  31:- 10102: Qdrant gRPC interface (sutazai-qdrant)
  32:- 10103: FAISS vector service (sutazai-faiss) **[DEFINED BUT NOT RUNNING]**
  33:- 10104: Ollama LLM server (sutazai-ollama) **[RESERVED - CRITICAL]**
  37:- 10200: Prometheus metrics collection (sutazai-prometheus)
  38:- 10201: Grafana dashboards (sutazai-grafana)
  39:- 10202: Loki log aggregation (sutazai-loki)
  40:- 10203: AlertManager notifications (sutazai-alertmanager)
  41:- 10204: Blackbox Exporter (sutazai-blackbox-exporter)
  42:- 10205: Node Exporter system metrics (sutazai-node-exporter)
  43:- 10206: cAdvisor container metrics (sutazai-cadvisor)
  44:- 10207: Postgres Exporter DB metrics (sutazai-postgres-exporter)
  45:- 10208: Redis Exporter cache metrics (sutazai-redis-exporter) **[DEFINED BUT NOT RUNNING]**
  46:- 10210: Jaeger tracing UI (sutazai-jaeger)
  47:- 10211: Jaeger collector (sutazai-jaeger)
  48:- 10212: Jaeger gRPC (sutazai-jaeger)
  49:- 10213: Jaeger Zipkin (sutazai-jaeger)
  50:- 10214: Jaeger OTLP gRPC (sutazai-jaeger)
  51:- 10215: Jaeger OTLP HTTP (sutazai-jaeger)
  56:- 11019: Hardware Resource Optimizer (sutazai-hardware-resource-optimizer) **[DEFINED BUT NOT RUNNING]**
  57:- 11069: Task Assignment Coordinator (sutazai-task-assignment-coordinator) **[DEFINED BUT NOT RUNNING]**
  58:- 11071: Ollama Integration Agent (sutazai-ollama-integration) **[DEFINED BUT NOT RUNNING]**
  59:- 11200: Ultra System Architect (sutazai-ultra-system-architect) **[RUNNING]**
  60:- 11201: Ultra Frontend UI Architect (sutazai-ultra-frontend-ui-architect) **[DEFINED BUT NOT RUNNING]**
  76:- 11090: MCP Consul UI (sutazai-mcp-consul)
  77:- 11091: MCP Network Monitor (sutazai-mcp-monitor)  
  78:- 11099: MCP HAProxy Stats (sutazai-mcp-haproxy)
  79:- 11100: MCP PostgreSQL Service (sutazai-mcp-postgres)
  80:- 11101: MCP Files Service (sutazai-mcp-files)
  81:- 11102: MCP HTTP Service (sutazai-mcp-http)
  82:- 11103: MCP DuckDuckGo Service (sutazai-mcp-ddg)
  83:- 11104: MCP GitHub Service (sutazai-mcp-github)
  84:- 11105: MCP Memory Service (sutazai-mcp-memory)
  119:- 11100: MCP Postgres Service (sutazai-mcp-postgres)
  120:- 11101: MCP Files Service (sutazai-mcp-files)
  121:- 11102: MCP HTTP Service (sutazai-mcp-http)
  122:- 11103: MCP DuckDuckGo Service (sutazai-mcp-ddg)
  123:- 11104: MCP GitHub Service (sutazai-mcp-github)
  124:- 11105: MCP Memory Service (sutazai-mcp-memory)

## Docker-related files (static inventory)
  - /opt/sutazaiapp/backups/deploy_20250813_103632/docker-compose.yml
  - /opt/sutazaiapp/docker/backend/Dockerfile
  - /opt/sutazaiapp/docker/base/unified-base.Dockerfile
  - /opt/sutazaiapp/docker/dind/docker-compose.dind.yml
  - /opt/sutazaiapp/docker/dind/mcp-containers/docker-compose.mcp-services.yml
  - /opt/sutazaiapp/docker/dind/orchestrator/manager/Dockerfile
  - /opt/sutazaiapp/docker/docker-compose.yml
  - /opt/sutazaiapp/docker/faiss/Dockerfile
  - /opt/sutazaiapp/docker/frontend/Dockerfile
  - /opt/sutazaiapp/docker/mcp-services/real-mcp-server/Dockerfile
  - /opt/sutazaiapp/docker/mcp-services/unified-dev/Dockerfile
  - /opt/sutazaiapp/docker/mcp-services/unified-memory/docker-compose.unified-memory.yml
  - /opt/sutazaiapp/docker/monitoring/mcp-monitoring.Dockerfile
  - /opt/sutazaiapp/docker/streamlit.Dockerfile
  - /opt/sutazaiapp/node_modules/getos/Dockerfile
  - /opt/sutazaiapp/node_modules/getos/tests/alpine/3.3/Dockerfile
  - /opt/sutazaiapp/node_modules/getos/tests/debian/7.3/Dockerfile
  - /opt/sutazaiapp/node_modules/getos/tests/debian/7.4/Dockerfile
  - /opt/sutazaiapp/node_modules/getos/tests/debian/7.5/Dockerfile
  - /opt/sutazaiapp/node_modules/getos/tests/debian/7.6/Dockerfile
  - /opt/sutazaiapp/node_modules/getos/tests/fedora/20/Dockerfile
  - /opt/sutazaiapp/node_modules/getos/tests/ubuntu/13.10/Dockerfile
  - /opt/sutazaiapp/node_modules/getos/tests/ubuntu/14.04/Dockerfile
  - /opt/sutazaiapp/node_modules/newman/docker/images/alpine/Dockerfile
  - /opt/sutazaiapp/node_modules/newman/docker/images/ubuntu/Dockerfile

## Compose Exposed Ports (heuristic parse)
### /opt/sutazaiapp/backups/deploy_20250813_103632/docker-compose.yml
    services:
      postgres:
        container_name: sutazai-postgres
        deploy:
          resources:
            limits:
              cpus: '2.0'
              memory: 2G
            reservations:
              cpus: '0.5'
              memory: 512M
        environment:
          POSTGRES_DB: ${POSTGRES_DB:-sutazai}
          POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
          POSTGRES_USER: ${POSTGRES_USER:-sutazai}
        healthcheck:
          interval: 10s
          retries: 5
          start_period: 60s
          test:
          - CMD-SHELL
          - pg_isready -U ${POSTGRES_USER:-sutazai}
          timeout: 5s
        image: sutazai-postgres-secure:latest
        networks:
        - sutazai-network
        ports:
        - 10000:5432
        restart: unless-stopped
        volumes:
        - postgres_data:/var/lib/postgresql/data
        - ./IMPORTANT/init_db.sql:/docker-entrypoint-initdb.d/init.sql:ro
      redis:
        command: redis-server /usr/local/etc/redis/redis.conf
        container_name: sutazai-redis
        deploy:
          resources:
            limits:
              cpus: '1.0'
              memory: 1G
            reservations:
              cpus: '0.25'
              memory: 256M
        healthcheck:
          interval: 10s
          retries: 5
          test:
          - CMD-SHELL
          - redis-cli ping
          timeout: 5s
        image: sutazai-redis-secure:latest
        networks:
        - sutazai-network
        ports:
        - 10001:6379
        restart: unless-stopped
        volumes:
        - redis_data:/data
        - ./config/redis-optimized.conf:/usr/local/etc/redis/redis.conf:ro
      neo4j:
        container_name: sutazai-neo4j
        deploy:
          resources:
            limits:
              cpus: '1.0'
              memory: 1G
            reservations:
              cpus: '0.25'
              memory: 512M
        environment:
          NEO4J_AUTH: neo4j/${NEO4J_PASSWORD}
          NEO4J_server_memory_heap_max__size: 512m
          NEO4J_server_memory_heap_initial__size: 256m
          NEO4J_server_memory_pagecache_size: 256m
          NEO4J_server_jvm_additional: -XX:+UseG1GC -XX:G1HeapRegionSize=4m -XX:+DisableExplicitGC
            -XX:+ExitOnOutOfMemoryError
          NEO4J_initial_dbms_default__database: sutazai
          NEO4J_db_checkpoint_interval_time: 30s
          NEO4J_db_transaction_timeout: 30s
          NEO4J_db_logs_query_enabled: "OFF"
          NEO4J_server_config_strict__validation_enabled: "false"
          NEO4J_db_transaction_bookmark_ready_timeout: 5s
          NEO4J_dbms_cluster_discovery_type: SINGLE
        healthcheck:
          interval: 60s
          retries: 3
          start_period: 45s
          test:
          - CMD-SHELL
          - wget --no-verbose --tries=1 --spider http://localhost:7474/ || exit 1
          timeout: 15s
        image: sutazai-neo4j-secure:latest
        networks:
        - sutazai-network
        ports:
        - 10002:7474
        - 10003:7687
        restart: unless-stopped
        volumes:
        - neo4j_data:/data
      ollama:
        container_name: sutazai-ollama
        deploy:
          resources:
            limits:
              cpus: '4.0'
              memory: 4G
            reservations:
              cpus: '1.0'
              memory: 1G
        environment:
          CLAUDE_RULES_PATH: /app/CLAUDE.md
          ENFORCE_CLAUDE_RULES: 'true'
          OLLAMA_API_KEY: local
          OLLAMA_BASE_URL: http://ollama:11434
          OLLAMA_DEBUG: "false"
          OLLAMA_FLASH_ATTENTION: '0'
          OLLAMA_HOST: 0.0.0.0
          OLLAMA_KEEP_ALIVE: 5m
          OLLAMA_MAX_LOADED_MODELS: '1'
          OLLAMA_MODELS: /home/ollama/.ollama/models
          OLLAMA_NUM_PARALLEL: '1'
          OLLAMA_NUM_THREADS: '8'
          OLLAMA_ORIGINS: '*'
          OLLAMA_RUNNERS_DIR: /tmp
          OLLAMA_TMPDIR: /tmp/ollama
          OLLAMA_MAX_QUEUE: '10'
          OLLAMA_TIMEOUT: 300s
          OLLAMA_REQUEST_TIMEOUT: '300'
          OLLAMA_CONNECTION_POOL: '10'
          OLLAMA_USE_MMAP: 'true'
          OLLAMA_USE_NUMA: 'false'
        healthcheck:
          test:
          - CMD-SHELL
          - ollama list || exit 1
          interval: 30s
          timeout: 10s
          retries: 3
          start_period: 60s
        image: sutazai-ollama-secure:latest
        networks:
        - sutazai-network
        ports:
        - 10104:11434
        restart: unless-stopped
        sysctls:
        - net.core.somaxconn=65535
        ulimits:
          nofile:
            hard: 65536
            soft: 65536
        volumes:
        - ollama_data:/home/ollama/.ollama
        - models_data:/models
        - /opt/sutazaiapp/CLAUDE.md:/app/CLAUDE.md:ro
        - /opt/sutazaiapp/config/ollama.yaml:/app/config/ollama.yaml:ro
        logging:
          driver: json-file
          options:
            max-size: 100m
            max-file: '5'
      chromadb:
        container_name: sutazai-chromadb
        deploy:
          resources:
            limits:
              cpus: '1'
              memory: 1G
            reservations:
              cpus: '0.25'
              memory: 256M
        environment:
        - CHROMA_SERVER_AUTH_PROVIDER=chromadb.auth.token.TokenAuthenticationServerProvider
        - CHROMA_SERVER_AUTH_CREDENTIALS=${CHROMADB_API_KEY}
        - CHROMA_SERVER_HOST=0.0.0.0
        - CHROMA_SERVER_HTTP_PORT=8000
        - CHROMA_SERVER_CORS_ALLOW_ORIGINS=["http://localhost:8501", "http://backend:8000"]
        healthcheck:
          interval: 60s
          retries: 5
          start_period: 120s
          test:
          - CMD
          - sh
          - -c
          - curl -f http://localhost:8000/api/v1/heartbeat || exit 1
          timeout: 30s
        image: sutazai-chromadb-secure:latest
        networks:
        - sutazai-network
        ports:
        - 10100:8000
        restart: unless-stopped
        volumes:
        - chromadb_data:/chroma/chroma
      qdrant:
        container_name: sutazai-qdrant
        deploy:
          resources:
            limits:
              cpus: '2'
              memory: 2G
            reservations:
              cpus: '0.5'
              memory: 512M
        environment:
          QDRANT__LOG_LEVEL: INFO
          QDRANT__SERVICE__GRPC_PORT: 6334
          QDRANT__SERVICE__HTTP_PORT: 6333
        healthcheck:
          interval: 60s
          retries: 5
          test:
          - CMD
          - sh
          - -c
          - echo 'use IO::Socket::INET; my $$s = IO::Socket::INET->new(PeerAddr => q{localhost:6333},
            Proto => q{tcp}, Timeout => 2); if ($$s) { print $$s qq{GET / HTTP/1.0\r\n\r\n};
            while (<$$s>) { if (/200 OK/) { exit 0; } } } exit 1;' | perl
          timeout: 30s
        image: sutazai-qdrant-secure:latest
        networks:
        - sutazai-network
        ports:
        - 10101:6333
        - 10102:6334
        restart: unless-stopped
        volumes:
        - qdrant_data:/qdrant/storage
      faiss:
        image: sutazaiapp-faiss:latest
        container_name: sutazai-faiss
        deploy:
          resources:
            limits:
              cpus: '1'
              memory: 512M
            reservations:
              cpus: '0.25'
              memory: 128M
        environment:
          SUTAZAI_ENV: ${SUTAZAI_ENV:-production}
          TZ: ${TZ:-UTC}
        healthcheck:
          interval: 60s
          retries: 5
          test:
          - CMD
          - python
          - -c
          - import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()
          timeout: 30s
        networks:
        - sutazai-network
        ports:
        - 10103:8000
        restart: unless-stopped
        volumes:
        - faiss_data:/data
      kong:
        image: kong:3.5
        container_name: sutazai-kong
        deploy:
          resources:
            limits:
              cpus: '0.5'
              memory: 512M
            reservations:
              cpus: '0.1'
              memory: 128M
        environment:
          KONG_DATABASE: 'off'
          KONG_DECLARATIVE_CONFIG: /etc/kong/kong-optimized.yml
          KONG_PROXY_LISTEN: 0.0.0.0:8000
          KONG_ADMIN_LISTEN: 0.0.0.0:8001
          KONG_LOG_LEVEL: error
          KONG_NGINX_WORKER_PROCESSES: '1'
          KONG_NGINX_WORKER_CONNECTIONS: '512'
          KONG_MEM_CACHE_SIZE: '64m'
          KONG_SSL_CIPHER_SUITE: 'intermediate'
          KONG_PROXY_ACCESS_LOG: 'off'
          KONG_ADMIN_ACCESS_LOG: 'off'
          KONG_PROXY_ERROR_LOG: '/dev/stderr'
          KONG_ADMIN_ERROR_LOG: '/dev/stderr'
          KONG_CLIENT_BODY_BUFFER_SIZE: '8k'
          KONG_CLIENT_MAX_BODY_SIZE: '8m'
          KONG_NGINX_HTTP_CLIENT_BODY_BUFFER_SIZE: '8k'
          KONG_NGINX_HTTP_CLIENT_MAX_BODY_SIZE: '8m'
        healthcheck:
          test:
          - CMD-SHELL
          - ps aux | grep '[k]ong' > /dev/null || exit 1
          interval: 30s
          timeout: 5s
          retries: 3
          start_period: 10s
        networks:
        - sutazai-network
        ports:
        - 10005:8000
        - 10015:8001
        restart: unless-stopped
        volumes:
        - ./config/kong/kong-optimized.yml:/etc/kong/kong-optimized.yml:ro
      consul:
        image: sutazai-consul-secure:latest
        container_name: sutazai-consul
        # ULTRA-FIX: Remove user: root - image now runs as consul user
        environment:
          CONSUL_ALLOW_PRIVILEGED_PORTS: 'true'
          CONSUL_DISABLE_PERM: 'true'
        command:
        - agent
        - -server
        - -bootstrap-expect=1
        - -ui
        - -client=0.0.0.0
        - -data-dir=/consul/data
        deploy:
          resources:
            limits:
              cpus: '0.5'
              memory: 512M
            reservations:
              cpus: '0.25'
              memory: 256M
        healthcheck:
          test:
          - CMD
          - wget
          - --no-verbose
          - --tries=1
          - --spider
          - http://localhost:8500/v1/status/leader
          interval: 30s
          timeout: 5s
          retries: 10
          start_period: 10s
        networks:
        - sutazai-network
        ports:
        - 10006:8500
        restart: unless-stopped
        volumes:
        - ./config/consul/consul.hcl:/consul/config/consul.hcl:ro
        - consul_data:/consul/data
      rabbitmq:
        image: sutazai-rabbitmq-secure:latest
        container_name: sutazai-rabbitmq
        deploy:
          resources:
            limits:
              cpus: '1.0'
              memory: 1G
            reservations:
              cpus: '0.5'
              memory: 512M
        environment:
          RABBITMQ_DEFAULT_USER: ${RABBITMQ_DEFAULT_USER:-sutazai}
          RABBITMQ_DEFAULT_PASS: ${RABBITMQ_DEFAULT_PASS}
        healthcheck:
          test:
          - CMD
          - rabbitmq-diagnostics
          - check_running
          interval: 30s
          timeout: 5s
          retries: 10
          start_period: 20s
        networks:
        - sutazai-network
        ports:
        - 10007:5672
        - 10008:15672
        restart: unless-stopped
        volumes:
        - rabbitmq_data:/var/lib/rabbitmq
      backend:
        image: sutazaiapp-backend:latest
        command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
        container_name: sutazai-backend
        depends_on:
          chromadb:
            condition: service_healthy
          neo4j:
            condition: service_started
          ollama:
            condition: service_healthy
          postgres:
            condition: service_healthy
          qdrant:
            condition: service_healthy
        deploy:
          resources:
            limits:
              cpus: '4'
              memory: 4G
            reservations:
              cpus: '1'
              memory: 1G
        environment:
          API_V1_STR: /api/v1
          BACKEND_CORS_ORIGINS: '["http://localhost:10011", "http://172.31.77.193:10011"]'
          CHROMADB_HOST: chromadb
          CHROMADB_PORT: 8000
          CHROMADB_URL: http://chromadb:8000
          DATABASE_URL: postgresql+asyncpg://${POSTGRES_USER:-sutazai}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB:-sutazai}
          FAISS_INDEX_PATH: /data/faiss
          GRAFANA_PASSWORD: ${GRAFANA_PASSWORD}
          JWT_SECRET: ${JWT_SECRET}
          JWT_SECRET_KEY: ${JWT_SECRET_KEY}
          NEO4J_HOST: neo4j
          NEO4J_PASSWORD: ${NEO4J_PASSWORD}
          NEO4J_PORT: 7687
          NEO4J_URI: bolt://neo4j:7687
          NEO4J_USER: neo4j
          OLLAMA_API_KEY: local
          OLLAMA_BASE_URL: http://ollama:11434
          OLLAMA_HOST: 0.0.0.0
          OLLAMA_ORIGINS: '*'
          POSTGRES_DB: ${POSTGRES_DB:-sutazai}
          POSTGRES_HOST: postgres
          POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
          POSTGRES_PORT: 5432
          POSTGRES_USER: ${POSTGRES_USER:-sutazai}
          QDRANT_HOST: qdrant
          QDRANT_PORT: 6333
          QDRANT_URL: http://qdrant:6333
          REDIS_HOST: sutazai-redis
          # REDIS_PASSWORD not needed - Redis runs without authentication
          REDIS_PORT: 6379
          REDIS_URL: redis://sutazai-redis:6379/0
          SECRET_KEY: ${SECRET_KEY}
          SUTAZAI_ENV: ${SUTAZAI_ENV:-production}
          TZ: ${TZ:-UTC}
        healthcheck:
          interval: 60s
          retries: 5
          start_period: 120s
          test:
          - CMD
          - python3
          - -c
          - import socket; s=socket.socket(); s.settimeout(5); exit(0 if s.connect_ex(("localhost",
            8000))==0 else 1)
          timeout: 30s
        networks:
        - sutazai-network
        ports:
        - 10010:8000
        restart: unless-stopped
        volumes:
        - ./backend:/app
        - ./data:/data
        - ./logs:/logs
        - agent_workspaces:/app/agent_workspaces
      frontend:
        image: sutazaiapp-frontend:latest
        command: streamlit run app.py --server.port 8501 --server.address 0.0.0.0
        container_name: sutazai-frontend
        depends_on:
          backend:
            condition: service_healthy
        deploy:
          resources:
            limits:
              cpus: '2'
              memory: 2G
            reservations:
              cpus: '0.5'
              memory: 512M
        environment:
          BACKEND_URL: http://backend:8000
          STREAMLIT_SERVER_ADDRESS: 0.0.0.0
          STREAMLIT_SERVER_PORT: 8501
          SUTAZAI_ENV: ${SUTAZAI_ENV:-production}
          TZ: ${TZ:-UTC}
        healthcheck:
          interval: 60s
          retries: 5
          start_period: 120s
          test:
          - CMD
          - python3
          - -c
          - import socket; s=socket.socket(); s.settimeout(5); exit(0 if s.connect_ex(("localhost",
            8501))==0 else 1)
          timeout: 30s
        networks:
        - sutazai-network
        ports:
        - 10011:8501
        restart: unless-stopped
        volumes:
        - ./frontend:/app
        - ./data:/data
      prometheus:
        command:
        - --config.file=/etc/prometheus/prometheus.yml
        - --storage.tsdb.path=/prometheus
        - --web.console.libraries=/usr/share/prometheus/console_libraries
        - --web.console.templates=/usr/share/prometheus/consoles
        - --web.enable-lifecycle
        - --storage.tsdb.retention.time=7d
        - --web.enable-admin-api
        - --storage.tsdb.max-block-duration=2h
        - --storage.tsdb.min-block-duration=2h
        - --storage.tsdb.retention.size=1GB
        container_name: sutazai-prometheus
        deploy:
          resources:
            limits:
              cpus: '1'
              memory: 1G
            reservations:
              cpus: '0.25'
              memory: 256M
        healthcheck:
          interval: 60s
          retries: 5
          start_period: 90s
          test:
          - CMD
          - wget
          - --no-verbose
          - --tries=1
          - --spider
          - http://localhost:9090/-/healthy
          timeout: 30s
        image: prom/prometheus:latest
        networks:
        - sutazai-network
        ports:
        - 10200:9090
        restart: unless-stopped
        volumes:
        - ./monitoring/prometheus:/etc/prometheus
        - prometheus_data:/prometheus
      grafana:
        container_name: sutazai-grafana
        depends_on:
        - prometheus
        - loki
        deploy:
          resources:
            limits:
              cpus: '1'
              memory: 512M
            reservations:
              cpus: '0.25'
              memory: 128M
        environment:
        - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
        - GF_USERS_ALLOW_SIGN_UP=false
        - GF_INSTALL_PLUGINS=
        - GF_ANALYTICS_REPORTING_ENABLED=false
        - GF_ANALYTICS_CHECK_FOR_UPDATES=false
        - GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH=/var/lib/grafana/dashboards/system-overview.json
        healthcheck:
          interval: 60s
          retries: 5
          start_period: 60s
          test:
          - CMD-SHELL
          - wget --no-verbose --tries=1 --spider http://localhost:3000/api/health || exit
            1
          timeout: 30s
        image: grafana/grafana:latest
        networks:
        - sutazai-network
        ports:
        - 10201:3000
        restart: unless-stopped
        volumes:
        - grafana_data:/var/lib/grafana
        - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
        - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards
      loki:
        command: -config.file=/etc/loki/local-config.yaml
        container_name: sutazai-loki
        deploy:
          resources:
            limits:
              cpus: '0.5'
              memory: 512M
            reservations:
              cpus: '0.1'
              memory: 128M
        healthcheck:
          interval: 60s
          retries: 5
          start_period: 60s
          test:
          - CMD-SHELL
          - wget --no-verbose --tries=1 --spider http://localhost:3100/ready || exit 1
          timeout: 30s
        image: grafana/loki:2.9.0
        networks:
        - sutazai-network
        ports:
        - 10202:3100
        restart: unless-stopped
        volumes:
        - loki_data:/loki
        - ./monitoring/loki/config.yml:/etc/loki/local-config.yaml
      alertmanager:
        command:
        - --config.file=/etc/alertmanager/config.yml
        - --storage.path=/alertmanager
        - --web.external-url=http://localhost:9093
        container_name: sutazai-alertmanager
        deploy:
          resources:
            limits:
              cpus: '0.5'
              memory: 512M
            reservations:
              cpus: '0.1'
              memory: 128M
        environment:
        - SLACK_WEBHOOK_URL=${SLACK_WEBHOOK_URL:-}
        - SLACK_AI_WEBHOOK_URL=${SLACK_AI_WEBHOOK_URL:-}
        - SLACK_SECURITY_WEBHOOK_URL=${SLACK_SECURITY_WEBHOOK_URL:-}
        - PAGERDUTY_SERVICE_KEY=${PAGERDUTY_SERVICE_KEY:-}
        healthcheck:
          test:
          - CMD
          - wget
          - --no-verbose
          - --tries=1
          - --spider
          - http://localhost:9093/-/ready
          interval: 60s
          timeout: 30s
          retries: 5
        image: prom/alertmanager:latest
        networks:
        - sutazai-network
        ports:
        - 10203:9093
        restart: unless-stopped
        volumes:
        - ./monitoring/alertmanager:/etc/alertmanager
        - alertmanager_data:/alertmanager
      blackbox-exporter:
        container_name: sutazai-blackbox-exporter
        user: nobody
        deploy:
          resources:
            limits:
              cpus: '0.25'
              memory: 256M
            reservations:
              cpus: '0.1'
              memory: 64M
        healthcheck:
          interval: 60s
          retries: 5
          test:
          - CMD
          - wget
          - --no-verbose
          - --tries=1
          - --spider
          - http://localhost:9115/
          timeout: 30s
        image: sutazai-blackbox-exporter-secure:latest
        networks:
        - sutazai-network
        ports:
        - 10204:9115
        restart: unless-stopped
        volumes:
        - ./monitoring/blackbox/config.yml:/etc/blackbox_exporter/config.yml
      node-exporter:
        command:
        - --path.procfs=/host/proc
        - --path.sysfs=/host/sys
        - --path.rootfs=/rootfs
        - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$$$|/)'
        container_name: sutazai-node-exporter
        deploy:
          resources:
            limits:
              cpus: '0.25'
              memory: 128M
            reservations:
              cpus: '0.1'
              memory: 32M
        image: prom/node-exporter:latest
        networks:
        - sutazai-network
        ports:
        - 10205:9100
        restart: unless-stopped
        volumes:
        - /proc:/host/proc:ro
        - /sys:/host/sys:ro
        - /:/rootfs:ro
      cadvisor:
        container_name: sutazai-cadvisor
        devices:
        - /dev/kmsg
        image: sutazai-cadvisor-secure:latest
        # ULTRA-FIX: Changed user from 'cadvisor' to 'nobody' to match Dockerfile
        user: nobody
        networks:
        - sutazai-network
        ports:
        - 10206:8080
        privileged: false
        cap_add:
        - SYS_ADMIN
        - SYS_RESOURCE
        - SYS_PTRACE
        - DAC_READ_SEARCH
        security_opt:
        - no-new-privileges:true
        # ULTRA-FIX: Set read_only to true with tmpfs for writable areas
        read_only: true
        restart: unless-stopped
        command:
        - --housekeeping_interval=60s
        - --max_housekeeping_interval=120s
        - --allow_dynamic_housekeeping=true
        - --global_housekeeping_interval=2m
        - --disable_metrics=advtcp,cpu_topology,disk,hugetlb,memory_numa,percpu,referenced_memory,resctrl,tcp,udp,process
        - --store_container_labels=false
        - --whitelisted_container_labels=io.kubernetes.container.name,io.kubernetes.pod.name
        deploy:
          resources:
            limits:
              cpus: '0.5'
              memory: 200M
            reservations:
              cpus: '0.1'
              memory: 50M
        volumes:
        - /:/rootfs:ro
        - /var/run:/var/run:ro
        - /sys:/sys:ro
        - /var/lib/docker/:/var/lib/docker:ro
        - /dev/disk/:/dev/disk:ro
        # ULTRA-FIX: Add tmpfs volumes for writable areas needed by cAdvisor
        tmpfs:
        - /tmp:size=100M,mode=1777
        - /var/cache:size=50M,mode=1777
      postgres-exporter:
        container_name: sutazai-postgres-exporter
        depends_on:
          postgres:
            condition: service_healthy
        deploy:
          resources:
            limits:
              cpus: '0.25'
              memory: 128M
            reservations:
              cpus: '0.1'
              memory: 32M
        environment:
          DATA_SOURCE_NAME: postgresql://${POSTGRES_USER:-sutazai}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB:-sutazai}?sslmode=disable
        image: prometheuscommunity/postgres-exporter:latest
        networks:
        - sutazai-network
        ports:
        - 10207:9187
        restart: unless-stopped
      redis-exporter:
        container_name: sutazai-redis-exporter
        depends_on:
          redis:
            condition: service_healthy
        deploy:
          resources:
            limits:
              cpus: '0.25'
              memory: 128M
            reservations:
              cpus: '0.1'
              memory: 32M
        environment:
          REDIS_ADDR: redis://redis:6379
          # REDIS_PASSWORD not needed - Redis runs without authentication
        image: sutazai-redis-exporter-secure:latest
        networks:
        - sutazai-network
        ports:
        - 10208:9121
        restart: unless-stopped
      ollama-integration:
        image: sutazaiapp-ollama-integration:latest
        container_name: sutazai-ollama-integration
        depends_on:
          ollama:
            condition: service_healthy
          redis:
            condition: service_healthy
        deploy:
          resources:
            limits:
              cpus: '1'
              memory: 256M
            reservations:
              cpus: '0.25'
              memory: 64M
        environment:
          AGENT_TYPE: ollama-integration
          API_ENDPOINT: http://backend:8000
          LOG_LEVEL: INFO
          OLLAMA_BASE_URL: http://ollama:11434
          PORT: 8090
          RABBITMQ_URL: amqp://${RABBITMQ_DEFAULT_USER:-sutazai}:${RABBITMQ_DEFAULT_PASS}@rabbitmq:5672/
          REDIS_URL: redis://sutazai-redis:6379/0
          SUTAZAI_ENV: ${SUTAZAI_ENV:-production}
          TZ: ${TZ:-UTC}
          MAX_RETRIES: 3
          BACKOFF_BASE: 2
          REQUEST_TIMEOUT: 30
          CONNECTION_POOL_SIZE: 10
        healthcheck:
          test:
          - CMD
          - curl
          - -f
          - http://localhost:8090/health
          interval: 30s
          timeout: 3s
          retries: 3
          start_period: 5s
        networks:
        - sutazai-network
        ports:
        - 8090:8090
        restart: unless-stopped
        volumes:
        - ./agents/core:/app/agents/core:ro
      hardware-resource-optimizer:
        image: sutazaiapp-hardware-resource-optimizer:latest
        container_name: sutazai-hardware-resource-optimizer
        depends_on:
          backend:
            condition: service_started
          ollama:
            condition: service_started
          redis:
            condition: service_healthy
        deploy:
          resources:
            limits:
              cpus: '2'
              memory: 1G
            reservations:
              cpus: '0.5'
              memory: 256M
        environment:
          AGENT_TYPE: hardware-resource-optimizer
          API_ENDPOINT: http://backend:8000
          DATABASE_URL: postgresql+asyncpg://${POSTGRES_USER:-sutazai}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB:-sutazai}
          LOG_LEVEL: INFO
          OLLAMA_API_KEY: local
          OLLAMA_BASE_URL: http://ollama:11434
          OLLAMA_HOST: 0.0.0.0
          OLLAMA_MODEL: tinyllama:latest
          OLLAMA_ORIGINS: '*'
          PORT: 8080
          RABBITMQ_URL: amqp://${RABBITMQ_DEFAULT_USER:-sutazai}:${RABBITMQ_DEFAULT_PASS}@rabbitmq:5672/
          REDIS_URL: redis://sutazai-redis:6379/0
          SUTAZAI_ENV: ${SUTAZAI_ENV:-production}
          TZ: ${TZ:-UTC}
        healthcheck:
          interval: 60s
          retries: 5
          start_period: 120s
          test:
          - CMD
          - curl
          - -f
          - http://localhost:8080/health
          timeout: 30s
        networks:
        - sutazai-network
        ports:
        - 11110:8080
        privileged: false
        cap_drop:
        - ALL
        cap_add:
        - SYS_PTRACE
        security_opt:
        - no-new-privileges:true
        - seccomp:unconfined
        restart: unless-stopped
        volumes:
        - ./data:/app/data:rw,noexec
        - ./configs:/app/configs:ro
        - ./logs:/app/logs:rw,noexec
      jarvis-hardware-resource-optimizer:
        image: sutazaiapp-jarvis-hardware-resource-optimizer:latest
        container_name: sutazai-jarvis-hardware-resource-optimizer
        depends_on:
          backend:
            condition: service_started
          rabbitmq:
            condition: service_healthy
          redis:
            condition: service_healthy
        deploy:
          resources:
            limits:
              cpus: '1'
              memory: 256M
            reservations:
              cpus: '0.25'
              memory: 64M
        environment:
          AGENT_TYPE: jarvis-hardware-resource-optimizer
          API_ENDPOINT: http://backend:8000
          LOG_LEVEL: INFO
          PORT: 8080
          RABBITMQ_URL: amqp://${RABBITMQ_DEFAULT_USER:-sutazai}:${RABBITMQ_DEFAULT_PASS}@rabbitmq:5672/
          REDIS_URL: redis://sutazai-redis:6379/0
          SUTAZAI_ENV: ${SUTAZAI_ENV:-production}
          TZ: ${TZ:-UTC}
        healthcheck:
          test:
          - CMD
          - curl
          - -f
          - http://localhost:8080/health
          interval: 60s
          timeout: 30s
          retries: 5
          start_period: 60s
        networks:
        - sutazai-network
        ports:
        - 11104:8080
        privileged: false
        cap_drop:
        - ALL
        cap_add:
        - SYS_PTRACE
        security_opt:
        - no-new-privileges:true
        - seccomp:unconfined
        user: 1001:1001
        restart: unless-stopped
        volumes:
        - ./agents/core:/app/agents/core:ro
        - ./data:/app/data:rw,noexec
        - ./configs:/app/configs:ro
        - ./logs:/app/logs:rw,noexec
      jarvis-automation-agent:
        build:
          context: ./agents/jarvis-automation-agent
          dockerfile: Dockerfile
        container_name: sutazai-jarvis-automation-agent
        depends_on:
          backend:
            condition: service_started
          rabbitmq:
            condition: service_healthy
          redis:
            condition: service_healthy
        deploy:
          resources:
            limits:
              cpus: '2'
              memory: 1G
            reservations:
              cpus: '0.5'
              memory: 256M
        environment:
          AGENT_TYPE: jarvis-automation-agent
          API_ENDPOINT: http://backend:8000
          DATABASE_URL: postgresql+asyncpg://${POSTGRES_USER:-sutazai}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB:-sutazai}
          LOG_LEVEL: INFO
          OLLAMA_BASE_URL: http://ollama:11434
          PORT: 8080
          RABBITMQ_URL: amqp://${RABBITMQ_DEFAULT_USER:-sutazai}:${RABBITMQ_DEFAULT_PASS}@rabbitmq:5672/
          REDIS_URL: redis://sutazai-redis:6379/0
          SUTAZAI_ENV: ${SUTAZAI_ENV:-production}
          TZ: ${TZ:-UTC}
        healthcheck:
          test:
          - CMD
          - curl
          - -f
          - http://localhost:8080/health
          interval: 60s
          timeout: 30s
          retries: 5
          start_period: 60s
        networks:
        - sutazai-network
        ports:
        - 11102:8080
        restart: unless-stopped
        volumes:
        - ./agents/core:/app/agents/core:ro
        - ./data:/app/data
        - ./configs:/app/configs
        - ./logs:/app/logs
        - /tmp:/tmp
        - /opt/sutazaiapp:/opt/sutazaiapp:ro
      ai-agent-orchestrator:
        build:
          context: ./agents/ai_agent_orchestrator
          dockerfile: Dockerfile
        container_name: sutazai-ai-agent-orchestrator
        depends_on:
          rabbitmq:
            condition: service_healthy
          redis:
            condition: service_healthy
        deploy:
          resources:
            limits:
              cpus: '1'
              memory: 256M
            reservations:
              cpus: '0.25'
              memory: 64M
        environment:
          AGENT_TYPE: ai-agent-orchestrator
          API_ENDPOINT: http://backend:8000
          LOG_LEVEL: INFO
          PORT: 8589
          RABBITMQ_URL: amqp://${RABBITMQ_DEFAULT_USER:-sutazai}:${RABBITMQ_DEFAULT_PASS}@rabbitmq:5672/
          REDIS_URL: redis://sutazai-redis:6379/0
          SUTAZAI_ENV: ${SUTAZAI_ENV:-production}
          TZ: ${TZ:-UTC}
        healthcheck:
          test:
          - CMD
          - curl
          - -f
          - http://localhost:8589/health
          interval: 30s
          timeout: 10s
          retries: 5
          start_period: 120s
        networks:
        - sutazai-network
        ports:
        - 8589:8589
        restart: unless-stopped
        volumes:
        - ./agents/core:/app/agents/core:ro
        - ./logs:/app/logs
      task-assignment-coordinator:
        image: sutazaiapp-task-assignment-coordinator:latest
        container_name: sutazai-task-assignment-coordinator
        depends_on:
          rabbitmq:
            condition: service_healthy
          redis:
            condition: service_healthy
        deploy:
          resources:
            limits:
              cpus: '1'
              memory: 256M
            reservations:
              cpus: '0.25'
              memory: 64M
        environment:
          AGENT_TYPE: task-assignment-coordinator
          API_ENDPOINT: http://backend:8000
          LOG_LEVEL: INFO
          PORT: 8551
          RABBITMQ_URL: amqp://${RABBITMQ_DEFAULT_USER:-sutazai}:${RABBITMQ_DEFAULT_PASS}@rabbitmq:5672/
          REDIS_URL: redis://sutazai-redis:6379/0
          SUTAZAI_ENV: ${SUTAZAI_ENV:-production}
          TZ: ${TZ:-UTC}
        healthcheck:
          test:
          - CMD
          - curl
          - -f
          - http://localhost:8551/health
          interval: 30s
          timeout: 10s
          retries: 5
          start_period: 120s
        networks:
        - sutazai-network
        ports:
        - 8551:8551
        restart: unless-stopped
        volumes:
        - ./agents/core:/app/agents/core:ro
        - ./logs:/app/logs
      resource-arbitration-agent:
        image: sutazaiapp-resource-arbitration-agent:latest
        container_name: sutazai-resource-arbitration-agent
        depends_on:
          rabbitmq:
            condition: service_healthy
          redis:
            condition: service_healthy
        deploy:
          resources:
            limits:
              cpus: '2'
              memory: 1G
            reservations:
              cpus: '0.5'
              memory: 256M
        environment:
          AGENT_TYPE: resource-arbitration-agent
          API_ENDPOINT: http://backend:8000
          LOG_LEVEL: INFO
          PORT: 8588
          RABBITMQ_URL: amqp://${RABBITMQ_DEFAULT_USER:-sutazai}:${RABBITMQ_DEFAULT_PASS}@rabbitmq:5672/
          REDIS_URL: redis://sutazai-redis:6379/0
          SUTAZAI_ENV: ${SUTAZAI_ENV:-production}
          TZ: ${TZ:-UTC}
        healthcheck:
          test:
          - CMD
          - curl
          - -f
          - http://localhost:8588/health
          interval: 30s
          timeout: 10s
          retries: 5
          start_period: 120s
        networks:
        - sutazai-network
        ports:
        - 8588:8588
        privileged: false
        cap_drop:
        - ALL
        cap_add:
        - SYS_PTRACE
        security_opt:
        - no-new-privileges:true
        - seccomp:unconfined
        user: 1001:1001
        restart: unless-stopped
        volumes:
        - ./agents/core:/app/agents/core:ro
        - ./data:/app/data:rw,noexec
        - ./logs:/app/logs:rw,noexec
    
      # Distributed Tracing
      jaeger:
        container_name: sutazai-jaeger
        image: sutazai-jaeger-secure:latest
        environment:
          - COLLECTOR_ZIPKIN_HTTP_PORT=9411
          - COLLECTOR_OTLP_ENABLED=true
          - JAEGER_DISABLED=false
          - METRICS_STORAGE_TYPE=prometheus
          - PROMETHEUS_SERVER_URL=http://sutazai-prometheus:9090
          - PROMETHEUS_QUERY_SUPPORT_SPANMETRICS_CONNECTOR=true
          - SPAN_STORAGE_TYPE=memory
          - MEMORY_MAX_TRACES=100000
        ports:
          - "10210:16686"  # Jaeger UI
          - "10211:14268"  # Jaeger Collector HTTP
          - "10212:14250"  # Jaeger Collector gRPC
          - "10213:9411"   # Zipkin Collector
          - "10214:4317"   # OTLP gRPC
          - "10215:4318"   # OTLP HTTP
        networks:
          - sutazai-network
        restart: unless-stopped
        deploy:
          resources:
            limits:
              cpus: '1'
              memory: 1G
            reservations:
              cpus: '0.25'
              memory: 256M
        healthcheck:
          test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:16686/"]
          interval: 30s
          timeout: 10s
          retries: 3
          start_period: 40s
    
      # Log Shipping (Promtail for Loki)
      promtail:
        container_name: sutazai-promtail
        image: sutazai-promtail-secure:latest
        user: promtail
        volumes:
          - /var/log:/var/log:ro
          - /var/lib/docker/containers:/var/lib/docker/containers:ro
          - /var/run/docker.sock:/var/run/docker.sock:ro
          - ./monitoring/promtail/config.yml:/etc/promtail/config.yml:ro
        networks:
          - sutazai-network
        restart: unless-stopped
        deploy:
          resources:
            limits:
              cpus: '0.5'
              memory: 256M
            reservations:
              cpus: '0.1'
              memory: 64M
        privileged: true
        depends_on:
          - loki
        healthcheck:
          test:
          - CMD-SHELL
          - pgrep promtail
          interval: 60s
          timeout: 10s
          retries: 3
          start_period: 30s
    
    32:    - 10000:5432
    59:    - 10001:6379
    100:    - 10002:7474
    101:    - 10003:7687
    149:    - 10104:11434
    197:    - 10100:8000
    230:    - 10101:6333
    231:    - 10102:6334
    261:    - 10103:8000
    305:    - 10005:8000
    306:    - 10015:8001
    347:    - 10006:8500
    378:    - 10007:5672
    379:    - 10008:15672
    455:    - 10010:8000
    497:    - 10011:8501
    539:    - 10200:9090
    577:    - 10201:3000
    606:    - 10202:3100
    645:    - 10203:9093
    676:    - 10204:9115
    699:    - 10205:9100
    715:    - 10206:8080
    772:    - 10207:9187
    794:    - 10208:9121
    839:    - 8090:8090
    889:    - 11110:8080
    943:    - 11104:8080
    1003:    - 11102:8080
    1052:    - 8589:8589
    1095:    - 8551:8551
    1138:    - 8588:8588
    1168:      - "10210:16686"  # Jaeger UI
    1169:      - "10211:14268"  # Jaeger Collector HTTP
    1170:      - "10212:14250"  # Jaeger Collector gRPC
    1171:      - "10213:9411"   # Zipkin Collector
    1172:      - "10214:4317"   # OTLP gRPC
    1173:      - "10215:4318"   # OTLP HTTP

### /opt/sutazaiapp/docker/dind/docker-compose.dind.yml
    services:
      # Docker-in-Docker MCP Orchestrator
      mcp-orchestrator:
        image: docker:25.0.5-dind-alpine3.19
        container_name: sutazai-mcp-orchestrator
        privileged: true
        environment:
          - DOCKER_TLS_CERTDIR=/certs
          - DOCKER_DRIVER=overlay2
          - DOCKER_STORAGE_DRIVER=overlay2
          - MCP_ORCHESTRATOR_PORT=2376
          - MCP_API_PORT=8080
          - MCP_METRICS_PORT=9090
        # Explicitly bind dockerd to both TLS (2376) and plain HTTP (2375) for intra-network access
        command:
          - dockerd
          - -H
          - tcp://0.0.0.0:2376
          - -H
          - tcp://0.0.0.0:2375
        volumes:
          # DinD internal Docker socket and certificates
          - mcp-docker-certs-ca:/certs/ca
          - mcp-docker-certs-client:/certs/client
          - mcp-orchestrator-data:/var/lib/docker
          # Shared Docker socket for manager access
          - mcp-docker-socket:/var/run
          # MCP configurations and manifests
          - ./orchestrator/mcp-manifests:/mcp-manifests:ro
          - ./orchestrator/scripts:/orchestrator-scripts:ro
          - ./orchestrator/configs:/orchestrator-configs:ro
          # Shared volumes for MCP containers
          - mcp-shared-data:/mcp-shared
          - mcp-logs:/var/log/mcp
        ports:
          - "12376:2376"  # Docker daemon API (TLS)
          - "12375:2375"  # Docker daemon API (no TLS, for orchestrator)
          - "18080:8080"  # MCP Orchestrator API
          - "19090:9090"  # MCP Metrics endpoint
        networks:
          - sutazai-dind-internal
          - sutazai-main
        healthcheck:
          test: ["CMD", "docker", "version"]
          interval: 30s
          timeout: 10s
          retries: 3
          start_period: 60s
        restart: unless-stopped
        deploy:
          resources:
            limits:
              cpus: '4.0'
              memory: 4G
            reservations:
              cpus: '1.0'
              memory: 1G
    
      # MCP Management Interface
      mcp-manager:
        build:
          context: ./orchestrator/manager
          dockerfile: Dockerfile
        image: sutazai-mcp-manager:v1.0.0
        container_name: sutazai-mcp-manager
        depends_on:
          mcp-orchestrator:
            condition: service_healthy
        environment:
          - DOCKER_HOST=unix:///var/run/docker.sock
          - MCP_REGISTRY_URL=http://mcp-orchestrator:8080
          - MESH_GATEWAY_URL=http://host.docker.internal:10005
          - CONSUL_URL=http://host.docker.internal:10006
        volumes:
          - ./orchestrator/manager/app:/app:ro
          - mcp-manager-logs:/var/log/manager
          # Shared Docker socket access (read-write for manager operations)
          - mcp-docker-socket:/var/run:rw
        ports:
          - "18081:8081"  # MCP Manager Web UI
        networks:
          - sutazai-dind-internal
          - sutazai-main
        healthcheck:
          test: ["CMD", "curl", "-f", "http://localhost:8081/health"]
          interval: 30s
          timeout: 10s
          retries: 3
          start_period: 30s
        restart: unless-stopped
        deploy:
          resources:
            limits:
              cpus: '1.0'
              memory: 512M
            reservations:
              cpus: '0.25'
              memory: 128M
    
    38:      - "12376:2376"  # Docker daemon API (TLS)
    39:      - "12375:2375"  # Docker daemon API (no TLS, for orchestrator)
    40:      - "18080:8080"  # MCP Orchestrator API
    41:      - "19090:9090"  # MCP Metrics endpoint
    82:      - "18081:8081"  # MCP Manager Web UI

### /opt/sutazaiapp/docker/dind/mcp-containers/docker-compose.mcp-services.yml
    services:
      # Node.js based MCP services
      mcp-claude-flow:
        image: sutazai-mcp-nodejs:latest
        container_name: mcp-claude-flow
        environment:
          - MCP_SERVICE=claude-flow
          - NODE_ENV=production
          - MCP_HOST=0.0.0.0
          - MCP_PORT=3001
        ports:
          - "3001:3001"
        volumes:
          - mcp-claude-flow-data:/var/lib/mcp
          - mcp-logs:/var/log/mcp
        networks:
          - mcp-bridge
        restart: unless-stopped
        healthcheck:
          test: ["CMD", "/opt/mcp/wrappers/claude-flow.sh", "health"]
          interval: 30s
          timeout: 10s
          retries: 3
          start_period: 30s
    
      mcp-ruv-swarm:
        image: sutazai-mcp-nodejs:latest
        container_name: mcp-ruv-swarm
        environment:
          - MCP_SERVICE=ruv-swarm
          - NODE_ENV=production
          - MCP_HOST=0.0.0.0
          - MCP_PORT=3002
        ports:
          - "3002:3002"
        volumes:
          - mcp-ruv-swarm-data:/var/lib/mcp
          - mcp-logs:/var/log/mcp
        networks:
          - mcp-bridge
        restart: unless-stopped
        healthcheck:
          test: ["CMD", "/opt/mcp/wrappers/ruv-swarm.sh", "health"]
          interval: 30s
          timeout: 10s
          retries: 3
          start_period: 30s
    
      mcp-files:
        image: sutazai-mcp-nodejs:latest
        container_name: mcp-files
        environment:
          - MCP_SERVICE=files
          - NODE_ENV=production
          - MCP_HOST=0.0.0.0
          - MCP_PORT=3003
        ports:
          - "3003:3003"
        volumes:
          - mcp-files-data:/var/lib/mcp
          - mcp-logs:/var/log/mcp
          - mcp-shared:/opt/shared:ro
        networks:
          - mcp-bridge
        restart: unless-stopped
        healthcheck:
          test: ["CMD", "/opt/mcp/wrappers/files.sh", "health"]
          interval: 30s
          timeout: 10s
          retries: 3
          start_period: 30s
    
      mcp-context7:
        image: sutazai-mcp-nodejs:latest
        container_name: mcp-context7
        environment:
          - MCP_SERVICE=context7
          - NODE_ENV=production
          - MCP_HOST=0.0.0.0
          - MCP_PORT=3004
        ports:
          - "3004:3004"
        volumes:
          - mcp-context7-data:/var/lib/mcp
          - mcp-logs:/var/log/mcp
        networks:
          - mcp-bridge
        restart: unless-stopped
        healthcheck:
          test: ["CMD", "/opt/mcp/wrappers/context7.sh", "health"]
          interval: 30s
          timeout: 10s
          retries: 3
          start_period: 30s
    
      mcp-ddg:
        image: sutazai-mcp-nodejs:latest
        container_name: mcp-ddg
        environment:
          - MCP_SERVICE=ddg
          - NODE_ENV=production
          - MCP_HOST=0.0.0.0
          - MCP_PORT=3006
        ports:
          - "3006:3006"
        volumes:
          - mcp-ddg-data:/var/lib/mcp
          - mcp-logs:/var/log/mcp
        networks:
          - mcp-bridge
        restart: unless-stopped
        healthcheck:
          test: ["CMD", "/opt/mcp/wrappers/ddg.sh", "health"]
          interval: 30s
          timeout: 10s
          retries: 3
          start_period: 30s
    
      # CONSOLIDATED into mcp-unified-dev above (sequentialthinking)
    
      mcp-nx-mcp:
        image: sutazai-mcp-nodejs:latest
        container_name: mcp-nx-mcp
        environment:
          - MCP_SERVICE=nx-mcp
          - NODE_ENV=production
          - MCP_HOST=0.0.0.0
          - MCP_PORT=3008
        ports:
          - "3008:3008"
        volumes:
          - mcp-nx-mcp-data:/var/lib/mcp
          - mcp-logs:/var/log/mcp
        networks:
          - mcp-bridge
        restart: unless-stopped
        healthcheck:
          test: ["CMD", "/opt/mcp/wrappers/nx-mcp.sh", "health"]
          interval: 30s
          timeout: 10s
          retries: 3
          start_period: 30s
    
      mcp-extended-memory:
        image: sutazai-mcp-nodejs:latest
        container_name: mcp-extended-memory
        environment:
          - MCP_SERVICE=extended-memory
          - NODE_ENV=production
          - MCP_HOST=0.0.0.0
          - MCP_PORT=3009
        ports:
          - "3009:3009"
        volumes:
          - mcp-extended-memory-data:/var/lib/mcp
          - mcp-logs:/var/log/mcp
        networks:
          - mcp-bridge
        restart: unless-stopped
        healthcheck:
          test: ["CMD", "/opt/mcp/wrappers/extended-memory.sh", "health"]
          interval: 30s
          timeout: 10s
          retries: 3
          start_period: 30s
    
      mcp-claude-task-runner:
        image: sutazai-mcp-nodejs:latest
        container_name: mcp-claude-task-runner
        environment:
          - MCP_SERVICE=claude-task-runner
          - NODE_ENV=production
          - MCP_HOST=0.0.0.0
          - MCP_PORT=3010
        ports:
          - "3010:3010"
        volumes:
          - mcp-claude-task-runner-data:/var/lib/mcp
          - mcp-logs:/var/log/mcp
        networks:
          - mcp-bridge
        restart: unless-stopped
        healthcheck:
          test: ["CMD", "/opt/mcp/wrappers/claude-task-runner.sh", "health"]
          interval: 30s
          timeout: 10s
          retries: 3
          start_period: 30s
    
      # Python based MCP services
      mcp-postgres:
        image: sutazai-mcp-python:latest
        container_name: mcp-postgres
        environment:
          - MCP_SERVICE=postgres
          - PYTHONPATH=/opt/mcp
          - MCP_HOST=0.0.0.0
          - MCP_PORT=4001
        ports:
          - "4001:4001"
        volumes:
          - mcp-postgres-data:/var/lib/mcp
          - mcp-logs:/var/log/mcp
        networks:
          - mcp-bridge
        restart: unless-stopped
        healthcheck:
          test: ["CMD", "/opt/mcp/wrappers/postgres.sh", "health"]
          interval: 30s
          timeout: 10s
          retries: 3
          start_period: 30s
    
      mcp-memory-bank-mcp:
        image: sutazai-mcp-python:latest
        container_name: mcp-memory-bank-mcp
        environment:
          - MCP_SERVICE=memory-bank-mcp
          - PYTHONPATH=/opt/mcp
          - MCP_HOST=0.0.0.0
          - MCP_PORT=4002
        ports:
          - "4002:4002"
        volumes:
          - mcp-memory-bank-mcp-data:/var/lib/mcp
          - mcp-logs:/var/log/mcp
        networks:
          - mcp-bridge
        restart: unless-stopped
        healthcheck:
          test: ["CMD", "/opt/mcp/wrappers/memory-bank-mcp.sh", "health"]
          interval: 30s
          timeout: 10s
          retries: 3
          start_period: 30s
    
      mcp-knowledge-graph-mcp:
        image: sutazai-mcp-python:latest
        container_name: mcp-knowledge-graph-mcp
        environment:
          - MCP_SERVICE=knowledge-graph-mcp
          - PYTHONPATH=/opt/mcp
          - MCP_HOST=0.0.0.0
          - MCP_PORT=4003
        ports:
          - "4003:4003"
        volumes:
          - mcp-knowledge-graph-mcp-data:/var/lib/mcp
          - mcp-logs:/var/log/mcp
        networks:
          - mcp-bridge
        restart: unless-stopped
        healthcheck:
          test: ["CMD", "/opt/mcp/wrappers/knowledge-graph-mcp.sh", "health"]
          interval: 30s
          timeout: 10s
          retries: 3
          start_period: 30s
    
      # CONSOLIDATED into mcp-unified-dev above (ultimatecoder)
    
      mcp-mcp-ssh:
        image: sutazai-mcp-python:latest
        container_name: mcp-mcp-ssh
        environment:
          - MCP_SERVICE=mcp_ssh
          - PYTHONPATH=/opt/mcp
          - MCP_HOST=0.0.0.0
          - MCP_PORT=4005
        ports:
          - "4005:4005"
        volumes:
          - mcp-mcp-ssh-data:/var/lib/mcp
          - mcp-logs:/var/log/mcp
        networks:
          - mcp-bridge
        restart: unless-stopped
        healthcheck:
          test: ["CMD", "/opt/mcp/wrappers/mcp_ssh.sh", "health"]
          interval: 30s
          timeout: 10s
          retries: 3
          start_period: 30s
    
      # Specialized browser and automation services
      mcp-playwright-mcp:
        image: sutazai-mcp-specialized:latest
        container_name: mcp-playwright-mcp
        environment:
          - MCP_SERVICE=playwright-mcp
          - NODE_ENV=production
          - MCP_HOST=0.0.0.0
          - MCP_PORT=5001
        ports:
          - "5001:5001"
        volumes:
          - mcp-playwright-mcp-data:/var/lib/mcp
          - mcp-logs:/var/log/mcp
        networks:
          - mcp-bridge
        restart: unless-stopped
        healthcheck:
          test: ["CMD", "/opt/mcp/wrappers/playwright-mcp.sh", "health"]
          interval: 30s
          timeout: 10s
          retries: 3
          start_period: 30s
    
      mcp-github:
        image: sutazai-mcp-specialized:latest
        container_name: mcp-github
        environment:
          - MCP_SERVICE=github
          - NODE_ENV=production
          - MCP_HOST=0.0.0.0
          - MCP_PORT=5003
        ports:
          - "5003:5003"
        volumes:
          - mcp-github-data:/var/lib/mcp
          - mcp-logs:/var/log/mcp
        networks:
          - mcp-bridge
        restart: unless-stopped
        healthcheck:
          test: ["CMD", "/opt/mcp/wrappers/github.sh", "health"]
          interval: 30s
          timeout: 10s
          retries: 3
          start_period: 30s
    
      mcp-compass-mcp:
        image: sutazai-mcp-specialized:latest
        container_name: mcp-compass-mcp
        environment:
          - MCP_SERVICE=compass-mcp
          - NODE_ENV=production
          - MCP_HOST=0.0.0.0
          - MCP_PORT=5004
        ports:
          - "5004:5004"
        volumes:
          - mcp-compass-mcp-data:/var/lib/mcp
          - mcp-logs:/var/log/mcp
        networks:
          - mcp-bridge
        restart: unless-stopped
        healthcheck:
          test: ["CMD", "/opt/mcp/wrappers/compass-mcp.sh", "health"]
          interval: 30s
          timeout: 10s
          retries: 3
          start_period: 30s
    
      # Unified Development Service (consolidates ultimatecoder, language-server, sequentialthinking)
      mcp-unified-dev:
        image: sutazai-mcp-unified:latest
        container_name: mcp-unified-dev
        environment:
          - MCP_SERVICE=unified-dev
          - NODE_ENV=production
          - MCP_HOST=0.0.0.0
          - MCP_PORT=4000
          - NODE_OPTIONS=--max-old-space-size=512
          - PYTHON_PATH=/opt/mcp/python
          - GO_PATH=/opt/mcp/go
          - MAX_INSTANCES=3
        ports:
          - "4000:4000"
        volumes:
          - mcp-unified-dev-data:/var/lib/mcp
          - mcp-logs:/var/log/mcp
        networks:
          - mcp-bridge
        restart: unless-stopped
        deploy:
          resources:
            limits:
              memory: 512M
            reservations:
              memory: 256M
        healthcheck:
          test: ["CMD", "/opt/mcp/wrappers/unified-dev.sh", "health"]
          interval: 30s
          timeout: 10s
          retries: 3
          start_period: 30s
    
    18:      - "3001:3001"
    41:      - "3002:3002"
    64:      - "3003:3003"
    88:      - "3004:3004"
    111:      - "3006:3006"
    136:      - "3008:3008"
    159:      - "3009:3009"
    182:      - "3010:3010"
    206:      - "4001:4001"
    229:      - "4002:4002"
    252:      - "4003:4003"
    277:      - "4005:4005"
    301:      - "5001:5001"
    324:      - "5003:5003"
    347:      - "5004:5004"
    375:      - "4000:4000"

### /opt/sutazaiapp/docker/docker-compose.yml
    services:
      postgres:
        container_name: sutazai-postgres
        environment:
          POSTGRES_DB: ${POSTGRES_DB:-sutazai}
          POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
          POSTGRES_USER: ${POSTGRES_USER:-sutazai}
        healthcheck:
          interval: 10s
          retries: 5
          start_period: 60s
          test:
          - CMD-SHELL
          - pg_isready -U ${POSTGRES_USER:-sutazai}
          timeout: 5s
        image: postgres:15-alpine
        networks:
        - sutazai-network
        ports:
        - 10000:5432
        restart: unless-stopped
        volumes:
        - postgres_data:/var/lib/postgresql/data
        - ./IMPORTANT/init_db.sql:/docker-entrypoint-initdb.d/init.sql:ro
      redis:
        command: redis-server /usr/local/etc/redis/redis.conf
        container_name: sutazai-redis
        healthcheck:
          interval: 10s
          retries: 5
          test:
          - CMD-SHELL
          - redis-cli ping
          timeout: 5s
        image: redis:7-alpine
        networks:
        - sutazai-network
        ports:
        - 10001:6379
        restart: unless-stopped
        volumes:
        - redis_data:/data
        - ./config/redis-optimized.conf:/usr/local/etc/redis/redis.conf:ro
      neo4j:
        container_name: sutazai-neo4j
        environment:
          NEO4J_AUTH: neo4j/${NEO4J_PASSWORD}
          NEO4J_server_memory_heap_max__size: 512m
          NEO4J_server_memory_heap_initial__size: 256m
          NEO4J_server_memory_pagecache_size: 256m
          NEO4J_server_jvm_additional: -XX:+UseG1GC -XX:G1HeapRegionSize=4m -XX:+DisableExplicitGC
            -XX:+ExitOnOutOfMemoryError
          NEO4J_initial_dbms_default__database: sutazai
          NEO4J_db_checkpoint_interval_time: 30s
          NEO4J_db_transaction_timeout: 30s
          NEO4J_db_logs_query_enabled: "OFF"
          NEO4J_server_config_strict__validation_enabled: "false"
          NEO4J_db_transaction_bookmark_ready_timeout: 5s
          NEO4J_dbms_cluster_discovery_type: SINGLE
        healthcheck:
          interval: 60s
          retries: 3
          start_period: 45s
          test:
          - CMD-SHELL
          - wget --no-verbose --tries=1 --spider http://localhost:7474/ || exit 1
          timeout: 15s
        image: neo4j:5
        networks:
        - sutazai-network
        ports:
        - 10002:7474
        - 10003:7687
        restart: unless-stopped
        volumes:
        - neo4j_data:/data
      ollama:
        container_name: sutazai-ollama
        environment:
          CLAUDE_RULES_PATH: /app/CLAUDE.md
          ENFORCE_CLAUDE_RULES: 'true'
          OLLAMA_API_KEY: local
          OLLAMA_BASE_URL: http://ollama:11434
          OLLAMA_DEBUG: "false"
          OLLAMA_FLASH_ATTENTION: '0'
          OLLAMA_HOST: 0.0.0.0
          OLLAMA_KEEP_ALIVE: 5m
          OLLAMA_MAX_LOADED_MODELS: '1'
          OLLAMA_MODELS: /home/ollama/.ollama/models
          OLLAMA_NUM_PARALLEL: '1'
          OLLAMA_NUM_THREADS: '8'
          OLLAMA_ORIGINS: '*'
          OLLAMA_RUNNERS_DIR: /tmp
          OLLAMA_TMPDIR: /tmp/ollama
          OLLAMA_MAX_QUEUE: '10'
          OLLAMA_TIMEOUT: 300s
          OLLAMA_REQUEST_TIMEOUT: '300'
          OLLAMA_CONNECTION_POOL: '10'
          OLLAMA_USE_MMAP: 'true'
          OLLAMA_USE_NUMA: 'false'
        healthcheck:
          test:
          - CMD-SHELL
          - ollama list || exit 1
          interval: 30s
          timeout: 10s
          retries: 3
          start_period: 60s
        image: ollama/ollama:latest
        networks:
        - sutazai-network
        ports:
        - 10104:11434
        restart: unless-stopped
        sysctls:
        - net.core.somaxconn=65535
        ulimits:
          nofile:
            hard: 65536
            soft: 65536
        volumes:
        - ollama_data:/home/ollama/.ollama
        - models_data:/models
        - /opt/sutazaiapp/CLAUDE.md:/app/CLAUDE.md:ro
        - /opt/sutazaiapp/config/ollama.yaml:/app/config/ollama.yaml:ro
        logging:
          driver: json-file
          options:
            max-size: 100m
            max-file: '5'
      chromadb:
        container_name: sutazai-chromadb
        environment:
        - CHROMA_SERVER_AUTH_PROVIDER=chromadb.auth.token.TokenAuthenticationServerProvider
        - CHROMA_SERVER_AUTH_CREDENTIALS=${CHROMADB_API_KEY}
        - CHROMA_SERVER_HOST=0.0.0.0
        - CHROMA_SERVER_HTTP_PORT=8000
        - CHROMA_SERVER_CORS_ALLOW_ORIGINS=["http://localhost:8501", "http://backend:8000"]
        healthcheck:
          interval: 60s
          retries: 5
          start_period: 120s
          test:
          - CMD
          - sh
          - -c
          - curl -f http://localhost:8000/api/v1/heartbeat || exit 1
          timeout: 30s
        image: chromadb/chroma:latest
        networks:
        - sutazai-network
        ports:
        - 10100:8000
        restart: unless-stopped
        volumes:
        - chromadb_data:/chroma/chroma
      qdrant:
        container_name: sutazai-qdrant
        environment:
          QDRANT__LOG_LEVEL: INFO
          QDRANT__SERVICE__GRPC_PORT: 6334
          QDRANT__SERVICE__HTTP_PORT: 6333
        healthcheck:
          interval: 60s
          retries: 5
          test:
          - CMD
          - sh
          - -c
          - echo 'use IO::Socket::INET; my $$s = IO::Socket::INET->new(PeerAddr => q{localhost:6333},
            Proto => q{tcp}, Timeout => 2); if ($$s) { print $$s qq{GET / HTTP/1.0\r\n\r\n};
            while (<$$s>) { if (/200 OK/) { exit 0; } } } exit 1;' | perl
          timeout: 30s
        image: qdrant/qdrant:latest
        networks:
        - sutazai-network
        ports:
        - 10101:6333
        - 10102:6334
        restart: unless-stopped
        volumes:
        - qdrant_data:/qdrant/storage
      faiss:
        image: sutazaiapp-faiss:latest
        build:
          context: ./docker/faiss
          dockerfile: ./docker/faiss/Dockerfile
        container_name: sutazai-faiss
        environment:
          SUTAZAI_ENV: ${SUTAZAI_ENV:-production}
          DIND_HOST: sutazai-mcp-orchestrator
          TZ: ${TZ:-UTC}
        healthcheck:
          interval: 60s
          retries: 5
          test:
          - CMD
          - python
          - -c
          - import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()
          timeout: 30s
        networks:
        - sutazai-network
        ports:
        - 10103:8000
        restart: unless-stopped
        volumes:
        - faiss_data:/data
      kong:
        image: kong:3.5
        container_name: sutazai-kong
        environment:
          KONG_DATABASE: 'off'
          KONG_DECLARATIVE_CONFIG: /etc/kong/kong-optimized.yml
          KONG_PROXY_LISTEN: 0.0.0.0:8000
          KONG_ADMIN_LISTEN: 0.0.0.0:8001
          KONG_LOG_LEVEL: error
          KONG_NGINX_WORKER_PROCESSES: '1'
          KONG_NGINX_WORKER_CONNECTIONS: '512'
          KONG_MEM_CACHE_SIZE: '64m'
          KONG_SSL_CIPHER_SUITE: 'intermediate'
          KONG_PROXY_ACCESS_LOG: 'off'
          KONG_ADMIN_ACCESS_LOG: 'off'
          KONG_PROXY_ERROR_LOG: '/dev/stderr'
          KONG_ADMIN_ERROR_LOG: '/dev/stderr'
          KONG_CLIENT_BODY_BUFFER_SIZE: '8k'
          KONG_CLIENT_MAX_BODY_SIZE: '8m'
          KONG_NGINX_HTTP_CLIENT_BODY_BUFFER_SIZE: '8k'
          KONG_NGINX_HTTP_CLIENT_MAX_BODY_SIZE: '8m'
        healthcheck:
          test:
          - CMD-SHELL
          - ps aux | grep '[k]ong' > /dev/null || exit 1
          interval: 30s
          timeout: 5s
          retries: 3
          start_period: 10s
        networks:
        - sutazai-network
        ports:
        - 10005:8000
        - 10015:8001
        restart: unless-stopped
        volumes:
        - ./config/kong/kong-optimized.yml:/etc/kong/kong-optimized.yml:ro
      consul:
        image: consul:latest
        container_name: sutazai-consul
        # ULTRA-FIX: Remove user: root - image now runs as consul user
        environment:
          CONSUL_ALLOW_PRIVILEGED_PORTS: 'true'
          CONSUL_DISABLE_PERM: 'true'
        command:
        - agent
        - -server
        - -bootstrap-expect=1
        - -ui
        - -client=0.0.0.0
        - -data-dir=/consul/data
        healthcheck:
          test:
          - CMD
          - wget
          - --no-verbose
          - --tries=1
          - --spider
          - http://localhost:8500/v1/status/leader
          interval: 30s
          timeout: 5s
          retries: 10
          start_period: 10s
        networks:
        - sutazai-network
        ports:
        - 10006:8500
        restart: unless-stopped
        volumes:
        - ./config/consul/consul.hcl:/consul/config/consul.hcl:ro
        - consul_data:/consul/data
      rabbitmq:
        image: rabbitmq:3.12-management-alpine
        container_name: sutazai-rabbitmq
        environment:
          RABBITMQ_DEFAULT_USER: ${RABBITMQ_DEFAULT_USER:-sutazai}
          RABBITMQ_DEFAULT_PASS: ${RABBITMQ_DEFAULT_PASS}
        healthcheck:
          test:
          - CMD
          - rabbitmq-diagnostics
          - check_running
          interval: 30s
          timeout: 5s
          retries: 10
          start_period: 20s
        networks:
        - sutazai-network
        ports:
        - 10007:5672
        - 10008:15672
        restart: unless-stopped
        volumes:
        - rabbitmq_data:/var/lib/rabbitmq
      backend:
        image: sutazaiapp-backend:latest
        build:
          context: ../backend
          dockerfile: ../docker/backend/Dockerfile
        command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
        container_name: sutazai-backend
        depends_on:
          chromadb:
            condition: service_healthy
          neo4j:
            condition: service_started
          ollama:
            condition: service_healthy
          postgres:
            condition: service_healthy
          qdrant:
            condition: service_healthy
        environment:
          API_V1_STR: /api/v1
          BACKEND_CORS_ORIGINS: '["http://localhost:10011", "http://172.31.77.193:10011"]'
          CHROMADB_HOST: chromadb
          CHROMADB_PORT: 8000
          CHROMADB_URL: http://chromadb:8000
          DATABASE_URL: postgresql+asyncpg://${POSTGRES_USER:-sutazai}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB:-sutazai}
          FAISS_INDEX_PATH: /data/faiss
          GRAFANA_PASSWORD: ${GRAFANA_PASSWORD}
          JWT_SECRET: ${JWT_SECRET}
          JWT_SECRET_KEY: ${JWT_SECRET_KEY}
          NEO4J_HOST: neo4j
          NEO4J_PASSWORD: ${NEO4J_PASSWORD}
          NEO4J_PORT: 7687
          NEO4J_URI: bolt://neo4j:7687
          NEO4J_USER: neo4j
          OLLAMA_API_KEY: local
          OLLAMA_BASE_URL: http://ollama:11434
          OLLAMA_HOST: 0.0.0.0
          OLLAMA_ORIGINS: '*'
          POSTGRES_DB: ${POSTGRES_DB:-sutazai}
          POSTGRES_HOST: postgres
          POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
          POSTGRES_PORT: 5432
          POSTGRES_USER: ${POSTGRES_USER:-sutazai}
          QDRANT_HOST: qdrant
          QDRANT_PORT: 6333
          QDRANT_URL: http://qdrant:6333
          REDIS_HOST: sutazai-redis
          # REDIS_PASSWORD not needed - Redis runs without authentication
          REDIS_PORT: 6379
          REDIS_URL: redis://sutazai-redis:6379/0
          SECRET_KEY: ${SECRET_KEY}
          SUTAZAI_ENV: ${SUTAZAI_ENV:-production}
          TZ: ${TZ:-UTC}
        healthcheck:
          interval: 60s
          retries: 5
          start_period: 120s
          test:
          - CMD
          - python3
          - -c
          - import socket; s=socket.socket(); s.settimeout(5); exit(0 if s.connect_ex(("localhost",
            8000))==0 else 1)
          timeout: 30s
        networks:
        - sutazai-network
        ports:
        - 10010:8000
        restart: unless-stopped
        volumes:
        - ../backend:/app
        - ../scripts:/opt/sutazaiapp/scripts:ro
        - ./data:/data
        - ./logs:/logs
        - agent_workspaces:/app/agent_workspaces
        # removed ./scripts:/scripts:ro (not needed)
        - ../scripts:/opt/sutazaiapp/scripts:ro
      frontend:
        image: sutazaiapp-frontend:latest
        build:
          context: ./frontend
          dockerfile: ./docker/frontend/Dockerfile
        command: streamlit run app.py --server.port 8501 --server.address 0.0.0.0
        container_name: sutazai-frontend
        depends_on:
          backend:
            condition: service_healthy
        environment:
          BACKEND_URL: http://backend:8000
          STREAMLIT_SERVER_ADDRESS: 0.0.0.0
          STREAMLIT_SERVER_PORT: 8501
          SUTAZAI_ENV: ${SUTAZAI_ENV:-production}
          TZ: ${TZ:-UTC}
        healthcheck:
          interval: 60s
          retries: 5
          start_period: 120s
          test:
          - CMD
          - python3
          - -c
          - import socket; s=socket.socket(); s.settimeout(5); exit(0 if s.connect_ex(("localhost",
            8501))==0 else 1)
          timeout: 30s
        networks:
        - sutazai-network
        ports:
        - 10011:8501
        restart: unless-stopped
        volumes:
        - ./frontend:/app
        - ./data:/data
      prometheus:
        command:
        - --config.file=/etc/prometheus/prometheus.yml
        - --storage.tsdb.path=/prometheus
        - --web.console.libraries=/usr/share/prometheus/console_libraries
        - --web.console.templates=/usr/share/prometheus/consoles
        - --web.enable-lifecycle
        - --storage.tsdb.retention.time=7d
        - --web.enable-admin-api
        - --storage.tsdb.max-block-duration=2h
        - --storage.tsdb.min-block-duration=2h
        - --storage.tsdb.retention.size=1GB
        container_name: sutazai-prometheus
        healthcheck:
          interval: 60s
          retries: 5
          start_period: 90s
          test:
          - CMD
          - wget
          - --no-verbose
          - --tries=1
          - --spider
          - http://localhost:9090/-/healthy
          timeout: 30s
        image: prom/prometheus:latest
        networks:
        - sutazai-network
        ports:
        - 10200:9090
        restart: unless-stopped
        volumes:
        - ./monitoring/prometheus:/etc/prometheus
        - prometheus_data:/prometheus
      grafana:
        container_name: sutazai-grafana
        depends_on:
        - prometheus
        - loki
        environment:
        - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
        - GF_USERS_ALLOW_SIGN_UP=false
        - GF_INSTALL_PLUGINS=
        - GF_ANALYTICS_REPORTING_ENABLED=false
        - GF_ANALYTICS_CHECK_FOR_UPDATES=false
        - GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH=/var/lib/grafana/dashboards/system-overview.json
        healthcheck:
          interval: 60s
          retries: 5
          start_period: 60s
          test:
          - CMD-SHELL
          - wget --no-verbose --tries=1 --spider http://localhost:3000/api/health || exit
            1
          timeout: 30s
        image: grafana/grafana:latest
        networks:
        - sutazai-network
        ports:
        - 10201:3000
        restart: unless-stopped
        volumes:
        - grafana_data:/var/lib/grafana
        - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
        - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards
      loki:
        command: -config.file=/etc/loki/local-config.yaml
        container_name: sutazai-loki
        healthcheck:
          interval: 60s
          retries: 5
          start_period: 60s
          test:
          - CMD-SHELL
          - wget --no-verbose --tries=1 --spider http://localhost:3100/ready || exit 1
          timeout: 30s
        image: grafana/loki:2.9.0
        networks:
        - sutazai-network
        ports:
        - 10202:3100
        restart: unless-stopped
        volumes:
        - loki_data:/loki
        - ./monitoring/loki/config.yml:/etc/loki/local-config.yaml
      alertmanager:
        command:
        - --config.file=/etc/alertmanager/config.yml
        - --storage.path=/alertmanager
        - --web.external-url=http://localhost:9093
        container_name: sutazai-alertmanager
        environment:
        - SLACK_WEBHOOK_URL=${SLACK_WEBHOOK_URL:-}
        - SLACK_AI_WEBHOOK_URL=${SLACK_AI_WEBHOOK_URL:-}
        - SLACK_SECURITY_WEBHOOK_URL=${SLACK_SECURITY_WEBHOOK_URL:-}
        - PAGERDUTY_SERVICE_KEY=${PAGERDUTY_SERVICE_KEY:-}
        healthcheck:
          test:
          - CMD
          - wget
          - --no-verbose
          - --tries=1
          - --spider
          - http://localhost:9093/-/ready
          interval: 60s
          timeout: 30s
          retries: 5
        image: prom/alertmanager:latest
        networks:
        - sutazai-network
        ports:
        - 10203:9093
        restart: unless-stopped
        volumes:
        - ./monitoring/alertmanager:/etc/alertmanager
        - alertmanager_data:/alertmanager
      blackbox-exporter:
        container_name: sutazai-blackbox-exporter
        user: nobody
        healthcheck:
          interval: 60s
          retries: 5
          test:
          - CMD
          - wget
          - --no-verbose
          - --tries=1
          - --spider
          - http://localhost:9115/
          timeout: 30s
        image: prom/blackbox-exporter:latest
        networks:
        - sutazai-network
        ports:
        - 10204:9115
        restart: unless-stopped
        volumes:
        - ./monitoring/blackbox/config.yml:/etc/blackbox_exporter/config.yml
      node-exporter:
        command:
        - --path.procfs=/host/proc
        - --path.sysfs=/host/sys
        - --path.rootfs=/rootfs
        - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$$$|/)'
        container_name: sutazai-node-exporter
        image: prom/node-exporter:latest
        networks:
        - sutazai-network
        ports:
        - 10205:9100
        restart: unless-stopped
        volumes:
        - /proc:/host/proc:ro
        - /sys:/host/sys:ro
        - /:/rootfs:ro
      cadvisor:
        container_name: sutazai-cadvisor
        devices:
        - /dev/kmsg
        image: gcr.io/cadvisor/cadvisor:latest
        # ULTRA-FIX: Changed user from 'cadvisor' to 'nobody' to match Dockerfile
        user: nobody
        networks:
        - sutazai-network
        ports:
        - 10206:8080
        privileged: false
        cap_add:
        - SYS_ADMIN
        - SYS_RESOURCE
        - SYS_PTRACE
        - DAC_READ_SEARCH
        security_opt:
        - no-new-privileges:true
        # ULTRA-FIX: Set read_only to true with tmpfs for writable areas
        read_only: true
        restart: unless-stopped
        command:
        - --housekeeping_interval=60s
        - --max_housekeeping_interval=120s
        - --allow_dynamic_housekeeping=true
        - --global_housekeeping_interval=2m
        - --disable_metrics=advtcp,cpu_topology,disk,hugetlb,memory_numa,percpu,referenced_memory,resctrl,tcp,udp,process
        - --store_container_labels=false
        - --whitelisted_container_labels=io.kubernetes.container.name,io.kubernetes.pod.name
        volumes:
        - /:/rootfs:ro
        - /var/run:/var/run:ro
        - /sys:/sys:ro
        - /var/lib/docker/:/var/lib/docker:ro
        - /dev/disk/:/dev/disk:ro
        # ULTRA-FIX: Add tmpfs volumes for writable areas needed by cAdvisor
        tmpfs:
        - /tmp:size=100M,mode=1777
        - /var/cache:size=50M,mode=1777
      postgres-exporter:
        container_name: sutazai-postgres-exporter
        depends_on:
          postgres:
            condition: service_healthy
        environment:
          DATA_SOURCE_NAME: postgresql://${POSTGRES_USER:-sutazai}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB:-sutazai}?sslmode=disable
        image: prometheuscommunity/postgres-exporter:latest
        networks:
        - sutazai-network
        ports:
        - 10207:9187
        restart: unless-stopped
      redis-exporter:
        container_name: sutazai-redis-exporter
        depends_on:
          redis:
            condition: service_healthy
        environment:
          REDIS_ADDR: redis://redis:6379
          # REDIS_PASSWORD not needed - Redis runs without authentication
        image: oliver006/redis_exporter:latest
        networks:
        - sutazai-network
        ports:
        - 10208:9121
        restart: unless-stopped
      ollama-integration:
        image: sutazai-mcp-unified-dev:latest
        container_name: sutazai-ollama-integration
        depends_on:
          ollama:
            condition: service_healthy
          redis:
            condition: service_healthy
        environment:
          AGENT_TYPE: ollama-integration
          API_ENDPOINT: http://backend:8000
          LOG_LEVEL: INFO
          OLLAMA_BASE_URL: http://ollama:11434
          PORT: 8090
          RABBITMQ_URL: amqp://${RABBITMQ_DEFAULT_USER:-sutazai}:${RABBITMQ_DEFAULT_PASS}@rabbitmq:5672/
          REDIS_URL: redis://sutazai-redis:6379/0
          SUTAZAI_ENV: ${SUTAZAI_ENV:-production}
          TZ: ${TZ:-UTC}
          MAX_RETRIES: 3
          BACKOFF_BASE: 2
          REQUEST_TIMEOUT: 30
          CONNECTION_POOL_SIZE: 10
        healthcheck:
          test:
          - CMD
          - curl
          - -f
          - http://localhost:8090/health
          interval: 30s
          timeout: 3s
          retries: 3
          start_period: 5s
        networks:
        - sutazai-network
        ports:
        - 8589:8589
        restart: unless-stopped
        volumes:
        - ./agents/core:/app/agents/core:ro
      hardware-resource-optimizer:
        image: sutazai-mcp-unified-dev:latest
        container_name: sutazai-hardware-resource-optimizer
        depends_on:
          backend:
            condition: service_started
          ollama:
            condition: service_started
          redis:
            condition: service_healthy
        environment:
          AGENT_TYPE: hardware-resource-optimizer
          API_ENDPOINT: http://backend:8000
          DATABASE_URL: postgresql+asyncpg://${POSTGRES_USER:-sutazai}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB:-sutazai}
          LOG_LEVEL: INFO
          OLLAMA_API_KEY: local
          OLLAMA_BASE_URL: http://ollama:11434
          OLLAMA_HOST: 0.0.0.0
          OLLAMA_MODEL: tinyllama:latest
          OLLAMA_ORIGINS: '*'
          PORT: 8080
          RABBITMQ_URL: amqp://${RABBITMQ_DEFAULT_USER:-sutazai}:${RABBITMQ_DEFAULT_PASS}@rabbitmq:5672/
          REDIS_URL: redis://sutazai-redis:6379/0
          SUTAZAI_ENV: ${SUTAZAI_ENV:-production}
          TZ: ${TZ:-UTC}
        healthcheck:
          interval: 60s
          retries: 5
          start_period: 120s
          test:
          - CMD
          - curl
          - -f
          - http://localhost:8080/health
          timeout: 30s
        networks:
        - sutazai-network
        ports:
        - 11110:8080
        privileged: false
        cap_drop:
        - ALL
        cap_add:
        - SYS_PTRACE
        security_opt:
        - no-new-privileges:true
        - seccomp:unconfined
        restart: unless-stopped
        volumes:
        - ./data:/app/data:rw,noexec
        - ./configs:/app/configs:ro
        - ./logs:/app/logs:rw,noexec
      jarvis-hardware-resource-optimizer:
        image: sutazaiapp-jarvis-hardware-resource-optimizer:latest
        container_name: sutazai-jarvis-hardware-resource-optimizer
        depends_on:
          backend:
            condition: service_started
          rabbitmq:
            condition: service_healthy
          redis:
            condition: service_healthy
        environment:
          AGENT_TYPE: jarvis-hardware-resource-optimizer
          API_ENDPOINT: http://backend:8000
          LOG_LEVEL: INFO
          PORT: 8080
          RABBITMQ_URL: amqp://${RABBITMQ_DEFAULT_USER:-sutazai}:${RABBITMQ_DEFAULT_PASS}@rabbitmq:5672/
          REDIS_URL: redis://sutazai-redis:6379/0
          SUTAZAI_ENV: ${SUTAZAI_ENV:-production}
          TZ: ${TZ:-UTC}
        healthcheck:
          test:
          - CMD
          - curl
          - -f
          - http://localhost:8080/health
          interval: 60s
          timeout: 30s
          retries: 5
          start_period: 60s
        networks:
        - sutazai-network
        ports:
        - 11104:8080
        privileged: false
        cap_drop:
        - ALL
        cap_add:
        - SYS_PTRACE
        security_opt:
        - no-new-privileges:true
        - seccomp:unconfined
        user: 1001:1001
        restart: unless-stopped
        volumes:
        - ./agents/core:/app/agents/core:ro
        - ./data:/app/data:rw,noexec
        - ./configs:/app/configs:ro
        - ./logs:/app/logs:rw,noexec
      jarvis-automation-agent:
        image: sutazaiapp-jarvis-hardware-resource-optimizer:latest
        container_name: sutazai-jarvis-automation-agent
        depends_on:
          backend:
            condition: service_started
          rabbitmq:
            condition: service_healthy
          redis:
            condition: service_healthy
        environment:
          AGENT_TYPE: jarvis-automation-agent
          API_ENDPOINT: http://backend:8000
          DATABASE_URL: postgresql+asyncpg://${POSTGRES_USER:-sutazai}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB:-sutazai}
          LOG_LEVEL: INFO
          OLLAMA_BASE_URL: http://ollama:11434
          PORT: 8080
          RABBITMQ_URL: amqp://${RABBITMQ_DEFAULT_USER:-sutazai}:${RABBITMQ_DEFAULT_PASS}@rabbitmq:5672/
          REDIS_URL: redis://sutazai-redis:6379/0
          SUTAZAI_ENV: ${SUTAZAI_ENV:-production}
          TZ: ${TZ:-UTC}
        healthcheck:
          test:
          - CMD
          - curl
          - -f
          - http://localhost:8080/health
          interval: 60s
          timeout: 30s
          retries: 5
          start_period: 60s
        networks:
        - sutazai-network
        ports:
        - 11102:8080
        restart: unless-stopped
        volumes:
        - ./agents/core:/app/agents/core:ro
        - ./data:/app/data
        - ./configs:/app/configs
        - ./logs:/app/logs
        - /tmp:/tmp
        - /opt/sutazaiapp:/opt/sutazaiapp:ro
      ai-agent-orchestrator:
        image: sutazai-mcp-unified-dev:latest
        container_name: sutazai-ai-agent-orchestrator
        depends_on:
          rabbitmq:
            condition: service_healthy
          redis:
            condition: service_healthy
        environment:
          AGENT_TYPE: ai-agent-orchestrator
          API_ENDPOINT: http://backend:8000
          LOG_LEVEL: INFO
          PORT: 8589
          RABBITMQ_URL: amqp://${RABBITMQ_DEFAULT_USER:-sutazai}:${RABBITMQ_DEFAULT_PASS}@rabbitmq:5672/
          REDIS_URL: redis://sutazai-redis:6379/0
          SUTAZAI_ENV: ${SUTAZAI_ENV:-production}
          TZ: ${TZ:-UTC}
        healthcheck:
          test:
          - CMD
          - curl
          - -f
          - http://localhost:8589/health
          interval: 30s
          timeout: 10s
          retries: 5
          start_period: 120s
        networks:
        - sutazai-network
        restart: unless-stopped
        volumes:
        - ./agents/core:/app/agents/core:ro
        - ./logs:/app/logs
      task-assignment-coordinator:
        image: sutazai-mcp-unified-dev:latest
        container_name: sutazai-task-assignment-coordinator
        depends_on:
          rabbitmq:
            condition: service_healthy
          redis:
            condition: service_healthy
        environment:
          AGENT_TYPE: task-assignment-coordinator
          API_ENDPOINT: http://backend:8000
          LOG_LEVEL: INFO
          PORT: 4000
          RABBITMQ_URL: amqp://${RABBITMQ_DEFAULT_USER:-sutazai}:${RABBITMQ_DEFAULT_PASS}@rabbitmq:5672/
          REDIS_URL: redis://sutazai-redis:6379/0
          SUTAZAI_ENV: ${SUTAZAI_ENV:-production}
          TZ: ${TZ:-UTC}
        healthcheck:
          test:
          - CMD
          - curl
          - -f
          - http://localhost:4000/health
          interval: 30s
          timeout: 10s
          retries: 5
          start_period: 120s
        networks:
        - sutazai-network
        ports:
        - 8551:4000
        restart: unless-stopped
        volumes:
        - ./agents/core:/app/agents/core:ro
        - ./logs:/app/logs
      resource-arbitration-agent:
        image: sutazai-mcp-unified-dev:latest
        container_name: sutazai-resource-arbitration-agent
        depends_on:
          rabbitmq:
            condition: service_healthy
          redis:
            condition: service_healthy
        environment:
          AGENT_TYPE: resource-arbitration-agent
          API_ENDPOINT: http://backend:8000
          LOG_LEVEL: INFO
          PORT: 8588
          RABBITMQ_URL: amqp://${RABBITMQ_DEFAULT_USER:-sutazai}:${RABBITMQ_DEFAULT_PASS}@rabbitmq:5672/
          REDIS_URL: redis://sutazai-redis:6379/0
          SUTAZAI_ENV: ${SUTAZAI_ENV:-production}
          TZ: ${TZ:-UTC}
        healthcheck:
          test:
          - CMD
          - curl
          - -f
          - http://localhost:8588/health
          interval: 30s
          timeout: 10s
          retries: 5
          start_period: 120s
        networks:
        - sutazai-network
        ports:
        - 8588:8588
        privileged: false
        cap_drop:
        - ALL
        cap_add:
        - SYS_PTRACE
        security_opt:
        - no-new-privileges:true
        - seccomp:unconfined
        user: 1001:1001
        restart: unless-stopped
        volumes:
        - ./agents/core:/app/agents/core:ro
        - ./data:/app/data:rw,noexec
        - ./logs:/app/logs:rw,noexec
    
      # Distributed Tracing
      jaeger:
        container_name: sutazai-jaeger
        image: jaegertracing/all-in-one:latest
        environment:
          - COLLECTOR_ZIPKIN_HTTP_PORT=9411
          - COLLECTOR_OTLP_ENABLED=true
          - JAEGER_DISABLED=false
          - METRICS_STORAGE_TYPE=prometheus
          - PROMETHEUS_SERVER_URL=http://sutazai-prometheus:9090
          - PROMETHEUS_QUERY_SUPPORT_SPANMETRICS_CONNECTOR=true
          - SPAN_STORAGE_TYPE=memory
          - MEMORY_MAX_TRACES=100000
        ports:
          - "10210:16686"  # Jaeger UI
          - "10211:14268"  # Jaeger Collector HTTP
          - "10212:14250"  # Jaeger Collector gRPC
          - "10213:9411"   # Zipkin Collector
          - "10214:4317"   # OTLP gRPC
          - "10215:4318"   # OTLP HTTP
        networks:
          - sutazai-network
        restart: unless-stopped
        healthcheck:
          test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:16686/"]
          interval: 30s
          timeout: 10s
          retries: 3
          start_period: 40s
    
      # Log Shipping (Promtail for Loki)
      promtail:
        container_name: sutazai-promtail
        image: grafana/promtail:latest
        user: promtail
        volumes:
          - /var/log:/var/log:ro
          - /var/lib/docker/containers:/var/lib/docker/containers:ro
          - /var/run/docker.sock:/var/run/docker.sock:ro
          - ./monitoring/promtail/config.yml:/etc/promtail/config.yml:ro
        networks:
          - sutazai-network
        restart: unless-stopped
        privileged: true
        depends_on:
          - loki
        healthcheck:
          test:
          - CMD-SHELL
          - pgrep promtail
          interval: 60s
          timeout: 10s
          retries: 3
          start_period: 30s
    
    24:    - 10000:5432
    43:    - 10001:6379
    76:    - 10002:7474
    77:    - 10003:7687
    117:    - 10104:11434
    157:    - 10100:8000
    182:    - 10101:6333
    183:    - 10102:6334
    209:    - 10103:8000
    245:    - 10005:8000
    246:    - 10015:8001
    279:    - 10006:8500
    302:    - 10007:5672
    303:    - 10008:15672
    374:    - 10010:8000
    414:    - 10011:8501
    448:    - 10200:9090
    478:    - 10201:3000
    499:    - 10202:3100
    530:    - 10203:9093
    553:    - 10204:9115
    568:    - 10205:9100
    584:    - 10206:8080
    625:    - 10207:9187
    639:    - 10208:9121
    676:    - 8589:8589
    718:    - 11110:8080
    764:    - 11104:8080
    814:    - 11102:8080
    886:    - 8551:4000
    921:    - 8588:8588
    951:      - "10210:16686"  # Jaeger UI
    952:      - "10211:14268"  # Jaeger Collector HTTP
    953:      - "10212:14250"  # Jaeger Collector gRPC
    954:      - "10213:9411"   # Zipkin Collector
    955:      - "10214:4317"   # OTLP gRPC
    956:      - "10215:4318"   # OTLP HTTP

### /opt/sutazaiapp/docker/mcp-services/unified-memory/docker-compose.unified-memory.yml
    services:
      mcp-unified-memory:
        build:
          context: .
          dockerfile: Dockerfile
        image: sutazai-mcp-unified-memory:latest
        container_name: mcp-unified-memory
        environment:
          - MCP_SERVICE=unified-memory
          - PYTHONPATH=/opt/memory
          - MCP_HOST=0.0.0.0
          - MCP_PORT=3009
          - SQLITE_PATH=/var/lib/mcp/unified_memory.db
          - VECTOR_STORE_ENABLED=true
          - CACHE_ENABLED=true
          - MAX_MEMORY_SIZE=2097152  # 2MB
        ports:
          - "3009:3009"
        volumes:
          - mcp-unified-memory-data:/var/lib/mcp
          - mcp-memory-vector-store:/var/lib/vector
          - mcp-logs:/var/log/mcp
        networks:
          - mcp-bridge
        restart: unless-stopped
        healthcheck:
          test: ["CMD", "curl", "-f", "http://localhost:3009/health"]
          interval: 30s
          timeout: 10s
          retries: 3
          start_period: 30s
        deploy:
          resources:
            limits:
              cpus: '1.0'
              memory: 512M
            reservations:
              cpus: '0.25'
              memory: 128M
        labels:
          - "mcp.service=unified-memory"
          - "mcp.consolidation=phase-2"
          - "mcp.replaces=extended-memory,memory-bank-mcp"
    
    23:      - "3009:3009"


## Preliminary Findings
- Static, read-only reconciliation. Does not verify runtime state.
- Next: correlate declared ports with compose exposure and produce mismatch table.

