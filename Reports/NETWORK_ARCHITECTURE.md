# SutazAI Docker Network Architecture

## Network Topology

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         External Access Layer                            │
│  Ports: 10000-11999 (Host → Container mapping)                          │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Primary Network: sutazai-network                      │
│                         172.20.0.0/16                                    │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌───────────────┐ │
│  │   Core Services       │  │   Application Layer  │  │   Agents      │ │
│  │   172.20.0.10-19     │  │   172.20.0.30-49    │  │  172.20.0.50+ │ │
│  ├──────────────────────┤  ├──────────────────────┤  ├───────────────┤ │
│  │ .10 PostgreSQL       │  │ .31 Frontend         │  │ .50+ Various  │ │
│  │ .11 Redis            │  │ .35 Kong Gateway     │  │     Agents    │ │
│  │ .12 Neo4j            │  │ .40 Backend API      │  └───────────────┘ │
│  │ .13 RabbitMQ         │  └──────────────────────┘                     │
│  │ .14 Consul           │                                               │
│  └──────────────────────┘                                               │
│                                                                           │
│  ┌──────────────────────┐                                               │
│  │   Vector Databases   │                                               │
│  │   172.20.0.20-29     │                                               │
│  ├──────────────────────┤                                               │
│  │ .20 ChromaDB         │                                               │
│  │ .21 Qdrant           │                                               │
│  │ .22 FAISS            │                                               │
│  └──────────────────────┘                                               │
└─────────────────────────────────────────────────────────────────────────┘

## Network Segmentation Plan (Future Implementation)

### Isolated Networks (To Be Implemented)
- **Database Network** (172.21.0.0/24): Isolated database communication
- **Agent Network** (172.22.0.0/24): Agent-to-agent communication
- **Vector Network** (172.23.0.0/24): Vector database cluster

## IP Address Allocation

### Core Infrastructure (172.20.0.10-19)
| Service | Container Name | IP Address | External Port | Internal Port |
|---------|---------------|------------|---------------|---------------|
| PostgreSQL | sutazai-postgres | 172.20.0.10 | 10000 | 5432 |
| Redis | sutazai-redis | 172.20.0.11 | 10001 | 6379 |
| Neo4j | sutazai-neo4j | 172.20.0.12 | 10002-10003 | 7474/7687 |
| RabbitMQ | sutazai-rabbitmq | 172.20.0.13 | 10004-10005 | 5672/15672 |
| Consul | sutazai-consul | 172.20.0.14 | 10006-10007 | 8500/8600 |

### Vector Databases (172.20.0.20-29)
| Service | Container Name | IP Address | External Port | Internal Port |
|---------|---------------|------------|---------------|---------------|
| ChromaDB | sutazai-chromadb | 172.20.0.20 | 10100 | 8000 |
| Qdrant | sutazai-qdrant | 172.20.0.21 | 10101-10102 | 6333/6334 |
| FAISS | sutazai-faiss | 172.20.0.22 | 10103 | 8000 |

### Application Layer (172.20.0.30-49)
| Service | Container Name | IP Address | External Port | Internal Port |
|---------|---------------|------------|---------------|---------------|
| Frontend | sutazai-jarvis-frontend | 172.20.0.31 | 11000 | 11000 |
| Kong Gateway | sutazai-kong | 172.20.0.35 | 10008-10009 | 8000/8001 |
| Backend API | sutazai-backend | 172.20.0.40 | 10200 | 8000 |

### AI Agents (Dynamic IPs)
| Service | Container Name | External Port | Internal Port |
|---------|---------------|---------------|---------------|
| Ollama | sutazai-ollama | 11435 | 11434 |
| Letta | sutazai-letta | 11401 | 8000 |
| AutoGPT | sutazai-autogpt | 11402 | 8000 |
| CrewAI | sutazai-crewai | 11403 | 8000 |
| Aider | sutazai-aider | 11404 | 8000 |
| LangChain | sutazai-langchain | 11405 | 8000 |
| LocalAGI | sutazai-localagi | 11406 | 8000 |
| BigAGI | sutazai-bigagi | 11407 | 8000 |
| AgentZero | sutazai-agentzero | 11408 | 8000 |
| Skyvern | sutazai-skyvern | 11409 | 8000 |
| FinRobot | sutazai-finrobot | 11410 | 8000 |
| ShellGPT | sutazai-shellgpt | 11413 | 8000 |
| Documind | sutazai-documind | 11414 | 8000 |
| AutoGen | sutazai-autogen | 11415 | 8000 |
| GPT-Engineer | sutazai-gpt-engineer | 11416 | 8000 |
| Browser Use | sutazai-browseruse | 11703 | 8000 |
| Semgrep | sutazai-semgrep | 11801 | 8000 |

## Service Dependencies

### Backend Service Dependencies
```

sutazai-backend
├── sutazai-postgres (Database)
├── sutazai-redis (Cache/Sessions)
├── sutazai-neo4j (Graph Database)
├── sutazai-rabbitmq (Message Queue)
├── sutazai-chromadb (Vector Store)
├── sutazai-qdrant (Vector Store)
└── sutazai-faiss (Vector Store)

```

### Agent Dependencies
```

AI Agents
├── sutazai-ollama (LLM Provider)
├── sutazai-mcp-bridge (Orchestration)
├── sutazai-rabbitmq (Message Queue)
├── sutazai-redis (Cache)
└── sutazai-consul (Service Discovery)

```

## Network Security Considerations

### Current State
- All services on single bridge network
- No network isolation between service tiers
- All containers can communicate freely

### Recommended Improvements
1. **Network Segmentation**: Implement separate networks for database, application, and agent tiers
2. **Network Policies**: Apply Kubernetes-style network policies to restrict inter-service communication
3. **TLS/SSL**: Enable encrypted communication between services
4. **API Gateway**: Route all external traffic through Kong gateway
5. **Service Mesh**: Consider implementing Consul Connect for mTLS

## Port Allocation Strategy

### Port Ranges
- **10000-10099**: Core infrastructure services
- **10100-10199**: Vector databases and storage
- **10200-10299**: Application services
- **10300-10999**: Reserved for future services
- **11000-11099**: Frontend and UI services
- **11100-11399**: MCP and orchestration services
- **11400-11899**: AI agents and automation tools

## Health Monitoring

### Service Health Endpoints
Most services expose health checks at:
- `http://[container]:8000/health` (Agents)
- `http://[container]:[port]/health` (Core services)

### Monitoring Commands
```bash
# Check all service health
docker ps --filter "name=sutazai" --format "table {{.Names}}\t{{.Status}}"

# Check network connections
docker network inspect sutazaiapp_sutazai-network

# Monitor resource usage
docker stats --no-stream $(docker ps --filter "name=sutazai" -q)
```

## Troubleshooting

### Common Issues and Solutions

#### IP Address Conflicts

**Problem**: Multiple containers assigned same IP
**Solution**:

1. Stop conflicting containers
2. Update docker-compose files with unique IPs
3. Recreate containers

#### Service Discovery Issues

**Problem**: Services cannot find each other
**Solution**:

1. Verify all services on same network
2. Use container names for internal communication
3. Check Consul service registry

#### Port Conflicts

**Problem**: Port already in use
**Solution**:

1. Check for conflicting processes: `netstat -tulpn | grep [port]`
2. Stop conflicting service
3. Update port mapping in docker-compose

## Maintenance Procedures

### Adding New Services

1. Assign IP from appropriate range
2. Add to relevant docker-compose file
3. Update this documentation
4. Test network connectivity

### Network Cleanup

```bash
# Remove orphaned containers
docker container prune

# Clean up unused networks
docker network prune

# Full cleanup (careful!)
docker system prune -a --volumes
```

## Performance Optimization

### Current Resource Allocations

- **Total Memory**: ~11GB allocated across all services
- **CPU**: ~15 CPU cores allocated
- **Disk**: Variable based on data volumes

### Optimization Opportunities

1. **Memory**: Reduce allocations for over-provisioned services
2. **CPU**: Implement CPU quotas to prevent resource contention
3. **Network**: Enable compression for inter-service communication
4. **Storage**: Implement volume pruning policies

## Future Roadmap

### Phase 1: Network Segmentation (Current Focus)

- [x] Document current architecture
- [x] Fix IP conflicts
- [ ] Implement network isolation
- [ ] Apply network policies

### Phase 2: Security Hardening

- [ ] Enable TLS between services
- [ ] Implement secrets management
- [ ] Add network intrusion detection

### Phase 3: High Availability

- [ ] Service replication
- [ ] Load balancing
- [ ] Failover mechanisms
- [ ] Distributed tracing

## Quick Reference

### Essential Commands

```bash
# View all SutazAI containers
docker ps --filter "name=sutazai"

# Check specific service logs
docker logs sutazai-[service] --tail 100 -f

# Restart a service
docker restart sutazai-[service]

# Scale a service (if using swarm mode)
docker service scale sutazai-[service]=3

# Network diagnostics
docker network inspect sutazaiapp_sutazai-network | jq '.Containers'
```

### Service URLs

- **Frontend**: <http://localhost:11000>
- **Backend API**: <http://localhost:10200>
- **Kong Admin**: <http://localhost:10009>
- **RabbitMQ Management**: <http://localhost:10005>
- **Neo4j Browser**: <http://localhost:10002>
- **Consul UI**: <http://localhost:10006>

## Contact and Support

For infrastructure issues or questions:

1. Check service logs first
2. Verify network connectivity
3. Review this documentation
4. Check container resource usage

Last Updated: 2025-08-29
Version: 1.0
