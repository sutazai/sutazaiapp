# Unified Memory Service Architecture
**Phase 2 Consolidation: Extended-Memory + Memory-Bank-MCP → Unified Memory Service**

## 🎯 Architecture Overview

### **Current State Analysis**
```yaml
Extended-Memory Service:
  Technology: Node.js (Python backend)
  Port: 3009
  Container: sutazai-mcp-nodejs
  Storage: SQLite + File system
  Features:
    - Project namespacing
    - Context tagging
    - TTL support
    - SQLite persistence
    - Popular tags management
    - Session context handling

Memory-Bank-MCP Service:
  Technology: Python
  Port: 4002  
  Container: sutazai-mcp-python
  Storage: ChromaDB/Vector database
  Features:
    - Vector-based memory storage
    - Semantic search capabilities
    - Memory banking operations
    - Advanced retrieval algorithms
```

### **Consolidation Decision: Keep Extended-Memory as Primary**

**Rationale:**
1. **Superior Architecture**: Dedicated venv, better isolation
2. **Proven Stability**: Already integrated and working
3. **Better Namespacing**: Project-based organization
4. **Comprehensive Features**: Tags, TTL, session management
5. **Performance**: Direct SQLite operations, faster response times

## 🏗️ Unified Memory Service Design

### **Container Architecture**
```dockerfile
# Unified Memory Service Container
FROM python:3.11-slim AS memory-unified

# Install both Python and Node.js for hybrid approach
RUN apt-get update && apt-get install -y \
    nodejs npm \
    sqlite3 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Extended Memory (Primary)
COPY extended-memory/ /opt/memory/extended/
RUN cd /opt/memory/extended && pip install -r requirements.txt

# Memory Bank features integration
COPY memory-bank-features/ /opt/memory/features/
RUN cd /opt/memory/features && pip install chromadb faiss-cpu

# Unified wrapper
COPY unified-memory-service.py /opt/memory/
COPY unified-memory-wrapper.sh /opt/mcp/wrappers/

EXPOSE 3009
CMD ["/opt/mcp/wrappers/unified-memory-wrapper.sh"]
```

### **Service Configuration**
```yaml
# Docker Compose Configuration
mcp-unified-memory:
  image: sutazai-mcp-unified-memory:latest
  container_name: mcp-unified-memory
  environment:
    - MCP_SERVICE=unified-memory
    - PYTHONPATH=/opt/memory
    - MCP_HOST=0.0.0.0
    - MCP_PORT=3009
    - MEMORY_BACKENDS=extended,vector
    - SQLITE_PATH=/var/lib/mcp/memory.db
    - VECTOR_STORE=chromadb
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
    test: ["CMD", "/opt/mcp/wrappers/unified-memory-wrapper.sh", "health"]
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
```

## 🔄 Unified API Design

### **Enhanced Memory API**
```python
# Unified Memory Service API
class UnifiedMemoryService:
    def __init__(self):
        self.extended_memory = ExtendedMemoryCore()
        self.vector_store = ChromaDBVectorStore()
        self.sqlite_store = SQLiteStore()
    
    # Core Operations (Extended Memory Based)
    async def store_memory(self, key: str, content: str, **kwargs):
        """Store memory with enhanced features from both services"""
        # Primary storage (Extended Memory)
        result = await self.extended_memory.store_context(
            key=key,
            content=content,
            namespace=kwargs.get('namespace', 'default'),
            tags=kwargs.get('tags', []),
            ttl=kwargs.get('ttl'),
            importance_level=kwargs.get('importance_level', 5)
        )
        
        # Vector storage for semantic search (Memory Bank features)
        if kwargs.get('enable_semantic_search', False):
            await self.vector_store.add_document(
                doc_id=result['context_id'],
                content=content,
                metadata={'namespace': kwargs.get('namespace'), 'tags': kwargs.get('tags')}
            )
        
        return result
    
    async def retrieve_memory(self, key: str, **kwargs):
        """Retrieve memory with fallback to vector search"""
        # Try exact key match first (Extended Memory)
        try:
            return await self.extended_memory.load_context(key, **kwargs)
        except KeyError:
            # Fallback to semantic search (Memory Bank features)
            if kwargs.get('semantic_search', True):
                return await self.semantic_search(key, limit=1)
            raise
    
    async def search_memory(self, query: str, **kwargs):
        """Enhanced search combining exact and semantic matching"""
        results = []
        
        # Extended Memory tag/content search
        extended_results = await self.extended_memory.search_contexts(
            pattern=query,
            namespace=kwargs.get('namespace'),
            limit=kwargs.get('limit', 10)
        )
        results.extend(extended_results)
        
        # Vector-based semantic search (Memory Bank features)
        if kwargs.get('include_semantic', True):
            semantic_results = await self.vector_store.similarity_search(
                query=query,
                limit=kwargs.get('semantic_limit', 5),
                threshold=kwargs.get('threshold', 0.7)
            )
            results.extend(semantic_results)
        
        return self._deduplicate_results(results)
    
    async def delete_memory(self, key: str, **kwargs):
        """Delete from both storage backends"""
        await self.extended_memory.forget_context(key)
        await self.vector_store.delete_document(key)
    
    # Enhanced Features
    async def get_popular_tags(self, **kwargs):
        """Enhanced tag management from Extended Memory"""
        return await self.extended_memory.get_popular_tags(**kwargs)
    
    async def memory_analytics(self, **kwargs):
        """Combined analytics from both backends"""
        extended_stats = await self.extended_memory.get_memory_stats()
        vector_stats = await self.vector_store.get_collection_stats()
        
        return {
            'total_contexts': extended_stats['total_contexts'],
            'total_projects': extended_stats['total_projects'],
            'vector_documents': vector_stats['document_count'],
            'storage_usage': {
                'sqlite_size_mb': extended_stats['db_size_mb'],
                'vector_size_mb': vector_stats['collection_size_mb']
            },
            'performance_metrics': await self._get_performance_metrics()
        }
```

## 📊 Migration Strategy

### **Phase 1: Data Export**
```bash
#!/bin/bash
# Export existing data from both services

echo "Exporting Extended Memory data..."
/opt/mcp/wrappers/extended-memory.sh export-all > /tmp/extended-memory-export.json

echo "Exporting Memory Bank data..."
/opt/mcp/wrappers/memory-bank-mcp.sh export-all > /tmp/memory-bank-export.json

echo "Creating unified migration file..."
python3 /opt/memory/scripts/create-migration-file.py
```

### **Phase 2: Unified Service Deployment**
```yaml
Migration Steps:
  1. Deploy unified service alongside existing services
  2. Import Extended Memory data (primary)
  3. Import Memory Bank data (vector features)
  4. Run parallel validation tests
  5. Switch API routing to unified service
  6. Monitor for 48 hours
  7. Decommission old services
```

### **Phase 3: Backend Integration**
```python
# Updated backend routing
@router.post("/api/v1/mcp/memory/{operation}")
async def unified_memory_operation(operation: str, request_data: dict):
    """Route all memory operations to unified service"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"http://mcp-unified-memory:3009/memory/{operation}",
            json=request_data
        )
        return response.json()
```

## 🔍 Performance Optimizations

### **Storage Optimization**
```python
# Intelligent storage routing
class StorageRouter:
    def route_operation(self, operation_type: str, data_size: int, access_pattern: str):
        """Route to optimal storage backend"""
        if operation_type == "store":
            if data_size > 1024 * 1024:  # >1MB
                return "file_system"
            elif access_pattern == "frequent":
                return "sqlite"
            else:
                return "sqlite"
        
        elif operation_type == "search":
            if "semantic" in access_pattern:
                return "vector_store"
            else:
                return "sqlite"
```

### **Caching Layer**
```python
# Redis caching for hot data
class MemoryCacheLayer:
    def __init__(self):
        self.redis_client = redis.Redis(host='redis', port=6379, db=2)
        self.cache_ttl = 300  # 5 minutes
    
    async def get_cached(self, key: str):
        """Get from cache with fallback to storage"""
        cached = await self.redis_client.get(f"memory:{key}")
        if cached:
            return json.loads(cached)
        
        # Fallback to unified storage
        result = await self.unified_memory.retrieve_memory(key)
        await self.redis_client.setex(f"memory:{key}", self.cache_ttl, json.dumps(result))
        return result
```

## 📈 Expected Benefits

### **Performance Improvements**
```yaml
Response Time:
  Current Extended-Memory: ~45ms average
  Current Memory-Bank: ~120ms average
  Unified Service Target: ~35ms average (20% improvement)

Memory Usage:
  Current Combined: ~768MB (512MB + 256MB)
  Unified Service: ~384MB (50% reduction)

Storage Efficiency:
  Deduplicated data storage
  Unified indexing
  Optimized vector operations
```

### **Feature Enhancements**
```yaml
New Capabilities:
  - Hybrid storage (SQL + Vector)
  - Semantic search across all memories
  - Unified tagging and namespacing
  - Enhanced analytics and insights
  - Intelligent storage routing
  - Performance-optimized caching

Maintained Features:
  - All Extended Memory features (primary)
  - Vector search capabilities (Memory Bank)
  - Project isolation
  - TTL and lifecycle management
```

## 🛡️ Risk Mitigation

### **Rollback Procedures**
```bash
#!/bin/bash
# Automated rollback script
echo "ROLLBACK: Unified Memory Service"

# Stop unified service
docker-compose stop mcp-unified-memory

# Restart original services
docker-compose up -d mcp-extended-memory mcp-memory-bank-mcp

# Restore API routing
/opt/scripts/restore-memory-api-routing.sh

# Validate original services
/opt/scripts/validate-memory-services.sh

echo "Rollback complete"
```

### **Data Integrity Validation**
```python
# Continuous validation during migration
async def validate_data_integrity():
    """Validate data consistency between old and new services"""
    test_cases = [
        {"operation": "store", "key": "test_key", "content": "test_content"},
        {"operation": "retrieve", "key": "test_key"},
        {"operation": "search", "query": "test"},
        {"operation": "delete", "key": "test_key"}
    ]
    
    for test in test_cases:
        old_result = await call_old_service(test)
        new_result = await call_unified_service(test)
        assert old_result == new_result, f"Data mismatch in {test['operation']}"
```

## 🎯 Implementation Timeline

### **Week 1: Development**
- Build unified memory service container
- Implement unified API with feature parity
- Create migration scripts
- Set up testing framework

### **Week 2: Testing & Validation**
- Deploy unified service in staging
- Run migration tests
- Performance benchmarking
- Integration testing

### **Week 3: Production Deployment**
- Deploy unified service in production
- Migrate data with zero downtime
- Switch API routing
- Monitor performance

### **Week 4: Cleanup & Optimization**
- Decommission old services
- Optimize performance based on metrics
- Update documentation
- Team training

---

**Architecture Status:** Ready for Implementation  
**Expected Outcome:** 18 → 17 services (6% additional reduction)  
**Performance Improvement:** 20% faster response times, 50% memory reduction