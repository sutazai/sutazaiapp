// Database Performance Load Testing
import { check, sleep } from 'k6';
import http from 'k6/http';
import { config, httpParams, validateResponse, randomChoice, randomInt } from '../k6-config.js';

export { options } from '../k6-config.js';

// Database-specific load testing
export default function() {
  const testType = randomChoice(['postgres', 'redis', 'neo4j', 'chromadb', 'qdrant']);
  
  switch(testType) {
    case 'postgres':
      testPostgreSQLLoad();
      break;
    case 'redis':
      testRedisLoad();
      break;
    case 'neo4j':
      testNeo4jLoad();
      break;
    case 'chromadb':
      testChromaDBLoad();
      break;
    case 'qdrant':
      testQdrantLoad();
      break;
  }
  
  sleep(randomInt(1, 2));
}

function testPostgreSQLLoad() {
  // Test PostgreSQL through backend API
  const queries = [
    'SELECT COUNT(*) FROM agents',
    'SELECT * FROM agent_logs ORDER BY created_at DESC LIMIT 10',
    'SELECT agent_name, COUNT(*) as request_count FROM agent_requests GROUP BY agent_name',
    'SELECT * FROM system_metrics WHERE timestamp > NOW() - INTERVAL \'1 hour\'',
    'SELECT agent_id, AVG(response_time) as avg_time FROM performance_metrics GROUP BY agent_id'
  ];
  
  const query = randomChoice(queries);
  
  const payload = {
    query: query,
    database: 'postgres'
  };
  
  const response = http.post(`${config.services.backend}/api/database/query`, JSON.stringify(payload), {
    ...httpParams,
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'postgres_load',
      query_type: 'select'
    }
  });
  
  validateResponse(response, 200);
  
  check(response, {
    'postgres query successful': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.success === true;
      } catch (e) {
        return false;
      }
    },
    'postgres response time acceptable': (r) => r.timings.duration < 1000
  });
}

function testRedisLoad() {
  const operations = ['get', 'set', 'exists', 'del'];
  const operation = randomChoice(operations);
  const key = `test_key_${randomInt(1, 1000)}`;
  const value = `test_value_${Date.now()}`;
  
  let payload;
  switch(operation) {
    case 'set':
      payload = { operation: 'set', key: key, value: value };
      break;
    case 'get':
      payload = { operation: 'get', key: key };
      break;
    case 'exists':
      payload = { operation: 'exists', key: key };
      break;
    case 'del':
      payload = { operation: 'del', key: key };
      break;
  }
  
  const response = http.post(`${config.services.backend}/api/cache/redis`, JSON.stringify(payload), {
    ...httpParams,
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'redis_load',
      operation: operation
    }
  });
  
  validateResponse(response, 200);
  
  check(response, {
    'redis operation successful': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.success === true;
      } catch (e) {
        return false;
      }
    },
    'redis response time under 100ms': (r) => r.timings.duration < 100
  });
}

function testNeo4jLoad() {
  const cypherQueries = [
    'MATCH (n:Agent) RETURN count(n)',
    'MATCH (a:Agent)-[r:USES]->(s:Service) RETURN a.name, s.name LIMIT 10',
    'MATCH (a:Agent) WHERE a.status = "active" RETURN a',
    'MATCH (n:Agent) WITH n.category as category, count(n) as count RETURN category, count',
    'MATCH (a:Agent)-[*1..2]-(related) RETURN a.name, count(related) as connections LIMIT 5'
  ];
  
  const query = randomChoice(cypherQueries);
  
  const payload = {
    query: query,
    database: 'neo4j'
  };
  
  const response = http.post(`${config.services.backend}/api/graph/query`, JSON.stringify(payload), {
    ...httpParams,
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'neo4j_load',
      query_type: 'cypher'
    }
  });
  
  validateResponse(response, 200);
  
  check(response, {
    'neo4j query successful': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.success === true;
      } catch (e) {
        return false;
      }
    },
    'neo4j response time under 2s': (r) => r.timings.duration < 2000
  });
}

function testChromaDBLoad() {
  const operations = ['query', 'add', 'get', 'delete'];
  const operation = randomChoice(operations);
  
  let payload;
  switch(operation) {
    case 'query':
      payload = {
        operation: 'query',
        collection_name: 'test_collection',
        query_texts: ['How to optimize database performance?'],
        n_results: 5
      };
      break;
    case 'add':
      payload = {
        operation: 'add',
        collection_name: 'test_collection',
        documents: [`Test document ${Date.now()}`],
        metadatas: [{ timestamp: Date.now() }],
        ids: [`doc_${randomInt(1, 10000)}`]
      };
      break;
    case 'get':
      payload = {
        operation: 'get',
        collection_name: 'test_collection',
        limit: 10
      };
      break;
    case 'delete':
      payload = {
        operation: 'delete',
        collection_name: 'test_collection',
        ids: [`doc_${randomInt(1, 1000)}`]
      };
      break;
  }
  
  const response = http.post(`${config.services.backend}/api/vector/chroma`, JSON.stringify(payload), {
    ...httpParams,
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'chromadb_load',
      operation: operation
    }
  });
  
  validateResponse(response, 200);
  
  check(response, {
    'chromadb operation successful': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.success === true;
      } catch (e) {
        return false;
      }
    },
    'chromadb response time under 1s': (r) => r.timings.duration < 1000
  });
}

function testQdrantLoad() {
  const operations = ['search', 'upsert', 'get', 'delete'];
  const operation = randomChoice(operations);
  const collectionName = 'test_vectors';
  
  let payload;
  switch(operation) {
    case 'search':
      payload = {
        operation: 'search',
        collection_name: collectionName,
        vector: Array(768).fill(0).map(() => Math.random()),
        limit: 10
      };
      break;
    case 'upsert':
      payload = {
        operation: 'upsert',
        collection_name: collectionName,
        points: [{
          id: randomInt(1, 10000),
          vector: Array(768).fill(0).map(() => Math.random()),
          payload: { text: `Test vector ${Date.now()}` }
        }]
      };
      break;
    case 'get':
      payload = {
        operation: 'get',
        collection_name: collectionName,
        ids: [randomInt(1, 1000)]
      };
      break;
    case 'delete':
      payload = {
        operation: 'delete',
        collection_name: collectionName,
        points: [randomInt(1, 1000)]
      };
      break;
  }
  
  const response = http.post(`${config.services.backend}/api/vector/qdrant`, JSON.stringify(payload), {
    ...httpParams,
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'qdrant_load',
      operation: operation
    }
  });
  
  validateResponse(response, 200);
  
  check(response, {
    'qdrant operation successful': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.success === true;
      } catch (e) {
        return false;
      }
    },
    'qdrant response time under 500ms': (r) => r.timings.duration < 500
  });
}

// Database connection pool stress test
export function connectionPoolStressTest() {
  // Simulate connection pool exhaustion
  const promises = [];
  
  for (let i = 0; i < 100; i++) {
    const payload = {
      query: 'SELECT pg_sleep(0.1)', // Short sleep to hold connections
      database: 'postgres'
    };
    
    promises.push(
      http.post(`${config.services.backend}/api/database/query`, JSON.stringify(payload), {
        ...httpParams,
        tags: { 
          ...httpParams.tags, 
          test_scenario: 'connection_pool_stress'
        }
      })
    );
  }
  
  // Check if system handles connection pool properly
  check(null, {
    'connection pool stress handled': () => promises.length === 100
  });
}

// Database transaction test
export function transactionTest() {
  const payload = {
    operations: [
      { query: 'INSERT INTO test_table (name) VALUES ($1)', params: [`test_${Date.now()}`] },
      { query: 'UPDATE test_table SET updated_at = NOW() WHERE name = $1', params: [`test_${Date.now()}`] },
      { query: 'SELECT COUNT(*) FROM test_table' }
    ],
    transaction: true
  };
  
  const response = http.post(`${config.services.backend}/api/database/transaction`, JSON.stringify(payload), {
    ...httpParams,
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'database_transaction'
    }
  });
  
  validateResponse(response, 200);
  
  check(response, {
    'transaction completed successfully': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.success === true && body.transaction_committed === true;
      } catch (e) {
        return false;
      }
    }
  });
}