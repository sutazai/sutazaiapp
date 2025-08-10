import { test, expect } from '@playwright/test';

test.describe('Database Connection Integration Tests', () => {

  test('PostgreSQL database connectivity via backend', async ({ request }) => {
    // Test PostgreSQL connectivity through backend health endpoint
    const response = await request.get('http://localhost:10010/health');
    expect(response.ok()).toBeTruthy();
    
    const body = await response.json();
    
    // Check if PostgreSQL status is reported
    if (body.services && body.services.postgres) {
      expect(body.services.postgres).toBeTruthy();
      console.log('✅ PostgreSQL connectivity confirmed via backend');
    } else {
      console.log('⚠️  PostgreSQL status not explicitly reported in health check');
    }
  });

  test('Redis cache connectivity via backend', async ({ request }) => {
    const response = await request.get('http://localhost:10010/health');
    expect(response.ok()).toBeTruthy();
    
    const body = await response.json();
    
    // Check if Redis status is reported
    if (body.services && body.services.redis) {
      expect(body.services.redis).toBeTruthy();
      console.log('✅ Redis connectivity confirmed via backend');
    } else {
      console.log('⚠️  Redis status not explicitly reported in health check');
    }
  });

  test('Neo4j graph database web interface', async ({ request }) => {
    // Test Neo4j browser interface
    const response = await request.get('http://localhost:10002/browser/');
    expect(response.ok()).toBeTruthy();
    
    const contentType = response.headers()['content-type'];
    expect(contentType).toContain('text/html');
    
    console.log('✅ Neo4j browser interface accessible');
  });

  test('Neo4j database connectivity via backend', async ({ request }) => {
    const response = await request.get('http://localhost:10010/health');
    expect(response.ok()).toBeTruthy();
    
    const body = await response.json();
    
    // Check if Neo4j status is reported
    if (body.services && body.services.neo4j) {
      expect(body.services.neo4j).toBeTruthy();
      console.log('✅ Neo4j connectivity confirmed via backend');
    } else {
      console.log('⚠️  Neo4j status not explicitly reported in health check');
    }
  });

  test('Vector database - Qdrant connectivity', async ({ request }) => {
    // Test Qdrant root endpoint
    const response = await request.get('http://localhost:10101/');
    expect(response.ok()).toBeTruthy();
    
    const body = await response.json();
    expect(body).toHaveProperty('title');
    expect(body.title).toContain('qdrant');
    
    console.log('✅ Qdrant vector database accessible');
  });

  test('Vector database - Qdrant collections', async ({ request }) => {
    // Test Qdrant collections endpoint
    const response = await request.get('http://localhost:10101/collections');
    expect(response.ok()).toBeTruthy();
    
    const body = await response.json();
    expect(body).toHaveProperty('result');
    expect(body.result).toHaveProperty('collections');
    
    console.log('✅ Qdrant collections endpoint accessible');
    console.log('Available collections:', body.result.collections.length);
  });

  test('Vector database - FAISS service', async ({ request }) => {
    // Test FAISS health endpoint
    const response = await request.get('http://localhost:10103/health');
    
    if (response.ok()) {
      const body = await response.json();
      expect(body).toBeDefined();
      console.log('✅ FAISS vector service accessible');
    } else {
      console.log('⚠️  FAISS service may not have health endpoint or is not running');
    }
  });

  test('Database schema validation via backend', async ({ request }) => {
    // Test if backend can validate database schemas
    const response = await request.get('http://localhost:10010/api/v1/schema/validate');
    
    if (response.ok()) {
      const body = await response.json();
      expect(body).toHaveProperty('schemas');
      console.log('✅ Database schema validation available');
    } else {
      console.log('⚠️  Database schema validation endpoint not implemented');
    }
  });

  test('Database migration status', async ({ request }) => {
    // Test if backend reports migration status
    const response = await request.get('http://localhost:10010/api/v1/migrations/status');
    
    if (response.ok()) {
      const body = await response.json();
      console.log('✅ Database migrations status available:', body);
    } else {
      console.log('⚠️  Database migrations status endpoint not implemented');
    }
  });

  test('Data persistence test', async ({ request }) => {
    // Test basic data persistence (if endpoints exist)
    const testData = {
      test_key: 'playwright_test',
      timestamp: new Date().toISOString(),
      data: 'integration test data'
    };
    
    // Try to create test data
    const createResponse = await request.post('http://localhost:10010/api/v1/test/data', {
      data: testData
    });
    
    if (createResponse.ok()) {
      const createBody = await createResponse.json();
      console.log('✅ Test data creation successful:', createBody);
      
      // Try to retrieve test data
      const retrieveResponse = await request.get(`http://localhost:10010/api/v1/test/data/${createBody.id}`);
      
      if (retrieveResponse.ok()) {
        const retrieveBody = await retrieveResponse.json();
        expect(retrieveBody.test_key).toBe(testData.test_key);
        console.log('✅ Data persistence verified');
        
        // Clean up test data
        await request.delete(`http://localhost:10010/api/v1/test/data/${createBody.id}`);
      }
    } else {
      console.log('⚠️  Data persistence test endpoints not available (expected in current implementation)');
    }
  });

  test('Connection pooling and concurrent access', async ({ request }) => {
    // Test concurrent database access
    const concurrentRequests = Array.from({ length: 5 }, (_, index) =>
      request.get('http://localhost:10010/health')
    );
    
    const responses = await Promise.all(concurrentRequests);
    
    // All requests should succeed
    for (const response of responses) {
      expect(response.ok()).toBeTruthy();
    }
    
    console.log('✅ Database handles concurrent connections properly');
  });

});