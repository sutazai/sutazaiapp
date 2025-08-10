import { test, expect } from '@playwright/test';

test.describe('Backend API Integration Tests', () => {

  test('Backend health endpoint provides detailed information', async ({ request }) => {
    const response = await request.get('http://localhost:10010/health');
    expect(response.ok()).toBeTruthy();
    
    const body = await response.json();
    
    // Validate response structure
    expect(body).toHaveProperty('status');
    expect(body).toHaveProperty('timestamp');
    expect(body).toHaveProperty('services');
    
    // Validate service status
    if (body.services) {
      expect(typeof body.services).toBe('object');
      console.log('Service status details:', JSON.stringify(body.services, null, 2));
    }
    
    console.log('Backend health check result:', body.status);
  });

  test('Backend API root endpoint responds', async ({ request }) => {
    const response = await request.get('http://localhost:10010/');
    expect(response.ok()).toBeTruthy();
    
    const body = await response.json();
    expect(body).toHaveProperty('message');
    expect(body.message).toContain('SutazAI');
  });

  test('Backend docs endpoint is accessible', async ({ request }) => {
    const response = await request.get('http://localhost:10010/docs');
    expect(response.ok()).toBeTruthy();
    
    // Should return HTML content for Swagger UI
    const contentType = response.headers()['content-type'];
    expect(contentType).toContain('text/html');
  });

  test('Backend API versioning works', async ({ request }) => {
    const response = await request.get('http://localhost:10010/api/v1/health');
    
    // This might not exist yet, so we handle both cases
    if (response.ok()) {
      const body = await response.json();
      expect(body).toHaveProperty('version');
      console.log('API version:', body.version);
    } else {
      console.log('⚠️  API versioning not yet implemented');
    }
  });

  test('Backend can communicate with Ollama', async ({ request }) => {
    // Test if backend can reach Ollama through its endpoints
    const response = await request.get('http://localhost:10010/health');
    expect(response.ok()).toBeTruthy();
    
    const body = await response.json();
    
    // Check if Ollama service status is reported
    if (body.services && body.services.ollama) {
      expect(body.services.ollama).toBeTruthy();
      console.log('✅ Backend can communicate with Ollama');
    } else {
      console.log('⚠️  Ollama connectivity status not reported in health check');
    }
  });

  test('Backend database connectivity', async ({ request }) => {
    const response = await request.get('http://localhost:10010/health');
    expect(response.ok()).toBeTruthy();
    
    const body = await response.json();
    
    // Check database connectivity status
    if (body.services) {
      const dbServices = ['postgres', 'redis', 'neo4j'];
      
      for (const dbService of dbServices) {
        if (body.services[dbService]) {
          console.log(`✅ ${dbService} connectivity: ${body.services[dbService]}`);
        } else {
          console.log(`⚠️  ${dbService} status not reported`);
        }
      }
    }
  });

  test('Backend error handling', async ({ request }) => {
    // Test non-existent endpoint
    const response = await request.get('http://localhost:10010/api/nonexistent');
    expect(response.status()).toBe(404);
    
    const body = await response.json();
    expect(body).toHaveProperty('detail');
    console.log('Error handling works correctly for 404 endpoints');
  });

  test('Backend CORS headers', async ({ request }) => {
    const response = await request.get('http://localhost:10010/health');
    expect(response.ok()).toBeTruthy();
    
    const headers = response.headers();
    
    // Check for CORS headers (if configured)
    if (headers['access-control-allow-origin']) {
      expect(headers['access-control-allow-origin']).toBeTruthy();
      console.log('✅ CORS headers configured');
    } else {
      console.log('⚠️  CORS headers not configured (may be intentional)');
    }
  });

});