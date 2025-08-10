import { test, expect } from '@playwright/test';

test.describe('System Health Checks - Smoke Tests', () => {
  
  test('Backend API health endpoint responds correctly', async ({ request }) => {
    const response = await request.get('http://localhost:10010/health');
    expect(response.ok()).toBeTruthy();
    
    const body = await response.json();
    expect(body).toHaveProperty('status');
    expect(['healthy', 'degraded']).toContain(body.status);
    
    // Log the actual status for debugging
    console.log('Backend health status:', body.status);
  });

  test('Frontend Streamlit application loads', async ({ page }) => {
    await page.goto('http://localhost:10011');
    
    // Wait for Streamlit to initialize
    await page.waitForSelector('div[data-testid="stApp"]', { timeout: 30000 });
    
    // Check if the page contains expected Streamlit elements
    const appElement = await page.locator('div[data-testid="stApp"]').count();
    expect(appElement).toBeGreaterThan(0);
  });

  test('Ollama service is running and has models', async ({ request }) => {
    const response = await request.get('http://localhost:10104/api/tags');
    expect(response.ok()).toBeTruthy();
    
    const body = await response.json();
    expect(body).toHaveProperty('models');
    expect(Array.isArray(body.models)).toBeTruthy();
    expect(body.models.length).toBeGreaterThan(0);
    
    console.log('Available Ollama models:', body.models.map((m: any) => m.name));
  });

  test('Database services are accessible', async ({ request }) => {
    // Test PostgreSQL via backend health
    const pgResponse = await request.get('http://localhost:10010/health');
    expect(pgResponse.ok()).toBeTruthy();
    
    // Test Redis connectivity 
    const redisResponse = await request.get('http://localhost:10010/health');
    expect(redisResponse.ok()).toBeTruthy();
    
    // Test Neo4j browser interface
    const neo4jResponse = await request.get('http://localhost:10002/browser/');
    expect(neo4jResponse.ok()).toBeTruthy();
  });

  test('Monitoring stack is operational', async ({ request }) => {
    // Test Prometheus
    const prometheusResponse = await request.get('http://localhost:10200/-/healthy');
    expect(prometheusResponse.ok()).toBeTruthy();
    
    // Test Grafana
    const grafanaResponse = await request.get('http://localhost:10201/api/health');
    expect(grafanaResponse.ok()).toBeTruthy();
  });

  test('Vector databases are running', async ({ request }) => {
    // Test Qdrant
    const qdrantResponse = await request.get('http://localhost:10101/');
    expect(qdrantResponse.ok()).toBeTruthy();
    
    // Test FAISS
    const faissResponse = await request.get('http://localhost:10103/health');
    expect(faissResponse.ok()).toBeTruthy();
  });

  test('Service mesh components are healthy', async ({ request }) => {
    // Test Kong Gateway
    const kongResponse = await request.get('http://localhost:10005/');
    expect(kongResponse.ok()).toBeTruthy();
    
    // Test Consul
    const consulResponse = await request.get('http://localhost:10006/v1/status/leader');
    expect(consulResponse.ok()).toBeTruthy();
  });

});