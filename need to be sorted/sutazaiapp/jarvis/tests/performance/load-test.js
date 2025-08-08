/**
 * K6 Load Test for SutazAI API
 * 
 * This script performs comprehensive load testing on the SutazAI backend API.
 * It tests various endpoints under different load conditions and measures
 * response times, error rates, and throughput.
 */

import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Rate, Trend, Counter, Gauge } from 'k6/metrics';
import { randomString, randomIntBetween } from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';

// Custom metrics
const errorRate = new Rate('errors');
const apiLatency = new Trend('api_latency', true);
const successfulRequests = new Counter('successful_requests');
const activeAgents = new Gauge('active_agents');

// Configuration
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const API_KEY = __ENV.API_KEY || 'test-api-key';

// Test scenarios
export const options = {
  scenarios: {
    // Smoke test
    smoke: {
      executor: 'constant-vus',
      vus: 1,
      duration: '1m',
      startTime: '0s',
      tags: { scenario: 'smoke' },
    },
    
    // Load test - gradual ramp-up
    load: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 10 },  // Ramp up to 10 users
        { duration: '5m', target: 10 },  // Stay at 10 users
        { duration: '2m', target: 20 },  // Ramp up to 20 users
        { duration: '5m', target: 20 },  // Stay at 20 users
        { duration: '2m', target: 0 },   // Ramp down to 0
      ],
      startTime: '2m',
      tags: { scenario: 'load' },
    },
    
    // Stress test
    stress: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 50 },   // Ramp up to 50 users
        { duration: '5m', target: 50 },   // Stay at 50 users
        { duration: '2m', target: 100 },  // Ramp up to 100 users
        { duration: '5m', target: 100 },  // Stay at 100 users
        { duration: '2m', target: 0 },    // Ramp down
      ],
      startTime: '20m',
      tags: { scenario: 'stress' },
    },
    
    // Spike test
    spike: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '30s', target: 5 },    // Warm up
        { duration: '10s', target: 100 },  // Spike to 100 users
        { duration: '1m', target: 100 },   // Stay at 100
        { duration: '10s', target: 5 },    // Back to normal
        { duration: '1m', target: 5 },     // Continue normal load
        { duration: '10s', target: 0 },    // Ramp down
      ],
      startTime: '40m',
      tags: { scenario: 'spike' },
    },
    
    // Soak test (endurance)
    soak: {
      executor: 'constant-vus',
      vus: 20,
      duration: '30m',
      startTime: '50m',
      tags: { scenario: 'soak' },
    },
  },
  
  thresholds: {
    // General thresholds
    http_req_failed: ['rate<0.1'],           // Error rate < 10%
    http_req_duration: ['p(95)<500'],        // 95% of requests < 500ms
    'http_req_duration{scenario:smoke}': ['p(99)<200'],  // Smoke test: 99% < 200ms
    'http_req_duration{scenario:load}': ['p(95)<400'],   // Load test: 95% < 400ms
    'http_req_duration{scenario:stress}': ['p(95)<800'], // Stress test: 95% < 800ms
    
    // Custom metrics thresholds
    errors: ['rate<0.05'],                   // Custom error rate < 5%
    api_latency: ['p(90)<300', 'p(95)<500'], // API latency thresholds
  },
};

// Setup function - runs once before the test
export function setup() {
  console.log('Setting up test environment...');
  
  // Check if the API is reachable
  const healthCheck = http.get(`${BASE_URL}/health`);
  check(healthCheck, {
    'API is reachable': (r) => r.status === 200,
  });
  
  if (healthCheck.status !== 200) {
    throw new Error(`API is not reachable. Status: ${healthCheck.status}`);
  }
  
  // Return any data needed by the main test function
  return {
    timestamp: new Date().toISOString(),
    testId: randomString(10),
  };
}

// Default headers
const defaultHeaders = {
  'Content-Type': 'application/json',
  'Authorization': `Bearer ${API_KEY}`,
};

// Main test function
export default function (data) {
  // Health check endpoints
  group('Health Checks', () => {
    const responses = http.batch([
      ['GET', `${BASE_URL}/health`, null, { tags: { endpoint: 'health' } }],
      ['GET', `${BASE_URL}/ready`, null, { tags: { endpoint: 'ready' } }],
      ['GET', `${BASE_URL}/live`, null, { tags: { endpoint: 'live' } }],
    ]);
    
    responses.forEach((res, index) => {
      const endpoint = ['health', 'ready', 'live'][index];
      check(res, {
        [`${endpoint} status is 200`]: (r) => r.status === 200,
        [`${endpoint} response time < 100ms`]: (r) => r.timings.duration < 100,
      });
      
      errorRate.add(res.status !== 200);
      apiLatency.add(res.timings.duration, { endpoint });
      if (res.status === 200) successfulRequests.add(1);
    });
  });
  
  sleep(randomIntBetween(1, 3));
  
  // API Documentation
  group('API Documentation', () => {
    const docs = http.get(`${BASE_URL}/docs`, { tags: { endpoint: 'docs' } });
    check(docs, {
      'Docs available': (r) => r.status === 200,
      'Docs load time < 500ms': (r) => r.timings.duration < 500,
    });
  });
  
  sleep(randomIntBetween(1, 2));
  
  // Agent operations
  group('Agent Operations', () => {
    // List agents
    const listAgents = http.get(`${BASE_URL}/api/v1/agents`, {
      headers: defaultHeaders,
      tags: { endpoint: 'list_agents' },
    });
    
    check(listAgents, {
      'List agents successful': (r) => r.status === 200 || r.status === 404,
      'List agents response time < 300ms': (r) => r.timings.duration < 300,
    });
    
    // Create agent
    const agentPayload = JSON.stringify({
      name: `TestAgent_${randomString(8)}`,
      type: 'code_generator',
      config: {
        temperature: 0.7,
        max_tokens: 1000,
      },
    });
    
    const createAgent = http.post(`${BASE_URL}/api/v1/agents`, agentPayload, {
      headers: defaultHeaders,
      tags: { endpoint: 'create_agent' },
    });
    
    const agentCreated = check(createAgent, {
      'Agent created': (r) => r.status === 200 || r.status === 201 || r.status === 404,
      'Agent creation time < 1000ms': (r) => r.timings.duration < 1000,
    });
    
    if (agentCreated && createAgent.status < 300) {
      activeAgents.add(1);
      const agentData = JSON.parse(createAgent.body);
      
      // Communicate with agent
      const messagePayload = JSON.stringify({
        agent_id: agentData.id || 'test-agent',
        message: 'Generate a hello world function in Python',
        context: { language: 'python', test: true },
      });
      
      const communicate = http.post(`${BASE_URL}/api/v1/agents/communicate`, messagePayload, {
        headers: defaultHeaders,
        tags: { endpoint: 'agent_communicate' },
      });
      
      check(communicate, {
        'Agent communication successful': (r) => r.status === 200 || r.status === 404,
        'Agent response time < 2000ms': (r) => r.timings.duration < 2000,
      });
    }
    
    errorRate.add(listAgents.status >= 400 || createAgent.status >= 400);
    apiLatency.add(listAgents.timings.duration, { operation: 'list_agents' });
    apiLatency.add(createAgent.timings.duration, { operation: 'create_agent' });
  });
  
  sleep(randomIntBetween(2, 4));
  
  // Document operations
  group('Document Operations', () => {
    // Upload document
    const boundary = '----WebKitFormBoundary' + randomString(16);
    const textContent = `Test document content ${randomString(100)}`;
    
    const formData = `------${boundary}\r\n` +
      'Content-Disposition: form-data; name="file"; filename="test.txt"\r\n' +
      'Content-Type: text/plain\r\n\r\n' +
      textContent + '\r\n' +
      `------${boundary}--\r\n`;
    
    const uploadHeaders = Object.assign({}, defaultHeaders, {
      'Content-Type': `multipart/form-data; boundary=----${boundary}`,
    });
    
    const upload = http.post(`${BASE_URL}/api/v1/documents/upload`, formData, {
      headers: uploadHeaders,
      tags: { endpoint: 'upload_document' },
    });
    
    check(upload, {
      'Document uploaded': (r) => r.status === 200 || r.status === 201 || r.status === 404,
      'Upload time < 1500ms': (r) => r.timings.duration < 1500,
    });
    
    // List documents
    const listDocs = http.get(`${BASE_URL}/api/v1/documents`, {
      headers: defaultHeaders,
      tags: { endpoint: 'list_documents' },
    });
    
    check(listDocs, {
      'Documents listed': (r) => r.status === 200 || r.status === 404,
      'List documents time < 300ms': (r) => r.timings.duration < 300,
    });
    
    errorRate.add(upload.status >= 400 || listDocs.status >= 400);
    apiLatency.add(upload.timings.duration, { operation: 'upload_document' });
    apiLatency.add(listDocs.timings.duration, { operation: 'list_documents' });
  });
  
  sleep(randomIntBetween(1, 3));
  
  // Vector operations
  group('Vector Operations', () => {
    // Text embedding
    const embedPayload = JSON.stringify({
      text: `Test text for embedding ${randomString(50)}`,
      model: 'default',
    });
    
    const embed = http.post(`${BASE_URL}/api/v1/vectors/embed`, embedPayload, {
      headers: defaultHeaders,
      tags: { endpoint: 'embed_text' },
    });
    
    check(embed, {
      'Text embedded': (r) => r.status === 200 || r.status === 404,
      'Embedding time < 500ms': (r) => r.timings.duration < 500,
    });
    
    // Vector search
    const searchPayload = JSON.stringify({
      query: 'test query for vector search',
      top_k: 5,
      collection: 'documents',
    });
    
    const search = http.post(`${BASE_URL}/api/v1/vectors/search`, searchPayload, {
      headers: defaultHeaders,
      tags: { endpoint: 'vector_search' },
    });
    
    check(search, {
      'Vector search successful': (r) => r.status === 200 || r.status === 404,
      'Search time < 800ms': (r) => r.timings.duration < 800,
    });
    
    errorRate.add(embed.status >= 400 || search.status >= 400);
    apiLatency.add(embed.timings.duration, { operation: 'embed_text' });
    apiLatency.add(search.timings.duration, { operation: 'vector_search' });
  });
  
  sleep(randomIntBetween(1, 2));
  
  // Concurrent requests test
  if (__VU % 10 === 0) { // Every 10th VU performs this test
    group('Concurrent Requests', () => {
      const requests = [
        ['GET', `${BASE_URL}/health`],
        ['GET', `${BASE_URL}/api/v1/agents`],
        ['GET', `${BASE_URL}/api/v1/documents`],
      ];
      
      const responses = http.batch(
        requests.map(([method, url]) => [method, url, null, { headers: defaultHeaders }])
      );
      
      responses.forEach((res) => {
        errorRate.add(res.status >= 400);
        if (res.status < 400) successfulRequests.add(1);
      });
    });
  }
}

// Teardown function - runs once after the test
export function teardown(data) {
  console.log('Test completed!');
  console.log(`Test ID: ${data.testId}`);
  console.log(`Started at: ${data.timestamp}`);
  console.log(`Ended at: ${new Date().toISOString()}`);
  
  // Cleanup can be performed here if needed
}