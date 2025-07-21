// K6 Performance Test Script for SutazAI
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const apiResponseTime = new Trend('api_response_time');
const brainThinkTime = new Trend('brain_think_time');
const authTime = new Trend('auth_time');

// Test configuration
export const options = {
  stages: [
    { duration: '2m', target: 10 },   // Ramp up to 10 users
    { duration: '5m', target: 50 },   // Ramp up to 50 users
    { duration: '10m', target: 100 }, // Stay at 100 users
    { duration: '5m', target: 50 },   // Ramp down to 50 users
    { duration: '2m', target: 0 },    // Ramp down to 0 users
  ],
  thresholds: {
    'http_req_duration': ['p(95)<500', 'p(99)<1000'], // 95% of requests under 500ms
    'errors': ['rate<0.1'], // Error rate under 10%
    'http_req_failed': ['rate<0.1'], // HTTP failure rate under 10%
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
let authToken = null;

// Setup - run once per VU
export function setup() {
  // Login to get auth token
  const loginRes = http.post(`${BASE_URL}/api/v1/security/login`, 
    JSON.stringify({
      username: 'admin',
      password: 'password'
    }),
    {
      headers: { 'Content-Type': 'application/json' },
    }
  );
  
  if (loginRes.status === 200) {
    const data = JSON.parse(loginRes.body);
    return { token: data.access_token };
  }
  
  console.error('Failed to authenticate during setup');
  return { token: null };
}

// Main test scenario
export default function(data) {
  const token = data.token;
  const headers = {
    'Content-Type': 'application/json',
    'Authorization': token ? `Bearer ${token}` : '',
  };

  // Test 1: Health Check
  const healthRes = http.get(`${BASE_URL}/health`);
  check(healthRes, {
    'health check status is 200': (r) => r.status === 200,
    'health check response time < 100ms': (r) => r.timings.duration < 100,
  });
  errorRate.add(healthRes.status !== 200);
  
  sleep(1);

  // Test 2: System Status
  const statusRes = http.get(`${BASE_URL}/api/v1/system/status`);
  check(statusRes, {
    'system status is 200': (r) => r.status === 200,
    'system is operational': (r) => {
      const body = JSON.parse(r.body);
      return body.status === 'operational';
    },
  });
  apiResponseTime.add(statusRes.timings.duration);
  errorRate.add(statusRes.status !== 200);
  
  sleep(1);

  // Test 3: AGI Brain Think (most important endpoint)
  const thinkPayload = {
    input_data: {
      text: `Performance test query ${__VU} - ${__ITER}`,
    },
    reasoning_type: 'deductive',
  };
  
  const thinkRes = http.post(
    `${BASE_URL}/api/v1/brain/think`,
    JSON.stringify(thinkPayload),
    { headers }
  );
  
  check(thinkRes, {
    'brain think status is 200': (r) => r.status === 200,
    'brain returns thought_id': (r) => {
      if (r.status !== 200) return false;
      const body = JSON.parse(r.body);
      return body.thought_id !== undefined;
    },
    'brain response time < 1000ms': (r) => r.timings.duration < 1000,
  });
  
  brainThinkTime.add(thinkRes.timings.duration);
  errorRate.add(thinkRes.status !== 200);
  
  sleep(2);

  // Test 4: Model List
  const modelsRes = http.get(`${BASE_URL}/api/v1/models/`, { headers });
  check(modelsRes, {
    'models list status is 200': (r) => r.status === 200,
    'models response contains array': (r) => {
      if (r.status !== 200) return false;
      const body = JSON.parse(r.body);
      return Array.isArray(body.models);
    },
  });
  apiResponseTime.add(modelsRes.timings.duration);
  errorRate.add(modelsRes.status !== 200);
  
  sleep(1);

  // Test 5: Agent Status
  const agentsRes = http.get(`${BASE_URL}/api/v1/agents/status`, { headers });
  check(agentsRes, {
    'agents status is 200': (r) => r.status === 200,
  });
  apiResponseTime.add(agentsRes.timings.duration);
  errorRate.add(agentsRes.status !== 200);
  
  sleep(1);

  // Test 6: Vector Search
  const searchPayload = {
    query: 'test query for performance testing',
    limit: 10,
  };
  
  const searchRes = http.post(
    `${BASE_URL}/api/v1/vectors/search`,
    JSON.stringify(searchPayload),
    { headers }
  );
  
  check(searchRes, {
    'vector search status is 200': (r) => r.status === 200,
    'vector search response time < 200ms': (r) => r.timings.duration < 200,
  });
  apiResponseTime.add(searchRes.timings.duration);
  errorRate.add(searchRes.status !== 200);
  
  sleep(1);

  // Test 7: Concurrent Brain Requests (stress test)
  const batch = [];
  for (let i = 0; i < 5; i++) {
    batch.push([
      'POST',
      `${BASE_URL}/api/v1/brain/think`,
      JSON.stringify({
        input_data: { text: `Batch request ${i}` },
        reasoning_type: 'creative',
      }),
      { headers }
    ]);
  }
  
  const batchRes = http.batch(batch);
  const batchErrors = batchRes.filter(r => r.status !== 200).length;
  errorRate.add(batchErrors > 0);
  
  sleep(3);

  // Test 8: Authentication Refresh
  if (Math.random() < 0.1) { // 10% of iterations test auth
    const authRes = http.post(
      `${BASE_URL}/api/v1/security/login`,
      JSON.stringify({
        username: 'admin',
        password: 'password',
      }),
      { headers: { 'Content-Type': 'application/json' } }
    );
    
    check(authRes, {
      'auth status is 200': (r) => r.status === 200,
      'auth returns tokens': (r) => {
        if (r.status !== 200) return false;
        const body = JSON.parse(r.body);
        return body.access_token && body.refresh_token;
      },
    });
    authTime.add(authRes.timings.duration);
    errorRate.add(authRes.status !== 200);
  }
  
  sleep(1);
}

// Teardown - run once after all VUs finish
export function teardown(data) {
  // Could perform cleanup here if needed
  console.log('Performance test completed');
}

// Custom summary
export function handleSummary(data) {
  return {
    'performance-results/summary.json': JSON.stringify(data),
    'performance-results/summary.txt': textSummary(data, { indent: ' ', enableColors: false }),
    stdout: textSummary(data, { indent: ' ', enableColors: true }),
  };
}

function textSummary(data, options) {
  let summary = '\n=== Performance Test Results ===\n\n';
  
  // Key metrics
  const metrics = data.metrics;
  if (metrics) {
    summary += 'Response Times:\n';
    summary += `  API Average: ${metrics.api_response_time?.avg?.toFixed(2) || 'N/A'} ms\n`;
    summary += `  Brain Think Average: ${metrics.brain_think_time?.avg?.toFixed(2) || 'N/A'} ms\n`;
    summary += `  Auth Average: ${metrics.auth_time?.avg?.toFixed(2) || 'N/A'} ms\n\n`;
    
    summary += 'Error Rates:\n';
    summary += `  Overall Error Rate: ${(metrics.errors?.rate * 100)?.toFixed(2) || '0'} %\n`;
    summary += `  HTTP Failures: ${(metrics.http_req_failed?.rate * 100)?.toFixed(2) || '0'} %\n\n`;
    
    summary += 'Throughput:\n';
    summary += `  Requests/sec: ${metrics.http_reqs?.rate?.toFixed(2) || 'N/A'}\n`;
    summary += `  Data Received: ${(metrics.data_received?.rate / 1024)?.toFixed(2) || 'N/A'} KB/s\n`;
    summary += `  Data Sent: ${(metrics.data_sent?.rate / 1024)?.toFixed(2) || 'N/A'} KB/s\n\n`;
  }
  
  // Threshold results
  summary += 'Threshold Results:\n';
  const thresholds = data.thresholds || {};
  for (const [metric, result] of Object.entries(thresholds)) {
    summary += `  ${metric}: ${result.ok ? '✓ PASS' : '✗ FAIL'}\n`;
  }
  
  return summary;
}