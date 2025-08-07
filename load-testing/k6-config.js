// K6 Load Testing Configuration for SutazAI
import { check, sleep } from 'k6';
import http from 'k6/http';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics
export const errorRate = new Rate('errors');
export const responseTime = new Trend('response_time');
export const throughput = new Counter('requests_total');

// Test configuration
export const options = {
  scenarios: {
    // Light load baseline
    baseline: {
      executor: 'constant-vus',
      vus: 10,
      duration: '2m',
      tags: { test_type: 'baseline' },
    },
    // Ramp up to normal load
    ramp_up: {
      executor: 'ramping-vus',
      startVUs: 10,
      stages: [
        { duration: '2m', target: 50 },
        { duration: '5m', target: 100 },
        { duration: '5m', target: 200 },
        { duration: '2m', target: 0 },
      ],
      tags: { test_type: 'ramp_up' },
    },
    // Sustained load
    sustained: {
      executor: 'constant-vus',
      vus: 100,
      duration: '10m',
      tags: { test_type: 'sustained' },
    },
    // Spike testing
    spike: {
      executor: 'ramping-vus',
      startVUs: 50,
      stages: [
        { duration: '1m', target: 50 },
        { duration: '30s', target: 500 }, // Spike
        { duration: '1m', target: 50 },
      ],
      tags: { test_type: 'spike' },
    },
    // Stress testing
    stress: {
      executor: 'ramping-vus',
      startVUs: 100,
      stages: [
        { duration: '2m', target: 200 },
        { duration: '2m', target: 400 },
        { duration: '2m', target: 600 },
        { duration: '2m', target: 800 },
        { duration: '2m', target: 1000 },
        { duration: '5m', target: 1000 },
        { duration: '2m', target: 0 },
      ],
      tags: { test_type: 'stress' },
    }
  },
  thresholds: {
    // Global thresholds
    http_req_duration: ['p(95)<2000', 'p(99)<5000'], // 95% of requests under 2s, 99% under 5s
    http_req_failed: ['rate<0.01'], // Error rate under 1%
    errors: ['rate<0.01'],
    
    // Per-scenario thresholds
    'http_req_duration{test_type:baseline}': ['p(95)<1000'],
    'http_req_duration{test_type:sustained}': ['p(95)<2000'],
    'http_req_duration{test_type:spike}': ['p(95)<3000'],
    'http_req_duration{test_type:stress}': ['p(95)<5000'],
  },
  // Output configuration
  summaryTrendStats: ['avg', 'min', 'med', 'max', 'p(90)', 'p(95)', 'p(99)'],
};

// SutazAI system endpoints
export const config = {
  baseUrl: __ENV.BASE_URL || 'http://localhost',
  
  // Core services
  services: {
    backend: `${__ENV.BASE_URL || 'http://localhost'}:8000`,
    frontend: `${__ENV.BASE_URL || 'http://localhost'}:8501`,
    ollama: `${__ENV.BASE_URL || 'http://localhost'}:10104`,
    postgres: `${__ENV.BASE_URL || 'http://localhost'}:10000`,
    redis: `${__ENV.BASE_URL || 'http://localhost'}:10001`,
    neo4j: `${__ENV.BASE_URL || 'http://localhost'}:10002`,
    chromadb: `${__ENV.BASE_URL || 'http://localhost'}:10100`,
    qdrant: `${__ENV.BASE_URL || 'http://localhost'}:10101`,
    prometheus: `${__ENV.BASE_URL || 'http://localhost'}:10200`,
    grafana: `${__ENV.BASE_URL || 'http://localhost'}:10201`,
  },
  
  // Agent endpoints (sample - will be dynamically generated)
  agents: {
    'ai-system-architect': 8080,
    'ai-senior-engineer': 8081,
    'ai-qa-team-lead': 8082,
    'testing-qa-validator': 8083,
    'deployment-automation-master': 8084,
    'infrastructure-devops-manager': 8085,
    'ollama-integration-specialist': 8086,
    'security-pentesting-specialist': 8087,
    'senior-backend-developer': 8088,
    'senior-frontend-developer': 8089,
  },
  
  // Test authentication
  auth: {
    token: __ENV.AUTH_TOKEN || 'test-token',
    username: __ENV.USERNAME || 'test-user',
    password: __ENV.PASSWORD || 'test-password',
  },
  
  // Test parameters
  testData: {
    samplePrompts: [
      'Analyze the current system architecture',
      'Generate a Python FastAPI endpoint',
      'Create comprehensive unit tests',
      'Deploy the application to production',
      'Optimize database performance',
      'Implement security best practices',
      'Design a microservices architecture',
      'Create monitoring dashboards',
      'Implement CI/CD pipeline',
      'Analyze code quality metrics'
    ],
    
    testPayloads: {
      small: JSON.stringify({ message: 'Hello', size: 'small' }),
      medium: JSON.stringify({ 
        message: 'Medium test payload with more data',
        data: Array(100).fill('test-data'),
        timestamp: Date.now()
      }),
      large: JSON.stringify({
        message: 'Large test payload for stress testing',
        data: Array(1000).fill('comprehensive-test-data-entry'),
        metadata: {
          testType: 'load-testing',
          timestamp: Date.now(),
          system: 'sutazai',
          version: '2.0'
        }
      })
    }
  }
};

// Common HTTP parameters
export const httpParams = {
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${config.auth.token}`,
    'User-Agent': 'K6-LoadTest/1.0',
  },
  timeout: '30s',
  tags: { name: 'sutazai-load-test' }
};

// Utility functions
export function randomChoice(array) {
  return array[Math.floor(Math.random() * array.length)];
}

export function randomInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

export function validateResponse(response, expectedStatus = 200) {
  const isValid = check(response, {
    [`status is ${expectedStatus}`]: (r) => r.status === expectedStatus,
    'response time < 5s': (r) => r.timings.duration < 5000,
    'response has body': (r) => r.body && r.body.length > 0,
  });
  
  errorRate.add(!isValid);
  responseTime.add(response.timings.duration);
  throughput.add(1);
  
  return isValid;
}