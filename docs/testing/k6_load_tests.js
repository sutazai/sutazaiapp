// K6 Load Testing Suite for Jarvis Endpoints
// File: load-tests/jarvis-load-test.js

import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Counter, Rate, Trend, Gauge } from 'k6/metrics';
import { htmlReport } from 'https://raw.githubusercontent.com/benc-uk/k6-reporter/main/dist/bundle.js';

// Custom metrics
const jarvisResponseTime = new Trend('jarvis_response_time');
const jarvisErrorRate = new Rate('jarvis_error_rate');
const jarvisRequestsPerSecond = new Rate('jarvis_rps');
const activeChatSessions = new Gauge('active_chat_sessions');
const concurrentVoiceRequests = new Counter('concurrent_voice_requests');
const knowledgeQueryLatency = new Trend('knowledge_query_latency');

// Test configuration
export const options = {
  scenarios: {
    // Scenario 1: Baseline load test
    baseline_load: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 20 },   // Ramp up to 20 users
        { duration: '5m', target: 20 },   // Stay at 20 users
        { duration: '2m', target: 50 },   // Ramp up to 50 users
        { duration: '5m', target: 50 },   // Stay at 50 users
        { duration: '2m', target: 0 },    // Ramp down
      ],
      gracefulRampDown: '30s',
      tags: { test_type: 'baseline' },
    },

    // Scenario 2: Stress test for peak load
    stress_test: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 50 },   // Ramp up
        { duration: '5m', target: 100 },  // Target 100 concurrent users
        { duration: '3m', target: 150 },  // Peak load
        { duration: '2m', target: 100 },  // Scale down
        { duration: '2m', target: 0 },    // Ramp down
      ],
      gracefulRampDown: '30s',
      tags: { test_type: 'stress' },
    },

    // Scenario 3: Spike test for sudden traffic bursts
    spike_test: {
      executor: 'ramping-vus',
      startVUs: 10,
      stages: [
        { duration: '1m', target: 10 },   // Normal load
        { duration: '30s', target: 200 }, // Spike to 200 users
        { duration: '1m', target: 200 },  // Maintain spike
        { duration: '30s', target: 10 },  // Quick recovery
        { duration: '2m', target: 10 },   // Recovery period
      ],
      tags: { test_type: 'spike' },
    },

    // Scenario 4: Soak test for stability
    soak_test: {
      executor: 'constant-vus',
      vus: 30,
      duration: '30m', // Extended duration for stability testing
      tags: { test_type: 'soak' },
    },
  },

  thresholds: {
    // Overall performance thresholds
    http_req_duration: ['p(95)<2000', 'p(99)<5000'], // 95% under 2s, 99% under 5s
    http_req_failed: ['rate<0.05'], // Error rate under 5%
    http_reqs: ['rate>10'], // Minimum 10 requests per second
    
    // Jarvis-specific thresholds
    jarvis_response_time: ['p(90)<3000', 'p(95)<5000'], // Jarvis responses
    jarvis_error_rate: ['rate<0.02'], // Jarvis error rate under 2%
    knowledge_query_latency: ['p(95)<4000'], // Knowledge queries under 4s
  },
};

// Configuration
const BASE_URL = __ENV.BASE_URL || 'http://localhost:10010';
const JARVIS_VOICE_URL = __ENV.JARVIS_VOICE_URL || 'http://localhost:11150';
const JARVIS_KNOWLEDGE_URL = __ENV.JARVIS_KNOWLEDGE_URL || 'http://localhost:11101';
const JARVIS_AUTOMATION_URL = __ENV.JARVIS_AUTOMATION_URL || 'http://localhost:11102';
const JARVIS_MULTIMODAL_URL = __ENV.JARVIS_MULTIMODAL_URL || 'http://localhost:11103';
const JARVIS_HARDWARE_URL = __ENV.JARVIS_HARDWARE_URL || 'http://localhost:11104';

// Test data
const chatMessages = [
  'Hello Jarvis, what can you help me with?',
  'Can you analyze system performance?',
  'What are the current system metrics?',
  'Help me optimize resource usage',
  'Explain machine learning concepts',
  'How do neural networks work?',
  'What is the current system status?',
  'Can you process this data for insights?',
  'Generate a performance report',
  'Analyze these log files',
];

const reasoningQueries = [
  'How can I improve system efficiency?',
  'What patterns do you see in this data?',
  'Analyze the relationship between CPU and memory usage',
  'Predict future resource requirements',
  'Identify optimization opportunities',
];

const taskDescriptions = [
  'Monitor system health and generate alerts',
  'Analyze log files for error patterns',
  'Optimize database query performance', 
  'Generate automated reports',
  'Process batch data analysis',
];

// Utility functions
function getRandomElement(array) {
  return array[Math.floor(Math.random() * array.length)];
}

function generateRandomUserId() {
  return `user_${Math.floor(Math.random() * 10000)}`;
}

// Test setup
export function setup() {
  console.log('Starting Jarvis Load Testing Suite');
  console.log(`Base URL: ${BASE_URL}`);
  console.log(`Test scenarios: ${Object.keys(options.scenarios).join(', ')}`);
  
  // Health check before starting tests
  const healthCheck = http.get(`${BASE_URL}/health`);
  if (!check(healthCheck, { 'health check passed': (r) => r.status === 200 })) {
    throw new Error('Health check failed - system may not be ready');
  }
  
  return {
    startTime: new Date().toISOString(),
    testConfig: options,
  };
}

// Main test function
export default function (data) {
  const userId = generateRandomUserId();
  const testType = __ENV.TEST_TYPE || 'mixed';
  
  // Update active sessions metric
  activeChatSessions.add(1);

  group('Backend API Tests', function () {
    // Test 1: Health Check
    group('Health Check', function () {
      const response = http.get(`${BASE_URL}/health`);
      
      check(response, {
        'health check status is 200': (r) => r.status === 200,
        'health check response time < 1000ms': (r) => r.timings.duration < 1000,
        'health check has status field': (r) => JSON.parse(r.body).status !== undefined,
      });
      
      jarvisRequestsPerSecond.add(1);
      jarvisResponseTime.add(response.timings.duration);
    });

    // Test 2: Chat API Load Test
    group('Chat API Load Test', function () {
      const message = getRandomElement(chatMessages);
      const payload = JSON.stringify({
        message: message,
        agent: 'task_coordinator',
        model: 'tinyllama',
        temperature: 0.7
      });
      
      const params = {
        headers: { 'Content-Type': 'application/json' },
        tags: { endpoint: 'chat', user_id: userId },
      };
      
      const response = http.post(`${BASE_URL}/chat`, payload, params);
      
      const responseTime = response.timings.duration;
      jarvisResponseTime.add(responseTime);
      jarvisRequestsPerSecond.add(1);
      
      const success = check(response, {
        'chat status is 200': (r) => r.status === 200,
        'chat response time < 5000ms': (r) => r.timings.duration < 5000,
        'chat has response field': (r) => {
          try {
            const body = JSON.parse(r.body);
            return body.response && body.response.length > 0;
          } catch {
            return false;
          }
        },
        'chat has model info': (r) => {
          try {
            const body = JSON.parse(r.body);
            return body.model !== undefined;
          } catch {
            return false;
          }
        },
      });
      
      if (!success) {
        jarvisErrorRate.add(1);
      }
    });

    // Test 3: Thinking API Load Test
    group('Thinking API Load Test', function () {
      const query = getRandomElement(reasoningQueries);
      const payload = JSON.stringify({
        query: query,
        reasoning_type: 'analytical'
      });
      
      const params = {
        headers: { 'Content-Type': 'application/json' },
        tags: { endpoint: 'think', user_id: userId },
      };
      
      const response = http.post(`${BASE_URL}/think`, payload, params);
      
      jarvisResponseTime.add(response.timings.duration);
      jarvisRequestsPerSecond.add(1);
      
      const success = check(response, {
        'think status is 200': (r) => r.status === 200,
        'think response time < 8000ms': (r) => r.timings.duration < 8000,
        'think has thought field': (r) => {
          try {
            const body = JSON.parse(r.body);
            return body.thought && body.thought.length > 0;
          } catch {
            return false;
          }
        },
        'think has confidence': (r) => {
          try {
            const body = JSON.parse(r.body);
            return body.confidence !== undefined;
          } catch {
            return false;
          }
        },
      });
      
      if (!success) {
        jarvisErrorRate.add(1);
      }
    });

    // Test 4: Task Execution Load Test
    group('Task Execution Load Test', function () {
      const description = getRandomElement(taskDescriptions);
      const payload = JSON.stringify({
        description: description,
        type: 'analysis'
      });
      
      const params = {
        headers: { 'Content-Type': 'application/json' },
        tags: { endpoint: 'execute', user_id: userId },
      };
      
      const response = http.post(`${BASE_URL}/execute`, payload, params);
      
      jarvisResponseTime.add(response.timings.duration);
      jarvisRequestsPerSecond.add(1);
      
      const success = check(response, {
        'execute status is 200': (r) => r.status === 200,
        'execute response time < 10000ms': (r) => r.timings.duration < 10000,
        'execute has result': (r) => {
          try {
            const body = JSON.parse(r.body);
            return body.result && body.result.length > 0;
          } catch {
            return false;
          }
        },
        'execute has task_id': (r) => {
          try {
            const body = JSON.parse(r.body);
            return body.task_id !== undefined;
          } catch {
            return false;
          }
        },
      });
      
      if (!success) {
        jarvisErrorRate.add(1);
      }
    });
  });

  group('Jarvis Service Tests', function () {
    // Test 5: Voice Interface Service
    group('Voice Interface Service', function () {
      const response = http.get(`${JARVIS_VOICE_URL}/health`, {
        tags: { service: 'jarvis-voice', user_id: userId },
      });
      
      concurrentVoiceRequests.add(1);
      
      check(response, {
        'voice service is healthy': (r) => r.status === 200,
        'voice service response time < 2000ms': (r) => r.timings.duration < 2000,
      });
    });

    // Test 6: Knowledge Management Service
    group('Knowledge Management Service', function () {
      const startTime = new Date().getTime();
      const response = http.get(`${JARVIS_KNOWLEDGE_URL}/health`, {
        tags: { service: 'jarvis-knowledge', user_id: userId },
      });
      
      const queryLatency = new Date().getTime() - startTime;
      knowledgeQueryLatency.add(queryLatency);
      
      check(response, {
        'knowledge service is healthy': (r) => r.status === 200,
        'knowledge service response time < 3000ms': (r) => r.timings.duration < 3000,
      });
    });

    // Test 7: Automation Agent Service
    group('Automation Agent Service', function () {
      const response = http.get(`${JARVIS_AUTOMATION_URL}/health`, {
        tags: { service: 'jarvis-automation', user_id: userId },
      });
      
      check(response, {
        'automation service is healthy': (r) => r.status === 200,
        'automation service response time < 2000ms': (r) => r.timings.duration < 2000,
      });
    });

    // Test 8: Multimodal AI Service
    group('Multimodal AI Service', function () {
      const response = http.get(`${JARVIS_MULTIMODAL_URL}/health`, {
        tags: { service: 'jarvis-multimodal', user_id: userId },
      });
      
      check(response, {
        'multimodal service is healthy': (r) => r.status === 200,
        'multimodal service response time < 2000ms': (r) => r.timings.duration < 2000,
      });
    });

    // Test 9: Hardware Resource Optimizer
    group('Hardware Resource Optimizer', function () {
      const response = http.get(`${JARVIS_HARDWARE_URL}/health`, {
        tags: { service: 'jarvis-hardware', user_id: userId },
      });
      
      check(response, {
        'hardware service is healthy': (r) => r.status === 200,
        'hardware service response time < 2000ms': (r) => r.timings.duration < 2000,
      });
    });
  });

  group('Error Handling Tests', function () {
    // Test 10: Invalid Request Handling
    group('Invalid Request Handling', function () {
      const invalidPayload = JSON.stringify({
        message: '', // Empty message should trigger validation error
        agent: 'invalid_agent'
      });
      
      const params = {
        headers: { 'Content-Type': 'application/json' },
        tags: { endpoint: 'chat_invalid', user_id: userId },
      };
      
      const response = http.post(`${BASE_URL}/chat`, invalidPayload, params);
      
      check(response, {
        'invalid request returns 400 or 422': (r) => r.status === 400 || r.status === 422,
        'invalid request response time < 1000ms': (r) => r.timings.duration < 1000,
        'invalid request has error detail': (r) => {
          try {
            const body = JSON.parse(r.body);
            return body.detail !== undefined;
          } catch {
            return false;
          }
        },
      });
    });

    // Test 11: Large Payload Handling
    group('Large Payload Handling', function () {
      const largeMessage = 'This is a large message test. '.repeat(1000); // ~30KB message
      const payload = JSON.stringify({
        message: largeMessage,
        agent: 'task_coordinator'
      });
      
      const params = {
        headers: { 'Content-Type': 'application/json' },
        tags: { endpoint: 'chat_large', user_id: userId },
      };
      
      const response = http.post(`${BASE_URL}/chat`, payload, params);
      
      check(response, {
        'large payload status is 200': (r) => r.status === 200,
        'large payload response time < 15000ms': (r) => r.timings.duration < 15000,
        'large payload has response': (r) => {
          try {
            const body = JSON.parse(r.body);
            return body.response && body.response.length > 0;
          } catch {
            return false;
          }
        },
      });
    });
  });

  // Cleanup
  activeChatSessions.add(-1);
  
  // Vary sleep time based on test scenario
  const sleepTime = Math.random() * 3 + 1; // 1-4 seconds
  sleep(sleepTime);
}

// Test teardown
export function teardown(data) {
  console.log('Load testing completed');
  console.log(`Test started at: ${data.startTime}`);
  console.log(`Test completed at: ${new Date().toISOString()}`);
}

// Generate HTML report
export function handleSummary(data) {
  return {
    'load-test-report.html': htmlReport(data),
    'load-test-results.json': JSON.stringify(data, null, 2),
  };
}

// Custom scenarios for specific testing needs
export const customScenarios = {
  // Voice interface specific load test
  voice_load: {
    executor: 'ramping-vus',
    startVUs: 0,
    stages: [
      { duration: '1m', target: 10 },
      { duration: '3m', target: 25 },
      { duration: '1m', target: 0 },
    ],
    exec: 'voiceOnlyTest',
  },
  
  // Knowledge query intensive test
  knowledge_intensive: {
    executor: 'constant-vus',
    vus: 20,
    duration: '5m',
    exec: 'knowledgeOnlyTest',
  },
};

// Specialized test functions
export function voiceOnlyTest() {
  const response = http.get(`${JARVIS_VOICE_URL}/health`);
  check(response, {
    'voice service available': (r) => r.status === 200,
  });
  concurrentVoiceRequests.add(1);
  sleep(2);
}

export function knowledgeOnlyTest() {
  const startTime = new Date().getTime();
  const response = http.get(`${JARVIS_KNOWLEDGE_URL}/health`);
  const queryLatency = new Date().getTime() - startTime;
  
  knowledgeQueryLatency.add(queryLatency);
  check(response, {
    'knowledge service available': (r) => r.status === 200,
  });
  sleep(1);
}