// Breaking Point and Stress Testing for SutazAI
// Identifies system limits and failure points under extreme load

import { check, sleep } from 'k6';
import http from 'k6/http';
import { config, httpParams, validateResponse, randomChoice, randomInt } from '../k6-config.js';

// Progressive load configuration for breaking point identification
export const options = {
  scenarios: {
    // Progressive stress test - gradually increase load until system breaks
    progressive_stress: {
      executor: 'ramping-arrival-rate',
      startRate: 10,
      timeUnit: '1s',
      preAllocatedVUs: 100,
      maxVUs: 2000,
      stages: [
        { duration: '2m', target: 50 },    // Warm up
        { duration: '3m', target: 200 },   // Normal load
        { duration: '3m', target: 500 },   // High load
        { duration: '3m', target: 1000 },  // Stress load  
        { duration: '3m', target: 2000 },  // Breaking point
        { duration: '3m', target: 3000 },  // Beyond breaking point
        { duration: '2m', target: 0 },     // Cool down
      ],
      tags: { test_type: 'progressive_stress' },
    },
    
    // Spike testing - sudden load increase
    spike_test: {
      executor: 'ramping-vus',
      startVUs: 100,
      stages: [
        { duration: '2m', target: 100 },   // Baseline
        { duration: '30s', target: 1500 }, // Sudden spike
        { duration: '1m', target: 1500 },  // Sustain spike
        { duration: '30s', target: 100 },  // Return to baseline
        { duration: '1m', target: 100 },   // Stabilize
      ],
      tags: { test_type: 'spike_test' },
    },
    
    // Soak testing - sustained load over time
    soak_test: {
      executor: 'constant-vus',
      vus: 300,
      duration: '30m',
      tags: { test_type: 'soak_test' },
    },
    
    // Volume testing - large data payloads
    volume_test: {
      executor: 'constant-vus',
      vus: 50,
      duration: '10m',
      tags: { test_type: 'volume_test' },
    }
  },
  
  thresholds: {
    // Breaking point identification thresholds
    http_req_duration: [
      { threshold: 'p(95)<5000', abortOnFail: false },
      { threshold: 'p(99)<10000', abortOnFail: false }
    ],
    http_req_failed: [
      { threshold: 'rate<0.1', abortOnFail: false },  // Allow up to 10% failures during stress
    ],
    
    // Scenario-specific thresholds
    'http_req_duration{test_type:progressive_stress}': ['p(95)<8000'],
    'http_req_duration{test_type:spike_test}': ['p(95)<10000'],
    'http_req_duration{test_type:soak_test}': ['p(95)<3000'],
    'http_req_failed{test_type:progressive_stress}': ['rate<0.2'],
  }
};

// Global variables to track breaking points
let systemBreakingPoint = {
  cpu_exhaustion: null,
  memory_exhaustion: null,
  connection_limit: null,
  database_saturation: null,
  agent_overload: null
};

export default function() {
  const testType = __ENV.K6_SCENARIO || 'progressive_stress';
  
  switch(testType) {
    case 'progressive_stress':
      progressiveStressTest();
      break;
    case 'spike_test':
      spikeLoadTest();
      break;
    case 'soak_test':
      soakTest();
      break;
    case 'volume_test':
      volumeTest();
      break;
    default:
      progressiveStressTest();
  }
}

function progressiveStressTest() {
  // Test different system components to find breaking points
  const stressTargets = [
    'agent_overload',
    'database_stress',
    'api_gateway_limit',
    'memory_pressure',
    'connection_exhaustion'
  ];
  
  const target = randomChoice(stressTargets);
  
  switch(target) {
    case 'agent_overload':
      testAgentOverload();
      break;
    case 'database_stress':
      testDatabaseStress();
      break;
    case 'api_gateway_limit':
      testAPIGatewayLimits();
      break;
    case 'memory_pressure':
      testMemoryPressure();
      break;
    case 'connection_exhaustion':
      testConnectionExhaustion();
      break;
  }
  
  sleep(randomInt(1, 3));
}

function testAgentOverload() {
  // Overload specific agents to find their breaking points
  const highLoadAgents = [
    'ai-system-architect',
    'ai-senior-backend-developer',
    'ai-qa-team-lead',
    'security-pentesting-specialist'
  ];
  
  const agent = randomChoice(highLoadAgents);
  const port = config.agents[agent] || 8080;
  
  // Send complex, resource-intensive requests
  const complexPrompt = `
    Perform comprehensive analysis of the following requirements:
    1. Design a distributed system architecture for 1 million concurrent users
    2. Create detailed implementation plan with code examples
    3. Identify all potential bottlenecks and solutions
    4. Generate comprehensive test strategy
    5. Create deployment and scaling recommendations
    6. Provide security analysis and recommendations
    7. Create monitoring and alerting strategy
    8. Design disaster recovery procedures
    
    Provide detailed, production-ready solutions with code examples, 
    configuration files, and step-by-step implementation guides.
  `;
  
  const payload = {
    prompt: complexPrompt,
    max_tokens: 4000,
    temperature: 0.5,
    stream: false,
    complexity: 'maximum',
    stress_test: true
  };
  
  const response = http.post(`${config.baseUrl}:${port}/api/chat`, JSON.stringify(payload), {
    ...httpParams,
    timeout: '120s', // Extended timeout for complex requests
    tags: { 
      ...httpParams.tags, 
      stress_target: 'agent_overload',
      agent: agent,
      complexity: 'maximum'
    }
  });
  
  // Check for overload indicators
  check(response, {
    'agent handles complex request': (r) => r.status === 200,
    'agent response within timeout': (r) => r.timings.duration < 120000,
    'no out-of-memory errors': (r) => !r.body.includes('OutOfMemoryError'),
    'no timeout errors': (r) => r.status !== 408 && r.status !== 504
  });
  
  // Record breaking point if detected
  if (response.status === 503 || response.status === 429 || response.timings.duration > 60000) {
    systemBreakingPoint.agent_overload = {
      agent: agent,
      timestamp: Date.now(),
      response_time: response.timings.duration,
      status: response.status
    };
  }
}

function testDatabaseStress() {
  // Stress test database connections and query performance
  const dbOperations = [
    'complex_join_query',
    'bulk_insert',
    'concurrent_updates',
    'transaction_stress',
    'connection_pool_exhaustion'
  ];
  
  const operation = randomChoice(dbOperations);
  
  let payload;
  switch(operation) {
    case 'complex_join_query':
      payload = {
        query: `
          SELECT a.*, u.*, i.*, m.* 
          FROM agents a 
          JOIN users u ON a.created_by = u.id 
          JOIN interactions i ON u.id = i.user_id 
          JOIN metrics m ON a.id = m.agent_id 
          WHERE i.created_at > NOW() - INTERVAL '1 day'
          ORDER BY m.response_time DESC 
          LIMIT 1000
        `,
        timeout: 30000
      };
      break;
    case 'bulk_insert':
      payload = {
        operation: 'bulk_insert',
        table: 'stress_test_data',
        rows: Array(1000).fill(0).map((_, i) => ({
          id: `stress_${Date.now()}_${i}`,
          data: JSON.stringify({ test: true, index: i, timestamp: Date.now() }),
          large_text: 'x'.repeat(10000) // 10KB text field
        }))
      };
      break;
    case 'concurrent_updates':
      payload = {
        operation: 'concurrent_update',
        table: 'agent_metrics',
        updates: Array(100).fill(0).map(() => ({
          id: randomInt(1, 1000),
          field: 'last_updated',
          value: Date.now()
        }))
      };
      break;
    case 'transaction_stress':
      payload = {
        operation: 'transaction',
        statements: Array(50).fill(0).map((_, i) => ({
          query: 'INSERT INTO stress_transactions (id, data) VALUES ($1, $2)',
          params: [`tx_${Date.now()}_${i}`, JSON.stringify({ index: i })]
        }))
      };
      break;
    case 'connection_pool_exhaustion':
      payload = {
        operation: 'hold_connection',
        duration: 30000, // Hold connection for 30 seconds
        query: 'SELECT pg_sleep(30)'
      };
      break;
  }
  
  const response = http.post(`${config.services.backend}/api/database/stress-test`, JSON.stringify(payload), {
    ...httpParams,
    timeout: '60s',
    tags: { 
      ...httpParams.tags, 
      stress_target: 'database_stress',
      operation: operation
    }
  });
  
  check(response, {
    'database handles stress operation': (r) => r.status === 200,
    'database responds within timeout': (r) => r.timings.duration < 60000,
    'no connection pool exhaustion': (r) => !r.body.includes('connection pool exhausted'),
    'no deadlock errors': (r) => !r.body.includes('deadlock')
  });
  
  // Record database breaking point
  if (response.status === 503 || response.body.includes('timeout') || response.body.includes('exhausted')) {
    systemBreakingPoint.database_saturation = {
      operation: operation,
      timestamp: Date.now(),
      response_time: response.timings.duration,
      error: response.body.substring(0, 200)
    };
  }
}

function testAPIGatewayLimits() {
  // Test API gateway rate limiting and throughput limits
  const gatewayEndpoints = [
    '/api/agents/list',
    '/api/system/status',
    '/api/user/dashboard',
    '/api/monitoring/metrics'
  ];
  
  const endpoint = randomChoice(gatewayEndpoints);
  
  // Send burst of requests to trigger rate limiting
  const requests = [];
  for (let i = 0; i < 100; i++) {
    requests.push(
      http.get(`${config.services.backend}${endpoint}`, {
        ...httpParams,
        tags: { 
          ...httpParams.tags, 
          stress_target: 'api_gateway_limit',
          endpoint: endpoint,
          burst_index: i
        }
      })
    );
  }
  
  // Analyze rate limiting behavior
  const successCount = requests.filter(r => r.status === 200).length;
  const rateLimitedCount = requests.filter(r => r.status === 429).length;
  const errorCount = requests.filter(r => r.status >= 500).length;
  
  check(null, {
    'some requests succeed in burst': () => successCount > 0,
    'rate limiting kicks in': () => rateLimitedCount > 0,
    'no server errors during rate limiting': () => errorCount < requests.length * 0.1
  });
  
  // Record API gateway limits
  if (errorCount > requests.length * 0.5) {
    systemBreakingPoint.connection_limit = {
      endpoint: endpoint,
      success_rate: successCount / requests.length,
      error_rate: errorCount / requests.length,
      timestamp: Date.now()
    };
  }
}

function testMemoryPressure() {
  // Create memory pressure by requesting large responses
  const memoryIntensiveAgents = [
    'document-knowledge-manager',
    'ai-system-architect',
    'code-generation-improver'
  ];
  
  const agent = randomChoice(memoryIntensiveAgents);
  const port = config.agents[agent] || 8080;
  
  const memoryStressPrompt = `
    Generate comprehensive documentation for a large-scale system including:
    ${Array(100).fill(0).map((_, i) => `${i + 1}. Detailed section about component ${i + 1} with examples and code`).join('\n')}
    
    Include detailed code examples, configuration files, API specifications,
    database schemas, deployment scripts, monitoring configurations,
    security policies, and troubleshooting guides for each component.
  `;
  
  const payload = {
    prompt: memoryStressPrompt,
    max_tokens: 8000, // Request large response
    temperature: 0.3,
    memory_stress: true
  };
  
  const response = http.post(`${config.baseUrl}:${port}/api/chat`, JSON.stringify(payload), {
    ...httpParams,
    timeout: '180s',
    tags: { 
      ...httpParams.tags, 
      stress_target: 'memory_pressure',
      agent: agent
    }
  });
  
  check(response, {
    'handles memory-intensive request': (r) => r.status === 200,
    'no out-of-memory errors': (r) => !r.body.includes('OutOfMemoryError') && !r.body.includes('memory limit'),
    'response within memory timeout': (r) => r.timings.duration < 180000
  });
  
  if (response.status === 503 || response.body.includes('memory')) {
    systemBreakingPoint.memory_exhaustion = {
      agent: agent,
      timestamp: Date.now(),
      response_size: response.body.length,
      error: response.body.substring(0, 100)
    };
  }
}

function testConnectionExhaustion() {
  // Test connection pool limits
  const connectionTests = [];
  
  // Create many simultaneous connections
  for (let i = 0; i < 200; i++) {
    connectionTests.push(
      http.get(`${config.services.backend}/api/health?connection_test=${i}`, {
        ...httpParams,
        timeout: '30s',
        tags: { 
          ...httpParams.tags, 
          stress_target: 'connection_exhaustion',
          connection_index: i
        }
      })
    );
  }
  
  const successfulConnections = connectionTests.filter(r => r.status === 200).length;
  const failedConnections = connectionTests.filter(r => r.status !== 200).length;
  
  check(null, {
    'handles multiple connections': () => successfulConnections > 150,
    'connection failures indicate limits': () => failedConnections > 0 && failedConnections < 100
  });
  
  if (failedConnections > successfulConnections) {
    systemBreakingPoint.connection_limit = {
      max_successful_connections: successfulConnections,
      failed_connections: failedConnections,
      timestamp: Date.now()
    };
  }
}

function spikeLoadTest() {
  // Test system behavior during sudden load spikes
  const spikeDuration = 60; // 1 minute spike
  const spikeIntensity = randomChoice(['medium', 'high', 'extreme']);
  
  let requestCount;
  switch(spikeIntensity) {
    case 'medium': requestCount = 10; break;
    case 'high': requestCount = 20; break;
    case 'extreme': requestCount = 50; break;
  }
  
  // Generate spike load
  const spikeRequests = [];
  for (let i = 0; i < requestCount; i++) {
    const agent = randomChoice(Object.keys(config.agents));
    const port = config.agents[agent];
    
    spikeRequests.push(
      http.post(`${config.baseUrl}:${port}/api/chat`, JSON.stringify({
        prompt: `Spike test request ${i} - quick response needed`,
        max_tokens: 100,
        spike_test: true
      }), {
        ...httpParams,
        timeout: '30s',
        tags: { 
          ...httpParams.tags, 
          test_type: 'spike_load',
          intensity: spikeIntensity,
          spike_index: i
        }
      })
    );
  }
  
  // Analyze spike handling
  const spikeSuccessRate = spikeRequests.filter(r => r.status === 200).length / spikeRequests.length;
  const avgResponseTime = spikeRequests.reduce((sum, r) => sum + r.timings.duration, 0) / spikeRequests.length;
  
  check(null, {
    'system handles spike load': () => spikeSuccessRate > 0.7, // 70% success during spike
    'spike response time reasonable': () => avgResponseTime < 15000, // Under 15 seconds average
    'no complete system failure': () => spikeSuccessRate > 0.1 // At least 10% success
  });
}

function soakTest() {
  // Long-running test to identify memory leaks and degradation
  const soakAgent = randomChoice(Object.keys(config.agents));
  const port = config.agents[soakAgent];
  
  const payload = {
    prompt: 'Soak test - sustained load testing for memory leak detection',
    max_tokens: 500,
    temperature: 0.5,
    soak_test: true
  };
  
  const response = http.post(`${config.baseUrl}:${port}/api/chat`, JSON.stringify(payload), {
    ...httpParams,
    timeout: '60s',
    tags: { 
      ...httpParams.tags, 
      test_type: 'soak_test',
      agent: soakAgent
    }
  });
  
  validateResponse(response, 200);
  
  // Check for performance degradation over time
  check(response, {
    'soak test response stable': (r) => r.status === 200,
    'no memory leak indicators': (r) => r.timings.duration < 10000,
    'consistent performance': (r) => !r.body.includes('timeout') && !r.body.includes('memory')
  });
}

function volumeTest() {
  // Test with large data volumes
  const largeDataPayload = {
    prompt: 'Process this large dataset and provide analysis',
    data: Array(1000).fill(0).map((_, i) => ({
      id: i,
      timestamp: Date.now(),
      data: `Large data entry ${i} with substantial content: ${'x'.repeat(1000)}`
    })),
    processing_instructions: Array(50).fill('Detailed analysis required'),
    max_tokens: 2000
  };
  
  const response = http.post(`${config.services.backend}/api/agents/data-analysis-processor`, JSON.stringify(largeDataPayload), {
    ...httpParams,
    timeout: '120s',
    tags: { 
      ...httpParams.tags, 
      test_type: 'volume_test',
      payload_size: 'large'
    }
  });
  
  check(response, {
    'handles large data volume': (r) => r.status === 200,
    'processes within timeout': (r) => r.timings.duration < 120000,
    'no data processing errors': (r) => !r.body.includes('processing error')
  });
}

// Export breaking point summary
export function handleSummary(data) {
  // Generate breaking point report
  const breakingPointReport = {
    timestamp: new Date().toISOString(),
    test_duration: data.state.testRunDurationMs,
    breaking_points: systemBreakingPoint,
    performance_metrics: {
      max_response_time: Math.max(...Object.values(data.metrics.http_req_duration.values)),
      avg_response_time: data.metrics.http_req_duration.avg,
      error_rate: data.metrics.http_req_failed.rate,
      total_requests: data.metrics.http_reqs.count
    },
    recommendations: generateOptimizationRecommendations(systemBreakingPoint, data.metrics)
  };
  
  return {
    '/opt/sutazaiapp/load-testing/reports/breaking_point_report.json': JSON.stringify(breakingPointReport, null, 2),
    '/opt/sutazaiapp/load-testing/reports/breaking_point_summary.txt': generateTextSummary(breakingPointReport)
  };
}

function generateOptimizationRecommendations(breakingPoints, metrics) {
  const recommendations = [];
  
  if (breakingPoints.agent_overload) {
    recommendations.push({
      issue: 'Agent Overload',
      priority: 'High',
      recommendation: 'Implement agent load balancing and request queuing',
      implementation: 'Add Redis queue for agent requests and horizontal scaling'
    });
  }
  
  if (breakingPoints.database_saturation) {
    recommendations.push({
      issue: 'Database Saturation',
      priority: 'Critical',
      recommendation: 'Optimize database queries and increase connection pool',
      implementation: 'Add read replicas, optimize slow queries, increase pool size'
    });
  }
  
  if (breakingPoints.memory_exhaustion) {
    recommendations.push({
      issue: 'Memory Exhaustion',
      priority: 'High',
      recommendation: 'Implement memory monitoring and garbage collection optimization',
      implementation: 'Add memory limits per agent, optimize model loading'
    });
  }
  
  if (breakingPoints.connection_limit) {
    recommendations.push({
      issue: 'Connection Limits',
      priority: 'Medium',
      recommendation: 'Increase connection limits and implement connection pooling',
      implementation: 'Configure nginx/HAProxy with higher limits, optimize keep-alive'
    });
  }
  
  // Performance-based recommendations
  if (metrics.http_req_duration.avg > 5000) {
    recommendations.push({
      issue: 'High Average Response Time',
      priority: 'Medium',
      recommendation: 'Implement caching and response optimization',
      implementation: 'Add Redis caching, optimize agent responses, CDN for static content'
    });
  }
  
  if (metrics.http_req_failed.rate > 0.05) {
    recommendations.push({
      issue: 'High Error Rate',
      priority: 'High',
      recommendation: 'Improve error handling and system resilience',
      implementation: 'Add circuit breakers, retry logic, better error responses'
    });
  }
  
  return recommendations;
}

function generateTextSummary(report) {
  let summary = `SutazAI Breaking Point Analysis Summary\n`;
  summary += `=====================================\n\n`;
  summary += `Test completed: ${report.timestamp}\n`;
  summary += `Test duration: ${Math.round(report.test_duration / 1000)}s\n\n`;
  
  summary += `Performance Metrics:\n`;
  summary += `- Average response time: ${Math.round(report.performance_metrics.avg_response_time)}ms\n`;
  summary += `- Maximum response time: ${Math.round(report.performance_metrics.max_response_time)}ms\n`;
  summary += `- Error rate: ${(report.performance_metrics.error_rate * 100).toFixed(2)}%\n`;
  summary += `- Total requests: ${report.performance_metrics.total_requests}\n\n`;
  
  summary += `Breaking Points Identified:\n`;
  Object.entries(report.breaking_points).forEach(([key, value]) => {
    if (value) {
      summary += `- ${key}: ${JSON.stringify(value)}\n`;
    }
  });
  
  summary += `\nOptimization Recommendations:\n`;
  report.recommendations.forEach((rec, index) => {
    summary += `${index + 1}. ${rec.issue} (${rec.priority})\n`;
    summary += `   Recommendation: ${rec.recommendation}\n`;
    summary += `   Implementation: ${rec.implementation}\n\n`;
  });
  
  return summary;
}