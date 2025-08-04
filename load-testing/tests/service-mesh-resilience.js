// Service Mesh Resilience and Failover Testing
import { check, sleep } from 'k6';
import http from 'k6/http';
import { config, httpParams, validateResponse, randomChoice, randomInt } from '../k6-config.js';

export { options } from '../k6-config.js';

// Service mesh and resilience testing
export default function() {
  const testScenario = randomChoice([
    'circuit_breaker',
    'retry_logic', 
    'timeout_handling',
    'load_balancing',
    'service_discovery',
    'fault_injection'
  ]);
  
  switch(testScenario) {
    case 'circuit_breaker':
      testCircuitBreaker();
      break;
    case 'retry_logic':
      testRetryLogic();
      break;
    case 'timeout_handling':
      testTimeoutHandling();
      break;
    case 'load_balancing':
      testLoadBalancing();
      break;
    case 'service_discovery':
      testServiceDiscovery();
      break;
    case 'fault_injection':
      testFaultInjection();
      break;
  }
  
  sleep(randomInt(1, 3));
}

function testCircuitBreaker() {
  // Test circuit breaker pattern by overwhelming a service
  const testService = 'ai-system-architect';
  const port = config.agents[testService];
  
  // Send requests rapidly to trigger circuit breaker
  for (let i = 0; i < 20; i++) {
    const response = http.post(`${config.baseUrl}:${port}/api/chat`, JSON.stringify({
      prompt: 'This is a circuit breaker test',
      max_tokens: 10
    }), {
      ...httpParams,
      timeout: '1s', // Short timeout to trigger failures
      tags: { 
        ...httpParams.tags, 
        test_scenario: 'circuit_breaker',
        service: testService,
        request_index: i
      }
    });
    
    // Don't validate all responses - expect some to fail as circuit opens
    if (i < 5) {
      // First few should succeed
      check(response, {
        'initial requests succeed': (r) => r.status === 200 || r.status === 429
      });
    } else if (i > 15) {
      // Later requests should be circuit-broken
      check(response, {
        'circuit breaker active': (r) => r.status === 503 || r.status === 429
      });
    }
    
    sleep(0.1); // Small delay between requests
  }
  
  // Wait for circuit to potentially reset
  sleep(5);
  
  // Test if circuit recovers
  const recoveryResponse = http.post(`${config.baseUrl}:${port}/api/chat`, JSON.stringify({
    prompt: 'Circuit recovery test',
    max_tokens: 10
  }), {
    ...httpParams,
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'circuit_breaker_recovery',
      service: testService
    }
  });
  
  check(recoveryResponse, {
    'circuit breaker recovers': (r) => r.status === 200
  });
}

function testRetryLogic() {
  // Test automatic retry mechanisms
  const payload = {
    prompt: 'Test retry logic with potential failures',
    max_tokens: 50,
    retry_config: {
      max_retries: 3,
      backoff_strategy: 'exponential',
      initial_delay: 100
    }
  };
  
  const response = http.post(`${config.services.backend}/api/agents/resilient-chat`, JSON.stringify(payload), {
    ...httpParams,
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'retry_logic'
    }
  });
  
  validateResponse(response, 200);
  
  check(response, {
    'retry logic executed': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.retry_count !== undefined;
      } catch (e) {
        return false;
      }
    },
    'eventual success after retries': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.success === true;
      } catch (e) {
        return false;
      }
    }
  });
}

function testTimeoutHandling() {
  // Test various timeout scenarios
  const timeoutScenarios = [
    { timeout: '100ms', expected: 'timeout' },
    { timeout: '1s', expected: 'success' },
    { timeout: '5s', expected: 'success' },
    { timeout: '30s', expected: 'success' }
  ];
  
  const scenario = randomChoice(timeoutScenarios);
  
  const payload = {
    prompt: 'This is a timeout test with configurable delay',
    processing_delay: scenario.timeout === '100ms' ? 200 : 50 // Introduce delay for timeout test
  };
  
  const response = http.post(`${config.services.backend}/api/agents/timeout-test`, JSON.stringify(payload), {
    ...httpParams,
    timeout: scenario.timeout,
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'timeout_handling',
      timeout_setting: scenario.timeout,
      expected_result: scenario.expected
    }
  });
  
  if (scenario.expected === 'timeout') {
    check(response, {
      'timeout handled correctly': (r) => r.status === 0 || r.error_code === 1050
    });
  } else {
    validateResponse(response, 200);
  }
}

function testLoadBalancing() {
  // Test load balancing across multiple instances
  const requests = [];
  const requestCount = 20;
  
  for (let i = 0; i < requestCount; i++) {
    requests.push(
      http.post(`${config.services.backend}/api/agents/load-balanced-chat`, JSON.stringify({
        prompt: `Load balancing test request ${i}`,
        request_id: `lb_test_${i}_${Date.now()}`
      }), {
        ...httpParams,
        tags: { 
          ...httpParams.tags, 
          test_scenario: 'load_balancing',
          request_index: i
        }
      })
    );
  }
  
  // Analyze response headers to verify load balancing
  const serverInstances = new Set();
  
  requests.forEach((response, index) => {
    check(response, {
      'load balanced request successful': (r) => r.status === 200
    });
    
    // Extract server instance identifier from response
    const serverHeader = response.headers['X-Server-Instance'] || response.headers['Server'];
    if (serverHeader) {
      serverInstances.add(serverHeader);
    }
  });
  
  check(null, {
    'multiple server instances used': () => serverInstances.size > 1,
    'load distribution reasonable': () => {
      // Check if requests were distributed (not all to same instance)
      return serverInstances.size >= Math.min(3, Math.ceil(requestCount / 5));
    }
  });
}

function testServiceDiscovery() {
  // Test service discovery and registration
  
  // 1. Query service registry
  const registryResponse = http.get(`${config.services.backend}/api/services/registry`, {
    ...httpParams,
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'service_discovery',
      operation: 'query_registry'
    }
  });
  
  validateResponse(registryResponse, 200);
  
  let discoveredServices = [];
  check(registryResponse, {
    'service registry accessible': (r) => {
      try {
        const body = JSON.parse(r.body);
        discoveredServices = body.services || [];
        return Array.isArray(discoveredServices) && discoveredServices.length > 0;
      } catch (e) {
        return false;
      }
    }
  });
  
  // 2. Test discovered service endpoints
  if (discoveredServices.length > 0) {
    const randomService = randomChoice(discoveredServices);
    
    const serviceResponse = http.get(`${randomService.endpoint}/health`, {
      ...httpParams,
      tags: { 
        ...httpParams.tags, 
        test_scenario: 'service_discovery',
        operation: 'test_discovered_service',
        service_name: randomService.name
      }
    });
    
    check(serviceResponse, {
      'discovered service responsive': (r) => r.status === 200,
      'service discovery functional': (r) => r.timings.duration < 5000
    });
  }
  
  // 3. Test service health check propagation
  const healthResponse = http.get(`${config.services.backend}/api/services/health`, {
    ...httpParams,
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'service_discovery',
      operation: 'health_aggregation'
    }
  });
  
  validateResponse(healthResponse, 200);
  
  check(healthResponse, {
    'aggregated health status available': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.overall_status && body.services && typeof body.services === 'object';
      } catch (e) {
        return false;
      }
    }
  });
}

function testFaultInjection() {
  // Test system behavior under various fault conditions
  const faultTypes = [
    'network_delay',
    'service_unavailable', 
    'partial_failure',
    'resource_exhaustion',
    'cascading_failure'
  ];
  
  const faultType = randomChoice(faultTypes);
  
  const faultPayload = {
    fault_type: faultType,
    intensity: randomChoice(['low', 'medium', 'high']),
    duration: randomInt(5, 30), // seconds
    target_services: randomChoice([
      ['ai-system-architect'],
      ['postgres', 'redis'],
      ['ollama'],
      ['all_agents']
    ])
  };
  
  // Inject fault
  const faultResponse = http.post(`${config.services.backend}/api/testing/inject-fault`, JSON.stringify(faultPayload), {
    ...httpParams,
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'fault_injection',
      fault_type: faultType,
      fault_intensity: faultPayload.intensity
    }
  });
  
  validateResponse(faultResponse, 200);
  
  sleep(2); // Allow fault to take effect
  
  // Test system behavior under fault
  const testResponse = http.post(`${config.services.backend}/api/jarvis/chat`, JSON.stringify({
    message: 'Test system behavior under fault conditions',
    session_id: `fault_test_${Date.now()}`
  }), {
    ...httpParams,
    timeout: '30s', // Longer timeout during fault conditions
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'fault_injection',
      operation: 'test_under_fault',
      fault_type: faultType
    }
  });
  
  // Behavior depends on fault type
  switch(faultType) {
    case 'network_delay':
      check(testResponse, {
        'system handles network delay': (r) => r.status === 200 && r.timings.duration > 1000
      });
      break;
    case 'service_unavailable':
      check(testResponse, {
        'graceful degradation on service unavailable': (r) => r.status === 200 || r.status === 503
      });
      break;
    case 'partial_failure':
      check(testResponse, {
        'partial failure handled gracefully': (r) => r.status === 200 || r.status === 206
      });
      break;
    default:
      check(testResponse, {
        'system resilient to faults': (r) => r.status < 500 || r.timings.duration < 30000
      });
  }
  
  // Clear fault
  http.post(`${config.services.backend}/api/testing/clear-faults`, '{}', {
    ...httpParams,
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'fault_injection',
      operation: 'clear_faults'
    }
  });
}

// Comprehensive resilience test
export function comprehensiveResilienceTest() {
  // Simulate multiple failure scenarios simultaneously
  const multiFailureScenario = {
    network_issues: true,
    high_load: true,
    service_failures: ['ai-qa-team-lead'],
    resource_constraints: {
      cpu: 90, // 90% CPU usage
      memory: 85 // 85% memory usage
    }
  };
  
  // Apply multiple stressors
  http.post(`${config.services.backend}/api/testing/multi-fault-injection`, JSON.stringify(multiFailureScenario), {
    ...httpParams,
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'comprehensive_resilience'
    }
  });
  
  sleep(5); // Allow faults to take effect
  
  // Test critical user journeys under stress
  const criticalJourneys = [
    'User authentication',
    'Agent query processing',
    'Database operations',
    'Service health monitoring'
  ];
  
  criticalJourneys.forEach(journey => {
    const response = http.post(`${config.services.backend}/api/testing/critical-journey`, JSON.stringify({
      journey: journey,
      validate_under_stress: true
    }), {
      ...httpParams,
      timeout: '60s',
      tags: { 
        ...httpParams.tags, 
        test_scenario: 'critical_journey_under_stress',
        journey: journey
      }
    });
    
    check(response, {
      [`${journey} survives multi-failure`]: (r) => r.status < 500,
      [`${journey} responds within SLA`]: (r) => r.timings.duration < 30000
    });
  });
  
  // Clear all faults
  http.post(`${config.services.backend}/api/testing/clear-all-faults`, '{}', httpParams);
}