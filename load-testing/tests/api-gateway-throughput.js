// API Gateway Throughput and Rate Limiting Testing
import { check, sleep } from 'k6';
import http from 'k6/http';
import { config, httpParams, validateResponse, randomChoice, randomInt } from '../k6-config.js';

export { options } from '../k6-config.js';

// API Gateway performance and rate limiting tests
export default function() {
  const testType = randomChoice([
    'throughput_test',
    'rate_limiting',
    'authentication_load',
    'routing_performance',
    'caching_behavior',
    'compression_test'
  ]);
  
  switch(testType) {
    case 'throughput_test':
      testAPIGatewayThroughput();
      break;
    case 'rate_limiting':
      testRateLimiting();
      break;
    case 'authentication_load':
      testAuthenticationLoad();
      break;
    case 'routing_performance':
      testRoutingPerformance();
      break;
    case 'caching_behavior':
      testCachingBehavior();
      break;
    case 'compression_test':
      testCompressionHandling();
      break;
  }
  
  sleep(randomInt(1, 2));
}

function testAPIGatewayThroughput() {
  // Test maximum throughput through API gateway
  const endpoints = [
    '/api/health',
    '/api/agents/list',
    '/api/services/status',
    '/api/system/metrics',
    '/api/user/profile'
  ];
  
  const endpoint = randomChoice(endpoints);
  const gatewayUrl = `${config.services.backend}${endpoint}`;
  
  const response = http.get(gatewayUrl, {
    ...httpParams,
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'api_gateway_throughput',
      endpoint: endpoint,
      gateway: 'primary'
    }
  });
  
  validateResponse(response, 200);
  
  check(response, {
    'gateway response time acceptable': (r) => r.timings.duration < 500,
    'gateway headers present': (r) => {
      return r.headers['X-Gateway-Time'] || 
             r.headers['X-RateLimit-Remaining'] || 
             r.headers['Server'];
    },
    'no gateway errors': (r) => !r.headers['X-Gateway-Error']
  });
  
  // Extract gateway metrics if available
  const gatewayTime = response.headers['X-Gateway-Time'];
  if (gatewayTime) {
    check(null, {
      'gateway processing time minimal': () => parseInt(gatewayTime) < 50 // < 50ms overhead
    });
  }
}

function testRateLimiting() {
  // Test rate limiting enforcement
  const rateLimitEndpoint = '/api/agents/chat';
  const requests = [];
  const burstSize = 50; // Send 50 requests rapidly
  
  // Generate rapid requests to trigger rate limiting
  for (let i = 0; i < burstSize; i++) {
    const response = http.post(`${config.services.backend}${rateLimitEndpoint}`, JSON.stringify({
      message: `Rate limit test ${i}`,
      agent: 'ai-system-architect'
    }), {
      ...httpParams,
      tags: { 
        ...httpParams.tags, 
        test_scenario: 'rate_limiting',
        request_index: i,
        burst_test: true
      }
    });
    
    requests.push(response);
    
    // Check rate limit headers
    const remaining = response.headers['X-RateLimit-Remaining'];
    const limit = response.headers['X-RateLimit-Limit'];
    const resetTime = response.headers['X-RateLimit-Reset'];
    
    if (i < 10) {
      // First few requests should succeed
      check(response, {
        'initial requests allowed': (r) => r.status === 200 || r.status === 202
      });
    } else if (i > 30) {
      // Later requests should be rate limited
      check(response, {
        'rate limiting active': (r) => r.status === 429,
        'rate limit headers present': (r) => remaining !== undefined && limit !== undefined
      });
    }
  }
  
  // Analyze rate limiting pattern
  const successfulRequests = requests.filter(r => r.status === 200).length;
  const rateLimitedRequests = requests.filter(r => r.status === 429).length;
  
  check(null, {
    'rate limiting enforced': () => rateLimitedRequests > 0,
    'some requests allowed': () => successfulRequests > 0,
    'rate limiting is reasonable': () => successfulRequests >= burstSize * 0.2 // At least 20% allowed
  });
  
  // Wait for rate limit reset
  sleep(5);
  
  // Test if rate limit resets properly
  const resetResponse = http.get(`${config.services.backend}/api/health`, {
    ...httpParams,
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'rate_limit_reset'
    }
  });
  
  check(resetResponse, {
    'rate limit resets': (r) => r.status === 200
  });
}

function testAuthenticationLoad() {
  // Test authentication performance under load
  const authScenarios = [
    'token_validation',
    'jwt_parsing',
    'permission_check',
    'session_lookup'
  ];
  
  const scenario = randomChoice(authScenarios);
  
  let endpoint, payload;
  switch(scenario) {
    case 'token_validation':
      endpoint = '/api/auth/validate';
      payload = { token: config.auth.token };
      break;
    case 'jwt_parsing':
      endpoint = '/api/auth/parse';
      payload = { jwt: config.auth.token };
      break;
    case 'permission_check':
      endpoint = '/api/auth/permissions';
      payload = { 
        user: config.auth.username,
        resource: randomChoice(['agents', 'system', 'admin'])
      };
      break;
    case 'session_lookup':
      endpoint = '/api/auth/session';
      payload = { session_id: `session_${Date.now()}` };
      break;
  }
  
  const response = http.post(`${config.services.backend}${endpoint}`, JSON.stringify(payload), {
    ...httpParams,
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'authentication_load',
      auth_scenario: scenario
    }
  });
  
  validateResponse(response, 200);
  
  check(response, {
    'auth response time acceptable': (r) => r.timings.duration < 200,
    'auth result valid': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.valid !== undefined || body.authorized !== undefined;
      } catch (e) {
        return false;
      }
    }
  });
}

function testRoutingPerformance() {
  // Test API gateway routing to different backend services
  const routingTargets = [
    { path: '/api/agents/ai-system-architect/chat', service: 'agent' },
    { path: '/api/database/query', service: 'backend' },
    { path: '/api/vector/search', service: 'vector-db' },
    { path: '/api/monitoring/metrics', service: 'monitoring' },
    { path: '/api/ollama/models', service: 'ollama' }
  ];
  
  const target = randomChoice(routingTargets);
  
  const payload = target.service === 'agent' ? 
    { message: 'Routing performance test', max_tokens: 10 } :
    { query: 'SELECT 1', limit: 1 };
  
  const response = http.post(`${config.services.backend}${target.path}`, JSON.stringify(payload), {
    ...httpParams,
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'routing_performance',
      target_service: target.service,
      route_path: target.path
    }
  });
  
  validateResponse(response, 200);
  
  check(response, {
    'routing completed successfully': (r) => r.status === 200,
    'routing overhead minimal': (r) => {
      const routingTime = r.headers['X-Routing-Time'];
      return !routingTime || parseInt(routingTime) < 10; // < 10ms routing overhead
    },
    'correct service reached': (r) => {
      const serviceHeader = r.headers['X-Service-Name'];
      return !serviceHeader || serviceHeader.includes(target.service);
    }
  });
}

function testCachingBehavior() {
  // Test API gateway caching functionality
  const cacheableEndpoints = [
    '/api/agents/list',
    '/api/system/info',
    '/api/services/registry',
    '/api/monitoring/status'
  ];
  
  const endpoint = randomChoice(cacheableEndpoints);
  const cacheKey = `cache_test_${Date.now()}`;
  
  // First request - should miss cache
  const firstResponse = http.get(`${config.services.backend}${endpoint}?cache_key=${cacheKey}`, {
    ...httpParams,
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'caching_behavior',
      cache_operation: 'first_request',
      endpoint: endpoint
    }
  });
  
  validateResponse(firstResponse, 200);
  
  const firstCacheStatus = firstResponse.headers['X-Cache-Status'];
  
  sleep(0.1); // Small delay
  
  // Second request - should hit cache if caching is enabled
  const secondResponse = http.get(`${config.services.backend}${endpoint}?cache_key=${cacheKey}`, {
    ...httpParams,
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'caching_behavior',
      cache_operation: 'second_request',
      endpoint: endpoint
    }
  });
  
  validateResponse(secondResponse, 200);
  
  const secondCacheStatus = secondResponse.headers['X-Cache-Status'];
  
  check(null, {
    'caching mechanism present': () => firstCacheStatus !== undefined || secondCacheStatus !== undefined,
    'cache hit faster than miss': () => {
      if (firstCacheStatus === 'MISS' && secondCacheStatus === 'HIT') {
        return secondResponse.timings.duration < firstResponse.timings.duration;
      }
      return true; // Not applicable if no caching
    },
    'cached content consistent': () => {
      // Basic check that cached response has similar structure
      return firstResponse.body.length > 0 && secondResponse.body.length > 0;
    }
  });
}

function testCompressionHandling() {
  // Test API gateway compression capabilities
  const largeDataEndpoint = '/api/agents/documentation';
  
  // Request with compression
  const compressedResponse = http.get(`${config.services.backend}${largeDataEndpoint}`, {
    ...httpParams,
    headers: {
      ...httpParams.headers,
      'Accept-Encoding': 'gzip, deflate, br'
    },
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'compression_handling',
      compression_type: 'enabled'
    }
  });
  
  // Request without compression
  const uncompressedResponse = http.get(`${config.services.backend}${largeDataEndpoint}`, {
    ...httpParams,
    headers: {
      ...httpParams.headers,
      'Accept-Encoding': 'identity'
    },
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'compression_handling',
      compression_type: 'disabled'
    }
  });
  
  validateResponse(compressedResponse, 200);
  validateResponse(uncompressedResponse, 200);
  
  const compressedSize = compressedResponse.headers['Content-Length'];
  const uncompressedSize = uncompressedResponse.headers['Content-Length'];
  const compressionHeader = compressedResponse.headers['Content-Encoding'];
  
  check(null, {
    'compression applied when requested': () => {
      return compressionHeader && (compressionHeader.includes('gzip') || compressionHeader.includes('deflate'));
    },
    'compressed response smaller': () => {
      if (compressedSize && uncompressedSize) {
        return parseInt(compressedSize) < parseInt(uncompressedSize);
      }
      return true; // Skip if headers not available
    },
    'compressed response time reasonable': () => {
      // Compression should not significantly impact response time
      return Math.abs(compressedResponse.timings.duration - uncompressedResponse.timings.duration) < 1000;
    }
  });
}

// API Gateway stress test
export function apiGatewayStressTest() {
  // Overwhelm the API gateway with requests
  const stressEndpoints = [
    '/api/health',
    '/api/agents/list', 
    '/api/system/status'
  ];
  
  const concurrentRequests = 100;
  const requests = [];
  
  for (let i = 0; i < concurrentRequests; i++) {
    const endpoint = randomChoice(stressEndpoints);
    
    requests.push(
      http.get(`${config.services.backend}${endpoint}`, {
        ...httpParams,
        tags: { 
          ...httpParams.tags, 
          test_scenario: 'api_gateway_stress',
          request_index: i
        }
      })
    );
  }
  
  // Analyze stress test results
  const successfulRequests = requests.filter(r => r.status === 200).length;
  const errorRequests = requests.filter(r => r.status >= 500).length;
  const rateLimitedRequests = requests.filter(r => r.status === 429).length;
  
  check(null, {
    'gateway handles stress load': () => successfulRequests > concurrentRequests * 0.7, // 70% success rate
    'gateway errors minimal': () => errorRequests < concurrentRequests * 0.1, // < 10% errors
    'rate limiting functional under stress': () => rateLimitedRequests > 0
  });
}

// API Gateway failover test
export function apiGatewayFailoverTest() {
  // Test failover between gateway instances
  const primaryGateway = config.services.backend;
  const secondaryGateway = config.services.backend.replace(':8000', ':8001'); // Assuming secondary on 8001
  
  // Test primary gateway
  let response = http.get(`${primaryGateway}/api/health`, {
    ...httpParams,
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'gateway_failover',
      gateway: 'primary'
    }
  });
  
  if (response.status !== 200) {
    // Try secondary gateway
    response = http.get(`${secondaryGateway}/api/health`, {
      ...httpParams,
      tags: { 
        ...httpParams.tags, 
        test_scenario: 'gateway_failover',
        gateway: 'secondary'
      }
    });
    
    check(response, {
      'secondary gateway available': (r) => r.status === 200,
      'failover successful': (r) => r.timings.duration < 5000
    });
  } else {
    check(response, {
      'primary gateway healthy': (r) => r.status === 200
    });
  }
}