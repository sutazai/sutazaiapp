// ARCHIVED COPY — moved from load-testing/tests/service-mesh-resilience.js
// See docs/decisions/2025-08-07-remove-service-mesh.md

// Service Mesh Resilience and Failover Testing (k6)
import { check, sleep } from 'k6';
import http from 'k6/http';
import { config, httpParams, validateResponse, randomChoice, randomInt } from '../k6-config.js';

export { options } from '../k6-config.js';

export default function() {
  const testScenario = randomChoice([
    'circuit_breaker',
    'retry_logic', 
    'timeout_handling',
    'load_balancing',
    'service_discovery',
    'fault_injection'
  ]);
  // ... trimmed for archive
}

