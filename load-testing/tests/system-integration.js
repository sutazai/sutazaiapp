// System-Wide Integration Load Testing
import { check, sleep } from 'k6';
import http from 'k6/http';
import { config, httpParams, validateResponse, randomChoice, randomInt } from '../k6-config.js';

export { options } from '../k6-config.js';

// Comprehensive system-wide integration testing
export default function() {
  const integrationScenario = randomChoice([
    'end_to_end_user_journey',
    'cross_agent_collaboration',
    'data_pipeline_integration',
    'monitoring_integration',
    'external_service_integration'
  ]);
  
  switch(integrationScenario) {
    case 'end_to_end_user_journey':
      testEndToEndUserJourney();
      break;
    case 'cross_agent_collaboration':
      testCrossAgentCollaboration();
      break;
    case 'data_pipeline_integration':
      testDataPipelineIntegration();
      break;
    case 'monitoring_integration':
      testMonitoringIntegration();
      break;
    case 'external_service_integration':
      testExternalServiceIntegration();
      break;
  }
  
  sleep(randomInt(2, 5));
}

function testEndToEndUserJourney() {
  // Simulate complete user journey from login to task completion
  const journeyId = `journey_${Date.now()}_${randomInt(1, 1000)}`;
  
  // Step 1: User Authentication
  const loginResponse = http.post(`${config.services.backend}/api/auth/login`, JSON.stringify({
    username: config.auth.username,
    password: config.auth.password
  }), {
    ...httpParams,
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'end_to_end_journey',
      journey_id: journeyId,
      step: 'authentication'
    }
  });
  
  validateResponse(loginResponse, 200);
  
  let authToken;
  try {
    const loginBody = JSON.parse(loginResponse.body);
    authToken = loginBody.token || config.auth.token;
  } catch (e) {
    authToken = config.auth.token;
  }
  
  sleep(1);
  
  // Step 2: Load User Dashboard
  const dashboardResponse = http.get(`${config.services.backend}/api/user/dashboard`, {
    ...httpParams,
    headers: {
      ...httpParams.headers,
      'Authorization': `Bearer ${authToken}`
    },
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'end_to_end_journey',
      journey_id: journeyId,
      step: 'dashboard_load'
    }
  });
  
  validateResponse(dashboardResponse, 200);
  
  sleep(2);
  
  // Step 3: Interact with AI Agent
  const agentRequest = http.post(`${config.services.backend}/api/agents/ai-system-architect/chat`, JSON.stringify({
    message: 'Design a scalable microservices architecture for an e-commerce platform',
    context: {
      journey_id: journeyId,
      user_level: 'experienced'
    }
  }), {
    ...httpParams,
    headers: {
      ...httpParams.headers,
      'Authorization': `Bearer ${authToken}`
    },
    timeout: '60s',
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'end_to_end_journey',
      journey_id: journeyId,
      step: 'agent_interaction'
    }
  });
  
  validateResponse(agentRequest, 200);
  
  sleep(3);
  
  // Step 4: Save and Review Results
  const saveResponse = http.post(`${config.services.backend}/api/user/save-interaction`, JSON.stringify({
    interaction_id: `${journeyId}_interaction_1`,
    agent: 'ai-system-architect',
    content: 'Architecture design request',
    timestamp: Date.now()
  }), {
    ...httpParams,
    headers: {
      ...httpParams.headers,
      'Authorization': `Bearer ${authToken}`
    },
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'end_to_end_journey',
      journey_id: journeyId,
      step: 'save_results'
    }
  });
  
  validateResponse(saveResponse, 200);
  
  sleep(1);
  
  // Step 5: Check History
  const historyResponse = http.get(`${config.services.backend}/api/user/history?limit=10`, {
    ...httpParams,
    headers: {
      ...httpParams.headers,
      'Authorization': `Bearer ${authToken}`
    },
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'end_to_end_journey',
      journey_id: journeyId,
      step: 'check_history'
    }
  });
  
  validateResponse(historyResponse, 200);
  
  // Validate end-to-end journey success
  check(null, {
    'complete user journey successful': () => {
      return loginResponse.status === 200 &&
             dashboardResponse.status === 200 &&
             agentRequest.status === 200 &&
             saveResponse.status === 200 &&
             historyResponse.status === 200;
    }
  });
}

function testCrossAgentCollaboration() {
  // Test collaboration between multiple agents
  const collaborationId = `collab_${Date.now()}`;
  
  // Step 1: System Architect creates design
  const architectResponse = http.post(`${config.services.backend}/api/agents/ai-system-architect/chat`, JSON.stringify({
    message: 'Create a high-level architecture for a real-time chat application',
    collaboration_id: collaborationId,
    output_format: 'structured'
  }), {
    ...httpParams,
    timeout: '60s',
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'cross_agent_collaboration',
      collaboration_id: collaborationId,
      agent: 'architect',
      step: 'design_creation'
    }
  });
  
  validateResponse(architectResponse, 200);
  
  let architectureDesign;
  try {
    const body = JSON.parse(architectResponse.body);
    architectureDesign = body.response || body.content;
  } catch (e) {
    architectureDesign = 'Architecture design placeholder';
  }
  
  sleep(2);
  
  // Step 2: Backend Developer implements from design
  const backendResponse = http.post(`${config.services.backend}/api/agents/ai-senior-backend-developer/chat`, JSON.stringify({
    message: `Implement the backend components based on this architecture: ${architectureDesign.substring(0, 500)}`,
    collaboration_id: collaborationId,
    context: {
      previous_agent: 'ai-system-architect',
      task_type: 'implementation'
    }
  }), {
    ...httpParams,
    timeout: '60s',
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'cross_agent_collaboration',
      collaboration_id: collaborationId,
      agent: 'backend',
      step: 'implementation'
    }
  });
  
  validateResponse(backendResponse, 200);
  
  sleep(2);
  
  // Step 3: QA Lead creates test strategy
  const qaResponse = http.post(`${config.services.backend}/api/agents/ai-qa-team-lead/chat`, JSON.stringify({
    message: 'Create comprehensive testing strategy for the chat application implementation',
    collaboration_id: collaborationId,
    context: {
      architecture_context: architectureDesign.substring(0, 300),
      implementation_ready: true
    }
  }), {
    ...httpParams,
    timeout: '60s',
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'cross_agent_collaboration',
      collaboration_id: collaborationId,
      agent: 'qa',
      step: 'test_strategy'
    }
  });
  
  validateResponse(qaResponse, 200);
  
  sleep(2);
  
  // Step 4: Security Specialist reviews
  const securityResponse = http.post(`${config.services.backend}/api/agents/security-pentesting-specialist/chat`, JSON.stringify({
    message: 'Perform security analysis of the chat application design and implementation',
    collaboration_id: collaborationId,
    context: {
      review_scope: 'full_application',
      focus_areas: ['authentication', 'data_protection', 'real_time_security']
    }
  }), {
    ...httpParams,
    timeout: '60s',
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'cross_agent_collaboration',
      collaboration_id: collaborationId,
      agent: 'security',
      step: 'security_review'
    }
  });
  
  validateResponse(securityResponse, 200);
  
  // Validate collaboration workflow
  check(null, {
    'cross-agent collaboration successful': () => {
      return architectResponse.status === 200 &&
             backendResponse.status === 200 &&
             qaResponse.status === 200 &&
             securityResponse.status === 200;
    },
    'collaboration maintains context': () => {
      // Check if responses show awareness of collaboration
      try {
        const backendBody = JSON.parse(backendResponse.body);
        const qaBody = JSON.parse(qaResponse.body);
        return (backendBody.response && backendBody.response.length > 100) &&
               (qaBody.response && qaBody.response.length > 100);
      } catch (e) {
        return false;
      }
    }
  });
}

function testDataPipelineIntegration() {
  // Test integration across data storage and processing systems
  const pipelineId = `pipeline_${Date.now()}`;
  
  // Step 1: Store data in PostgreSQL
  const postgresData = {
    pipeline_id: pipelineId,
    user_id: randomInt(1, 1000),
    interaction_type: 'agent_query',
    content: 'Test data pipeline integration',
    timestamp: new Date().toISOString()
  };
  
  const postgresResponse = http.post(`${config.services.backend}/api/database/insert`, JSON.stringify({
    table: 'user_interactions',
    data: postgresData
  }), {
    ...httpParams,
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'data_pipeline_integration',
      pipeline_id: pipelineId,
      step: 'postgres_insert'
    }
  });
  
  validateResponse(postgresResponse, 200);
  
  sleep(1);
  
  // Step 2: Cache frequently accessed data in Redis
  const redisResponse = http.post(`${config.services.backend}/api/cache/set`, JSON.stringify({
    key: `user_${postgresData.user_id}_recent`,
    value: JSON.stringify({
      last_interaction: postgresData.content,
      timestamp: postgresData.timestamp
    }),
    ttl: 3600
  }), {
    ...httpParams,
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'data_pipeline_integration',
      pipeline_id: pipelineId,
      step: 'redis_cache'
    }
  });
  
  validateResponse(redisResponse, 200);
  
  sleep(1);
  
  // Step 3: Store relationships in Neo4j
  const neo4jResponse = http.post(`${config.services.backend}/api/graph/create`, JSON.stringify({
    nodes: [
      { type: 'User', id: postgresData.user_id, properties: { last_active: postgresData.timestamp } },
      { type: 'Interaction', id: pipelineId, properties: { content: postgresData.content } }
    ],
    relationships: [
      { from: postgresData.user_id, to: pipelineId, type: 'PERFORMED', properties: { when: postgresData.timestamp } }
    ]
  }), {
    ...httpParams,
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'data_pipeline_integration',
      pipeline_id: pipelineId,
      step: 'neo4j_relationships'
    }
  });
  
  validateResponse(neo4jResponse, 200);
  
  sleep(1);
  
  // Step 4: Index content for vector search
  const vectorResponse = http.post(`${config.services.backend}/api/vector/index`, JSON.stringify({
    collection: 'user_interactions',
    documents: [{
      id: pipelineId,
      content: postgresData.content,
      metadata: {
        user_id: postgresData.user_id,
        timestamp: postgresData.timestamp,
        type: postgresData.interaction_type
      }
    }]
  }), {
    ...httpParams,
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'data_pipeline_integration',
      pipeline_id: pipelineId,
      step: 'vector_indexing'
    }
  });
  
  validateResponse(vectorResponse, 200);
  
  sleep(2);
  
  // Step 5: Test integrated query across all systems
  const integratedQuery = http.post(`${config.services.backend}/api/search/integrated`, JSON.stringify({
    query: 'data pipeline integration',
    user_id: postgresData.user_id,
    include_relations: true,
    include_cache: true,
    limit: 10
  }), {
    ...httpParams,
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'data_pipeline_integration',
      pipeline_id: pipelineId,
      step: 'integrated_query'
    }
  });
  
  validateResponse(integratedQuery, 200);
  
  // Validate data pipeline integration
  check(null, {
    'data pipeline integration successful': () => {
      return postgresResponse.status === 200 &&
             redisResponse.status === 200 &&
             neo4jResponse.status === 200 &&
             vectorResponse.status === 200 &&
             integratedQuery.status === 200;
    },
    'cross-system data consistency': () => {
      try {
        const queryResult = JSON.parse(integratedQuery.body);
        return queryResult.results && queryResult.results.length > 0;
      } catch (e) {
        return false;
      }
    }
  });
}

function testMonitoringIntegration() {
  // Test monitoring system integration
  const monitoringSessionId = `monitoring_${Date.now()}`;
  
  // Step 1: Generate system metrics
  const metricsResponse = http.get(`${config.services.prometheus}/api/v1/query?query=up`, {
    ...httpParams,
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'monitoring_integration',
      session_id: monitoringSessionId,
      step: 'metrics_collection'
    }
  });
  
  check(metricsResponse, {
    'prometheus metrics accessible': (r) => r.status === 200
  });
  
  sleep(1);
  
  // Step 2: Check Grafana dashboard
  const grafanaResponse = http.get(`${config.services.grafana}/api/health`, {
    ...httpParams,
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'monitoring_integration',
      session_id: monitoringSessionId,
      step: 'dashboard_health'
    }
  });
  
  check(grafanaResponse, {
    'grafana dashboard accessible': (r) => r.status === 200
  });
  
  sleep(1);
  
  // Step 3: Test alerting integration
  const alertResponse = http.post(`${config.services.backend}/api/monitoring/test-alert`, JSON.stringify({
    alert_type: 'integration_test',
    severity: 'info',
    message: 'Load testing monitoring integration',
    session_id: monitoringSessionId
  }), {
    ...httpParams,
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'monitoring_integration',
      session_id: monitoringSessionId,
      step: 'alert_testing'
    }
  });
  
  validateResponse(alertResponse, 200);
  
  sleep(2);
  
  // Step 4: Verify alert propagation
  const alertStatusResponse = http.get(`${config.services.backend}/api/monitoring/alerts?session_id=${monitoringSessionId}`, {
    ...httpParams,
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'monitoring_integration',
      session_id: monitoringSessionId,
      step: 'alert_verification'
    }
  });
  
  validateResponse(alertStatusResponse, 200);
  
  // Validate monitoring integration
  check(null, {
    'monitoring integration functional': () => {
      return metricsResponse && grafanaResponse && 
             alertResponse.status === 200 && alertStatusResponse.status === 200;
    }
  });
}

function testExternalServiceIntegration() {
  // Test integration with external services
  const integrationId = `external_${Date.now()}`;
  
  // Test Ollama integration
  const ollamaResponse = http.post(`${config.services.ollama}/api/generate`, JSON.stringify({
    model: 'gpt-oss',
    prompt: 'Test external service integration',
    stream: false
  }), {
    ...httpParams,
    timeout: '30s',
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'external_service_integration',
      integration_id: integrationId,
      service: 'ollama'
    }
  });
  
  check(ollamaResponse, {
    'ollama integration working': (r) => r.status === 200
  });
  
  sleep(2);
  
  // Test external API through proxy
  const proxyResponse = http.get(`${config.services.backend}/api/external/status`, {
    ...httpParams,
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'external_service_integration',
      integration_id: integrationId,
      service: 'external_proxy'
    }
  });
  
  validateResponse(proxyResponse, 200);
  
  // Validate external service integration
  check(null, {
    'external services integrated': () => {
      return (ollamaResponse.status === 200 || ollamaResponse.status === 404) && // Ollama might not be available
             proxyResponse.status === 200;
    }
  });
}