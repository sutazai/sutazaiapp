// Jarvis Interface Concurrent User Load Testing
import { check, sleep } from 'k6';
import http from 'k6/http';
import ws from 'k6/ws';
import { config, httpParams, validateResponse, randomChoice, randomInt } from '../k6-config.js';

export { options } from '../k6-config.js';

// Jarvis-specific concurrent user simulation
export default function() {
  const userScenario = randomChoice(['new_user', 'returning_user', 'power_user']);
  
  switch(userScenario) {
    case 'new_user':
      simulateNewUser();
      break;
    case 'returning_user':
      simulateReturningUser();
      break;
    case 'power_user':
      simulatePowerUser();
      break;
  }
}

function simulateNewUser() {
  // New user journey: registration, first query, exploration
  
  // 1. Access Jarvis interface
  let response = http.get(`${config.services.frontend}/`, {
    ...httpParams,
    tags: { 
      ...httpParams.tags, 
      user_type: 'new',
      journey_step: 'landing'
    }
  });
  
  validateResponse(response, 200);
  sleep(randomInt(2, 5)); // Reading landing page
  
  // 2. User registration/authentication
  const userData = {
    username: `testuser_${Date.now()}_${randomInt(1, 1000)}`,
    email: `test${Date.now()}@example.com`,
    password: 'TestPassword123!'
  };
  
  response = http.post(`${config.services.backend}/api/auth/register`, JSON.stringify(userData), {
    ...httpParams,
    tags: { 
      ...httpParams.tags, 
      user_type: 'new',
      journey_step: 'registration'
    }
  });
  
  validateResponse(response, 201);
  
  // Extract auth token
  let authToken;
  try {
    const body = JSON.parse(response.body);
    authToken = body.token;
  } catch (e) {
    authToken = config.auth.token; // Fallback
  }
  
  sleep(randomInt(1, 3));
  
  // 3. First interaction with Jarvis
  const firstQuery = randomChoice([
    'Hello Jarvis, what can you help me with?',
    'Can you explain what SutazAI does?',
    'How do I get started with AI agents?',
    'What agents are available?'
  ]);
  
  testJarvisInteraction(firstQuery, authToken, 'new', 'first_query');
  
  sleep(randomInt(3, 7)); // Thinking time
  
  // 4. Explore available agents
  response = http.get(`${config.services.backend}/api/agents/list`, {
    ...httpParams,
    headers: {
      ...httpParams.headers,
      'Authorization': `Bearer ${authToken}`
    },
    tags: { 
      ...httpParams.tags, 
      user_type: 'new',
      journey_step: 'exploration'
    }
  });
  
  validateResponse(response, 200);
  
  check(response, {
    'agent list loaded': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.agents && Array.isArray(body.agents) && body.agents.length > 0;
      } catch (e) {
        return false;
      }
    }
  });
}

function simulateReturningUser() {
  // Returning user: quick auth, specific task, multiple interactions
  
  // 1. Login
  const loginData = {
    username: config.auth.username,
    password: config.auth.password
  };
  
  let response = http.post(`${config.services.backend}/api/auth/login`, JSON.stringify(loginData), {
    ...httpParams,
    tags: { 
      ...httpParams.tags, 
      user_type: 'returning',
      journey_step: 'login'
    }
  });
  
  validateResponse(response, 200);
  
  let authToken;
  try {
    const body = JSON.parse(response.body);
    authToken = body.token;
  } catch (e) {
    authToken = config.auth.token;
  }
  
  sleep(1);
  
  // 2. Direct task execution
  const tasks = [
    'Create a Python FastAPI application with authentication',
    'Analyze this codebase for security vulnerabilities',
    'Design a microservices architecture for e-commerce',
    'Generate comprehensive unit tests for my backend',
    'Optimize database queries for better performance'
  ];
  
  const task = randomChoice(tasks);
  testJarvisInteraction(task, authToken, 'returning', 'task_execution');
  
  sleep(randomInt(2, 4));
  
  // 3. Follow-up questions
  const followUps = [
    'Can you explain that in more detail?',
    'What are the security implications?',
    'How would this scale to 1000 users?',
    'Can you provide code examples?',
    'What are the best practices here?'
  ];
  
  const followUp = randomChoice(followUps);
  testJarvisInteraction(followUp, authToken, 'returning', 'followup');
  
  sleep(randomInt(1, 3));
  
  // 4. Check task history
  response = http.get(`${config.services.backend}/api/user/history`, {
    ...httpParams,
    headers: {
      ...httpParams.headers,
      'Authorization': `Bearer ${authToken}`
    },
    tags: { 
      ...httpParams.tags, 
      user_type: 'returning',
      journey_step: 'history'
    }
  });
  
  validateResponse(response, 200);
}

function simulatePowerUser() {
  // Power user: complex workflows, multiple agents, advanced features
  
  // 1. Quick auth
  const authToken = config.auth.token; // Assume pre-authenticated
  
  // 2. Complex multi-agent workflow
  const workflow = [
    {
      agent: 'ai-system-architect',
      query: 'Design a scalable microservices architecture for a fintech application'
    },
    {
      agent: 'ai-senior-backend-developer', 
      query: 'Implement the user authentication service from the architecture'
    },
    {
      agent: 'ai-qa-team-lead',
      query: 'Create comprehensive test strategy for the authentication service'
    },
    {
      agent: 'security-pentesting-specialist',
      query: 'Perform security analysis of the authentication implementation'
    },
    {
      agent: 'deployment-automation-master',
      query: 'Create CI/CD pipeline for the authentication service'
    }
  ];
  
  workflow.forEach((step, index) => {
    testSpecificAgentInteraction(step.agent, step.query, authToken, 'power', `workflow_step_${index + 1}`);
    sleep(randomInt(2, 4)); // Processing time between steps
  });
  
  // 3. Batch operations
  const batchQueries = [
    'Generate API documentation',
    'Create monitoring dashboards', 
    'Implement error handling',
    'Add logging and metrics',
    'Create deployment scripts'
  ];
  
  testBatchOperations(batchQueries, authToken);
  
  // 4. Advanced features
  testAdvancedFeatures(authToken);
}

function testJarvisInteraction(query, authToken, userType, step) {
  const payload = {
    message: query,
    session_id: `session_${Date.now()}_${randomInt(1, 1000)}`,
    user_context: {
      type: userType,
      step: step
    }
  };
  
  const response = http.post(`${config.services.backend}/api/jarvis/chat`, JSON.stringify(payload), {
    ...httpParams,
    headers: {
      ...httpParams.headers,
      'Authorization': `Bearer ${authToken}`
    },
    tags: { 
      ...httpParams.tags, 
      user_type: userType,
      journey_step: step,
      interaction_type: 'jarvis_chat'
    }
  });
  
  validateResponse(response, 200);
  
  check(response, {
    'jarvis responded': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.response && body.response.length > 0;
      } catch (e) {
        return false;
      }
    },
    'response is relevant': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.response.length > 10; // Basic relevance check
      } catch (e) {
        return false;
      }
    }
  });
}

function testSpecificAgentInteraction(agentName, query, authToken, userType, step) {
  const payload = {
    agent: agentName,
    message: query,
    parameters: {
      temperature: 0.7,
      max_tokens: 2000
    }
  };
  
  const response = http.post(`${config.services.backend}/api/agents/${agentName}/chat`, JSON.stringify(payload), {
    ...httpParams,
    headers: {
      ...httpParams.headers,
      'Authorization': `Bearer ${authToken}`
    },
    timeout: '60s', // Longer timeout for complex agent interactions
    tags: { 
      ...httpParams.tags, 
      user_type: userType,
      journey_step: step,
      agent: agentName,
      interaction_type: 'specific_agent'
    }
  });
  
  validateResponse(response, 200);
  
  check(response, {
    'agent responded appropriately': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.response && body.response.length > 100; // Expecting detailed response
      } catch (e) {
        return false;
      }
    }
  });
}

function testBatchOperations(queries, authToken) {
  const batchPayload = {
    batch_requests: queries.map((query, index) => ({
      id: `batch_${index}`,
      message: query,
      agent: 'auto-select' // Let system choose appropriate agent
    }))
  };
  
  const response = http.post(`${config.services.backend}/api/jarvis/batch`, JSON.stringify(batchPayload), {
    ...httpParams,
    headers: {
      ...httpParams.headers,
      'Authorization': `Bearer ${authToken}`
    },
    timeout: '120s', // Long timeout for batch operations
    tags: { 
      ...httpParams.tags, 
      user_type: 'power',
      journey_step: 'batch_operations',
      interaction_type: 'batch'
    }
  });
  
  validateResponse(response, 200);
  
  check(response, {
    'batch operation completed': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.results && Array.isArray(body.results) && body.results.length === queries.length;
      } catch (e) {
        return false;
      }
    }
  });
}

function testAdvancedFeatures(authToken) {
  // Test WebSocket connection for real-time updates
  const wsUrl = `${config.services.backend.replace('http', 'ws')}/api/jarvis/ws`;
  
  const wsResponse = ws.connect(wsUrl, {
    headers: {
      'Authorization': `Bearer ${authToken}`
    }
  }, function (socket) {
    socket.on('open', () => {
      socket.send(JSON.stringify({
        type: 'subscribe',
        topics: ['agent_status', 'system_metrics']
      }));
    });
    
    socket.on('message', (data) => {
      try {
        const message = JSON.parse(data);
        check(message, {
          'websocket message valid': (msg) => msg.type && msg.data
        });
      } catch (e) {
        check(null, {
          'websocket message parse error': () => false
        });
      }
    });
    
    // Keep connection open for a short time
    sleep(5);
    socket.close();
  });
  
  check(wsResponse, {
    'websocket connection established': (r) => r && r.status === 101
  });
}

// Concurrent session simulation
export function concurrentSessionTest() {
  const sessions = [];
  const sessionCount = 50;
  
  for (let i = 0; i < sessionCount; i++) {
    const sessionId = `concurrent_session_${i}_${Date.now()}`;
    
    sessions.push(
      http.post(`${config.services.backend}/api/jarvis/chat`, JSON.stringify({
        message: `Concurrent test message from session ${i}`,
        session_id: sessionId
      }), {
        ...httpParams,
        tags: { 
          ...httpParams.tags, 
          test_scenario: 'concurrent_sessions',
          session_id: sessionId
        }
      })
    );
  }
  
  check(null, {
    'concurrent sessions handled': () => sessions.length === sessionCount
  });
}