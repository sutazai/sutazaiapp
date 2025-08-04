// Individual Agent Performance Testing
import { check, sleep } from 'k6';
import http from 'k6/http';
import { config, httpParams, validateResponse, randomChoice, randomInt } from '../k6-config.js';

export { options } from '../k6-config.js';

// Agent-specific test scenarios
export default function() {
  const testScenario = randomChoice(['health_check', 'simple_request', 'complex_request']);
  
  switch(testScenario) {
    case 'health_check':
      testAgentHealthCheck();
      break;
    case 'simple_request':
      testSimpleAgentRequest();
      break;
    case 'complex_request':
      testComplexAgentRequest();
      break;
  }
  
  sleep(randomInt(1, 3)); // Random think time
}

function testAgentHealthCheck() {
  const agents = Object.keys(config.agents);
  const agent = randomChoice(agents);
  const port = config.agents[agent];
  
  const response = http.get(`${config.baseUrl}:${port}/health`, {
    ...httpParams,
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'health_check',
      agent: agent 
    }
  });
  
  validateResponse(response, 200);
  
  check(response, {
    'health check returns status': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.status === 'healthy' || body.status === 'ok';
      } catch (e) {
        return false;
      }
    }
  });
}

function testSimpleAgentRequest() {
  const agents = Object.keys(config.agents);
  const agent = randomChoice(agents);
  const port = config.agents[agent];
  const prompt = randomChoice(config.testData.samplePrompts);
  
  const payload = {
    prompt: prompt,
    max_tokens: 100,
    temperature: 0.7,
    stream: false
  };
  
  const response = http.post(`${config.baseUrl}:${port}/api/chat`, JSON.stringify(payload), {
    ...httpParams,
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'simple_request',
      agent: agent 
    }
  });
  
  validateResponse(response, 200);
  
  check(response, {
    'simple request returns response': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.response && body.response.length > 0;
      } catch (e) {
        return false;
      }
    }
  });
}

function testComplexAgentRequest() {
  const agents = Object.keys(config.agents);
  const agent = randomChoice(agents);
  const port = config.agents[agent];
  
  const complexPrompt = `
    Create a comprehensive analysis of the following system requirements:
    1. Scalability requirements for 1000+ concurrent users
    2. Database performance optimization strategies
    3. Security implementation best practices
    4. Monitoring and alerting configuration
    5. CI/CD pipeline design recommendations
    
    Provide detailed implementation steps and code examples.
  `;
  
  const payload = {
    prompt: complexPrompt,
    max_tokens: 2000,
    temperature: 0.5,
    stream: false,
    context: {
      system: 'sutazai',
      complexity: 'high',
      requirements: ['scalability', 'performance', 'security']
    }
  };
  
  const response = http.post(`${config.baseUrl}:${port}/api/chat`, JSON.stringify(payload), {
    ...httpParams,
    timeout: '60s', // Longer timeout for complex requests
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'complex_request',
      agent: agent 
    }
  });
  
  validateResponse(response, 200);
  
  check(response, {
    'complex request returns detailed response': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.response && body.response.length > 500; // Expecting detailed response
      } catch (e) {
        return false;
      }
    },
    'response contains implementation details': (r) => {
      try {
        const body = JSON.parse(r.body);
        const response = body.response.toLowerCase();
        return response.includes('implementation') || 
               response.includes('code') || 
               response.includes('configuration');
      } catch (e) {
        return false;
      }
    }
  });
}

// Agent-specific stress test
export function agentStressTest() {
  const agents = Object.keys(config.agents);
  
  // Test all agents simultaneously
  agents.forEach(agent => {
    const port = config.agents[agent];
    
    const payload = {
      prompt: 'Perform a quick system check and return status',
      max_tokens: 50,
      temperature: 0.1
    };
    
    http.post(`${config.baseUrl}:${port}/api/chat`, JSON.stringify(payload), {
      ...httpParams,
      tags: { 
        ...httpParams.tags, 
        test_scenario: 'stress_test',
        agent: agent 
      }
    });
  });
}

// Agent failover test
export function agentFailoverTest() {
  const primaryAgent = 'ai-system-architect';
  const fallbackAgent = 'ai-senior-engineer';
  
  // Try primary agent
  let response = http.get(`${config.baseUrl}:${config.agents[primaryAgent]}/health`, {
    ...httpParams,
    tags: { 
      ...httpParams.tags, 
      test_scenario: 'failover_test',
      agent: primaryAgent 
    }
  });
  
  if (response.status !== 200) {
    // Fallback to secondary agent
    response = http.get(`${config.baseUrl}:${config.agents[fallbackAgent]}/health`, {
      ...httpParams,
      tags: { 
        ...httpParams.tags, 
        test_scenario: 'failover_test',
        agent: fallbackAgent 
      }
    });
    
    check(response, {
      'failover successful': (r) => r.status === 200
    });
  }
}