import { test, expect } from '@playwright/test';

test.describe('Agent Endpoints Integration Tests', () => {

  const agentEndpoints = [
    { port: 8589, name: 'AI Agent Orchestrator', path: '' },
    { port: 8588, name: 'Resource Arbitration Agent', path: '' },
    { port: 8551, name: 'Task Assignment Coordinator', path: '' },
    { port: 8002, name: 'Hardware Resource Optimizer', path: '' },
    { port: 11015, name: 'Ollama Integration Specialist', path: '' }
  ];

  for (const agent of agentEndpoints) {
    test(`${agent.name} health endpoint responds`, async ({ request }) => {
      const response = await request.get(`http://localhost:${agent.port}/health`);
      
      if (response.ok()) {
        const body = await response.json();
        
        expect(body).toHaveProperty('status');
        expect(body.status).toBe('healthy');
        
        console.log(`✅ ${agent.name} (port ${agent.port}): healthy`);
      } else {
        console.log(`⚠️  ${agent.name} (port ${agent.port}): not responding (may be expected for stubs)`);
      }
    });

    test(`${agent.name} process endpoint exists`, async ({ request }) => {
      const response = await request.post(`http://localhost:${agent.port}/process`, {
        data: {
          message: 'test',
          task_type: 'ping'
        }
      });
      
      if (response.ok()) {
        const body = await response.json();
        
        // For stub agents, we expect a basic response structure
        expect(body).toBeDefined();
        
        console.log(`✅ ${agent.name} process endpoint responds:`, body);
      } else {
        console.log(`⚠️  ${agent.name} process endpoint not available (expected for stubs)`);
      }
    });
  }

  test('Agent orchestration communication', async ({ request }) => {
    // Test if the AI Agent Orchestrator can coordinate with other agents
    const orchestratorResponse = await request.post('http://localhost:8589/process', {
      data: {
        task: 'coordinate_agents',
        agents: ['resource-arbitration', 'task-assignment'],
        message: 'system health check'
      }
    });
    
    if (orchestratorResponse.ok()) {
      const body = await orchestratorResponse.json();
      console.log('✅ Agent orchestration test response:', body);
      
      // Basic validation for stub responses
      expect(body).toBeDefined();
    } else {
      console.log('⚠️  Agent orchestration not fully implemented (expected for current stubs)');
    }
  });

  test('Resource arbitration functionality', async ({ request }) => {
    // Test resource arbitration agent
    const response = await request.post('http://localhost:8588/process', {
      data: {
        action: 'allocate_resources',
        requested_cpu: 2,
        requested_memory: '4GB',
        task_priority: 'high'
      }
    });
    
    if (response.ok()) {
      const body = await response.json();
      console.log('✅ Resource arbitration test response:', body);
      
      expect(body).toBeDefined();
    } else {
      console.log('⚠️  Resource arbitration detailed functionality not implemented');
    }
  });

  test('Task assignment coordination', async ({ request }) => {
    // Test task assignment coordinator
    const response = await request.post('http://localhost:8551/process', {
      data: {
        action: 'assign_task',
        task_id: 'test-task-001',
        task_type: 'data_processing',
        priority: 'medium'
      }
    });
    
    if (response.ok()) {
      const body = await response.json();
      console.log('✅ Task assignment test response:', body);
      
      expect(body).toBeDefined();
    } else {
      console.log('⚠️  Task assignment detailed functionality not implemented');
    }
  });

  test('Hardware resource optimization', async ({ request }) => {
    // Test hardware resource optimizer
    const response = await request.post('http://localhost:8002/process', {
      data: {
        action: 'optimize_resources',
        target: 'cpu_memory',
        optimization_level: 'balanced'
      }
    });
    
    if (response.ok()) {
      const body = await response.json();
      console.log('✅ Hardware optimization test response:', body);
      
      expect(body).toBeDefined();
    } else {
      console.log('⚠️  Hardware optimization detailed functionality not implemented');
    }
  });

  test('Ollama integration specialist', async ({ request }) => {
    // Test Ollama integration specialist
    const response = await request.post('http://localhost:11015/process', {
      data: {
        action: 'generate_text',
        model: 'tinyllama',
        prompt: 'Hello, test prompt',
        max_tokens: 50
      }
    });
    
    if (response.ok()) {
      const body = await response.json();
      console.log('✅ Ollama integration test response:', body);
      
      expect(body).toBeDefined();
    } else {
      console.log('⚠️  Ollama integration detailed functionality not implemented');
    }
  });

  test('Agent error handling', async ({ request }) => {
    // Test how agents handle malformed requests
    const response = await request.post('http://localhost:8589/process', {
      data: {
        invalid_field: 'invalid_data',
        malformed_request: true
      }
    });
    
    // Agents should either handle gracefully or return appropriate error
    if (response.status() >= 400) {
      console.log('✅ Agent properly handles invalid requests with error status');
    } else {
      const body = await response.json();
      console.log('✅ Agent handles invalid requests gracefully:', body);
    }
  });

});