#!/usr/bin/env node
/**
 * Test script for ruv-swarm MCP server
 * Tests swarm initialization, status checking, and agent spawning
 */

const { spawn } = require('child_process');
const readline = require('readline');

// MCP message helper functions
function createMCPRequest(method, params, id = 1) {
  return JSON.stringify({
    jsonrpc: '2.0',
    method: method,
    params: params || {},
    id: id
  });
}

function parseMCPResponse(data) {
  try {
    return JSON.parse(data);
  } catch (e) {
    console.error('Failed to parse response:', data);
    return null;
  }
}

// Test runner
async function testRuvSwarm() {
  console.log('=== RUV-SWARM MCP TEST SUITE ===\n');
  console.log('Starting ruv-swarm MCP server...');
  
  // Start the MCP server with full path
  const npxPath = 'C:/Program Files/nodejs/npx.cmd';
  const mcpProcess = spawn(npxPath, ['ruv-swarm@latest', 'mcp', 'start'], {
    stdio: ['pipe', 'pipe', 'pipe'],
    shell: true
  });

  // Set up output handling
  const rl = readline.createInterface({
    input: mcpProcess.stdout,
    crlfDelay: Infinity
  });

  let requestId = 1;
  const pendingRequests = new Map();

  // Handle responses
  rl.on('line', (line) => {
    const response = parseMCPResponse(line);
    if (response && response.id) {
      const resolver = pendingRequests.get(response.id);
      if (resolver) {
        resolver(response);
        pendingRequests.delete(response.id);
      }
    }
    console.log('Response:', line);
  });

  // Handle errors
  mcpProcess.stderr.on('data', (data) => {
    console.error('Error:', data.toString());
  });

  // Send request helper
  const sendRequest = (method, params) => {
    return new Promise((resolve) => {
      const id = requestId++;
      pendingRequests.set(id, resolve);
      const request = createMCPRequest(method, params, id);
      console.log('\nSending request:', request);
      mcpProcess.stdin.write(request + '\n');
      
      // Timeout after 10 seconds
      setTimeout(() => {
        if (pendingRequests.has(id)) {
          pendingRequests.delete(id);
          resolve({ error: 'Request timeout' });
        }
      }, 10000);
    });
  };

  // Wait for server to initialize
  await new Promise(resolve => setTimeout(resolve, 3000));

  console.log('\n=== TEST 1: Initialize Swarm ===');
  const initResponse = await sendRequest('tools/swarm_init', {
    topology: 'mesh',
    agents: 3
  });
  console.log('Init Response:', JSON.stringify(initResponse, null, 2));

  console.log('\n=== TEST 2: Check Swarm Status ===');
  const statusResponse = await sendRequest('tools/swarm_status', {});
  console.log('Status Response:', JSON.stringify(statusResponse, null, 2));

  console.log('\n=== TEST 3: Spawn Agent ===');
  const spawnResponse = await sendRequest('tools/agent_spawn', {
    name: 'test-agent-1',
    type: 'worker'
  });
  console.log('Spawn Response:', JSON.stringify(spawnResponse, null, 2));

  console.log('\n=== TEST 4: Final Status Check ===');
  const finalStatusResponse = await sendRequest('tools/swarm_status', {});
  console.log('Final Status:', JSON.stringify(finalStatusResponse, null, 2));

  // Clean up
  console.log('\n=== Shutting down MCP server ===');
  mcpProcess.kill();
  
  process.exit(0);
}

// Error handling
process.on('unhandledRejection', (error) => {
  console.error('Unhandled error:', error);
  process.exit(1);
});

// Run tests
testRuvSwarm().catch(console.error);