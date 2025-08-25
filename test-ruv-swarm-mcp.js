#!/usr/bin/env node
/**
 * MCP Test Suite for ruv-swarm
 * Tests swarm initialization, status checking, and agent spawning via MCP protocol
 */

const { spawn } = require('child_process');
const path = require('path');

// Configuration
const TEST_CONFIG = {
  swarmTopology: 'mesh',
  initialAgents: 3,
  testTimeout: 60000
};

// MCP Protocol Helper
class MCPClient {
  constructor(process) {
    this.process = process;
    this.requestId = 1;
    this.pendingRequests = new Map();
    this.buffer = '';
    
    // Handle stdout data
    this.process.stdout.on('data', (data) => {
      this.buffer += data.toString();
      this.processBuffer();
    });
    
    // Handle stderr
    this.process.stderr.on('data', (data) => {
      console.error('[MCP Server Error]:', data.toString());
    });
  }
  
  processBuffer() {
    const lines = this.buffer.split('\n');
    this.buffer = lines.pop() || '';
    
    for (const line of lines) {
      if (line.trim()) {
        try {
          const message = JSON.parse(line);
          console.log('[MCP Response]:', JSON.stringify(message, null, 2));
          
          if (message.id && this.pendingRequests.has(message.id)) {
            const resolver = this.pendingRequests.get(message.id);
            this.pendingRequests.delete(message.id);
            resolver(message);
          }
        } catch (e) {
          console.log('[MCP Raw Output]:', line);
        }
      }
    }
  }
  
  async request(method, params = {}) {
    const id = this.requestId++;
    const request = {
      jsonrpc: '2.0',
      method,
      params,
      id
    };
    
    console.log(`\n[MCP Request #${id}]:`, JSON.stringify(request, null, 2));
    
    return new Promise((resolve, reject) => {
      this.pendingRequests.set(id, resolve);
      
      // Send request
      this.process.stdin.write(JSON.stringify(request) + '\n');
      
      // Timeout handler
      setTimeout(() => {
        if (this.pendingRequests.has(id)) {
          this.pendingRequests.delete(id);
          reject(new Error(`Request ${id} timed out after 15 seconds`));
        }
      }, 15000);
    });
  }
  
  close() {
    this.process.kill();
  }
}

// Test Suite
async function runTests() {
  console.log('===========================================');
  console.log('   RUV-SWARM MCP INTEGRATION TEST SUITE   ');
  console.log('===========================================\n');
  
  // Start MCP server
  console.log('📡 Starting ruv-swarm MCP server...');
  const mcpProcess = spawn('npx', ['ruv-swarm@latest', 'mcp', 'start', '--stability'], {
    stdio: ['pipe', 'pipe', 'pipe'],
    cwd: 'C:\\Users\\root\\sutazaiapp',
    shell: true
  });
  
  const mcp = new MCPClient(mcpProcess);
  
  // Wait for server initialization
  console.log('⏳ Waiting for server initialization...');
  await new Promise(resolve => setTimeout(resolve, 5000));
  
  try {
    // Test 1: Initialize protocol
    console.log('\n=== TEST 1: Protocol Initialization ===');
    const initResult = await mcp.request('initialize', {
      protocolVersion: '0.1.0',
      capabilities: {}
    });
    console.log('✅ Protocol initialized:', initResult.result ? 'Success' : 'Failed');
    
    // Test 2: List available tools
    console.log('\n=== TEST 2: List Available Tools ===');
    const toolsResult = await mcp.request('tools/list', {});
    if (toolsResult.result && toolsResult.result.tools) {
      console.log(`✅ Found ${toolsResult.result.tools.length} tools`);
      toolsResult.result.tools.slice(0, 5).forEach(tool => {
        console.log(`   - ${tool.name}: ${tool.description}`);
      });
    }
    
    // Test 3: Initialize Swarm
    console.log('\n=== TEST 3: Initialize Swarm ===');
    console.log(`Configuration: Topology=${TEST_CONFIG.swarmTopology}, Agents=${TEST_CONFIG.initialAgents}`);
    const swarmInitResult = await mcp.request('tools/call', {
      name: 'swarm_init',
      arguments: {
        topology: TEST_CONFIG.swarmTopology,
        maxAgents: TEST_CONFIG.initialAgents
      }
    });
    
    if (swarmInitResult.result) {
      console.log('✅ Swarm initialized successfully');
      console.log('   Swarm Details:', JSON.stringify(swarmInitResult.result, null, 2));
    } else if (swarmInitResult.error) {
      console.log('❌ Swarm initialization failed:', swarmInitResult.error.message);
    }
    
    // Test 4: Check Swarm Status
    console.log('\n=== TEST 4: Swarm Status ===');
    const statusResult = await mcp.request('tools/call', {
      name: 'swarm_status',
      arguments: {}
    });
    
    if (statusResult.result) {
      console.log('✅ Swarm status retrieved');
      console.log('   Status:', JSON.stringify(statusResult.result, null, 2));
    } else if (statusResult.error) {
      console.log('❌ Status check failed:', statusResult.error.message);
    }
    
    // Test 5: Spawn Agent
    console.log('\n=== TEST 5: Spawn Test Agent ===');
    const spawnResult = await mcp.request('tools/call', {
      name: 'agent_spawn',
      arguments: {
        type: 'researcher',
        name: 'test-researcher-001'
      }
    });
    
    if (spawnResult.result) {
      console.log('✅ Agent spawned successfully');
      console.log('   Agent Details:', JSON.stringify(spawnResult.result, null, 2));
    } else if (spawnResult.error) {
      console.log('❌ Agent spawn failed:', spawnResult.error.message);
    }
    
    // Test 6: Final Status Check
    console.log('\n=== TEST 6: Final Status Check ===');
    const finalStatusResult = await mcp.request('tools/call', {
      name: 'swarm_status',
      arguments: { verbose: true }
    });
    
    if (finalStatusResult.result) {
      console.log('✅ Final status retrieved');
      console.log('   Active Swarms:', finalStatusResult.result.activeSwarms || 0);
      console.log('   Total Agents:', finalStatusResult.result.totalAgents || 0);
      console.log('   Full Status:', JSON.stringify(finalStatusResult.result, null, 2));
    }
    
  } catch (error) {
    console.error('\n❌ Test failed:', error.message);
  } finally {
    // Cleanup
    console.log('\n=== Shutting down MCP server ===');
    mcp.close();
    
    // Summary
    console.log('\n===========================================');
    console.log('         TEST SUITE COMPLETED              ');
    console.log('===========================================');
  }
  
  process.exit(0);
}

// Error handling
process.on('unhandledRejection', (error) => {
  console.error('Unhandled error:', error);
  process.exit(1);
});

// Run the test suite
console.log('Starting test suite in 2 seconds...\n');
setTimeout(runTests, 2000);