#!/usr/bin/env node

const { spawn } = require('child_process');
const readline = require('readline');

console.log('Testing GitMCP server for SutazAI repository...\n');

// Start the MCP remote connection
const mcpProcess = spawn('npx', ['mcp-remote', 'https://gitmcp.io/sutazai/sutazaiapp'], {
  stdio: ['pipe', 'pipe', 'pipe']
});

let connected = false;
let testPassed = false;

// Handle stdout
mcpProcess.stdout.on('data', (data) => {
  const output = data.toString();
  console.log(`[STDOUT]: ${output}`);
  
  if (output.includes('Connected to remote server')) {
    connected = true;
    console.log('✅ Successfully connected to GitMCP server');
  }
  
  if (output.includes('Proxy established successfully')) {
    testPassed = true;
    console.log('✅ Proxy established - GitMCP is ready to use');
  }
});

// Handle stderr (mcp-remote outputs to stderr)
mcpProcess.stderr.on('data', (data) => {
  const output = data.toString();
  console.log(`[STDERR]: ${output}`);
  
  if (output.includes('Connected to remote server')) {
    connected = true;
    console.log('✅ Successfully connected to GitMCP server');
  }
  
  if (output.includes('Proxy established successfully')) {
    testPassed = true;
    console.log('✅ Proxy established - GitMCP is ready to use');
  }
});

// Handle process exit
mcpProcess.on('close', (code) => {
  console.log(`\nMCP process exited with code ${code}`);
  
  if (testPassed) {
    console.log('\n✅ GitMCP server test PASSED');
    console.log('The server is properly configured for the SutazAI repository');
    process.exit(0);
  } else {
    console.log('\n❌ GitMCP server test FAILED');
    process.exit(1);
  }
});

// Timeout after 15 seconds
setTimeout(() => {
  if (!connected) {
    console.log('\n❌ Test timeout - could not connect to GitMCP server');
    mcpProcess.kill();
    process.exit(1);
  } else {
    console.log('\n✅ Test completed successfully');
    mcpProcess.kill();
    process.exit(0);
  }
}, 15000);

// Handle process termination
process.on('SIGINT', () => {
  console.log('\nTerminating test...');
  mcpProcess.kill();
  process.exit(0);
});