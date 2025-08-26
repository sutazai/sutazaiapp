#!/usr/bin/env node

const { spawn } = require('child_process');

console.log('Testing GitMCP Integration with SutazAI system...\n');

// Define test scenarios
const tests = [
  {
    name: 'Repository Documentation Access',
    query: 'README.md',
    expected: 'SutazAI'
  },
  {
    name: 'Backend API Documentation',
    query: 'api/v1/endpoints',
    expected: 'FastAPI'
  },
  {
    name: 'Frontend Documentation',
    query: 'streamlit',
    expected: 'frontend'
  }
];

let testsPassed = 0;
let testsFailed = 0;

async function runTest(test) {
  return new Promise((resolve) => {
    console.log(`Running test: ${test.name}`);
    
    const mcpProcess = spawn('npx', [
      'mcp-remote',
      'https://gitmcp.io/sutazai/sutazaiapp',
      '--query',
      test.query
    ], { timeout: 10000 });
    
    let output = '';
    
    mcpProcess.stdout.on('data', (data) => {
      output += data.toString();
    });
    
    mcpProcess.stderr.on('data', (data) => {
      output += data.toString();
    });
    
    mcpProcess.on('close', (code) => {
      if (output.toLowerCase().includes(test.expected.toLowerCase())) {
        console.log(`✅ ${test.name}: PASSED`);
        testsPassed++;
      } else {
        console.log(`❌ ${test.name}: FAILED`);
        testsFailed++;
      }
      resolve();
    });
    
    setTimeout(() => {
      mcpProcess.kill();
      resolve();
    }, 5000);
  });
}

async function runAllTests() {
  console.log('Starting integration tests...\n');
  
  for (const test of tests) {
    await runTest(test);
  }
  
  console.log('\n========================================');
  console.log(`Tests Passed: ${testsPassed}`);
  console.log(`Tests Failed: ${testsFailed}`);
  console.log('========================================');
  
  if (testsFailed === 0) {
    console.log('\n✅ All integration tests PASSED');
    process.exit(0);
  } else {
    console.log('\n⚠️  Some tests failed');
    process.exit(1);
  }
}

// Run tests
runAllTests().catch(console.error);
