const { spawn } = require('child_process');
const readline = require('readline');

async function testGitMCP() {
  console.log('Testing GitMCP connection to sutazai/sutazaiapp...\n');
  
  const mcpProcess = spawn('npx.cmd', ['-y', 'mcp-remote', 'https://gitmcp.io/sutazai/sutazaiapp'], {
    stdio: ['pipe', 'pipe', 'pipe'],
    shell: true
  });

  // Handle stderr (logging messages)
  mcpProcess.stderr.on('data', (data) => {
    const msg = data.toString();
    if (msg.includes('Proxy established successfully')) {
      console.log('✅ MCP Proxy established successfully!');
      sendInitializeRequest();
    }
  });

  // Handle stdout (JSON-RPC responses)
  const rl = readline.createInterface({
    input: mcpProcess.stdout,
    crlfDelay: Infinity
  });

  rl.on('line', (line) => {
    if (line.trim() && !line.startsWith('[')) {
      try {
        const response = JSON.parse(line);
        handleResponse(response);
      } catch (e) {
        // Not JSON, ignore
      }
    }
  });

  function sendInitializeRequest() {
    const initRequest = {
      jsonrpc: "2.0",
      method: "initialize",
      id: 1,
      params: {
        protocolVersion: "2024-11-05",
        capabilities: {
          roots: { listChanged: true }
        },
        clientInfo: {
          name: "test-client",
          version: "1.0.0"
        }
      }
    };
    
    console.log('Sending initialize request...');
    mcpProcess.stdin.write(JSON.stringify(initRequest) + '\n');
  }

  function sendListResourcesRequest() {
    const listRequest = {
      jsonrpc: "2.0",
      method: "resources/list",
      id: 2,
      params: {}
    };
    
    console.log('\nSending list resources request...');
    mcpProcess.stdin.write(JSON.stringify(listRequest) + '\n');
  }

  function sendReadResourceRequest(uri) {
    const readRequest = {
      jsonrpc: "2.0",
      method: "resources/read",
      id: 3,
      params: { uri }
    };
    
    console.log(`\nReading resource: ${uri}`);
    mcpProcess.stdin.write(JSON.stringify(readRequest) + '\n');
  }

  function handleResponse(response) {
    if (response.id === 1) {
      console.log('✅ Initialize response received!');
      console.log('Server name:', response.result?.serverInfo?.name);
      console.log('Server version:', response.result?.serverInfo?.version);
      console.log('Capabilities:', JSON.stringify(response.result?.capabilities, null, 2));
      
      // After initialization, list resources
      sendListResourcesRequest();
    } else if (response.id === 2) {
      console.log('✅ Resources list received!');
      const resources = response.result?.resources || [];
      console.log(`Found ${resources.length} resources:`);
      
      resources.slice(0, 5).forEach(resource => {
        console.log(`  - ${resource.name} (${resource.uri})`);
      });
      
      // Read the first resource
      if (resources.length > 0) {
        sendReadResourceRequest(resources[0].uri);
      } else {
        console.log('\n❌ No resources found!');
        process.exit(1);
      }
    } else if (response.id === 3) {
      console.log('✅ Resource content received!');
      const content = response.result?.contents?.[0];
      if (content) {
        console.log('Content type:', content.mimeType);
        console.log('Content preview (first 500 chars):');
        console.log(content.text?.substring(0, 500) + '...');
        console.log('\n✅ All tests passed! GitMCP is working correctly.');
      }
      
      // Clean exit
      mcpProcess.kill();
      process.exit(0);
    }
  }

  // Handle errors
  mcpProcess.on('error', (error) => {
    console.error('❌ Error spawning MCP process:', error);
    process.exit(1);
  });

  // Timeout after 30 seconds
  setTimeout(() => {
    console.error('❌ Test timeout after 30 seconds');
    mcpProcess.kill();
    process.exit(1);
  }, 30000);
}

testGitMCP().catch(console.error);