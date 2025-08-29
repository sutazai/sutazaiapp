const { spawn } = require('child_process');
const readline = require('readline');

async function testGitMCPTools() {
  console.log('Testing GitMCP tools for sutazai/sutazaiapp...\n');
  
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
          tools: { listChanged: true }
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

  function sendListToolsRequest() {
    const listRequest = {
      jsonrpc: "2.0",
      method: "tools/list",
      id: 2,
      params: {}
    };
    
    console.log('\nSending list tools request...');
    mcpProcess.stdin.write(JSON.stringify(listRequest) + '\n');
  }

  function sendCallToolRequest(toolName, args) {
    const callRequest = {
      jsonrpc: "2.0",
      method: "tools/call",
      id: 3,
      params: {
        name: toolName,
        arguments: args
      }
    };
    
    console.log(`\nCalling tool: ${toolName}`);
    mcpProcess.stdin.write(JSON.stringify(callRequest) + '\n');
  }

  function handleResponse(response) {
    if (response.error) {
      console.error('❌ Error:', response.error);
      return;
    }

    if (response.id === 1) {
      console.log('✅ Initialize response received!');
      console.log('Server:', response.result?.serverInfo?.name, response.result?.serverInfo?.version);
      console.log('Tools capability:', response.result?.capabilities?.tools);
      
      // After initialization, list tools
      sendListToolsRequest();
    } else if (response.id === 2) {
      console.log('✅ Tools list received!');
      const tools = response.result?.tools || [];
      console.log(`Found ${tools.length} tools:`);
      
      tools.forEach(tool => {
        console.log(`  - ${tool.name}: ${tool.description}`);
      });
      
      // Test the search_repositories tool
      if (tools.find(t => t.name === 'search_repositories')) {
        sendCallToolRequest('search_repositories', { query: 'sutazai' });
      } else if (tools.find(t => t.name === 'get_file_contents')) {
        // Try to get README
        sendCallToolRequest('get_file_contents', { 
          owner: 'sutazai',
          repo: 'sutazaiapp',
          path: 'README.md'
        });
      } else if (tools.length > 0) {
        // Call the first available tool
        const firstTool = tools[0];
        console.log(`\nTesting first tool: ${firstTool.name}`);
        sendCallToolRequest(firstTool.name, {});
      } else {
        console.log('\n❌ No tools available!');
        process.exit(1);
      }
    } else if (response.id === 3) {
      console.log('✅ Tool call response received!');
      const content = response.result?.content;
      if (content) {
        console.log('Response type:', Array.isArray(content) ? 'array' : typeof content);
        if (Array.isArray(content) && content[0]) {
          console.log('First result:', JSON.stringify(content[0], null, 2).substring(0, 500));
        } else {
          console.log('Response:', JSON.stringify(content, null, 2).substring(0, 500));
        }
        console.log('\n✅ All tests passed! GitMCP tools are working correctly.');
      } else {
        console.log('Response:', JSON.stringify(response.result, null, 2));
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

testGitMCPTools().catch(console.error);