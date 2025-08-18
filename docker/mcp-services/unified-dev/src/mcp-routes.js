#!/usr/bin/env node
/**
 * MCP Integration Routes Module
 * Modular MCP endpoint handlers for unified development service
 */

const logger = {
  info: (msg, data = {}) => console.log(`[${new Date().toISOString()}] [INFO] ${msg}`, data),
  warn: (msg, data = {}) => console.warn(`[${new Date().toISOString()}] [WARN] ${msg}`, data),
  error: (msg, data = {}) => console.error(`[${new Date().toISOString()}] [ERROR] ${msg}`, data)
};

/**
 * Setup MCP routes for the unified development service
 * @param {Object} app - Express app instance
 * @param {Object} mcpClient - MCP client instance
 */
function setupMCPRoutes(app, mcpClient) {
  if (!mcpClient) {
    logger.warn('MCP client not available, skipping MCP route setup');
    return;
  }

  // MCP health check
  app.get('/api/mcp/health', async (req, res) => {
    try {
      const mcpHealth = await mcpClient.checkBackendHealth();
      res.json({
        success: true,
        mcp: mcpHealth,
        metrics: mcpClient.getMetrics()
      });
    } catch (error) {
      logger.error('MCP health check error:', error.message);
      res.status(500).json({ success: false, error: error.message });
    }
  });

  // MCP tool execution endpoint
  app.post('/api/mcp/tools/:server/:tool', async (req, res) => {
    try {
      const { server, tool } = req.params;
      const arguments_obj = req.body;
      
      logger.info(`Executing MCP tool: ${server}/${tool}`);
      
      const result = await mcpClient.callMCPTool(server, tool, arguments_obj);
      
      if (result.success) {
        res.json({
          success: true,
          server,
          tool,
          result: result.data,
          responseTime: result.responseTime,
          timestamp: new Date().toISOString()
        });
      } else {
        res.status(500).json({
          success: false,
          server,
          tool,
          error: result.error,
          responseTime: result.responseTime,
          timestamp: new Date().toISOString()
        });
      }
    } catch (error) {
      logger.error('MCP tool execution error:', error.message);
      res.status(500).json({ success: false, error: error.message });
    }
  });

  // MCP resource access endpoint
  app.get('/api/mcp/resources/:server', async (req, res) => {
    try {
      const { server } = req.params;
      const { uri } = req.query;
      
      if (!uri) {
        return res.status(400).json({
          success: false,
          error: 'Resource URI is required as query parameter'
        });
      }
      
      logger.info(`Accessing MCP resource: ${server}/${uri}`);
      
      const result = await mcpClient.accessMCPResource(server, uri);
      
      if (result.success) {
        res.json({
          success: true,
          server,
          resource: uri,
          data: result.data,
          responseTime: result.responseTime,
          timestamp: new Date().toISOString()
        });
      } else {
        res.status(500).json({
          success: false,
          server,
          resource: uri,
          error: result.error,
          responseTime: result.responseTime,
          timestamp: new Date().toISOString()
        });
      }
    } catch (error) {
      logger.error('MCP resource access error:', error.message);
      res.status(500).json({ success: false, error: error.message });
    }
  });

  // MCP server discovery endpoint
  app.get('/api/mcp/servers', async (req, res) => {
    try {
      const servers = await mcpClient.loadServerRegistry();
      const metrics = mcpClient.getMetrics();
      
      res.json({
        success: true,
        servers,
        health_summary: {
          total: Object.keys(servers).length,
          healthy: Object.values(servers).filter(s => s.healthy).length,
          circuit_breakers: metrics.circuit_breakers
        },
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      logger.error('MCP server discovery error:', error.message);
      res.status(500).json({ success: false, error: error.message });
    }
  });

  // Enhanced ultimatecoder with MCP memory integration
  app.post('/api/mcp/ultimatecoder/enhanced', async (req, res) => {
    try {
      const { code, language, action, context = {} } = req.body;
      
      // Store context in extended-memory for learning
      const contextKey = `ultimatecoder-${Date.now()}`;
      const memoryResult = await mcpClient.callMCPTool('extended-memory', 'save_context', {
        content: JSON.stringify({ code, language, action, context }),
        tags: ['ultimatecoder', 'code-generation', language],
        importance_level: 5
      });
      
      if (memoryResult.success) {
        context.memoryId = contextKey;
        logger.info(`Stored context in memory: ${contextKey}`);
      }
      
      // Get related context from memory
      const relatedMemory = await mcpClient.callMCPTool('extended-memory', 'load_contexts', {
        tags_filter: [language, action],
        limit: 3
      });
      
      if (relatedMemory.success && relatedMemory.data) {
        context.relatedExamples = relatedMemory.data;
      }
      
      // Return enhanced context response
      res.json({
        success: true,
        service: 'ultimatecoder-enhanced',
        context: {
          memoryStored: memoryResult.success,
          relatedExamples: context.relatedExamples ? context.relatedExamples.length : 0,
          enhancedProcessing: true
        },
        input: { code, language, action },
        timestamp: new Date().toISOString()
      });
      
    } catch (error) {
      logger.error('Enhanced ultimatecoder error:', error.message);
      res.status(500).json({ success: false, error: error.message });
    }
  });

  // MCP batch operations endpoint
  app.post('/api/mcp/batch', async (req, res) => {
    try {
      const { operations } = req.body;
      
      if (!Array.isArray(operations) || operations.length === 0) {
        return res.status(400).json({
          success: false,
          error: 'Operations array is required'
        });
      }
      
      logger.info(`Executing ${operations.length} MCP batch operations`);
      
      const results = [];
      const startTime = Date.now();
      
      for (const operation of operations) {
        const { type, server, tool, resource, arguments: args } = operation;
        
        try {
          let result;
          if (type === 'tool') {
            result = await mcpClient.callMCPTool(server, tool, args || {});
          } else if (type === 'resource') {
            result = await mcpClient.accessMCPResource(server, resource);
          } else {
            result = { success: false, error: `Unknown operation type: ${type}` };
          }
          
          results.push({
            operation,
            result,
            success: result.success
          });
        } catch (error) {
          results.push({
            operation,
            result: { success: false, error: error.message },
            success: false
          });
        }
      }
      
      const totalTime = Date.now() - startTime;
      const successCount = results.filter(r => r.success).length;
      
      res.json({
        success: true,
        batch_results: results,
        summary: {
          total_operations: operations.length,
          successful: successCount,
          failed: operations.length - successCount,
          success_rate: (successCount / operations.length) * 100,
          total_time: totalTime
        },
        timestamp: new Date().toISOString()
      });
      
    } catch (error) {
      logger.error('MCP batch operations error:', error.message);
      res.status(500).json({ success: false, error: error.message });
    }
  });

  // MCP metrics and monitoring endpoint
  app.get('/api/mcp/metrics', async (req, res) => {
    try {
      const metrics = mcpClient.getMetrics();
      const health = await mcpClient.checkBackendHealth();
      
      res.json({
        success: true,
        metrics,
        health,
        integration_status: {
          client_initialized: true,
          backend_connected: health.success,
          servers_available: Object.keys(metrics.servers).length,
          circuit_breakers: metrics.circuit_breakers
        },
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      logger.error('MCP metrics error:', error.message);
      res.status(500).json({ success: false, error: error.message });
    }
  });

  logger.info('MCP routes initialized successfully');
}

module.exports = { setupMCPRoutes };