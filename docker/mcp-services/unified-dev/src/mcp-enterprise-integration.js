#!/usr/bin/env node
/**
 * MCP Enterprise Integration - Unified Enterprise MCP Management
 * Integrates Smart Router, Security Manager, and Event Stream for production deployment
 */

const MCPClient = require('./mcp-client');
const MCPSmartRouter = require('./mcp-smart-router');
const MCPSecurityManager = require('./mcp-security-manager');
const MCPEventStream = require('./mcp-event-stream');
const EventEmitter = require('events');

class MCPEnterpriseIntegration extends EventEmitter {
  constructor(options = {}) {
    super();
    
    this.config = {
      ...options,
      enableSmartRouting: options.enableSmartRouting !== false,
      enableSecurity: options.enableSecurity !== false,
      enableEventStream: options.enableEventStream !== false,
      enableMetrics: options.enableMetrics !== false,
      healthCheckInterval: options.healthCheckInterval || 30000,
      performanceThresholds: options.performanceThresholds || {
        responseTime: 2000,
        errorRate: 5,
        memoryUsage: 85
      }
    };
    
    this.logger = options.logger || console;
    this.components = {};
    this.metrics = {
      requests: { total: 0, successful: 0, failed: 0 },
      performance: { avgResponseTime: 0, p95ResponseTime: 0 },
      security: { authAttempts: 0, authFailures: 0 },
      routing: { routedRequests: 0, failovers: 0 }
    };
    
    // Initialize enterprise components
    this.initializeEnterprise();
  }

  /**
   * Initialize all enterprise components
   */
  async initializeEnterprise() {
    try {
      this.logger.info('Initializing MCP Enterprise Integration...');
      
      // Initialize base MCP client
      await this.initializeMCPClient();
      
      // Initialize enterprise components
      if (this.config.enableSmartRouting) {
        await this.initializeSmartRouter();
      }
      
      if (this.config.enableSecurity) {
        await this.initializeSecurityManager();
      }
      
      if (this.config.enableEventStream) {
        await this.initializeEventStream();
      }
      
      // Setup component integrations
      this.setupComponentIntegrations();
      
      // Start monitoring
      this.startEnterpriseMonitoring();
      
      this.logger.info('MCP Enterprise Integration initialized successfully');
      this.emit('enterprise_ready');
      
    } catch (error) {
      this.logger.error('Enterprise initialization failed:', error.message);
      this.emit('enterprise_error', error);
      throw error;
    }
  }

  /**
   * Initialize MCP client
   */
  async initializeMCPClient() {
    this.components.mcpClient = new MCPClient({
      ...this.config,
      logger: this.logger,
      enableMetrics: true
    });
    
    await this.components.mcpClient.initialize();
    this.logger.info('MCP Client initialized');
  }

  /**
   * Initialize smart router
   */
  async initializeSmartRouter() {
    this.components.smartRouter = new MCPSmartRouter({
      ...this.config,
      logger: this.logger,
      routingStrategy: 'predictive',
      enablePredictiveRouting: true,
      enableCaching: true
    });
    
    // Register MCP servers with the router
    await this.registerMCPServersWithRouter();
    
    this.logger.info('Smart Router initialized');
  }

  /**
   * Initialize security manager
   */
  async initializeSecurityManager() {
    this.components.securityManager = new MCPSecurityManager({
      ...this.config,
      logger: this.logger,
      auditLogEnabled: true,
      enableEncryption: true
    });
    
    this.logger.info('Security Manager initialized');
  }

  /**
   * Initialize event stream
   */
  async initializeEventStream() {
    this.components.eventStream = new MCPEventStream({
      ...this.config,
      logger: this.logger,
      enableWebSocket: true,
      websocketPort: this.config.websocketPort || 4002
    });
    
    this.logger.info('Event Stream initialized');
  }

  /**
   * Register MCP servers with smart router
   */
  async registerMCPServersWithRouter() {
    try {
      // Get server registry from MCP client
      const servers = await this.components.mcpClient.loadServerRegistry();
      
      // Group servers by service type for load balancing
      const serviceGroups = this.groupServersByService(servers);
      
      for (const [serviceName, serverList] of Object.entries(serviceGroups)) {
        this.components.smartRouter.registerServerPool(serviceName, serverList);
      }
      
      this.logger.info(`Registered ${Object.keys(serviceGroups).length} service pools with Smart Router`);
      
    } catch (error) {
      this.logger.error('Failed to register servers with router:', error.message);
      throw error;
    }
  }

  /**
   * Group servers by service type
   */
  groupServersByService(servers) {
    const groups = {};
    
    for (const [serverName, serverInfo] of Object.entries(servers)) {
      const serviceType = this.determineServiceType(serverName);
      
      if (!groups[serviceType]) {
        groups[serviceType] = [];
      }
      
      groups[serviceType].push({
        id: serverName,
        name: serverName,
        healthy: serverInfo.healthy,
        available: serverInfo.available,
        weight: this.calculateServerWeight(serverInfo),
        region: this.determineServerRegion(serverName),
        priority: this.calculateServerPriority(serverInfo)
      });
    }
    
    return groups;
  }

  /**
   * Determine service type from server name
   */
  determineServiceType(serverName) {
    const typeMap = {
      'claude-flow': 'orchestration',
      'ruv-swarm': 'orchestration',
      'ultimatecoder': 'development',
      'language-server': 'development',
      'sequentialthinking': 'reasoning',
      'files': 'storage',
      'context7': 'documentation',
      'extended-memory': 'memory',
      'postgres': 'database',
      'github': 'vcs',
      'playwright-mcp': 'testing',
      'puppeteer-mcp': 'testing'
    };
    
    return typeMap[serverName] || 'utility';
  }

  /**
   * Calculate server weight based on performance
   */
  calculateServerWeight(serverInfo) {
    let weight = 1;
    
    if (serverInfo.healthy) weight += 2;
    if (serverInfo.available) weight += 1;
    
    // Adjust based on historical performance
    if (serverInfo.averageResponseTime) {
      if (serverInfo.averageResponseTime < 100) weight += 2;
      else if (serverInfo.averageResponseTime < 500) weight += 1;
    }
    
    return Math.max(1, weight);
  }

  /**
   * Determine server region (for geographic routing)
   */
  determineServerRegion(serverName) {
    // In production, determine based on actual server location
    return 'us-east-1'; // Default region
  }

  /**
   * Calculate server priority
   */
  calculateServerPriority(serverInfo) {
    if (!serverInfo.healthy) return 3; // Low priority
    if (!serverInfo.available) return 2; // Medium priority
    return 1; // High priority
  }

  /**
   * Setup integrations between components
   */
  setupComponentIntegrations() {
    // Smart Router events
    if (this.components.smartRouter) {
      this.components.smartRouter.on('server_health_changed', (event) => {
        this.handleServerHealthChange(event);
      });
      
      this.components.smartRouter.on('performance_analysis', (event) => {
        this.handlePerformanceAnalysis(event);
      });
    }
    
    // Security Manager events
    if (this.components.securityManager) {
      this.components.securityManager.on('security_audit', (event) => {
        this.handleSecurityAudit(event);
      });
      
      this.components.securityManager.on('circuit_breaker_opened', (event) => {
        this.handleCircuitBreakerEvent(event);
      });
    }
    
    // Event Stream integration
    if (this.components.eventStream) {
      this.components.eventStream.on('client_authenticated', (event) => {
        this.handleEventStreamAuth(event);
      });
      
      this.components.eventStream.on('event_processed', (event) => {
        this.handleEventProcessed(event);
      });
    }
    
    // MCP Client events
    if (this.components.mcpClient) {
      this.components.mcpClient.on('circuit_breaker_opened', (event) => {
        this.handleCircuitBreakerEvent(event);
      });
    }
  }

  /**
   * Execute MCP tool with enterprise features
   */
  async executeMCPTool(serviceName, toolName, arguments_obj = {}, context = {}) {
    const startTime = Date.now();
    let result;
    
    try {
      this.metrics.requests.total++;
      
      // Security check
      if (this.components.securityManager && context.authToken) {
        await this.components.securityManager.authorize(
          context.authToken,
          serviceName,
          'execute',
          context
        );
      }
      
      // Smart routing
      let selectedServer = null;
      if (this.components.smartRouter) {
        try {
          selectedServer = await this.components.smartRouter.selectServer(
            this.determineServiceType(serviceName),
            context
          );
          this.metrics.routing.routedRequests++;
        } catch (routingError) {
          this.logger.warn('Smart routing failed, falling back to direct call:', routingError.message);
        }
      }
      
      // Execute the tool
      if (selectedServer) {
        result = await this.executeOnSelectedServer(selectedServer, toolName, arguments_obj);
      } else {
        result = await this.components.mcpClient.callMCPTool(serviceName, toolName, arguments_obj);
      }
      
      const responseTime = Date.now() - startTime;
      
      // Update metrics
      if (result.success) {
        this.metrics.requests.successful++;
      } else {
        this.metrics.requests.failed++;
      }
      
      // Update smart router metrics
      if (selectedServer && this.components.smartRouter) {
        this.components.smartRouter.recordRequestCompletion(
          selectedServer,
          responseTime,
          result.success
        );
      }
      
      // Stream event
      if (this.components.eventStream) {
        this.components.eventStream.processEvent('mcp_request', {
          serviceName,
          toolName,
          success: result.success,
          responseTime,
          selectedServer: selectedServer?.id,
          arguments: arguments_obj
        });
      }
      
      return {
        ...result,
        enterprise: {
          responseTime,
          selectedServer: selectedServer?.id,
          routedViaSmartRouter: !!selectedServer
        }
      };
      
    } catch (error) {
      const responseTime = Date.now() - startTime;
      this.metrics.requests.failed++;
      
      // Log security events
      if (this.components.securityManager) {
        this.components.securityManager.auditSecurity('MCP_TOOL_ERROR', {
          serviceName,
          toolName,
          error: error.message,
          responseTime
        });
      }
      
      // Stream error event
      if (this.components.eventStream) {
        this.components.eventStream.processEvent('mcp_request', {
          serviceName,
          toolName,
          success: false,
          error: error.message,
          responseTime
        });
      }
      
      throw error;
    }
  }

  /**
   * Execute tool on selected server
   */
  async executeOnSelectedServer(server, toolName, arguments_obj) {
    // In production, implement server-specific execution logic
    // For now, use the standard MCP client
    return await this.components.mcpClient.callMCPTool(
      server.name || server.id,
      toolName,
      arguments_obj
    );
  }

  /**
   * Access MCP resource with enterprise features
   */
  async accessMCPResource(serviceName, resourceUri, context = {}) {
    const startTime = Date.now();
    
    try {
      // Security check
      if (this.components.securityManager && context.authToken) {
        await this.components.securityManager.authorize(
          context.authToken,
          serviceName,
          'read',
          context
        );
      }
      
      // Execute resource access
      const result = await this.components.mcpClient.accessMCPResource(serviceName, resourceUri);
      
      const responseTime = Date.now() - startTime;
      
      // Stream event
      if (this.components.eventStream) {
        this.components.eventStream.processEvent('mcp_request', {
          serviceName,
          resourceUri,
          operation: 'resource_access',
          success: result.success,
          responseTime
        });
      }
      
      return result;
      
    } catch (error) {
      const responseTime = Date.now() - startTime;
      
      // Stream error event
      if (this.components.eventStream) {
        this.components.eventStream.processEvent('mcp_request', {
          serviceName,
          resourceUri,
          operation: 'resource_access',
          success: false,
          error: error.message,
          responseTime
        });
      }
      
      throw error;
    }
  }

  /**
   * Handle server health change
   */
  handleServerHealthChange(event) {
    this.logger.info('Server health changed:', event);
    
    // Update MCP client circuit breaker if needed
    if (this.components.mcpClient) {
      this.components.mcpClient.updateServerHealth(
        event.serviceName,
        event.serverId,
        event.healthy
      );
    }
    
    // Stream health event
    if (this.components.eventStream) {
      this.components.eventStream.processEvent('health', {
        serverName: event.serverId,
        healthy: event.healthy,
        serviceName: event.serviceName
      });
    }
  }

  /**
   * Handle performance analysis
   */
  handlePerformanceAnalysis(event) {
    // Stream performance event
    if (this.components.eventStream) {
      this.components.eventStream.processEvent('performance', {
        metric: 'response_time',
        value: event.averageResponseTime,
        serverKey: event.serverKey,
        successRate: event.successRate
      });
    }
    
    // Check performance thresholds
    if (event.averageResponseTime > this.config.performanceThresholds.responseTime) {
      this.logger.warn('Performance threshold exceeded:', event);
    }
  }

  /**
   * Handle security audit
   */
  handleSecurityAudit(event) {
    // Stream security event
    if (this.components.eventStream) {
      this.components.eventStream.processEvent('security', {
        event: event.event,
        details: event.details,
        timestamp: event.timestamp
      });
    }
    
    // Update security metrics
    this.metrics.security.authAttempts++;
    if (event.event === 'AUTH_FAILED') {
      this.metrics.security.authFailures++;
    }
  }

  /**
   * Handle circuit breaker events
   */
  handleCircuitBreakerEvent(event) {
    this.logger.warn('Circuit breaker event:', event);
    this.metrics.routing.failovers++;
    
    // Stream circuit breaker event
    if (this.components.eventStream) {
      this.components.eventStream.processEvent('health', {
        circuitBreaker: event.serverName,
        state: 'open',
        failures: event.failures
      });
    }
  }

  /**
   * Handle event stream authentication
   */
  handleEventStreamAuth(event) {
    this.logger.debug('Event stream client authenticated:', event.connectionId);
  }

  /**
   * Handle processed events
   */
  handleEventProcessed(event) {
    // Could implement additional processing logic here
    // For example, alerting, notifications, etc.
  }

  /**
   * Start enterprise monitoring
   */
  startEnterpriseMonitoring() {
    setInterval(() => {
      this.updatePerformanceMetrics();
      this.checkHealthThresholds();
      this.emitSystemMetrics();
    }, this.config.healthCheckInterval);
    
    this.logger.info('Enterprise monitoring started');
  }

  /**
   * Update performance metrics
   */
  updatePerformanceMetrics() {
    // Calculate average response time from recent requests
    if (this.components.mcpClient) {
      const mcpMetrics = this.components.mcpClient.getMetrics();
      this.metrics.performance.avgResponseTime = mcpMetrics.performance.avg_response_time;
      this.metrics.performance.p95ResponseTime = mcpMetrics.performance.p95_response_time;
    }
  }

  /**
   * Check health thresholds
   */
  checkHealthThresholds() {
    const thresholds = this.config.performanceThresholds;
    
    // Check response time
    if (this.metrics.performance.avgResponseTime > thresholds.responseTime) {
      this.emit('threshold_exceeded', {
        metric: 'response_time',
        value: this.metrics.performance.avgResponseTime,
        threshold: thresholds.responseTime
      });
    }
    
    // Check error rate
    const errorRate = this.metrics.requests.total > 0 ?
      (this.metrics.requests.failed / this.metrics.requests.total) * 100 : 0;
    
    if (errorRate > thresholds.errorRate) {
      this.emit('threshold_exceeded', {
        metric: 'error_rate',
        value: errorRate,
        threshold: thresholds.errorRate
      });
    }
  }

  /**
   * Emit system metrics
   */
  emitSystemMetrics() {
    if (this.components.eventStream) {
      this.components.eventStream.processEvent('performance', {
        metric: 'enterprise_metrics',
        value: this.metrics,
        timestamp: new Date().toISOString()
      });
    }
  }

  /**
   * Get comprehensive enterprise status
   */
  getEnterpriseStatus() {
    const status = {
      initialized: true,
      components: {},
      metrics: this.metrics,
      health: 'healthy',
      timestamp: new Date().toISOString()
    };
    
    // Component status
    if (this.components.mcpClient) {
      status.components.mcpClient = {
        initialized: true,
        metrics: this.components.mcpClient.getMetrics()
      };
    }
    
    if (this.components.smartRouter) {
      status.components.smartRouter = {
        initialized: true,
        stats: this.components.smartRouter.getRoutingStats()
      };
    }
    
    if (this.components.securityManager) {
      status.components.securityManager = {
        initialized: true,
        metrics: this.components.securityManager.getSecurityMetrics()
      };
    }
    
    if (this.components.eventStream) {
      status.components.eventStream = {
        initialized: true,
        stats: this.components.eventStream.getStreamStats()
      };
    }
    
    // Overall health determination
    const errorRate = this.metrics.requests.total > 0 ?
      (this.metrics.requests.failed / this.metrics.requests.total) * 100 : 0;
    
    if (errorRate > 15) status.health = 'unhealthy';
    else if (errorRate > 5) status.health = 'degraded';
    else if (this.metrics.performance.avgResponseTime > 2000) status.health = 'warning';
    
    return status;
  }

  /**
   * Graceful shutdown
   */
  async shutdown() {
    this.logger.info('Shutting down MCP Enterprise Integration...');
    
    // Shutdown components in reverse order
    const shutdownPromises = [];
    
    if (this.components.eventStream) {
      shutdownPromises.push(this.components.eventStream.shutdown());
    }
    
    if (this.components.securityManager) {
      shutdownPromises.push(this.components.securityManager.shutdown());
    }
    
    if (this.components.smartRouter) {
      shutdownPromises.push(this.components.smartRouter.shutdown());
    }
    
    if (this.components.mcpClient) {
      shutdownPromises.push(this.components.mcpClient.shutdown());
    }
    
    await Promise.all(shutdownPromises);
    
    this.removeAllListeners();
    this.logger.info('MCP Enterprise Integration shutdown complete');
    
    return { success: true };
  }
}

module.exports = MCPEnterpriseIntegration;