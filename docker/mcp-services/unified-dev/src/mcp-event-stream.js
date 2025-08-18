#!/usr/bin/env node
/**
 * MCP Event Stream - Real-time Event Processing & WebSocket Integration
 * Provides live event streaming for MCP operations and system monitoring
 */

const EventEmitter = require('events');
const WebSocket = require('ws');

class MCPEventStream extends EventEmitter {
  constructor(options = {}) {
    super();
    
    this.config = {
      websocketPort: options.websocketPort || 4002,
      enableWebSocket: options.enableWebSocket !== false,
      maxConnections: options.maxConnections || 1000,
      heartbeatInterval: options.heartbeatInterval || 30000,
      maxEventHistory: options.maxEventHistory || 10000,
      enableEventPersistence: options.enableEventPersistence || false,
      eventFilters: options.eventFilters || [],
      compressionEnabled: options.compressionEnabled !== false
    };
    
    this.connections = new Map();
    this.eventHistory = [];
    this.eventStats = new Map();
    this.subscriptions = new Map();
    this.eventProcessors = new Map();
    this.logger = options.logger || console;
    
    // Initialize event streaming
    this.initializeEventStream();
  }

  /**
   * Initialize event streaming system
   */
  initializeEventStream() {
    // Setup event processors
    this.setupEventProcessors();
    
    // Initialize WebSocket server if enabled
    if (this.config.enableWebSocket) {
      this.initializeWebSocketServer();
    }
    
    // Setup event persistence
    if (this.config.enableEventPersistence) {
      this.setupEventPersistence();
    }
    
    // Start monitoring tasks
    this.startMonitoringTasks();
    
    this.logger.info('MCP Event Stream initialized');
  }

  /**
   * Setup event processors for different event types
   */
  setupEventProcessors() {
    // MCP request/response processor
    this.eventProcessors.set('mcp_request', (event) => {
      return {
        ...event,
        type: 'mcp_request',
        enriched: {
          duration: event.endTime ? event.endTime - event.startTime : null,
          success: event.success,
          responseSize: event.responseSize || 0
        }
      };
    });
    
    // Performance metrics processor
    this.eventProcessors.set('performance', (event) => {
      return {
        ...event,
        type: 'performance',
        enriched: {
          severity: this.classifyPerformanceEvent(event),
          trend: this.calculateTrend(event),
          threshold: this.getPerformanceThreshold(event.metric)
        }
      };
    });
    
    // Security event processor
    this.eventProcessors.set('security', (event) => {
      return {
        ...event,
        type: 'security',
        enriched: {
          riskLevel: this.classifySecurityRisk(event),
          actionRequired: this.determineSecurityAction(event)
        }
      };
    });
    
    // System health processor
    this.eventProcessors.set('health', (event) => {
      return {
        ...event,
        type: 'health',
        enriched: {
          healthScore: this.calculateHealthScore(event),
          serviceStatus: this.determineServiceStatus(event)
        }
      };
    });
  }

  /**
   * Initialize WebSocket server
   */
  initializeWebSocketServer() {
    this.wss = new WebSocket.Server({
      port: this.config.websocketPort,
      maxPayload: 1024 * 1024, // 1MB max message size
      perMessageDeflate: this.config.compressionEnabled
    });
    
    this.wss.on('connection', (ws, req) => {
      this.handleNewConnection(ws, req);
    });
    
    this.wss.on('error', (error) => {
      this.logger.error('WebSocket server error:', error.message);
    });
    
    // Setup heartbeat mechanism
    this.setupHeartbeat();
    
    this.logger.info(`WebSocket server listening on port ${this.config.websocketPort}`);
  }

  /**
   * Handle new WebSocket connection
   */
  handleNewConnection(ws, req) {
    const connectionId = this.generateConnectionId();
    const clientInfo = {
      id: connectionId,
      socket: ws,
      connected: Date.now(),
      lastActivity: Date.now(),
      subscriptions: new Set(),
      filters: {},
      authenticated: false,
      ipAddress: req.connection.remoteAddress,
      userAgent: req.headers['user-agent']
    };
    
    this.connections.set(connectionId, clientInfo);
    
    ws.on('message', (data) => {
      this.handleClientMessage(connectionId, data);
    });
    
    ws.on('close', () => {
      this.handleClientDisconnect(connectionId);
    });
    
    ws.on('error', (error) => {
      this.logger.error(`WebSocket error for connection ${connectionId}:`, error.message);
      this.handleClientDisconnect(connectionId);
    });
    
    // Send welcome message
    this.sendToClient(connectionId, {
      type: 'connection_established',
      connectionId,
      timestamp: new Date().toISOString(),
      availableEvents: Array.from(this.eventProcessors.keys())
    });
    
    this.logger.debug(`New WebSocket connection: ${connectionId}`);
  }

  /**
   * Handle client message
   */
  handleClientMessage(connectionId, data) {
    try {
      const message = JSON.parse(data.toString());
      const client = this.connections.get(connectionId);
      
      if (!client) return;
      
      client.lastActivity = Date.now();
      
      switch (message.type) {
        case 'authenticate':
          this.handleAuthentication(connectionId, message);
          break;
          
        case 'subscribe':
          this.handleSubscription(connectionId, message);
          break;
          
        case 'unsubscribe':
          this.handleUnsubscription(connectionId, message);
          break;
          
        case 'filter':
          this.handleFilter(connectionId, message);
          break;
          
        case 'history':
          this.handleHistoryRequest(connectionId, message);
          break;
          
        case 'ping':
          this.sendToClient(connectionId, { type: 'pong', timestamp: new Date().toISOString() });
          break;
          
        default:
          this.sendToClient(connectionId, {
            type: 'error',
            message: `Unknown message type: ${message.type}`
          });
      }
      
    } catch (error) {
      this.logger.error(`Error handling client message for ${connectionId}:`, error.message);
      this.sendToClient(connectionId, {
        type: 'error',
        message: 'Invalid message format'
      });
    }
  }

  /**
   * Handle client authentication
   */
  handleAuthentication(connectionId, message) {
    const client = this.connections.get(connectionId);
    if (!client) return;
    
    // Simple authentication - in production, integrate with your auth system
    if (message.token && this.validateAuthToken(message.token)) {
      client.authenticated = true;
      client.userId = message.userId;
      
      this.sendToClient(connectionId, {
        type: 'authenticated',
        success: true,
        permissions: this.getUserPermissions(message.userId)
      });
      
      this.emit('client_authenticated', { connectionId, userId: message.userId });
    } else {
      this.sendToClient(connectionId, {
        type: 'authentication_failed',
        message: 'Invalid authentication token'
      });
    }
  }

  /**
   * Handle event subscription
   */
  handleSubscription(connectionId, message) {
    const client = this.connections.get(connectionId);
    if (!client) return;
    
    const eventTypes = Array.isArray(message.events) ? message.events : [message.events];
    
    for (const eventType of eventTypes) {
      if (this.eventProcessors.has(eventType)) {
        client.subscriptions.add(eventType);
        
        // Add to global subscriptions tracking
        if (!this.subscriptions.has(eventType)) {
          this.subscriptions.set(eventType, new Set());
        }
        this.subscriptions.get(eventType).add(connectionId);
      }
    }
    
    this.sendToClient(connectionId, {
      type: 'subscription_confirmed',
      events: Array.from(client.subscriptions),
      timestamp: new Date().toISOString()
    });
    
    this.logger.debug(`Client ${connectionId} subscribed to: ${eventTypes.join(', ')}`);
  }

  /**
   * Handle event unsubscription
   */
  handleUnsubscription(connectionId, message) {
    const client = this.connections.get(connectionId);
    if (!client) return;
    
    const eventTypes = Array.isArray(message.events) ? message.events : [message.events];
    
    for (const eventType of eventTypes) {
      client.subscriptions.delete(eventType);
      
      // Remove from global subscriptions
      if (this.subscriptions.has(eventType)) {
        this.subscriptions.get(eventType).delete(connectionId);
        
        // Clean up empty subscription sets
        if (this.subscriptions.get(eventType).size === 0) {
          this.subscriptions.delete(eventType);
        }
      }
    }
    
    this.sendToClient(connectionId, {
      type: 'unsubscription_confirmed',
      events: Array.from(client.subscriptions),
      timestamp: new Date().toISOString()
    });
  }

  /**
   * Handle event filtering
   */
  handleFilter(connectionId, message) {
    const client = this.connections.get(connectionId);
    if (!client) return;
    
    client.filters = {
      ...client.filters,
      ...message.filters
    };
    
    this.sendToClient(connectionId, {
      type: 'filter_applied',
      filters: client.filters,
      timestamp: new Date().toISOString()
    });
  }

  /**
   * Handle history request
   */
  handleHistoryRequest(connectionId, message) {
    const client = this.connections.get(connectionId);
    if (!client) return;
    
    const limit = Math.min(message.limit || 100, 1000);
    const eventType = message.eventType;
    
    let history = this.eventHistory.slice(-limit);
    
    if (eventType) {
      history = history.filter(event => event.type === eventType);
    }
    
    // Apply client filters
    history = this.applyClientFilters(history, client.filters);
    
    this.sendToClient(connectionId, {
      type: 'history_response',
      events: history,
      count: history.length,
      timestamp: new Date().toISOString()
    });
  }

  /**
   * Handle client disconnect
   */
  handleClientDisconnect(connectionId) {
    const client = this.connections.get(connectionId);
    if (!client) return;
    
    // Remove from all subscriptions
    for (const eventType of client.subscriptions) {
      if (this.subscriptions.has(eventType)) {
        this.subscriptions.get(eventType).delete(connectionId);
        
        if (this.subscriptions.get(eventType).size === 0) {
          this.subscriptions.delete(eventType);
        }
      }
    }
    
    this.connections.delete(connectionId);
    this.emit('client_disconnected', { connectionId });
    this.logger.debug(`Client disconnected: ${connectionId}`);
  }

  /**
   * Process and broadcast event
   */
  processEvent(eventType, eventData) {
    try {
      // Get processor for this event type
      const processor = this.eventProcessors.get(eventType);
      if (!processor) {
        this.logger.warn(`No processor found for event type: ${eventType}`);
        return;
      }
      
      // Process the event
      const processedEvent = processor({
        ...eventData,
        id: this.generateEventId(),
        timestamp: new Date().toISOString(),
        type: eventType
      });
      
      // Add to history
      this.addToHistory(processedEvent);
      
      // Update statistics
      this.updateEventStats(eventType);
      
      // Broadcast to subscribers
      this.broadcastEvent(eventType, processedEvent);
      
      // Emit for internal processing
      this.emit('event_processed', processedEvent);
      
    } catch (error) {
      this.logger.error(`Error processing event ${eventType}:`, error.message);
    }
  }

  /**
   * Add event to history
   */
  addToHistory(event) {
    this.eventHistory.push(event);
    
    // Trim history if too large
    if (this.eventHistory.length > this.config.maxEventHistory) {
      this.eventHistory.shift();
    }
  }

  /**
   * Update event statistics
   */
  updateEventStats(eventType) {
    if (!this.eventStats.has(eventType)) {
      this.eventStats.set(eventType, {
        count: 0,
        lastSeen: null,
        rate: 0,
        history: []
      });
    }
    
    const stats = this.eventStats.get(eventType);
    stats.count++;
    stats.lastSeen = Date.now();
    
    // Track rate (events per minute)
    stats.history.push(Date.now());
    const oneMinuteAgo = Date.now() - 60000;
    stats.history = stats.history.filter(timestamp => timestamp > oneMinuteAgo);
    stats.rate = stats.history.length;
  }

  /**
   * Broadcast event to subscribed clients
   */
  broadcastEvent(eventType, event) {
    const subscribers = this.subscriptions.get(eventType);
    if (!subscribers || subscribers.size === 0) return;
    
    for (const connectionId of subscribers) {
      const client = this.connections.get(connectionId);
      if (!client || client.socket.readyState !== WebSocket.OPEN) {
        // Clean up dead connections
        subscribers.delete(connectionId);
        continue;
      }
      
      // Apply client filters
      if (this.eventMatchesFilters(event, client.filters)) {
        this.sendToClient(connectionId, {
          type: 'event',
          eventType,
          data: event
        });
      }
    }
  }

  /**
   * Send message to specific client
   */
  sendToClient(connectionId, message) {
    const client = this.connections.get(connectionId);
    if (!client || client.socket.readyState !== WebSocket.OPEN) return false;
    
    try {
      client.socket.send(JSON.stringify(message));
      return true;
    } catch (error) {
      this.logger.error(`Error sending to client ${connectionId}:`, error.message);
      this.handleClientDisconnect(connectionId);
      return false;
    }
  }

  /**
   * Check if event matches client filters
   */
  eventMatchesFilters(event, filters) {
    if (!filters || Object.keys(filters).length === 0) return true;
    
    for (const [key, value] of Object.entries(filters)) {
      if (event[key] !== value && event.enriched?.[key] !== value) {
        return false;
      }
    }
    
    return true;
  }

  /**
   * Apply client filters to event array
   */
  applyClientFilters(events, filters) {
    if (!filters || Object.keys(filters).length === 0) return events;
    
    return events.filter(event => this.eventMatchesFilters(event, filters));
  }

  /**
   * Classify performance event severity
   */
  classifyPerformanceEvent(event) {
    const thresholds = {
      response_time: { warning: 1000, critical: 5000 },
      memory_usage: { warning: 80, critical: 95 },
      cpu_usage: { warning: 80, critical: 95 },
      error_rate: { warning: 5, critical: 15 }
    };
    
    const threshold = thresholds[event.metric];
    if (!threshold) return 'info';
    
    if (event.value >= threshold.critical) return 'critical';
    if (event.value >= threshold.warning) return 'warning';
    return 'info';
  }

  /**
   * Classify security risk level
   */
  classifySecurityRisk(event) {
    const highRiskEvents = ['AUTH_FAILED', 'IP_BLOCKED', 'RATE_LIMIT_EXCEEDED'];
    const mediumRiskEvents = ['UNAUTHORIZED_ACCESS', 'SUSPICIOUS_ACTIVITY'];
    
    if (highRiskEvents.includes(event.event)) return 'high';
    if (mediumRiskEvents.includes(event.event)) return 'medium';
    return 'low';
  }

  /**
   * Determine security action required
   */
  determineSecurityAction(event) {
    const riskLevel = this.classifySecurityRisk(event);
    
    switch (riskLevel) {
      case 'high': return 'immediate_review';
      case 'medium': return 'investigate';
      default: return 'monitor';
    }
  }

  /**
   * Calculate health score
   */
  calculateHealthScore(event) {
    // Simple health scoring - customize based on your requirements
    let score = 100;
    
    if (event.errors && event.errors > 0) score -= event.errors * 10;
    if (event.responseTime && event.responseTime > 1000) score -= 20;
    if (event.memoryUsage && event.memoryUsage > 80) score -= 15;
    
    return Math.max(0, score);
  }

  /**
   * Determine service status
   */
  determineServiceStatus(event) {
    const healthScore = this.calculateHealthScore(event);
    
    if (healthScore >= 90) return 'healthy';
    if (healthScore >= 70) return 'warning';
    if (healthScore >= 50) return 'degraded';
    return 'unhealthy';
  }

  /**
   * Calculate trend for metrics
   */
  calculateTrend(event) {
    // Simple trend calculation - in production, use more sophisticated analysis
    const history = this.eventHistory
      .filter(e => e.type === 'performance' && e.metric === event.metric)
      .slice(-10);
    
    if (history.length < 2) return 'stable';
    
    const recent = history.slice(-3).reduce((sum, e) => sum + e.value, 0) / 3;
    const older = history.slice(0, -3).reduce((sum, e) => sum + e.value, 0) / Math.max(1, history.length - 3);
    
    const change = ((recent - older) / older) * 100;
    
    if (change > 10) return 'increasing';
    if (change < -10) return 'decreasing';
    return 'stable';
  }

  /**
   * Get performance threshold for metric
   */
  getPerformanceThreshold(metric) {
    const thresholds = {
      response_time: 1000,
      memory_usage: 80,
      cpu_usage: 80,
      error_rate: 5
    };
    
    return thresholds[metric] || 100;
  }

  /**
   * Setup heartbeat mechanism
   */
  setupHeartbeat() {
    setInterval(() => {
      this.wss.clients.forEach((ws) => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.ping();
        }
      });
    }, this.config.heartbeatInterval);
  }

  /**
   * Setup event persistence
   */
  setupEventPersistence() {
    // In production, implement persistent storage (database, file system, etc.)
    this.logger.info('Event persistence enabled (memory-based)');
  }

  /**
   * Start monitoring tasks
   */
  startMonitoringTasks() {
    // Clean up dead connections
    setInterval(() => {
      this.cleanupDeadConnections();
    }, 60000); // 1 minute
    
    // Generate system metrics events
    setInterval(() => {
      this.generateSystemMetrics();
    }, 30000); // 30 seconds
  }

  /**
   * Clean up dead connections
   */
  cleanupDeadConnections() {
    let cleanedCount = 0;
    
    for (const [connectionId, client] of this.connections.entries()) {
      if (client.socket.readyState !== WebSocket.OPEN) {
        this.handleClientDisconnect(connectionId);
        cleanedCount++;
      }
    }
    
    if (cleanedCount > 0) {
      this.logger.debug(`Cleaned up ${cleanedCount} dead connections`);
    }
  }

  /**
   * Generate system metrics events
   */
  generateSystemMetrics() {
    const memoryUsage = process.memoryUsage();
    
    this.processEvent('performance', {
      metric: 'memory_usage',
      value: Math.round((memoryUsage.heapUsed / memoryUsage.heapTotal) * 100),
      unit: 'percentage'
    });
    
    this.processEvent('health', {
      connections: this.connections.size,
      subscriptions: this.subscriptions.size,
      eventHistory: this.eventHistory.length,
      memoryUsage: memoryUsage.heapUsed
    });
  }

  /**
   * Validate authentication token (mock implementation)
   */
  validateAuthToken(token) {
    // In production, integrate with your authentication system
    return token && token.startsWith('mcp-auth-');
  }

  /**
   * Get user permissions (mock implementation)
   */
  getUserPermissions(userId) {
    // In production, fetch from your authorization system
    return {
      canSubscribe: ['mcp_request', 'performance', 'health'],
      canFilter: true,
      canAccessHistory: true
    };
  }

  /**
   * Generate unique connection ID
   */
  generateConnectionId() {
    return `conn_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Generate unique event ID
   */
  generateEventId() {
    return `evt_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Get stream statistics
   */
  getStreamStats() {
    return {
      connections: this.connections.size,
      subscriptions: Object.fromEntries(
        Array.from(this.subscriptions.entries()).map(([eventType, connections]) => [
          eventType,
          connections.size
        ])
      ),
      eventStats: Object.fromEntries(this.eventStats),
      historySize: this.eventHistory.length,
      websocketEnabled: this.config.enableWebSocket,
      port: this.config.websocketPort,
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Graceful shutdown
   */
  async shutdown() {
    this.logger.info('Shutting down MCP Event Stream...');
    
    // Close WebSocket server
    if (this.wss) {
      this.wss.close();
    }
    
    // Clear data structures
    this.connections.clear();
    this.subscriptions.clear();
    this.eventHistory = [];
    this.eventStats.clear();
    
    this.removeAllListeners();
    return { success: true };
  }
}

module.exports = MCPEventStream;