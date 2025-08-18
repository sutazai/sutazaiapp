#!/usr/bin/env node
/**
 * MCP Security Manager - Enterprise Authentication & Authorization
 * Provides comprehensive security controls for MCP server access
 */

const crypto = require('crypto');
const jwt = require('jsonwebtoken');
const EventEmitter = require('events');

class MCPSecurityManager extends EventEmitter {
  constructor(options = {}) {
    super();
    
    this.config = {
      jwtSecret: options.jwtSecret || process.env.MCP_JWT_SECRET || this.generateSecret(),
      tokenExpiry: options.tokenExpiry || '1h',
      rateLimitWindow: options.rateLimitWindow || 60000, // 1 minute
      maxRequestsPerWindow: options.maxRequestsPerWindow || 100,
      enableEncryption: options.enableEncryption !== false,
      encryptionAlgorithm: options.encryptionAlgorithm || 'aes-256-gcm',
      auditLogEnabled: options.auditLogEnabled !== false,
      ipWhitelist: options.ipWhitelist || [],
      ipBlacklist: options.ipBlacklist || []
    };
    
    this.authTokens = new Map();
    this.rateLimits = new Map();
    this.auditLog = [];
    this.encryptionKeys = new Map();
    this.permissions = new Map();
    this.sessions = new Map();
    this.logger = options.logger || console;
    
    // Initialize security features
    this.initializeSecurity();
  }

  /**
   * Initialize security subsystems
   */
  initializeSecurity() {
    // Generate master encryption key
    this.masterKey = this.generateSecret(32);
    
    // Setup default permissions
    this.setupDefaultPermissions();
    
    // Start cleanup tasks
    this.startSecurityTasks();
    
    this.logger.info('MCP Security Manager initialized');
  }

  /**
   * Generate cryptographically secure random secret
   */
  generateSecret(length = 64) {
    return crypto.randomBytes(length).toString('hex');
  }

  /**
   * Setup default permission schemes
   */
  setupDefaultPermissions() {
    // Define permission levels
    const permissions = {
      'guest': {
        allowedServices: ['health'],
        allowedOperations: ['read'],
        rateLimit: 10,
        timeWindow: 60000
      },
      'user': {
        allowedServices: ['ultimatecoder', 'language-server', 'sequentialthinking'],
        allowedOperations: ['read', 'execute'],
        rateLimit: 50,
        timeWindow: 60000
      },
      'admin': {
        allowedServices: ['*'],
        allowedOperations: ['*'],
        rateLimit: 200,
        timeWindow: 60000
      },
      'service': {
        allowedServices: ['*'],
        allowedOperations: ['*'],
        rateLimit: 1000,
        timeWindow: 60000,
        noAudit: false
      }
    };
    
    for (const [role, perms] of Object.entries(permissions)) {
      this.permissions.set(role, perms);
    }
  }

  /**
   * Authenticate user and generate JWT token
   */
  async authenticate(credentials) {
    try {
      // Validate credentials (implement your auth logic here)
      const user = await this.validateCredentials(credentials);
      
      if (!user) {
        this.auditSecurity('AUTH_FAILED', { credentials: credentials.username });
        throw new Error('Invalid credentials');
      }
      
      // Generate JWT token
      const token = jwt.sign(
        {
          userId: user.id,
          username: user.username,
          role: user.role,
          permissions: this.permissions.get(user.role),
          iat: Math.floor(Date.now() / 1000)
        },
        this.config.jwtSecret,
        { expiresIn: this.config.tokenExpiry }
      );
      
      // Store token session
      const sessionId = this.generateSecret(16);
      this.sessions.set(sessionId, {
        userId: user.id,
        token,
        createdAt: Date.now(),
        lastActivity: Date.now(),
        ipAddress: credentials.ipAddress,
        userAgent: credentials.userAgent
      });
      
      this.auditSecurity('AUTH_SUCCESS', { userId: user.id, role: user.role });
      
      return {
        token,
        sessionId,
        expiresIn: this.config.tokenExpiry,
        user: {
          id: user.id,
          username: user.username,
          role: user.role
        }
      };
      
    } catch (error) {
      this.auditSecurity('AUTH_ERROR', { error: error.message });
      throw error;
    }
  }

  /**
   * Validate user credentials (implement based on your auth system)
   */
  async validateCredentials(credentials) {
    // Mock implementation - replace with your authentication logic
    const { username, password, apiKey } = credentials;
    
    // API Key authentication
    if (apiKey) {
      return this.validateApiKey(apiKey);
    }
    
    // Username/password authentication
    if (username && password) {
      return this.validateUserPassword(username, password);
    }
    
    return null;
  }

  /**
   * Validate API key
   */
  async validateApiKey(apiKey) {
    // Mock implementation - replace with your API key validation
    const validApiKeys = {
      'mcp-admin-key-12345': { id: 'admin-1', username: 'admin', role: 'admin' },
      'mcp-user-key-67890': { id: 'user-1', username: 'user', role: 'user' },
      'mcp-service-key-11111': { id: 'service-1', username: 'service', role: 'service' }
    };
    
    return validApiKeys[apiKey] || null;
  }

  /**
   * Validate username/password
   */
  async validateUserPassword(username, password) {
    // Mock implementation - replace with your user database
    const users = {
      'admin': { id: 'admin-1', username: 'admin', password: 'admin123', role: 'admin' },
      'user': { id: 'user-1', username: 'user', password: 'user123', role: 'user' }
    };
    
    const user = users[username];
    if (user && user.password === password) {
      return { id: user.id, username: user.username, role: user.role };
    }
    
    return null;
  }

  /**
   * Authorize request based on JWT token and permissions
   */
  async authorize(token, serviceName, operation, context = {}) {
    try {
      // Verify JWT token
      const decoded = jwt.verify(token, this.config.jwtSecret);
      
      // Check if user exists and is active
      const session = this.findSessionByToken(token);
      if (!session) {
        throw new Error('Invalid session');
      }
      
      // Update last activity
      session.lastActivity = Date.now();
      
      // Check IP restrictions
      if (context.ipAddress && !this.isIpAllowed(context.ipAddress)) {
        this.auditSecurity('IP_BLOCKED', { ip: context.ipAddress, userId: decoded.userId });
        throw new Error('IP address not allowed');
      }
      
      // Check rate limiting
      await this.checkRateLimit(decoded.userId, decoded.permissions);
      
      // Check service permissions
      if (!this.hasServicePermission(decoded.permissions, serviceName)) {
        this.auditSecurity('SERVICE_DENIED', { 
          userId: decoded.userId, 
          service: serviceName,
          role: decoded.role 
        });
        throw new Error(`Access denied to service: ${serviceName}`);
      }
      
      // Check operation permissions
      if (!this.hasOperationPermission(decoded.permissions, operation)) {
        this.auditSecurity('OPERATION_DENIED', { 
          userId: decoded.userId, 
          operation: operation,
          role: decoded.role 
        });
        throw new Error(`Operation not allowed: ${operation}`);
      }
      
      // Log successful authorization
      this.auditSecurity('ACCESS_GRANTED', {
        userId: decoded.userId,
        service: serviceName,
        operation: operation,
        role: decoded.role
      });
      
      return {
        authorized: true,
        user: decoded,
        permissions: decoded.permissions
      };
      
    } catch (error) {
      this.auditSecurity('AUTH_FAILED', { error: error.message, token: token.substring(0, 20) + '...' });
      throw new Error(`Authorization failed: ${error.message}`);
    }
  }

  /**
   * Find session by token
   */
  findSessionByToken(token) {
    for (const [sessionId, session] of this.sessions.entries()) {
      if (session.token === token) {
        return session;
      }
    }
    return null;
  }

  /**
   * Check if IP address is allowed
   */
  isIpAllowed(ipAddress) {
    // Check blacklist first
    if (this.config.ipBlacklist.length > 0 && this.config.ipBlacklist.includes(ipAddress)) {
      return false;
    }
    
    // Check whitelist (if configured)
    if (this.config.ipWhitelist.length > 0) {
      return this.config.ipWhitelist.includes(ipAddress);
    }
    
    return true; // Allow all IPs if no restrictions configured
  }

  /**
   * Check rate limiting
   */
  async checkRateLimit(userId, permissions) {
    const key = `rate_limit:${userId}`;
    const now = Date.now();
    const windowStart = now - this.config.rateLimitWindow;
    
    // Get or create rate limit tracking
    let rateLimit = this.rateLimits.get(key);
    if (!rateLimit) {
      rateLimit = { requests: [], window: now };
      this.rateLimits.set(key, rateLimit);
    }
    
    // Remove old requests outside the window
    rateLimit.requests = rateLimit.requests.filter(timestamp => timestamp > windowStart);
    
    // Check if limit exceeded
    const maxRequests = permissions.rateLimit || this.config.maxRequestsPerWindow;
    if (rateLimit.requests.length >= maxRequests) {
      this.auditSecurity('RATE_LIMIT_EXCEEDED', { userId, requestCount: rateLimit.requests.length });
      throw new Error('Rate limit exceeded');
    }
    
    // Add current request
    rateLimit.requests.push(now);
  }

  /**
   * Check service permission
   */
  hasServicePermission(permissions, serviceName) {
    const allowedServices = permissions.allowedServices || [];
    return allowedServices.includes('*') || allowedServices.includes(serviceName);
  }

  /**
   * Check operation permission
   */
  hasOperationPermission(permissions, operation) {
    const allowedOperations = permissions.allowedOperations || [];
    return allowedOperations.includes('*') || allowedOperations.includes(operation);
  }

  /**
   * Encrypt sensitive data
   */
  encrypt(data, keyId = 'default') {
    if (!this.config.enableEncryption) {
      return { encrypted: false, data };
    }
    
    try {
      const key = this.getEncryptionKey(keyId);
      const iv = crypto.randomBytes(12);
      const cipher = crypto.createCipher(this.config.encryptionAlgorithm, key);
      cipher.setIV(iv);
      
      let encrypted = cipher.update(JSON.stringify(data), 'utf8', 'hex');
      encrypted += cipher.final('hex');
      
      const authTag = cipher.getAuthTag();
      
      return {
        encrypted: true,
        data: encrypted,
        iv: iv.toString('hex'),
        authTag: authTag.toString('hex'),
        keyId
      };
      
    } catch (error) {
      this.logger.error('Encryption failed:', error.message);
      throw new Error('Data encryption failed');
    }
  }

  /**
   * Decrypt sensitive data
   */
  decrypt(encryptedData) {
    if (!encryptedData.encrypted) {
      return encryptedData.data;
    }
    
    try {
      const key = this.getEncryptionKey(encryptedData.keyId);
      const decipher = crypto.createDecipher(this.config.encryptionAlgorithm, key);
      decipher.setIV(Buffer.from(encryptedData.iv, 'hex'));
      decipher.setAuthTag(Buffer.from(encryptedData.authTag, 'hex'));
      
      let decrypted = decipher.update(encryptedData.data, 'hex', 'utf8');
      decrypted += decipher.final('utf8');
      
      return JSON.parse(decrypted);
      
    } catch (error) {
      this.logger.error('Decryption failed:', error.message);
      throw new Error('Data decryption failed');
    }
  }

  /**
   * Get or generate encryption key
   */
  getEncryptionKey(keyId) {
    if (!this.encryptionKeys.has(keyId)) {
      const key = crypto.pbkdf2Sync(this.masterKey, keyId, 10000, 32, 'sha256');
      this.encryptionKeys.set(keyId, key);
    }
    
    return this.encryptionKeys.get(keyId);
  }

  /**
   * Audit security events
   */
  auditSecurity(event, details = {}) {
    if (!this.config.auditLogEnabled) return;
    
    const auditEntry = {
      timestamp: new Date().toISOString(),
      event,
      details,
      id: this.generateSecret(8)
    };
    
    this.auditLog.push(auditEntry);
    
    // Keep only last 10000 entries
    if (this.auditLog.length > 10000) {
      this.auditLog.shift();
    }
    
    // Emit event for external logging systems
    this.emit('security_audit', auditEntry);
    
    // Log critical security events
    if (['AUTH_FAILED', 'RATE_LIMIT_EXCEEDED', 'IP_BLOCKED'].includes(event)) {
      this.logger.warn('Security Event:', auditEntry);
    }
  }

  /**
   * Get security metrics
   */
  getSecurityMetrics() {
    const now = Date.now();
    const lastHour = now - 3600000;
    
    const recentAudits = this.auditLog.filter(entry => 
      new Date(entry.timestamp).getTime() > lastHour
    );
    
    const eventCounts = {};
    recentAudits.forEach(entry => {
      eventCounts[entry.event] = (eventCounts[entry.event] || 0) + 1;
    });
    
    return {
      activeSessions: this.sessions.size,
      rateLimitTracking: this.rateLimits.size,
      auditLogSize: this.auditLog.length,
      recentEvents: eventCounts,
      encryptionEnabled: this.config.enableEncryption,
      encryptionKeys: this.encryptionKeys.size,
      lastCleanup: this.lastCleanup || null,
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Get audit log entries
   */
  getAuditLog(filters = {}) {
    let filtered = this.auditLog;
    
    if (filters.event) {
      filtered = filtered.filter(entry => entry.event === filters.event);
    }
    
    if (filters.userId) {
      filtered = filtered.filter(entry => entry.details.userId === filters.userId);
    }
    
    if (filters.since) {
      const since = new Date(filters.since).getTime();
      filtered = filtered.filter(entry => new Date(entry.timestamp).getTime() > since);
    }
    
    return filtered.slice(-(filters.limit || 100));
  }

  /**
   * Invalidate session
   */
  invalidateSession(sessionId) {
    const session = this.sessions.get(sessionId);
    if (session) {
      this.sessions.delete(sessionId);
      this.auditSecurity('SESSION_INVALIDATED', { sessionId });
      return true;
    }
    return false;
  }

  /**
   * Revoke all sessions for user
   */
  revokeUserSessions(userId) {
    let revokedCount = 0;
    
    for (const [sessionId, session] of this.sessions.entries()) {
      if (session.userId === userId) {
        this.sessions.delete(sessionId);
        revokedCount++;
      }
    }
    
    if (revokedCount > 0) {
      this.auditSecurity('USER_SESSIONS_REVOKED', { userId, count: revokedCount });
    }
    
    return revokedCount;
  }

  /**
   * Start security background tasks
   */
  startSecurityTasks() {
    // Cleanup expired sessions and rate limits
    setInterval(() => {
      this.cleanupExpiredSessions();
      this.cleanupRateLimits();
      this.lastCleanup = new Date().toISOString();
    }, 300000); // 5 minutes
    
    // Audit log rotation
    setInterval(() => {
      this.rotateAuditLog();
    }, 3600000); // 1 hour
  }

  /**
   * Cleanup expired sessions
   */
  cleanupExpiredSessions() {
    const now = Date.now();
    const maxAge = 24 * 60 * 60 * 1000; // 24 hours
    
    let cleanedCount = 0;
    
    for (const [sessionId, session] of this.sessions.entries()) {
      if (now - session.lastActivity > maxAge) {
        this.sessions.delete(sessionId);
        cleanedCount++;
      }
    }
    
    if (cleanedCount > 0) {
      this.logger.debug(`Cleaned up ${cleanedCount} expired sessions`);
    }
  }

  /**
   * Cleanup old rate limit entries
   */
  cleanupRateLimits() {
    const now = Date.now();
    const windowStart = now - this.config.rateLimitWindow;
    
    for (const [key, rateLimit] of this.rateLimits.entries()) {
      rateLimit.requests = rateLimit.requests.filter(timestamp => timestamp > windowStart);
      
      // Remove empty rate limits
      if (rateLimit.requests.length === 0) {
        this.rateLimits.delete(key);
      }
    }
  }

  /**
   * Rotate audit log to prevent memory bloat
   */
  rotateAuditLog() {
    if (this.auditLog.length > 5000) {
      const archived = this.auditLog.splice(0, this.auditLog.length - 5000);
      this.emit('audit_log_rotated', { archivedCount: archived.length });
      this.logger.info(`Rotated audit log: ${archived.length} entries archived`);
    }
  }

  /**
   * Graceful shutdown
   */
  async shutdown() {
    this.logger.info('Shutting down MCP Security Manager...');
    this.authTokens.clear();
    this.rateLimits.clear();
    this.sessions.clear();
    this.encryptionKeys.clear();
    this.removeAllListeners();
    return { success: true };
  }
}

module.exports = MCPSecurityManager;