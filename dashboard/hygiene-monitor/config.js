/**
 * Hygiene Monitor Dashboard Configuration
 * Purpose: Configuration for connecting to backend API and WebSocket
 */

window.HYGIENE_CONFIG = {
    // Backend API configuration
    BACKEND_API_URL: 'http://localhost:8080/api/hygiene',
    
    // WebSocket configuration for real-time updates
    WEBSOCKET_URL: 'ws://localhost:8080/ws',
    
    // Rule control API (if different from main API)
    RULE_API_URL: 'http://localhost:8081/api',
    
    // Dashboard settings
    DEFAULT_REFRESH_INTERVAL: 1000, // 1 second
    MAX_RECONNECT_ATTEMPTS: 10,
    RECONNECT_DELAY: 2000, // 2 seconds
    
    // Feature flags
    ENABLE_WEBSOCKET: true,
    ENABLE_REAL_TIME_UPDATES: true,
    ENABLE_AUTO_REFRESH: true,
    
    // UI settings
    DEFAULT_THEME: 'dark',
    CHART_ANIMATION_DURATION: 750,
    NOTIFICATION_TIMEOUT: 5000,
    
    // Data limits
    MAX_RECENT_ACTIONS: 100,
    MAX_VIOLATION_HISTORY: 500,
    CHART_DATA_POINTS: 50
};