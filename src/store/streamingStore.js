/**
 * Streaming Response State Management
 * 
 * Handles WebSocket connections, streaming AI responses, and real-time
 * communication with performant chunk handling and automatic reconnection.
 */

import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import { StreamingStatus } from './types';

// Default configuration
const DEFAULT_CONFIG = {
  maxReconnectAttempts: 5,
  reconnectDelay: 3000,
  pingInterval: 30000,
  maxChunks: 1000,
  chunkBufferSize: 50,
  connectionTimeout: 10000
};

export const useStreamingStore = create(
  immer((set, get) => ({
    // State
    status: StreamingStatus.IDLE,
    chunks: [],
    currentResponse: '',
    connection: null,
    error: null,
    autoReconnect: true,
    reconnectAttempts: 0,
    maxReconnectAttempts: DEFAULT_CONFIG.maxReconnectAttempts,
    lastActivity: null,
    connectionId: null,
    
    // Performance optimization
    chunkBuffer: [],
    bufferFlushTimer: null,
    
    // Timers
    reconnectTimer: null,
    pingTimer: null,
    connectionTimer: null,

    // Actions
    connect: async (url, options = {}) => {
      const state = get();
      
      const {
        protocols = [],
        headers = {},
        timeout = DEFAULT_CONFIG.connectionTimeout,
        onMessage = null,
        onError = null,
        onClose = null
      } = options;

      // Don't connect if already connected or connecting
      if (state.status === StreamingStatus.CONNECTING || state.connection) {
        return;
      }

      set((draft) => {
        draft.status = StreamingStatus.CONNECTING;
        draft.error = null;
        draft.connectionId = Date.now() + Math.random();
      });

      try {
        // Create WebSocket connection
        const ws = new WebSocket(url, protocols);
        
        // Connection timeout
        const connectionTimer = setTimeout(() => {
          if (ws.readyState === WebSocket.CONNECTING) {
            ws.close();
            get().setError('Connection timeout');
          }
        }, timeout);

        // WebSocket event handlers
        ws.onopen = () => {
          clearTimeout(connectionTimer);
          
          set((draft) => {
            draft.connection = ws;
            draft.status = StreamingStatus.IDLE;
            draft.reconnectAttempts = 0;
            draft.lastActivity = new Date();
            draft.error = null;
          });

          // Start ping timer for connection health
          get().startPingTimer();
          
          console.log('WebSocket connected to:', url);
        };

        ws.onmessage = (event) => {
          set((draft) => {
            draft.lastActivity = new Date();
          });

          try {
            const data = JSON.parse(event.data);
            get().handleMessage(data);
            
            // Call custom message handler
            if (onMessage) {
              onMessage(data);
            }
          } catch (error) {
            console.error('Error parsing WebSocket message:', error);
            const textData = event.data;
            get().handleMessage({ type: 'text', content: textData });
          }
        };

        ws.onerror = (error) => {
          clearTimeout(connectionTimer);
          console.error('WebSocket error:', error);
          
          const errorMessage = `WebSocket error: ${error.message || 'Unknown error'}`;
          get().setError(errorMessage);
          
          if (onError) {
            onError(error);
          }
        };

        ws.onclose = (event) => {
          clearTimeout(connectionTimer);
          get().stopPingTimer();
          
          set((draft) => {
            draft.connection = null;
            draft.status = StreamingStatus.IDLE;
          });

          console.log('WebSocket closed:', event.code, event.reason);

          // Handle reconnection
          if (get().autoReconnect && !event.wasClean && get().reconnectAttempts < get().maxReconnectAttempts) {
            get().scheduleReconnect(url, options);
          } else {
            set((draft) => {
              draft.status = StreamingStatus.IDLE;
              if (!event.wasClean) {
                draft.error = `Connection closed: ${event.reason || 'Unknown reason'}`;
              }
            });
          }

          if (onClose) {
            onClose(event);
          }
        };

      } catch (error) {
        console.error('Failed to create WebSocket:', error);
        set((draft) => {
          draft.error = `Connection failed: ${error.message}`;
          draft.status = StreamingStatus.ERROR;
        });
      }
    },

    disconnect: () => {
      const state = get();
      
      // Clear timers
      if (state.reconnectTimer) {
        clearTimeout(state.reconnectTimer);
      }
      if (state.connectionTimer) {
        clearTimeout(state.connectionTimer);
      }
      
      get().stopPingTimer();
      get().flushChunkBuffer();

      // Close connection
      if (state.connection) {
        state.connection.close(1000, 'Manual disconnect');
      }

      set((draft) => {
        draft.connection = null;
        draft.status = StreamingStatus.IDLE;
        draft.autoReconnect = false;
        draft.reconnectTimer = null;
        draft.connectionTimer = null;
        draft.error = null;
      });
    },

    sendMessage: (message) => {
      const state = get();
      
      if (!state.connection || state.connection.readyState !== WebSocket.OPEN) {
        get().setError('Not connected to server');
        return false;
      }

      try {
        const messageData = typeof message === 'string' 
          ? { type: 'text', content: message }
          : message;
        
        state.connection.send(JSON.stringify(messageData));
        
        set((draft) => {
          draft.lastActivity = new Date();
        });
        
        return true;
      } catch (error) {
        console.error('Error sending message:', error);
        get().setError(`Failed to send message: ${error.message}`);
        return false;
      }
    },

    handleMessage: (data) => {
      switch (data.type) {
        case 'chunk':
        case 'streaming':
          get().addChunk(data);
          break;
        
        case 'complete':
          set((draft) => {
            draft.status = StreamingStatus.COMPLETE;
          });
          get().flushChunkBuffer();
          break;
        
        case 'error':
          get().setError(data.message || 'Server error');
          break;
        
        case 'ping':
          // Respond to ping
          get().sendMessage({ type: 'pong' });
          break;
          
        case 'pong':
          // Ping response received
          break;
        
        default:
          // Handle as text chunk
          if (data.content) {
            get().addChunk({
              type: 'chunk',
              content: data.content,
              timestamp: Date.now()
            });
          }
          break;
      }
    },

    addChunk: (chunk) => {
      const state = get();
      
      if (!chunk || !chunk.content) return;

      // Process chunk data
      const processedChunk = {
        id: Date.now() + Math.random(),
        content: chunk.content,
        timestamp: chunk.timestamp || Date.now(),
        metadata: chunk.metadata || {}
      };

      // Add to buffer for performance
      set((draft) => {
        draft.chunkBuffer.push(processedChunk);
        draft.status = StreamingStatus.STREAMING;
      });

      // Flush buffer when it reaches capacity
      if (state.chunkBuffer.length >= DEFAULT_CONFIG.chunkBufferSize) {
        get().flushChunkBuffer();
      } else {
        // Schedule flush if not already scheduled
        if (!state.bufferFlushTimer) {
          const timer = setTimeout(() => {
            get().flushChunkBuffer();
          }, 100); // Flush every 100ms

          set((draft) => {
            draft.bufferFlushTimer = timer;
          });
        }
      }
    },

    flushChunkBuffer: () => {
      const state = get();
      
      if (state.bufferFlushTimer) {
        clearTimeout(state.bufferFlushTimer);
      }

      if (state.chunkBuffer.length === 0) {
        set((draft) => {
          draft.bufferFlushTimer = null;
        });
        return;
      }

      set((draft) => {
        // Add buffered chunks to main chunks array
        draft.chunks.push(...draft.chunkBuffer);
        
        // Update current response
        const newContent = draft.chunkBuffer.map(c => c.content).join('');
        draft.currentResponse += newContent;
        
        // Maintain chunk limit
        if (draft.chunks.length > DEFAULT_CONFIG.maxChunks) {
          const overflow = draft.chunks.length - DEFAULT_CONFIG.maxChunks;
          draft.chunks.splice(0, overflow);
        }
        
        // Clear buffer
        draft.chunkBuffer = [];
        draft.bufferFlushTimer = null;
      });
    },

    clearChunks: () => {
      const state = get();
      
      if (state.bufferFlushTimer) {
        clearTimeout(state.bufferFlushTimer);
      }

      set((draft) => {
        draft.chunks = [];
        draft.currentResponse = '';
        draft.chunkBuffer = [];
        draft.bufferFlushTimer = null;
        if (draft.status === StreamingStatus.COMPLETE) {
          draft.status = StreamingStatus.IDLE;
        }
      });
    },

    setError: (error) => {
      set((draft) => {
        draft.error = error;
        draft.status = StreamingStatus.ERROR;
      });
    },

    resetError: () => {
      set((draft) => {
        draft.error = null;
        if (draft.status === StreamingStatus.ERROR) {
          draft.status = draft.connection ? StreamingStatus.IDLE : StreamingStatus.IDLE;
        }
      });
    },

    toggleAutoReconnect: () => {
      set((draft) => {
        draft.autoReconnect = !draft.autoReconnect;
      });
    },

    // Private helper methods
    scheduleReconnect: (url, options) => {
      const state = get();
      
      set((draft) => {
        draft.reconnectAttempts += 1;
      });

      const delay = DEFAULT_CONFIG.reconnectDelay * Math.pow(2, state.reconnectAttempts - 1);
      
      console.log(`Attempting reconnect ${state.reconnectAttempts}/${state.maxReconnectAttempts} in ${delay}ms`);
      
      const timer = setTimeout(() => {
        get().connect(url, options);
      }, delay);

      set((draft) => {
        draft.reconnectTimer = timer;
      });
    },

    startPingTimer: () => {
      const timer = setInterval(() => {
        const state = get();
        if (state.connection && state.connection.readyState === WebSocket.OPEN) {
          get().sendMessage({ type: 'ping' });
        } else {
          get().stopPingTimer();
        }
      }, DEFAULT_CONFIG.pingInterval);

      set((draft) => {
        draft.pingTimer = timer;
      });
    },

    stopPingTimer: () => {
      const state = get();
      if (state.pingTimer) {
        clearInterval(state.pingTimer);
        set((draft) => {
          draft.pingTimer = null;
        });
      }
    },

    // Utility getters
    isConnected: () => {
      const state = get();
      return state.connection && state.connection.readyState === WebSocket.OPEN;
    },

    isStreaming: () => get().status === StreamingStatus.STREAMING,

    hasError: () => !!get().error,

    getConnectionState: () => {
      const state = get();
      if (!state.connection) return 'disconnected';
      
      switch (state.connection.readyState) {
        case WebSocket.CONNECTING: return 'connecting';
        case WebSocket.OPEN: return 'open';
        case WebSocket.CLOSING: return 'closing';
        case WebSocket.CLOSED: return 'closed';
        default: return 'unknown';
      }
    },

    getStats: () => {
      const state = get();
      return {
        totalChunks: state.chunks.length,
        responseLength: state.currentResponse.length,
        reconnectAttempts: state.reconnectAttempts,
        lastActivity: state.lastActivity,
        connectionId: state.connectionId,
        isConnected: get().isConnected(),
        status: state.status
      };
    },

    // Configuration
    setMaxReconnectAttempts: (max) => {
      set((draft) => {
        draft.maxReconnectAttempts = Math.max(0, max);
      });
    },

    // Cleanup on unmount
    cleanup: () => {
      get().disconnect();
      get().clearChunks();
    }
  }))
);