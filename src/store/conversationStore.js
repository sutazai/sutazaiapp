/**
 * Conversation History State Management
 * 
 * Manages conversation sessions, messages, persistence, and search functionality
 * with support for tags, metadata, and localStorage/IndexedDB persistence.
 */

import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import { MessageType } from './types';

// Default configuration
const DEFAULT_CONFIG = {
  maxSessions: 100,
  maxMessagesPerSession: 1000,
  autoSaveInterval: 30000, // 30 seconds
  storageKey: 'sutazai_conversations',
  useIndexedDB: true
};

// Utility functions for persistence
const StorageUtils = {
  // IndexedDB operations
  async initIndexedDB() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open('SutazAI_Conversations', 1);
      
      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve(request.result);
      
      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        if (!db.objectStoreNames.contains('conversations')) {
          const store = db.createObjectStore('conversations', { keyPath: 'id' });
          store.createIndex('updatedAt', 'updatedAt', { unique: false });
          store.createIndex('tags', 'tags', { unique: false, multiEntry: true });
        }
      };
    });
  },

  async saveToIndexedDB(sessions) {
    try {
      const db = await this.initIndexedDB();
      const transaction = db.transaction(['conversations'], 'readwrite');
      const store = transaction.objectStore('conversations');
      
      // Clear existing data
      await new Promise((resolve, reject) => {
        const clearRequest = store.clear();
        clearRequest.onsuccess = () => resolve();
        clearRequest.onerror = () => reject(clearRequest.error);
      });
      
      // Save new data
      for (const session of sessions) {
        await new Promise((resolve, reject) => {
          const addRequest = store.add(session);
          addRequest.onsuccess = () => resolve();
          addRequest.onerror = () => reject(addRequest.error);
        });
      }
      
      return true;
    } catch (error) {
      console.error('IndexedDB save error:', error);
      return false;
    }
  },

  async loadFromIndexedDB() {
    try {
      const db = await this.initIndexedDB();
      const transaction = db.transaction(['conversations'], 'readonly');
      const store = transaction.objectStore('conversations');
      
      return new Promise((resolve, reject) => {
        const request = store.getAll();
        request.onsuccess = () => resolve(request.result || []);
        request.onerror = () => reject(request.error);
      });
    } catch (error) {
      console.error('IndexedDB load error:', error);
      return [];
    }
  },

  // LocalStorage fallback
  saveToLocalStorage(sessions) {
    try {
      const data = {
        sessions,
        timestamp: new Date().toISOString(),
        version: '1.0'
      };
      localStorage.setItem(DEFAULT_CONFIG.storageKey, JSON.stringify(data));
      return true;
    } catch (error) {
      console.error('LocalStorage save error:', error);
      return false;
    }
  },

  loadFromLocalStorage() {
    try {
      const stored = localStorage.getItem(DEFAULT_CONFIG.storageKey);
      if (!stored) return [];
      
      const data = JSON.parse(stored);
      return Array.isArray(data.sessions) ? data.sessions : [];
    } catch (error) {
      console.error('LocalStorage load error:', error);
      return [];
    }
  }
};

export const useConversationStore = create(
  immer((set, get) => ({
    // State
    sessions: [],
    currentSessionId: null,
    currentSession: null,
    isLoading: false,
    error: null,
    maxSessions: DEFAULT_CONFIG.maxSessions,
    maxMessagesPerSession: DEFAULT_CONFIG.maxMessagesPerSession,
    autoPersist: true,
    autoSaveTimer: null,
    lastSavedAt: null,
    searchResults: [],
    isSearching: false,
    
    // Actions
    createSession: (title = null, metadata = {}) => {
      const sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      const now = new Date();
      
      const session = {
        id: sessionId,
        title: title || `Conversation ${new Date().toLocaleString()}`,
        messages: [],
        createdAt: now,
        updatedAt: now,
        tags: [],
        metadata: {
          messageCount: 0,
          lastActivity: now,
          ...metadata
        }
      };

      set((draft) => {
        // Add new session to beginning
        draft.sessions.unshift(session);
        
        // Maintain max sessions limit
        if (draft.sessions.length > draft.maxSessions) {
          draft.sessions = draft.sessions.slice(0, draft.maxSessions);
        }
        
        // Set as current session
        draft.currentSessionId = sessionId;
        draft.currentSession = session;
        
        draft.error = null;
      });

      // Auto-save if enabled
      if (get().autoPersist) {
        get().scheduleAutoSave();
      }

      return sessionId;
    },

    setCurrentSession: (sessionId) => {
      const state = get();
      const session = state.sessions.find(s => s.id === sessionId);
      
      if (session) {
        set((draft) => {
          draft.currentSessionId = sessionId;
          draft.currentSession = session;
          draft.error = null;
        });
      } else {
        set((draft) => {
          draft.error = `Session not found: ${sessionId}`;
        });
      }
    },

    addMessage: (content, type = MessageType.TEXT, metadata = {}) => {
      const state = get();
      
      // Create session if none exists
      if (!state.currentSession) {
        get().createSession();
      }

      const messageId = `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      const now = new Date();
      
      const message = {
        id: messageId,
        type,
        content,
        timestamp: now,
        metadata: {
          sender: type === MessageType.ASSISTANT ? 'assistant' : 'user',
          ...metadata
        },
        isStreaming: metadata.isStreaming || false
      };

      set((draft) => {
        const session = draft.sessions.find(s => s.id === draft.currentSessionId);
        if (session) {
          // Add message
          session.messages.push(message);
          
          // Maintain message limit
          if (session.messages.length > draft.maxMessagesPerSession) {
            session.messages = session.messages.slice(-draft.maxMessagesPerSession);
          }
          
          // Update session metadata
          session.updatedAt = now;
          session.metadata.messageCount = session.messages.length;
          session.metadata.lastActivity = now;
          
          // Update current session reference
          draft.currentSession = session;
          
          // Auto-generate title from first user message
          if (!session.title.startsWith('Conversation') && 
              type === MessageType.TEXT && 
              session.messages.length <= 2) {
            const preview = content.slice(0, 50);
            session.title = preview.length < content.length ? `${preview}...` : preview;
          }
        }
      });

      // Auto-save if enabled
      if (get().autoPersist) {
        get().scheduleAutoSave();
      }

      return messageId;
    },

    updateMessage: (messageId, updates) => {
      set((draft) => {
        const session = draft.sessions.find(s => s.id === draft.currentSessionId);
        if (session) {
          const message = session.messages.find(m => m.id === messageId);
          if (message) {
            Object.assign(message, {
              ...updates,
              updatedAt: new Date()
            });
            
            session.updatedAt = new Date();
            session.metadata.lastActivity = new Date();
            draft.currentSession = session;
          }
        }
      });

      if (get().autoPersist) {
        get().scheduleAutoSave();
      }
    },

    deleteMessage: (messageId) => {
      set((draft) => {
        const session = draft.sessions.find(s => s.id === draft.currentSessionId);
        if (session) {
          session.messages = session.messages.filter(m => m.id !== messageId);
          session.metadata.messageCount = session.messages.length;
          session.updatedAt = new Date();
          draft.currentSession = session;
        }
      });

      if (get().autoPersist) {
        get().scheduleAutoSave();
      }
    },

    deleteSession: (sessionId) => {
      set((draft) => {
        draft.sessions = draft.sessions.filter(s => s.id !== sessionId);
        
        // Update current session if deleted
        if (draft.currentSessionId === sessionId) {
          draft.currentSessionId = draft.sessions.length > 0 ? draft.sessions[0].id : null;
          draft.currentSession = draft.sessions.length > 0 ? draft.sessions[0] : null;
        }
      });

      if (get().autoPersist) {
        get().scheduleAutoSave();
      }
    },

    clearAllSessions: () => {
      if (get().autoSaveTimer) {
        clearTimeout(get().autoSaveTimer);
      }

      set((draft) => {
        draft.sessions = [];
        draft.currentSessionId = null;
        draft.currentSession = null;
        draft.searchResults = [];
        draft.autoSaveTimer = null;
      });

      // Clear persisted data
      get().persistSessions();
    },

    persistSessions: async () => {
      const state = get();
      
      set((draft) => {
        draft.isLoading = true;
      });

      try {
        let success = false;
        
        // Try IndexedDB first
        if (DEFAULT_CONFIG.useIndexedDB) {
          success = await StorageUtils.saveToIndexedDB(state.sessions);
        }
        
        // Fallback to localStorage
        if (!success) {
          success = StorageUtils.saveToLocalStorage(state.sessions);
        }
        
        if (success) {
          set((draft) => {
            draft.lastSavedAt = new Date();
            draft.error = null;
          });
        } else {
          throw new Error('Failed to save conversations');
        }
        
      } catch (error) {
        console.error('Persistence error:', error);
        set((draft) => {
          draft.error = `Failed to save: ${error.message}`;
        });
      } finally {
        set((draft) => {
          draft.isLoading = false;
        });
      }
    },

    loadSessions: async () => {
      set((draft) => {
        draft.isLoading = true;
        draft.error = null;
      });

      try {
        let sessions = [];
        
        // Try IndexedDB first
        if (DEFAULT_CONFIG.useIndexedDB) {
          sessions = await StorageUtils.loadFromIndexedDB();
        }
        
        // Fallback to localStorage if IndexedDB failed or empty
        if (sessions.length === 0) {
          sessions = StorageUtils.loadFromLocalStorage();
        }
        
        // Convert date strings back to Date objects
        sessions.forEach(session => {
          session.createdAt = new Date(session.createdAt);
          session.updatedAt = new Date(session.updatedAt);
          session.metadata.lastActivity = new Date(session.metadata.lastActivity);
          
          session.messages.forEach(message => {
            message.timestamp = new Date(message.timestamp);
            if (message.updatedAt) {
              message.updatedAt = new Date(message.updatedAt);
            }
          });
        });
        
        // Sort by last activity
        sessions.sort((a, b) => b.updatedAt - a.updatedAt);
        
        set((draft) => {
          draft.sessions = sessions;
          draft.currentSessionId = sessions.length > 0 ? sessions[0].id : null;
          draft.currentSession = sessions.length > 0 ? sessions[0] : null;
        });
        
      } catch (error) {
        console.error('Load error:', error);
        set((draft) => {
          draft.error = `Failed to load: ${error.message}`;
        });
      } finally {
        set((draft) => {
          draft.isLoading = false;
        });
      }
    },

    searchSessions: (query, filters = {}) => {
      if (!query.trim() && Object.keys(filters).length === 0) {
        set((draft) => {
          draft.searchResults = [];
          draft.isSearching = false;
        });
        return [];
      }

      set((draft) => {
        draft.isSearching = true;
      });

      const state = get();
      const searchTerm = query.toLowerCase();
      
      const results = state.sessions.filter(session => {
        // Text search in title and messages
        let matches = false;
        
        if (searchTerm) {
          matches = session.title.toLowerCase().includes(searchTerm) ||
                   session.messages.some(msg => 
                     msg.content.toLowerCase().includes(searchTerm)
                   );
        } else {
          matches = true; // Include all for filter-only searches
        }
        
        // Apply filters
        if (filters.tags && filters.tags.length > 0) {
          matches = matches && filters.tags.some(tag => 
            session.tags.includes(tag)
          );
        }
        
        if (filters.dateRange) {
          const { start, end } = filters.dateRange;
          if (start) {
            matches = matches && session.updatedAt >= start;
          }
          if (end) {
            matches = matches && session.updatedAt <= end;
          }
        }
        
        if (filters.messageType) {
          matches = matches && session.messages.some(msg => 
            msg.type === filters.messageType
          );
        }
        
        return matches;
      });

      // Sort by relevance (could be enhanced with scoring)
      results.sort((a, b) => b.updatedAt - a.updatedAt);
      
      set((draft) => {
        draft.searchResults = results;
        draft.isSearching = false;
      });

      return results;
    },

    tagSession: (sessionId, tags) => {
      set((draft) => {
        const session = draft.sessions.find(s => s.id === sessionId);
        if (session) {
          session.tags = [...new Set([...session.tags, ...tags])];
          session.updatedAt = new Date();
          
          if (draft.currentSessionId === sessionId) {
            draft.currentSession = session;
          }
        }
      });

      if (get().autoPersist) {
        get().scheduleAutoSave();
      }
    },

    removeTag: (sessionId, tag) => {
      set((draft) => {
        const session = draft.sessions.find(s => s.id === sessionId);
        if (session) {
          session.tags = session.tags.filter(t => t !== tag);
          session.updatedAt = new Date();
          
          if (draft.currentSessionId === sessionId) {
            draft.currentSession = session;
          }
        }
      });

      if (get().autoPersist) {
        get().scheduleAutoSave();
      }
    },

    setError: (error) => {
      set((draft) => {
        draft.error = error;
      });
    },

    resetError: () => {
      set((draft) => {
        draft.error = null;
      });
    },

    // Auto-save functionality
    scheduleAutoSave: () => {
      const state = get();
      
      if (state.autoSaveTimer) {
        clearTimeout(state.autoSaveTimer);
      }

      const timer = setTimeout(() => {
        get().persistSessions();
      }, DEFAULT_CONFIG.autoSaveInterval);

      set((draft) => {
        draft.autoSaveTimer = timer;
      });
    },

    toggleAutoPersist: () => {
      set((draft) => {
        draft.autoPersist = !draft.autoPersist;
        if (!draft.autoPersist && draft.autoSaveTimer) {
          clearTimeout(draft.autoSaveTimer);
          draft.autoSaveTimer = null;
        }
      });
    },

    // Utility getters
    getSessionById: (sessionId) => {
      return get().sessions.find(s => s.id === sessionId) || null;
    },

    getAllTags: () => {
      const allTags = get().sessions.flatMap(s => s.tags);
      return [...new Set(allTags)].sort();
    },

    getStats: () => {
      const state = get();
      return {
        totalSessions: state.sessions.length,
        totalMessages: state.sessions.reduce((sum, s) => sum + s.messages.length, 0),
        currentSessionMessages: state.currentSession ? state.currentSession.messages.length : 0,
        lastSavedAt: state.lastSavedAt,
        autoPersist: state.autoPersist
      };
    },

    exportSessions: (format = 'json') => {
      const state = get();
      const exportData = {
        sessions: state.sessions,
        exportedAt: new Date(),
        version: '1.0',
        format
      };

      if (format === 'json') {
        return JSON.stringify(exportData, null, 2);
      }
      
      // Could add other formats (CSV, Markdown, etc.)
      return exportData;
    },

    importSessions: (data, options = {}) => {
      const { merge = false, validate = true } = options;
      
      try {
        let importedSessions = [];
        
        if (typeof data === 'string') {
          const parsed = JSON.parse(data);
          importedSessions = parsed.sessions || [];
        } else if (Array.isArray(data)) {
          importedSessions = data;
        } else if (data.sessions) {
          importedSessions = data.sessions;
        }
        
        // Validate structure
        if (validate) {
          importedSessions.forEach(session => {
            if (!session.id || !session.messages || !Array.isArray(session.messages)) {
              throw new Error('Invalid session format');
            }
          });
        }
        
        set((draft) => {
          if (merge) {
            // Merge with existing sessions
            const existingIds = draft.sessions.map(s => s.id);
            const newSessions = importedSessions.filter(s => !existingIds.includes(s.id));
            draft.sessions.push(...newSessions);
          } else {
            // Replace all sessions
            draft.sessions = importedSessions;
          }
          
          // Sort by update date
          draft.sessions.sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt));
          
          // Set current session
          if (draft.sessions.length > 0 && !draft.currentSession) {
            draft.currentSessionId = draft.sessions[0].id;
            draft.currentSession = draft.sessions[0];
          }
        });
        
        // Auto-save imported data
        if (get().autoPersist) {
          get().persistSessions();
        }
        
        return true;
      } catch (error) {
        console.error('Import error:', error);
        get().setError(`Import failed: ${error.message}`);
        return false;
      }
    },

    // Cleanup
    cleanup: () => {
      const state = get();
      if (state.autoSaveTimer) {
        clearTimeout(state.autoSaveTimer);
      }
    }
  }))
);