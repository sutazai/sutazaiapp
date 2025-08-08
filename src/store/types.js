/**
 * Type Definitions for Zustand Stores
 * 
 * Centralized type definitions for all store states and actions
 */

// Voice Recording Types
export const VoiceRecordingStatus = {
  IDLE: 'idle',
  RECORDING: 'recording', 
  PROCESSING: 'processing',
  ERROR: 'error'
};

export const MessageType = {
  TEXT: 'text',
  VOICE: 'voice',
  SYSTEM: 'system',
  ASSISTANT: 'assistant'
};

export const StreamingStatus = {
  IDLE: 'idle',
  CONNECTING: 'connecting',
  STREAMING: 'streaming', 
  COMPLETE: 'complete',
  ERROR: 'error'
};

export const FilterType = {
  ALL: 'all',
  TODAY: 'today',
  WEEK: 'week',
  MONTH: 'month',
  KEYWORD: 'keyword',
  TAG: 'tag'
};

// Voice Store Types (for TypeScript documentation)
/**
 * @typedef {Object} VoiceState
 * @property {string} status - Current recording status
 * @property {Blob|null} audioBlob - Recorded audio blob
 * @property {MediaRecorder|null} mediaRecorder - MediaRecorder instance
 * @property {MediaStream|null} stream - Media stream
 * @property {Array} audioChunks - Audio data chunks
 * @property {string|null} transcript - Transcribed text
 * @property {string|null} error - Error message
 * @property {boolean} isSupported - Voice input support
 * @property {number} recordingDuration - Duration in seconds
 */

/**
 * @typedef {Object} VoiceActions  
 * @property {Function} startRecording - Start voice recording
 * @property {Function} stopRecording - Stop voice recording
 * @property {Function} processAudio - Process recorded audio
 * @property {Function} clearAudio - Clear audio data
 * @property {Function} setError - Set error state
 * @property {Function} resetError - Clear error state
 */

// Text Input Store Types
/**
 * @typedef {Object} TextInputState
 * @property {string} currentText - Current input text
 * @property {boolean} isSubmitting - Submission status
 * @property {Array} history - Input history
 * @property {string|null} error - Error message
 * @property {number} maxHistory - Max history items
 */

/**
 * @typedef {Object} TextInputActions
 * @property {Function} setText - Set input text
 * @property {Function} clearText - Clear input text
 * @property {Function} submitText - Submit text message
 * @property {Function} addToHistory - Add to input history
 * @property {Function} getFromHistory - Get from history by index
 * @property {Function} clearHistory - Clear input history
 * @property {Function} setError - Set error state
 * @property {Function} resetError - Clear error state
 */

// Streaming Store Types
/**
 * @typedef {Object} StreamingState
 * @property {string} status - Streaming status
 * @property {Array} chunks - Streaming response chunks
 * @property {string} currentResponse - Accumulated response
 * @property {WebSocket|null} connection - WebSocket connection
 * @property {string|null} error - Error message
 * @property {boolean} autoReconnect - Auto reconnect flag
 * @property {number} reconnectAttempts - Reconnect attempt count
 * @property {number} maxReconnectAttempts - Max reconnect attempts
 */

/**
 * @typedef {Object} StreamingActions
 * @property {Function} connect - Connect to streaming service
 * @property {Function} disconnect - Disconnect from service
 * @property {Function} sendMessage - Send streaming message
 * @property {Function} addChunk - Add response chunk
 * @property {Function} clearChunks - Clear response chunks
 * @property {Function} setError - Set error state
 * @property {Function} resetError - Clear error state
 * @property {Function} toggleAutoReconnect - Toggle auto reconnect
 */

// Conversation Store Types
/**
 * @typedef {Object} Message
 * @property {string} id - Unique message ID
 * @property {string} type - Message type (text/voice/system/assistant)
 * @property {string} content - Message content
 * @property {Date} timestamp - Message timestamp
 * @property {Object} metadata - Additional metadata
 * @property {boolean} isStreaming - Streaming status
 */

/**
 * @typedef {Object} ConversationSession
 * @property {string} id - Unique session ID
 * @property {string} title - Session title
 * @property {Array<Message>} messages - Session messages
 * @property {Date} createdAt - Creation timestamp
 * @property {Date} updatedAt - Last update timestamp
 * @property {Array<string>} tags - Session tags
 * @property {Object} metadata - Additional metadata
 */

/**
 * @typedef {Object} ConversationState
 * @property {Array<ConversationSession>} sessions - All conversation sessions
 * @property {string|null} currentSessionId - Active session ID
 * @property {ConversationSession|null} currentSession - Active session
 * @property {boolean} isLoading - Loading state
 * @property {string|null} error - Error message
 * @property {number} maxSessions - Maximum stored sessions
 * @property {boolean} autoPersist - Auto persistence flag
 */

/**
 * @typedef {Object} ConversationActions
 * @property {Function} createSession - Create new conversation session
 * @property {Function} setCurrentSession - Set active session
 * @property {Function} addMessage - Add message to current session
 * @property {Function} updateMessage - Update existing message
 * @property {Function} deleteMessage - Delete message
 * @property {Function} deleteSession - Delete conversation session
 * @property {Function} clearAllSessions - Clear all sessions
 * @property {Function} persistSessions - Save sessions to storage
 * @property {Function} loadSessions - Load sessions from storage
 * @property {Function} searchSessions - Search through sessions
 * @property {Function} tagSession - Add tags to session
 * @property {Function} setError - Set error state
 * @property {Function} resetError - Clear error state
 */

// Sidebar Store Types  
/**
 * @typedef {Object} SidebarState
 * @property {boolean} isOpen - Sidebar open state
 * @property {string} activeFilter - Current active filter
 * @property {string} searchQuery - Search query string
 * @property {Array<string>} selectedTags - Selected filter tags
 * @property {Object} dateRange - Date filter range
 * @property {Array<ConversationSession>} filteredSessions - Filtered sessions
 * @property {boolean} isLoading - Loading state
 * @property {string|null} error - Error message
 */

/**
 * @typedef {Object} SidebarActions
 * @property {Function} toggleSidebar - Toggle sidebar open/close
 * @property {Function} setFilter - Set active filter
 * @property {Function} setSearchQuery - Set search query
 * @property {Function} addSelectedTag - Add tag to selection
 * @property {Function} removeSelectedTag - Remove tag from selection
 * @property {Function} clearSelectedTags - Clear all selected tags
 * @property {Function} setDateRange - Set date filter range
 * @property {Function} applyFilters - Apply all active filters
 * @property {Function} clearFilters - Clear all filters
 * @property {Function} setError - Set error state
 * @property {Function} resetError - Clear error state
 */