# SutazAI State Management Documentation

This documentation provides comprehensive information about the Zustand-based state management architecture implemented for the SutazAI frontend application.

## Overview

The state management system is built using [Zustand](https://zustand-demo.pmnd.rs/) for simplicity and performance, with [Immer](https://immerjs.github.io/immer/) middleware for immutable state updates. The architecture is designed to handle:

- Voice recording and processing
- Text input and submission
- Real-time streaming responses
- Conversation history and persistence
- Sidebar navigation and filtering

## Architecture

### Store Structure

```
src/store/
├── index.js                 # Central exports
├── types.js                 # Type definitions and constants
├── voiceStore.js           # Voice recording state
├── textInputStore.js       # Text input management
├── streamingStore.js       # WebSocket streaming
├── conversationStore.js    # Conversation history
├── sidebarStore.js         # Sidebar and filtering
├── README.md               # This documentation
└── __tests__/              # Test suite
    ├── voiceStore.test.js
    ├── conversationStore.test.js
    └── integration.test.js
```

### State Shape

Each store manages a specific domain of the application state:

```javascript
// Voice Store
{
  status: 'idle' | 'recording' | 'processing' | 'error',
  audioBlob: Blob | null,
  transcript: string | null,
  recordingDuration: number,
  error: string | null,
  // ... additional state
}

// Text Input Store
{
  currentText: string,
  isSubmitting: boolean,
  history: Array<HistoryItem>,
  validationErrors: Array<string>,
  // ... additional state
}

// Streaming Store
{
  status: 'idle' | 'connecting' | 'streaming' | 'complete' | 'error',
  chunks: Array<Chunk>,
  currentResponse: string,
  connection: WebSocket | null,
  // ... additional state
}

// Conversation Store
{
  sessions: Array<ConversationSession>,
  currentSessionId: string | null,
  currentSession: ConversationSession | null,
  isLoading: boolean,
  // ... additional state
}

// Sidebar Store
{
  isOpen: boolean,
  activeFilter: FilterType,
  searchQuery: string,
  selectedTags: Array<string>,
  filteredSessions: Array<ConversationSession>,
  // ... additional state
}
```

## Store Documentation

### Voice Store (`voiceStore.js`)

Handles voice input recording, processing, and audio blob storage using the Web Audio API and MediaRecorder.

#### Key Features
- **Voice Recording**: Start/stop recording with MediaRecorder API
- **Audio Processing**: Send audio to backend for transcription
- **Error Handling**: Comprehensive error handling for microphone access and API calls
- **Audio Management**: Download, play, and manage recorded audio
- **Performance**: Optimized for real-time recording with configurable constraints

#### Usage Example
```javascript
import { useVoiceStore } from './store/voiceStore';

function VoiceComponent() {
  const {
    status,
    audioBlob,
    transcript,
    startRecording,
    stopRecording,
    processAudio,
    clearAudio,
    isRecording,
    hasError
  } = useVoiceStore();

  const handleRecord = async () => {
    if (isRecording()) {
      stopRecording();
    } else {
      await startRecording();
    }
  };

  const handleProcess = async () => {
    const result = await processAudio('http://api.example.com');
    if (result) {
      console.log('Transcript:', result.transcript);
    }
  };

  return (
    <div>
      <button onClick={handleRecord}>
        {isRecording() ? 'Stop Recording' : 'Start Recording'}
      </button>
      {audioBlob && (
        <button onClick={handleProcess}>Process Audio</button>
      )}
      {transcript && <p>Transcript: {transcript}</p>}
    </div>
  );
}
```

#### Audio Constraints
The voice store uses optimized audio constraints for voice recording:
```javascript
{
  audio: {
    sampleRate: 16000,        // Optimal for speech recognition
    channelCount: 1,          // Mono audio
    echoCancellation: true,   // Reduce echo
    noiseSuppression: true,   // Reduce background noise
    autoGainControl: true     // Automatic volume adjustment
  }
}
```

### Text Input Store (`textInputStore.js`)

Manages text input state, validation, history, and submission workflows.

#### Key Features
- **Input Management**: Real-time text input with validation
- **History**: Command history with navigation (↑/↓ arrows)
- **Validation**: Configurable input validation with error messages
- **Debouncing**: Debounced validation for performance
- **Keyboard Shortcuts**: Built-in keyboard navigation support

#### Usage Example
```javascript
import { useTextInputStore } from './store/textInputStore';

function TextInput() {
  const {
    currentText,
    isSubmitting,
    history,
    validationErrors,
    setText,
    submitText,
    navigateHistory,
    canSubmit,
    handleKeyDown
  } = useTextInputStore();

  const onSubmit = async (text) => {
    // Send to backend
    const response = await fetch('/api/message', {
      method: 'POST',
      body: JSON.stringify({ message: text })
    });
    return response.json();
  };

  const onKeyDown = (e) => {
    const action = handleKeyDown(e);
    if (action === 'submit') {
      submitText(onSubmit);
    }
  };

  return (
    <div>
      <textarea
        value={currentText}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKeyDown}
        disabled={isSubmitting}
      />
      <button
        onClick={() => submitText(onSubmit)}
        disabled={!canSubmit()}
      >
        {isSubmitting ? 'Sending...' : 'Send'}
      </button>
      {validationErrors.map((error, i) => (
        <div key={i} className="error">{error}</div>
      ))}
    </div>
  );
}
```

### Streaming Store (`streamingStore.js`)

Manages WebSocket connections for real-time streaming responses with automatic reconnection.

#### Key Features
- **WebSocket Management**: Connection, reconnection, and error handling
- **Chunk Processing**: Efficient handling of streaming response chunks
- **Performance**: Buffered chunk processing for optimal rendering
- **Reliability**: Automatic reconnection with exponential backoff
- **Health Monitoring**: Ping/pong for connection health

#### Usage Example
```javascript
import { useStreamingStore } from './store/streamingStore';

function StreamingChat() {
  const {
    status,
    currentResponse,
    chunks,
    connect,
    disconnect,
    sendMessage,
    isConnected,
    isStreaming
  } = useStreamingStore();

  useEffect(() => {
    connect('ws://localhost:8888/ws');
    return () => disconnect();
  }, []);

  const sendChat = (message) => {
    if (isConnected()) {
      sendMessage({ type: 'chat', content: message });
    }
  };

  return (
    <div>
      <div>Status: {status}</div>
      <div>Response: {currentResponse}</div>
      {isStreaming() && <div>Streaming... ({chunks.length} chunks)</div>}
    </div>
  );
}
```

### Conversation Store (`conversationStore.js`)

Manages conversation sessions, messages, and persistence with IndexedDB/localStorage.

#### Key Features
- **Session Management**: Create, update, delete conversation sessions
- **Message Handling**: Add, update, delete messages within sessions
- **Persistence**: Automatic saving to IndexedDB with localStorage fallback
- **Search**: Full-text search across conversations and messages
- **Tagging**: Tag-based organization and filtering
- **Import/Export**: Data portability and backup functionality

#### Data Models

```javascript
// ConversationSession
{
  id: string,
  title: string,
  messages: Array<Message>,
  createdAt: Date,
  updatedAt: Date,
  tags: Array<string>,
  metadata: {
    messageCount: number,
    lastActivity: Date,
    // ... additional metadata
  }
}

// Message
{
  id: string,
  type: 'text' | 'voice' | 'assistant' | 'system',
  content: string,
  timestamp: Date,
  metadata: {
    sender: string,
    confidence?: number,
    // ... additional metadata
  }
}
```

#### Usage Example
```javascript
import { useConversationStore } from './store/conversationStore';

function ConversationManager() {
  const {
    sessions,
    currentSession,
    createSession,
    addMessage,
    searchSessions,
    tagSession,
    loadSessions
  } = useConversationStore();

  useEffect(() => {
    loadSessions();
  }, []);

  const startNewConversation = () => {
    const sessionId = createSession('New Conversation');
    addMessage('Hello!', 'text');
  };

  const findConversations = (query) => {
    return searchSessions(query, {
      tags: ['important'],
      dateRange: { start: new Date('2024-01-01') }
    });
  };

  return (
    <div>
      <button onClick={startNewConversation}>New Conversation</button>
      <div>Total sessions: {sessions.length}</div>
      {currentSession && (
        <div>
          <h3>{currentSession.title}</h3>
          <div>Messages: {currentSession.messages.length}</div>
        </div>
      )}
    </div>
  );
}
```

### Sidebar Store (`sidebarStore.js`)

Manages sidebar visibility, filtering, and search functionality.

#### Key Features
- **Filtering**: Multiple filter types (time, tags, search)
- **Search**: Real-time search with debouncing
- **Tag Management**: Tag-based filtering with multi-select
- **Date Filtering**: Flexible date range filtering
- **Performance**: Debounced filter application for optimal UX

#### Filter Types
```javascript
export const FilterType = {
  ALL: 'all',
  TODAY: 'today',
  WEEK: 'week', 
  MONTH: 'month',
  KEYWORD: 'keyword',
  TAG: 'tag'
};
```

#### Usage Example
```javascript
import { useSidebarStore } from './store/sidebarStore';
import { useConversationStore } from './store/conversationStore';

function Sidebar() {
  const { sessions } = useConversationStore();
  const {
    isOpen,
    filteredSessions,
    searchQuery,
    selectedTags,
    setSearchQuery,
    addSelectedTag,
    applyFilters,
    toggleSidebar
  } = useSidebarStore();

  useEffect(() => {
    applyFilters(sessions);
  }, [sessions, applyFilters]);

  return (
    <aside className={`sidebar ${isOpen ? 'open' : 'closed'}`}>
      <input
        value={searchQuery}
        onChange={(e) => setSearchQuery(e.target.value)}
        placeholder="Search conversations..."
      />
      <div>
        {filteredSessions.map(session => (
          <div key={session.id}>{session.title}</div>
        ))}
      </div>
    </aside>
  );
}
```

## Integration Patterns

### Cross-Store Communication

Stores are designed to be used together in complex workflows:

```javascript
// Complete workflow example
function useCompleteWorkflow() {
  const voiceStore = useVoiceStore();
  const textInputStore = useTextInputStore();
  const streamingStore = useStreamingStore();
  const conversationStore = useConversationStore();

  const handleVoiceMessage = async () => {
    // 1. Record voice
    await voiceStore.startRecording();
    // ... user speaks ...
    voiceStore.stopRecording();
    
    // 2. Process voice
    const result = await voiceStore.processAudio();
    
    // 3. Add to conversation
    if (result?.transcript) {
      const messageId = conversationStore.addMessage(
        result.transcript,
        'voice',
        { confidence: result.confidence }
      );
      
      // 4. Send for AI response via streaming
      streamingStore.sendMessage({
        type: 'voice',
        content: result.transcript,
        messageId
      });
    }
  };

  const handleTextMessage = async () => {
    // 1. Submit text
    const submitResult = await textInputStore.submitText(async (text) => {
      // 2. Add to conversation
      const messageId = conversationStore.addMessage(text, 'text');
      
      // 3. Send via streaming
      return streamingStore.sendMessage({
        type: 'text',
        content: text,
        messageId
      });
    });
  };

  return { handleVoiceMessage, handleTextMessage };
}
```

### Error Handling Patterns

Each store implements consistent error handling:

```javascript
// Error handling example
function useErrorHandling() {
  const stores = [
    useVoiceStore(),
    useTextInputStore(),
    useStreamingStore(),
    useConversationStore(),
    useSidebarStore()
  ];

  const hasAnyError = stores.some(store => 
    store.hasError && store.hasError()
  );

  const getAllErrors = () => stores
    .filter(store => store.error)
    .map(store => store.error);

  const clearAllErrors = () => {
    stores.forEach(store => {
      if (store.resetError) {
        store.resetError();
      }
    });
  };

  return { hasAnyError, getAllErrors, clearAllErrors };
}
```

## Performance Considerations

### Optimization Strategies

1. **Debounced Updates**: Search and filter operations are debounced to prevent excessive re-renders
2. **Chunk Buffering**: Streaming responses use buffered chunk processing
3. **Selective Re-renders**: Zustand's selector pattern minimizes unnecessary re-renders
4. **Memory Management**: Automatic cleanup of resources and timers

### Best Practices

```javascript
// Use selectors to prevent unnecessary re-renders
const currentText = useTextInputStore(state => state.currentText);
const isSubmitting = useTextInputStore(state => state.isSubmitting);

// Instead of:
const { currentText, isSubmitting, ...everything } = useTextInputStore();

// Cleanup resources in useEffect
useEffect(() => {
  const cleanup = () => {
    voiceStore.clearAudio();
    streamingStore.disconnect();
  };
  
  return cleanup;
}, []);
```

## Testing

The state management system includes comprehensive tests:

- **Unit Tests**: Individual store functionality
- **Integration Tests**: Cross-store workflows
- **Performance Tests**: Large dataset handling
- **Error Scenarios**: Error handling and recovery

### Running Tests

```bash
# Run all tests
npm test

# Run with coverage
npm run test:coverage

# Run specific test file
npm test voiceStore.test.js

# Run in watch mode
npm test --watch
```

### Test Structure

```javascript
// Example test structure
describe('useVoiceStore', () => {
  describe('Initial State', () => {
    test('should have correct initial state', () => {
      // Test initial state
    });
  });

  describe('Voice Recording', () => {
    test('should start recording successfully', () => {
      // Test recording functionality
    });
  });

  describe('Error Handling', () => {
    test('should handle microphone access denied', () => {
      // Test error scenarios
    });
  });
});
```

## Migration Guide

### From Redux to Zustand

If migrating from Redux, here are the key differences:

```javascript
// Redux pattern
const mapStateToProps = (state) => ({
  text: state.textInput.currentText,
  isSubmitting: state.textInput.isSubmitting
});

const mapDispatchToProps = {
  setText: setTextAction,
  submitText: submitTextAction
};

export default connect(mapStateToProps, mapDispatchToProps)(Component);

// Zustand pattern
function Component() {
  const { currentText, isSubmitting, setText, submitText } = useTextInputStore();
  // Use directly in component
}
```

### Key Migration Benefits

- **Reduced Boilerplate**: No actions, reducers, or middleware setup
- **Better TypeScript**: Automatic type inference
- **Simpler Testing**: Direct function calls instead of action dispatching
- **Performance**: Built-in selector optimization

## Troubleshooting

### Common Issues

1. **WebSocket Connection Fails**
   ```javascript
   // Check connection status
   const { isConnected, error } = useStreamingStore();
   if (!isConnected() && error) {
     console.log('Connection error:', error);
   }
   ```

2. **Voice Recording Not Working**
   ```javascript
   // Check browser support and permissions
   const { isSupported, error } = useVoiceStore();
   if (!isSupported) {
     console.log('Voice recording not supported');
   }
   ```

3. **Persistence Issues**
   ```javascript
   // Check storage availability
   const { error } = useConversationStore();
   if (error?.includes('storage')) {
     console.log('Storage error - try clearing browser data');
   }
   ```

### Debug Tools

Enable debug logging:

```javascript
// Add to your app initialization
if (process.env.NODE_ENV === 'development') {
  window.debugStores = {
    voice: useVoiceStore.getState(),
    textInput: useTextInputStore.getState(),
    streaming: useStreamingStore.getState(),
    conversation: useConversationStore.getState(),
    sidebar: useSidebarStore.getState()
  };
}
```

## Contributing

### Adding New Stores

1. Create new store file in `src/store/`
2. Follow existing patterns for structure and naming
3. Add comprehensive tests
4. Update this documentation
5. Add to central exports in `index.js`

### Store Template

```javascript
import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';

export const useNewStore = create(
  immer((set, get) => ({
    // State
    someState: initialValue,
    
    // Actions
    someAction: (param) => {
      set((draft) => {
        draft.someState = param;
      });
    },
    
    // Getters
    getSomeValue: () => get().someState,
    
    // Cleanup
    cleanup: () => {
      // Cleanup logic
    }
  }))
);
```

## Roadmap

### Planned Features

- [ ] Offline support with service worker
- [ ] Real-time collaboration features
- [ ] Advanced search with fuzzy matching
- [ ] Voice commands for navigation
- [ ] Conversation templates and macros
- [ ] Analytics and usage tracking
- [ ] Plugin system for extending stores

### Performance Improvements

- [ ] Virtual scrolling for large conversation lists
- [ ] Background persistence with web workers
- [ ] Optimistic updates for better UX
- [ ] Streaming response caching

---

For more information or questions about the state management system, please refer to the individual store files or create an issue in the project repository.