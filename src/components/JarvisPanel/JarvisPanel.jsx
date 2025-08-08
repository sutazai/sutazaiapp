import React, { useState, useEffect, useRef, useCallback } from 'react';
import './JarvisPanel.css';

/**
 * JarvisPanel - Advanced AI Assistant Interface Component
 * 
 * Features:
 * - Voice input using Web Audio API
 * - Real-time streaming responses via WebSocket
 * - File upload support (PDF, DOCX, XLSX)
 * - Responsive design with TailwindCSS
 * - Full accessibility support
 * 
 * API Endpoints:
 * - POST /jarvis/voice/process - Process voice audio
 * - POST /jarvis/task/plan - Plan and execute text tasks
 * - WebSocket /ws - Real-time communication
 * 
 * Usage:
 * ```jsx
 * import { JarvisPanel } from './components/JarvisPanel/JarvisPanel';
 * 
 * function App() {
 *   return (
 *     <div className="app">
 *       <JarvisPanel 
 *         apiBaseUrl="http://localhost:8888"
 *         theme="dark"
 *         onResponse={(response) => console.log(response)}
 *       />
 *     </div>
 *   );
 * }
 * ```
 */

const JarvisPanel = ({ 
  apiBaseUrl = 'http://localhost:8888',
  theme = 'dark',
  onResponse,
  onError,
  maxTranscriptLines = 100,
  enableVoiceInput = true,
  enableFileUpload = true
}) => {
  // Component state
  const [isConnected, setIsConnected] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [transcript, setTranscript] = useState([]);
  const [inputText, setInputText] = useState('');
  const [dragActive, setDragActive] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [status, setStatus] = useState({ text: 'Initializing...', type: 'info' });

  // Refs
  const websocketRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const transcriptEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const streamRef = useRef(null);

  // Audio constraints for voice input
  const audioConstraints = {
    audio: {
      sampleRate: 16000,
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true
    }
  };

  // Supported file types
  const supportedFileTypes = {
    'application/pdf': '.pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
    'text/plain': '.txt',
    'application/json': '.json'
  };

  /**
   * Initialize WebSocket connection
   */
  const initializeWebSocket = useCallback(() => {
    try {
      const wsUrl = apiBaseUrl.replace(/^http/, 'ws') + '/ws';
      websocketRef.current = new WebSocket(wsUrl);

      websocketRef.current.onopen = () => {
        setIsConnected(true);
        setStatus({ text: 'Connected to Jarvis', type: 'success' });
        addToTranscript('system', 'Connected to Jarvis AI Assistant');
      };

      websocketRef.current.onmessage = (event) => {
        try {
          const response = JSON.parse(event.data);
          handleWebSocketMessage(response);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
          setStatus({ text: 'Error processing response', type: 'error' });
        }
      };

      websocketRef.current.onclose = (event) => {
        setIsConnected(false);
        setStatus({ text: 'Disconnected', type: 'warning' });
        
        // Attempt reconnection after 3 seconds
        if (!event.wasClean) {
          setTimeout(() => {
            initializeWebSocket();
          }, 3000);
        }
      };

      websocketRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setStatus({ text: 'Connection error', type: 'error' });
        if (onError) onError(error);
      };
    } catch (error) {
      console.error('Failed to initialize WebSocket:', error);
      setStatus({ text: 'Failed to connect', type: 'error' });
    }
  }, [apiBaseUrl, onError]);

  /**
   * Handle incoming WebSocket messages
   */
  const handleWebSocketMessage = (response) => {
    if (response.result) {
      addToTranscript('assistant', response.result);
      if (onResponse) onResponse(response);
    }
    
    if (response.status === 'processing') {
      setStatus({ text: 'Processing...', type: 'info' });
    } else if (response.status === 'completed') {
      setStatus({ text: 'Ready', type: 'success' });
      setIsProcessing(false);
    } else if (response.status === 'error') {
      setStatus({ text: 'Error occurred', type: 'error' });
      setIsProcessing(false);
    }

    // Handle voice response
    if (response.voice_response && 'speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(response.voice_response);
      utterance.rate = 0.9;
      utterance.pitch = 1.0;
      speechSynthesis.speak(utterance);
    }
  };

  /**
   * Add message to transcript
   */
  const addToTranscript = (sender, message, metadata = {}) => {
    const entry = {
      id: Date.now() + Math.random(),
      sender,
      message,
      timestamp: new Date(),
      metadata
    };

    setTranscript(prev => {
      const updated = [...prev, entry];
      return updated.slice(-maxTranscriptLines);
    });

    // Auto-scroll to bottom
    setTimeout(() => {
      transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, 100);
  };

  /**
   * Start voice recording
   */
  const startRecording = async () => {
    if (!enableVoiceInput || !navigator.mediaDevices?.getUserMedia) {
      setStatus({ text: 'Voice input not supported', type: 'error' });
      return;
    }

    try {
      streamRef.current = await navigator.mediaDevices.getUserMedia(audioConstraints);
      
      mediaRecorderRef.current = new MediaRecorder(streamRef.current, {
        mimeType: 'audio/webm;codecs=opus'
      });

      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current.onstop = () => {
        processVoiceRecording();
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      setStatus({ text: 'Recording...', type: 'info' });
      addToTranscript('system', '🎤 Recording started - speak now');
    } catch (error) {
      console.error('Error starting recording:', error);
      setStatus({ text: 'Microphone access denied', type: 'error' });
    }
  };

  /**
   * Stop voice recording
   */
  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    }
  };

  /**
   * Process recorded voice audio
   */
  const processVoiceRecording = async () => {
    if (audioChunksRef.current.length === 0) return;

    setIsProcessing(true);
    setStatus({ text: 'Processing voice...', type: 'info' });

    try {
      const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
      const formData = new FormData();
      formData.append('audio', audioBlob, 'voice_input.webm');

      const response = await fetch(`${apiBaseUrl}/jarvis/voice/process`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      if (result.transcript) {
        addToTranscript('user', result.transcript, { type: 'voice' });
      }

      if (result.result) {
        addToTranscript('assistant', result.result);
        if (onResponse) onResponse(result);
      }

    } catch (error) {
      console.error('Error processing voice:', error);
      setStatus({ text: 'Voice processing failed', type: 'error' });
      addToTranscript('system', '❌ Voice processing failed');
    } finally {
      setIsProcessing(false);
      audioChunksRef.current = [];
    }
  };

  /**
   * Send text message
   */
  const sendTextMessage = async (message = inputText.trim()) => {
    if (!message || !isConnected) return;

    setIsProcessing(true);
    addToTranscript('user', message, { type: 'text' });
    setInputText('');

    try {
      if (websocketRef.current?.readyState === WebSocket.OPEN) {
        // Use WebSocket for real-time streaming
        websocketRef.current.send(JSON.stringify({
          command: message,
          context: {
            uploaded_files: uploadedFiles.map(f => f.name),
            timestamp: new Date().toISOString()
          },
          voice_enabled: false
        }));
      } else {
        // Fallback to REST API
        const response = await fetch(`${apiBaseUrl}/jarvis/task/plan`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            command: message,
            context: {
              uploaded_files: uploadedFiles.map(f => f.name)
            },
            voice_enabled: false
          })
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        addToTranscript('assistant', result.result);
        if (onResponse) onResponse(result);
        setIsProcessing(false);
      }
    } catch (error) {
      console.error('Error sending message:', error);
      setStatus({ text: 'Failed to send message', type: 'error' });
      addToTranscript('system', '❌ Failed to send message');
      setIsProcessing(false);
    }
  };

  /**
   * Handle file upload
   */
  const handleFileUpload = (files) => {
    if (!enableFileUpload) return;

    const validFiles = Array.from(files).filter(file => {
      return Object.keys(supportedFileTypes).includes(file.type);
    });

    if (validFiles.length === 0) {
      setStatus({ text: 'Unsupported file type', type: 'error' });
      return;
    }

    setUploadedFiles(prev => [...prev, ...validFiles]);
    validFiles.forEach(file => {
      addToTranscript('system', `📎 Uploaded: ${file.name} (${(file.size / 1024).toFixed(1)}KB)`);
    });
  };

  /**
   * Handle drag and drop events
   */
  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDragIn = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer.items && e.dataTransfer.items.length > 0) {
      setDragActive(true);
    }
  }, []);

  const handleDragOut = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files);
    }
  }, []);

  /**
   * Handle keyboard events
   */
  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      if (e.shiftKey) {
        // Allow line break with Shift+Enter
        return;
      } else {
        e.preventDefault();
        sendTextMessage();
      }
    }
    
    // Handle voice activation with Ctrl+Space
    if (e.ctrlKey && e.code === 'Space') {
      e.preventDefault();
      if (isRecording) {
        stopRecording();
      } else {
        startRecording();
      }
    }
  };

  /**
   * Remove uploaded file
   */
  const removeFile = (index) => {
    setUploadedFiles(prev => prev.filter((_, i) => i !== index));
  };

  /**
   * Clear transcript
   */
  const clearTranscript = () => {
    setTranscript([]);
    addToTranscript('system', 'Transcript cleared');
  };

  // Initialize component
  useEffect(() => {
    initializeWebSocket();

    // Cleanup on unmount
    return () => {
      if (websocketRef.current) {
        websocketRef.current.close();
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, [initializeWebSocket]);

  // Add drag event listeners
  useEffect(() => {
    const container = document.querySelector('.jarvis-panel');
    if (container) {
      container.addEventListener('dragenter', handleDragIn);
      container.addEventListener('dragleave', handleDragOut);
      container.addEventListener('dragover', handleDrag);
      container.addEventListener('drop', handleDrop);

      return () => {
        container.removeEventListener('dragenter', handleDragIn);
        container.removeEventListener('dragleave', handleDragOut);
        container.removeEventListener('dragover', handleDrag);
        container.removeEventListener('drop', handleDrop);
      };
    }
  }, [handleDrag, handleDragIn, handleDragOut, handleDrop]);

  const renderStatusIndicator = () => {
    const statusIcons = {
      success: '✅',
      error: '❌',
      warning: '⚠️',
      info: '💡'
    };

    return (
      <div className={`status-indicator status-${status.type}`}>
        <span className="status-icon">{statusIcons[status.type]}</span>
        <span className="status-text">{status.text}</span>
        <div className={`connection-dot ${isConnected ? 'connected' : 'disconnected'}`} />
      </div>
    );
  };

  const renderTranscript = () => {
    return (
      <div className="transcript-container" role="log" aria-live="polite" aria-label="Conversation transcript">
        {transcript.length === 0 ? (
          <div className="transcript-placeholder">
            <p>Welcome to Jarvis AI Assistant</p>
            <p>Start a conversation by typing below or using voice input</p>
          </div>
        ) : (
          transcript.map((entry) => (
            <div 
              key={entry.id}
              className={`transcript-entry ${entry.sender}`}
              role="article"
              aria-label={`${entry.sender} message`}
            >
              <div className="entry-header">
                <span className="entry-sender">
                  {entry.sender === 'user' ? '👤' : entry.sender === 'assistant' ? '🤖' : '⚙️'}
                  {entry.sender}
                </span>
                <span className="entry-timestamp">
                  {entry.timestamp.toLocaleTimeString()}
                </span>
                {entry.metadata?.type && (
                  <span className="entry-type">{entry.metadata.type}</span>
                )}
              </div>
              <div className="entry-content">{entry.message}</div>
            </div>
          ))
        )}
        <div ref={transcriptEndRef} />
      </div>
    );
  };

  const renderFileArea = () => {
    if (!enableFileUpload) return null;

    return (
      <div className={`file-upload-area ${dragActive ? 'drag-active' : ''}`}>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept={Object.values(supportedFileTypes).join(',')}
          onChange={(e) => handleFileUpload(e.target.files)}
          className="file-input"
          aria-label="Upload files"
        />
        
        {uploadedFiles.length > 0 && (
          <div className="uploaded-files">
            {uploadedFiles.map((file, index) => (
              <div key={index} className="uploaded-file">
                <span className="file-name">{file.name}</span>
                <button
                  onClick={() => removeFile(index)}
                  className="remove-file"
                  aria-label={`Remove ${file.name}`}
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        )}

        {dragActive && (
          <div className="drop-overlay">
            <div className="drop-message">
              📎 Drop files here to upload
            </div>
          </div>
        )}

        <button
          onClick={() => fileInputRef.current?.click()}
          className="upload-button"
          disabled={!enableFileUpload}
          aria-label="Choose files to upload"
        >
          📎 Upload Files
        </button>
      </div>
    );
  };

  return (
    <div className={`jarvis-panel ${theme}`} role="application" aria-label="Jarvis AI Assistant">
      {/* Header */}
      <header className="jarvis-header">
        <div className="header-title">
          <h1>🤖 Jarvis AI Assistant</h1>
          {renderStatusIndicator()}
        </div>
        <div className="header-controls">
          <button
            onClick={clearTranscript}
            className="control-button"
            aria-label="Clear conversation transcript"
          >
            🗑️ Clear
          </button>
          <button
            onClick={() => setIsConnected(prev => !prev)}
            className={`control-button ${isConnected ? 'connected' : 'disconnected'}`}
            aria-label={isConnected ? 'Disconnect' : 'Connect'}
          >
            {isConnected ? '🔌 Connected' : '🔌 Disconnected'}
          </button>
        </div>
      </header>

      {/* Transcript Area */}
      <main className="transcript-area">
        {renderTranscript()}
      </main>

      {/* File Upload Area */}
      {renderFileArea()}

      {/* Input Area */}
      <footer className="input-area">
        <div className="input-container">
          <textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type your message here... (Enter to send, Shift+Enter for new line, Ctrl+Space for voice)"
            className="text-input"
            disabled={!isConnected || isProcessing}
            rows={3}
            aria-label="Message input"
          />
          
          <div className="input-controls">
            {enableVoiceInput && (
              <button
                onClick={isRecording ? stopRecording : startRecording}
                className={`voice-button ${isRecording ? 'recording' : ''}`}
                disabled={!isConnected || isProcessing}
                aria-label={isRecording ? 'Stop recording' : 'Start voice recording'}
              >
                {isRecording ? '🛑 Stop' : '🎤 Voice'}
              </button>
            )}
            
            <button
              onClick={() => sendTextMessage()}
              className="send-button"
              disabled={!inputText.trim() || !isConnected || isProcessing}
              aria-label="Send message"
            >
              {isProcessing ? '⏳ Processing...' : '📤 Send'}
            </button>
          </div>
        </div>

        {/* Keyboard shortcuts hint */}
        <div className="shortcuts-hint">
          <small>
            💡 Tips: Enter to send, Shift+Enter for new line, Ctrl+Space for voice input
          </small>
        </div>
      </footer>
    </div>
  );
};

export default JarvisPanel;
