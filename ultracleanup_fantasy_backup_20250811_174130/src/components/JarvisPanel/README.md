# JarvisPanel React Component

A comprehensive React component for interfacing with the Jarvis AI system, featuring voice input, real-time streaming, file uploads, and full accessibility support.

## Features

### Core Functionality
- **Voice Input**: Web Audio API integration with microphone access
- **Real-time Communication**: WebSocket-based streaming responses
- **File Upload**: Drag-and-drop support for PDF, DOCX, XLSX files
- **Responsive Design**: Mobile-first design with TailwindCSS
- **Accessibility**: Full WCAG compliance with ARIA labels and keyboard navigation
- **Theme Support**: Dark/Light theme switching

### API Integration
- **Voice Processing**: `POST /jarvis/voice/process` - Process audio uploads
- **Task Planning**: `POST /jarvis/task/plan` - Execute text-based tasks
- **Real-time Streaming**: `WebSocket /ws` - Live communication channel

## Installation

### Prerequisites
- React 16.8+ (uses hooks)
- TailwindCSS configured
- Modern browser with WebSocket and Web Audio API support

### Installation Steps

1. **Copy component files** to your React project:
```bash
cp -r src/components/JarvisPanel /path/to/your/react-app/src/components/
```

2. **Install peer dependencies** (if not already installed):
```bash
npm install react react-dom
```

3. **Configure TailwindCSS** in your project's `tailwind.config.js`:
```javascript
module.exports = {
  content: [
    // ... your existing paths
    "./src/components/JarvisPanel/**/*.{js,jsx,ts,tsx}",
  ],
  darkMode: 'class', // Enable dark mode support
  theme: {
    extend: {
      // Custom theme extensions if needed
    },
  },
  plugins: [],
}
```

4. **Import component styles** in your main CSS file:
```css
@import './components/JarvisPanel/JarvisPanel.css';
```

## Usage

### Basic Usage

```jsx
import React from 'react';
import JarvisPanel from './components/JarvisPanel/JarvisPanel';

function App() {
  return (
    <div className="min-h-screen bg-gray-100">
      <JarvisPanel 
        apiBaseUrl="http://localhost:8888"
        theme="dark"
      />
    </div>
  );
}
```

### Advanced Usage with Event Handlers

```jsx
import React, { useState } from 'react';
import JarvisPanel from './components/JarvisPanel/JarvisPanel';

function AIAssistantApp() {
  const [responses, setResponses] = useState([]);
  const [errors, setErrors] = useState([]);

  const handleResponse = (response) => {
    console.log('Jarvis Response:', response);
    setResponses(prev => [...prev, response]);
    
    // Custom response handling
    if (response.result) {
      // Process the response
      console.log('Task completed:', response.result);
    }
  };

  const handleError = (error) => {
    console.error('Jarvis Error:', error);
    setErrors(prev => [...prev, error]);
  };

  return (
    <div className="min-h-screen bg-gray-900 p-4">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-3xl font-bold text-white mb-8 text-center">
          🤖 AI Assistant Dashboard
        </h1>
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Jarvis Panel */}
          <div className="lg:col-span-2">
            <JarvisPanel
              apiBaseUrl={process.env.REACT_APP_JARVIS_API || "http://localhost:8888"}
              theme="dark"
              maxTranscriptLines={200}
              enableVoiceInput={true}
              enableFileUpload={true}
              onResponse={handleResponse}
              onError={handleError}
            />
          </div>
          
          {/* Side Panel */}
          <div className="space-y-4">
            {/* Response History */}
            <div className="bg-gray-800 p-4 rounded-lg">
              <h3 className="text-lg font-semibold text-white mb-3">
                Recent Responses ({responses.length})
              </h3>
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {responses.slice(-5).map((response, index) => (
                  <div key={index} className="text-sm text-gray-300 p-2 bg-gray-700 rounded">
                    <div className="text-xs text-gray-400 mb-1">
                      {new Date().toLocaleTimeString()}
                    </div>
                    <div className="truncate">{response.result}</div>
                  </div>
                ))}
              </div>
            </div>
            
            {/* Error Log */}
            {errors.length > 0 && (
              <div className="bg-red-900 p-4 rounded-lg">
                <h3 className="text-lg font-semibold text-red-200 mb-3">
                  Errors ({errors.length})
                </h3>
                <div className="space-y-2 max-h-32 overflow-y-auto">
                  {errors.slice(-3).map((error, index) => (
                    <div key={index} className="text-sm text-red-300 p-2 bg-red-800 rounded">
                      {error.message || error.toString()}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default AIAssistantApp;
```

## Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `apiBaseUrl` | string | `'http://localhost:8888'` | Base URL for the Jarvis API server |
| `theme` | string | `'dark'` | Theme mode: `'dark'` or `'light'` |
| `onResponse` | function | `undefined` | Callback fired when receiving responses |
| `onError` | function | `undefined` | Callback fired when errors occur |
| `maxTranscriptLines` | number | `100` | Maximum number of transcript entries to keep |
| `enableVoiceInput` | boolean | `true` | Enable/disable voice input functionality |
| `enableFileUpload` | boolean | `true` | Enable/disable file upload functionality |

## API Endpoints

### Voice Processing
```
POST /jarvis/voice/process
Content-Type: multipart/form-data

FormData:
  - audio: File (WebM/WAV audio file)

Response:
{
  "transcript": "recognized speech text",
  "result": "AI response",
  "status": "completed",
  "execution_time": 1.23
}
```

### Task Planning
```
POST /jarvis/task/plan
Content-Type: application/json

Body:
{
  "command": "user input text",
  "context": {
    "uploaded_files": ["file1.pdf", "file2.docx"],
    "timestamp": "2025-08-08T10:30:00Z"
  },
  "voice_enabled": false
}

Response:
{
  "result": "AI response",
  "status": "completed",
  "execution_time": 0.85,
  "agents_used": ["jarvis-core", "document-processor"]
}
```

### WebSocket Real-time Communication
```
WebSocket: /ws

Send:
{
  "command": "user message",
  "context": {},
  "voice_enabled": false
}

Receive:
{
  "result": "streaming response",
  "status": "processing|completed|error",
  "voice_response": "optional TTS text"
}
```

## Accessibility Features

### Keyboard Navigation
- **Enter**: Send message
- **Shift + Enter**: New line in input
- **Ctrl + Space**: Toggle voice recording
- **Tab**: Navigate between interactive elements
- **Escape**: Clear current input/stop recording

### Screen Reader Support
- **ARIA labels** on all interactive elements
- **Role attributes** for semantic structure
- **Live regions** for dynamic content updates
- **Focus management** for optimal navigation

### High Contrast Support
- **prefers-contrast: high** media query support
- **Border enhancement** in high contrast mode
- **Color-blind friendly** status indicators

### Responsive Design
- **Mobile-first** approach with touch-friendly controls
- **Flexible layouts** that adapt to different screen sizes
- **Readable font sizes** across all devices

## File Upload Support

### Supported File Types
- **PDF**: `.pdf` files up to 10MB
- **Word Documents**: `.docx` files
- **Excel Spreadsheets**: `.xlsx` files
- **Text Files**: `.txt` files
- **JSON**: `.json` configuration files

### Drag & Drop
1. **Drag files** over the component
2. **Drop zone** highlights when files are dragged over
3. **Visual feedback** confirms successful uploads
4. **Error handling** for unsupported file types

### File Management
- **File list** shows all uploaded files
- **Remove button** for each file
- **File size** display in KB/MB
- **Upload progress** indication

## Voice Input Features

### Audio Processing
- **Sample Rate**: 16kHz optimized for speech
- **Format**: WebM with Opus codec
- **Channel Count**: Mono (1 channel)
- **Audio Enhancements**: Echo cancellation, noise suppression, auto gain control

### Recording Controls
- **Start/Stop** with visual feedback
- **Recording indicator** with pulsing animation
- **Audio level** monitoring
- **Timeout handling** for long recordings

### Error Handling
- **Microphone permissions** with user-friendly messages
- **Browser compatibility** checks
- **Fallback options** when Web Audio API isn't available
- **Network error** recovery

## Performance Optimizations

### React Optimizations
- **useCallback** hooks for stable event handlers
- **useMemo** for expensive calculations
- **Lazy loading** for large transcripts
- **Virtual scrolling** for performance with many messages

### CSS Optimizations
- **GPU acceleration** for animations
- **Efficient selectors** for faster rendering
- **Minimal repaints** with optimized transitions
- **Responsive images** and assets

### Memory Management
- **Transcript limits** to prevent memory bloat
- **WebSocket cleanup** on component unmount
- **Audio stream** cleanup after recording
- **File cleanup** after processing

## Troubleshooting

### Common Issues

#### WebSocket Connection Fails
```javascript
// Check if Jarvis API server is running
curl http://localhost:8888/health

// Verify WebSocket endpoint
curl -i -N -H "Connection: Upgrade" \
     -H "Upgrade: websocket" \
     -H "Sec-WebSocket-Version: 13" \
     -H "Sec-WebSocket-Key: test" \
     http://localhost:8888/ws
```

#### Voice Input Not Working
- **Check microphone permissions** in browser settings
- **Ensure HTTPS** for production (required for getUserMedia)
- **Test audio devices** in browser developer tools
- **Verify codec support** (WebM/Opus)

#### File Upload Issues
- **Check file size limits** (default 10MB)
- **Verify MIME types** are supported
- **Test CORS settings** if API is on different domain
- **Monitor network requests** in browser dev tools

#### Styling Problems
- **Ensure TailwindCSS is configured** correctly
- **Check CSS import** order
- **Verify dark mode** class is applied to parent element
- **Test responsive breakpoints** in browser dev tools

### Debug Mode

Enable debug logging:
```javascript
// Add to your component
<JarvisPanel 
  {...props}
  debug={true}  // Enables console logging
/>
```

### Performance Monitoring

Monitor component performance:
```javascript
// Add React DevTools Profiler
import { Profiler } from 'react';

function onRenderCallback(id, phase, actualDuration) {
  console.log('JarvisPanel render:', {
    id, phase, actualDuration
  });
}

<Profiler id="JarvisPanel" onRender={onRenderCallback}>
  <JarvisPanel {...props} />
</Profiler>
```

## Browser Compatibility

| Browser | Voice Input | WebSocket | File Upload | Rating |
|---------|-------------|-----------|-------------|---------|
| Chrome 90+ | ✅ Full | ✅ Full | ✅ Full | ⭐⭐⭐⭐⭐ |
| Firefox 88+ | ✅ Full | ✅ Full | ✅ Full | ⭐⭐⭐⭐⭐ |
| Safari 14+ | ⚠️ Limited | ✅ Full | ✅ Full | ⭐⭐⭐⭐ |
| Edge 90+ | ✅ Full | ✅ Full | ✅ Full | ⭐⭐⭐⭐⭐ |
| Mobile Chrome | ⚠️ Touch only | ✅ Full | ✅ Full | ⭐⭐⭐⭐ |
| Mobile Safari | ❌ Not supported | ✅ Full | ✅ Full | ⭐⭐⭐ |

## Contributing

### Development Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd sutazaiapp
```

2. **Install dependencies**:
```bash
npm install
```

3. **Start development server**:
```bash
# Start Jarvis API server
docker-compose up jarvis-voice-interface

# In another terminal, start React development
npm start
```

### Testing

Run component tests:
```bash
# Unit tests
npm run test

# Integration tests
npm run test:integration

# Accessibility tests
npm run test:a11y
```

### Code Style

- **ESLint** configuration for React best practices
- **Prettier** for consistent code formatting
- **TypeScript** support (optional)
- **JSDoc** comments for all public methods

## License

This component is part of the SutazAI system and follows the project's licensing terms.

## Support

For issues and support:
1. **Check troubleshooting section** above
2. **Review browser console** for error messages
3. **Test API endpoints** directly
4. **Open issue** with detailed reproduction steps

## Changelog

### v1.0.0 (2025-08-08)
- ✅ Initial release
- ✅ Voice input with Web Audio API
- ✅ WebSocket real-time communication
- ✅ File upload with drag & drop
- ✅ Full accessibility support
- ✅ Responsive TailwindCSS design
- ✅ Dark/Light theme support