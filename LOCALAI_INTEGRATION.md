# LocalAI Integration with SutazAI

This guide explains how to set up and use the LocalAI integration with SutazAI's web interface. This integration allows you to chat directly with your local AI models through a user-friendly chat interface.

## Setup Instructions

### 1. Install and Run LocalAI

If you haven't already, you need to set up LocalAI on your system. Follow the official instructions at [LocalAI GitHub Repository](https://github.com/go-skynet/LocalAI).

Basic installation steps:
```bash
# Using Docker
docker run -p 8080:8080 -v $PWD/models:/models -v $PWD/configs:/configs -e THREADS=4 localai/localai:latest
```

Make sure LocalAI is running and accessible on port 8080 (or modify the port in the following steps if you're using a different port).

### 2. Install Dependencies

The integration requires additional dependencies for the SutazAI web UI. Run the following in the web_ui directory:

```bash
cd web_ui
npm install react-markdown react-syntax-highlighter
```

### 3. Configure Environment

Make sure your web_ui/.env.local file contains the LocalAI URL configuration:

```
NEXT_PUBLIC_LOCAL_AI_URL=http://${LOCAL_AI_HOST:-localhost}:${LOCAL_AI_PORT:-8080}
```

You can customize the host and port by setting the LOCAL_AI_HOST and LOCAL_AI_PORT environment variables before starting the web UI.

### 4. Start the Web UI

Start the SutazAI web UI with:

```bash
cd web_ui
npm run dev
```

This will start the web UI on port 3000 (default). If you're running the start_services.sh script, it will handle this for you.

## Using the LocalAI Chat

1. Open your browser and navigate to `http://localhost:3000`
2. You'll see the "AI Chat" tab as the first tab in the interface
3. The interface will automatically fetch available models from your LocalAI instance
4. Select a model from the dropdown menu
5. Type your message in the text field and press "Send" to chat with the AI
6. Your conversation history will appear in the chat window

## Troubleshooting

### No Models Found

If the web UI can't find any models:

1. Make sure LocalAI is running and accessible
2. Check that the URL in your .env.local file matches your LocalAI setup
3. Verify you have properly loaded models in your LocalAI instance
4. Try using the "Refresh" button in the model selection area

### API Connection Issues

If you're experiencing connection issues:

1. Check that LocalAI is running and the port is accessible
2. Verify there are no CORS issues by checking your browser console
3. Make sure your firewall isn't blocking connections

### Model Loading Problems

If models load but don't respond:

1. Check the LocalAI logs for any errors
2. Verify the model is properly configured in LocalAI
3. Make sure the model is compatible with chat completion API

## Advanced Configuration

### Custom Model Settings

You can modify the model parameters in the `LocalAIChat.jsx` component:

```javascript
// Find this section in src/components/LocalAIChat.jsx
const response = await fetch(`${LOCAL_AI_URL}/v1/chat/completions`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: selectedModel,
    messages: [...messages, userMessage].map(msg => ({
      role: msg.role,
      content: msg.content
    })),
    temperature: 0.7,  // Adjust this value
    max_tokens: 1000,  // Adjust this value
  }),
});
```

### Adding Authentication

If your LocalAI instance requires authentication, modify the fetch requests in the `LocalAIChat.jsx` component to include the necessary headers. 