#!/bin/bash
# Restart SutazAI Frontend with Enhanced Features

echo "🔄 Restarting SutazAI Frontend with Enhanced Features..."

# Stop existing frontend
echo "Stopping existing frontend..."
pkill -f "streamlit run intelligent_chat_app.py" || true
sleep 3

# Start enhanced frontend
echo "Starting enhanced frontend with Enter key and Voice features..."
source venv/bin/activate
streamlit run intelligent_chat_app.py --server.address 0.0.0.0 --server.port 8501 > frontend_enhanced.log 2>&1 &

echo "✅ Enhanced frontend started!"
echo "🌐 Access at: http://localhost:8501"
echo "⌨️ Enter key functionality: ENABLED"
echo "🎤 Voice input: AVAILABLE"
echo "🔊 Voice output: AVAILABLE"
echo ""
echo "New Features:"
echo "• Press Enter to send messages (Shift+Enter for new line)"
echo "• Click 'Voice Input' button to speak your message"
echo "• Enable 'Voice Output' to hear AI responses"
echo "• All features work with your existing 26 AI agents"