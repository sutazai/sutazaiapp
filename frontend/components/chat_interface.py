"""
Chat Interface Module
Handles chat UI components and message management
"""

import streamlit as st
from typing import List, Dict, Optional
import time
from datetime import datetime

class ChatInterface:
    """Advanced chat interface with streaming and animations"""
    
    def __init__(self):
        self.messages = []
        self.typing_speed = 0.03  # Seconds between characters
        
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to the chat history"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.messages.append(message)
        return message
    
    def display_message(self, message: Dict, animated: bool = False):
        """Display a single message with optional typing animation"""
        role = message["role"]
        content = message["content"]
        timestamp = message.get("timestamp", "")
        
        # Message container styling based on role
        if role == "user":
            st.markdown(f"""
            <div style="display: flex; justify-content: flex-end; margin: 10px 0;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                          color: white; padding: 12px 18px; border-radius: 18px 18px 5px 18px;
                          max-width: 70%; word-wrap: break-word;">
                    <div style="font-size: 0.9em; opacity: 0.8; margin-bottom: 5px;">You</div>
                    <div>{content}</div>
                    <div style="font-size: 0.8em; opacity: 0.6; margin-top: 5px;">{timestamp}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            if animated:
                # Typing animation for assistant messages
                placeholder = st.empty()
                displayed_text = ""
                
                for char in content:
                    displayed_text += char
                    placeholder.markdown(f"""
                    <div style="display: flex; justify-content: flex-start; margin: 10px 0;">
                        <div style="background: linear-gradient(135deg, #00D4FF 0%, #0099CC 100%);
                                  color: white; padding: 12px 18px; border-radius: 18px 18px 18px 5px;
                                  max-width: 70%; word-wrap: break-word;">
                            <div style="font-size: 0.9em; opacity: 0.8; margin-bottom: 5px;">JARVIS</div>
                            <div>{displayed_text}▌</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    time.sleep(self.typing_speed)
                
                # Final display without cursor
                placeholder.markdown(f"""
                <div style="display: flex; justify-content: flex-start; margin: 10px 0;">
                    <div style="background: linear-gradient(135deg, #00D4FF 0%, #0099CC 100%);
                              color: white; padding: 12px 18px; border-radius: 18px 18px 18px 5px;
                              max-width: 70%; word-wrap: break-word;">
                        <div style="font-size: 0.9em; opacity: 0.8; margin-bottom: 5px;">JARVIS</div>
                        <div>{content}</div>
                        <div style="font-size: 0.8em; opacity: 0.6; margin-top: 5px;">{timestamp}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="display: flex; justify-content: flex-start; margin: 10px 0;">
                    <div style="background: linear-gradient(135deg, #00D4FF 0%, #0099CC 100%);
                              color: white; padding: 12px 18px; border-radius: 18px 18px 18px 5px;
                              max-width: 70%; word-wrap: break-word;">
                        <div style="font-size: 0.9em; opacity: 0.8; margin-bottom: 5px;">JARVIS</div>
                        <div>{content}</div>
                        <div style="font-size: 0.8em; opacity: 0.6; margin-top: 5px;">{timestamp}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    def display_chat_history(self, messages: List[Dict], animated_last: bool = True):
        """Display entire chat history"""
        for i, message in enumerate(messages):
            # Animate only the last assistant message
            should_animate = (
                animated_last and 
                i == len(messages) - 1 and 
                message["role"] == "assistant"
            )
            self.display_message(message, animated=should_animate)
    
    def display_thinking_indicator(self):
        """Display thinking/processing animation"""
        return st.markdown("""
        <div style="display: flex; align-items: center; margin: 20px 0;">
            <div style="background: rgba(0, 212, 255, 0.1); padding: 10px 20px; 
                      border-radius: 20px; border: 1px solid #00D4FF;">
                <span style="color: #00D4FF; margin-right: 10px;">JARVIS is thinking</span>
                <span class="thinking-dots">
                    <span>.</span><span>.</span><span>.</span>
                </span>
            </div>
        </div>
        <style>
        .thinking-dots span {
            animation: blink 1.4s linear infinite;
            animation-fill-mode: both;
        }
        .thinking-dots span:nth-child(2) {
            animation-delay: 0.2s;
        }
        .thinking-dots span:nth-child(3) {
            animation-delay: 0.4s;
        }
        @keyframes blink {
            0%, 60%, 100% { opacity: 0.3; }
            30% { opacity: 1; }
        }
        </style>
        """, unsafe_allow_html=True)
    
    def display_error_message(self, error: str):
        """Display error message in chat style"""
        st.markdown(f"""
        <div style="display: flex; justify-content: center; margin: 20px 0;">
            <div style="background: linear-gradient(135deg, #FF6B6B 0%, #FF5252 100%);
                      color: white; padding: 12px 20px; border-radius: 10px;
                      max-width: 70%; text-align: center;">
                <div style="font-size: 1.1em; margin-bottom: 5px;">⚠️ Error</div>
                <div>{error}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def display_system_message(self, message: str, type: str = "info"):
        """Display system messages (info, warning, success)"""
        colors = {
            "info": ("#00D4FF", "#0099CC"),
            "warning": ("#FFC107", "#FF9800"),
            "success": ("#4CAF50", "#45a049"),
            "error": ("#F44336", "#da190b")
        }
        
        color1, color2 = colors.get(type, colors["info"])
        
        st.markdown(f"""
        <div style="display: flex; justify-content: center; margin: 20px 0;">
            <div style="background: linear-gradient(135deg, {color1} 0%, {color2} 100%);
                      color: white; padding: 10px 20px; border-radius: 25px;
                      text-align: center; font-size: 0.9em;">
                {message}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def clear_chat(self):
        """Clear all messages"""
        self.messages = []
        
    def export_chat(self) -> str:
        """Export chat history as formatted text"""
        export = f"JARVIS Chat Export - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        export += "=" * 60 + "\n\n"
        
        for msg in self.messages:
            role = "You" if msg["role"] == "user" else "JARVIS"
            timestamp = msg.get("timestamp", "")
            export += f"[{timestamp}] {role}:\n{msg['content']}\n\n"
        
        return export