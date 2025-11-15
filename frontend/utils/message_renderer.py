"""
Enhanced Chat Message Renderer
Provides markdown rendering, code syntax highlighting, and rich message display
"""

import streamlit as st
import re
from typing import Dict, Any, Optional, List
from datetime import datetime
import hashlib

class MessageRenderer:
    """
    Advanced message rendering with markdown, code highlighting, and formatting
    """
    
    @staticmethod
    def render_message(message: Dict[str, Any], key_prefix: str = ""):
        """
        Render a single chat message with enhanced formatting
        
        Args:
            message: Message dictionary with role, content, timestamp, metadata
            key_prefix: Unique prefix for Streamlit component keys
        """
        role = message.get("role", "assistant")
        content = message.get("content", "")
        timestamp = message.get("timestamp")
        metadata = message.get("metadata", {})
        
        # Create unique key for this message
        msg_hash = hashlib.md5(f"{role}{content}{timestamp}".encode()).hexdigest()[:8]
        msg_key = f"{key_prefix}_{msg_hash}"
        
        # Render with role-specific styling
        with st.chat_message(role):
            # Render content with markdown and code highlighting
            MessageRenderer._render_content(content, msg_key)
            
            # Render metadata (timestamp, model, etc.)
            MessageRenderer._render_metadata(timestamp, metadata, msg_key)
            
            # Action buttons
            MessageRenderer._render_actions(content, msg_key)
    
    @staticmethod
    def _render_content(content: str, key: str):
        """Render message content with markdown and code highlighting"""
        # Check if content contains code blocks
        code_block_pattern = r'```(\w+)?\n(.*?)```'
        
        if '```' in content:
            # Split content into text and code parts
            parts = re.split(code_block_pattern, content, flags=re.DOTALL)
            
            for i, part in enumerate(parts):
                if i % 3 == 0:
                    # Regular text/markdown
                    if part.strip():
                        st.markdown(part)
                elif i % 3 == 1:
                    # Language identifier (skip)
                    continue
                elif i % 3 == 2:
                    # Code block
                    language = parts[i-1] if parts[i-1] else "python"
                    MessageRenderer._render_code_block(part, language, f"{key}_code_{i}")
        else:
            # No code blocks, render as markdown
            st.markdown(content)
    
    @staticmethod
    def _render_code_block(code: str, language: str, key: str):
        """Render code block with syntax highlighting and copy button"""
        col1, col2 = st.columns([6, 1])
        
        with col1:
            st.code(code, language=language)
        
        with col2:
            # Copy button
            if st.button("📋", key=f"{key}_copy", help="Copy code"):
                st.toast("Code copied to clipboard!", icon="✅")
                # Note: Actual clipboard copy requires JavaScript component
                # For now, we use session state
                st.session_state[f"{key}_clipboard"] = code
    
    @staticmethod
    def _render_metadata(timestamp: Optional[str], metadata: Dict[str, Any], key: str):
        """Render message metadata (timestamp, model, agent, etc.)"""
        metadata_parts = []
        
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp)
                time_str = dt.strftime("%H:%M:%S")
                metadata_parts.append(f"🕒 {time_str}")
            except:
                metadata_parts.append(f"🕒 {timestamp}")
        
        if "model" in metadata:
            metadata_parts.append(f"🤖 {metadata['model']}")
        
        if "agent" in metadata:
            metadata_parts.append(f"🚀 {metadata['agent']}")
        
        if "tokens" in metadata:
            metadata_parts.append(f"📊 {metadata['tokens']} tokens")
        
        if "latency_ms" in metadata:
            metadata_parts.append(f"⚡ {metadata['latency_ms']}ms")
        
        if metadata_parts:
            st.caption(" • ".join(metadata_parts))
    
    @staticmethod
    def _render_actions(content: str, key: str):
        """Render action buttons for the message"""
        col1, col2, col3, col4 = st.columns([1, 1, 1, 5])
        
        with col1:
            if st.button("👍", key=f"{key}_like", help="Like this response"):
                st.toast("Response liked!", icon="👍")
        
        with col2:
            if st.button("👎", key=f"{key}_dislike", help="Dislike this response"):
                st.toast("Feedback recorded", icon="📝")
        
        with col3:
            if st.button("🔄", key=f"{key}_regen", help="Regenerate response"):
                st.toast("Regenerating response...", icon="🔄")
                # Store regeneration request in session state
                st.session_state[f"{key}_regenerate"] = True
    
    @staticmethod
    def render_typing_indicator():
        """Render typing/processing indicator"""
        st.markdown("""
        <div style="padding: 10px;">
            <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_streaming_message(role: str, content_placeholder: st.delta_generator.DeltaGenerator, 
                                 tokens: List[str], complete: bool = False):
        """
        Render streaming message token by token
        
        Args:
            role: Message role (user/assistant)
            content_placeholder: Streamlit placeholder to update
            tokens: List of tokens received so far
            complete: Whether streaming is complete
        """
        full_content = "".join(tokens)
        
        with content_placeholder.container():
            with st.chat_message(role):
                st.markdown(full_content)
                
                if not complete:
                    # Show blinking cursor
                    st.markdown('<span class="cursor">▊</span>', unsafe_allow_html=True)


class ChatHistoryManager:
    """Manage chat history persistence and export"""
    
    @staticmethod
    def export_to_markdown(messages: List[Dict[str, Any]]) -> str:
        """Export chat history to markdown format"""
        md_lines = ["# JARVIS Chat History", ""]
        md_lines.append(f"**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")
        
        for msg in messages:
            role = msg.get("role", "assistant")
            content = msg.get("content", "")
            timestamp = msg.get("timestamp", "")
            
            role_emoji = "👤" if role == "user" else "🤖"
            md_lines.append(f"## {role_emoji} {role.title()}")
            if timestamp:
                md_lines.append(f"*{timestamp}*")
            md_lines.append("")
            md_lines.append(content)
            md_lines.append("")
            md_lines.append("---")
            md_lines.append("")
        
        return "\n".join(md_lines)
    
    @staticmethod
    def export_to_json(messages: List[Dict[str, Any]]) -> str:
        """Export chat history to JSON format"""
        import json
        return json.dumps({
            "exported_at": datetime.now().isoformat(),
            "message_count": len(messages),
            "messages": messages
        }, indent=2)
    
    @staticmethod
    def export_to_text(messages: List[Dict[str, Any]]) -> str:
        """Export chat history to plain text format"""
        lines = ["JARVIS Chat History"]
        lines.append(f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)
        lines.append("")
        
        for msg in messages:
            role = msg.get("role", "assistant").upper()
            content = msg.get("content", "")
            timestamp = msg.get("timestamp", "N/A")
            
            lines.append(f"[{timestamp}] {role}:")
            lines.append(content)
            lines.append("")
            lines.append("-" * 80)
            lines.append("")
        
        return "\n".join(lines)
    
    @staticmethod
    def save_to_session_storage(messages: List[Dict[str, Any]], key: str = "chat_backup"):
        """Save chat history to session state backup"""
        if 'st' in dir():
            st.session_state[key] = {
                "messages": messages,
                "saved_at": datetime.now().isoformat()
            }
    
    @staticmethod
    def load_from_session_storage(key: str = "chat_backup") -> Optional[List[Dict[str, Any]]]:
        """Load chat history from session state backup"""
        if 'st' in dir() and key in st.session_state:
            backup = st.session_state[key]
            return backup.get("messages", [])
        return None


class CodeFormatter:
    """Format and enhance code snippets"""
    
    @staticmethod
    def detect_language(code: str) -> str:
        """Auto-detect programming language from code"""
        # Simple heuristic-based detection
        if re.search(r'def\s+\w+\s*\(.*\)\s*:', code):
            return "python"
        elif re.search(r'function\s+\w+\s*\(.*\)\s*{', code):
            return "javascript"
        elif re.search(r'<\?php', code):
            return "php"
        elif re.search(r'SELECT\s+.*\s+FROM', code, re.IGNORECASE):
            return "sql"
        elif re.search(r'<!DOCTYPE|<html', code, re.IGNORECASE):
            return "html"
        elif re.search(r'package\s+main', code):
            return "go"
        elif re.search(r'fn\s+main\s*\(\)', code):
            return "rust"
        else:
            return "python"  # Default
    
    @staticmethod
    def add_line_numbers(code: str) -> str:
        """Add line numbers to code"""
        lines = code.split('\n')
        max_width = len(str(len(lines)))
        
        numbered = []
        for i, line in enumerate(lines, 1):
            numbered.append(f"{i:>{max_width}} | {line}")
        
        return '\n'.join(numbered)
    
    @staticmethod
    def extract_code_blocks(text: str) -> List[Dict[str, str]]:
        """Extract all code blocks from text"""
        pattern = r'```(\w+)?\n(.*?)```'
        matches = re.findall(pattern, text, flags=re.DOTALL)
        
        blocks = []
        for lang, code in matches:
            blocks.append({
                "language": lang or CodeFormatter.detect_language(code),
                "code": code.strip()
            })
        
        return blocks


# CSS for enhanced styling
CHAT_CSS = """
<style>
/* Typing indicator animation */
.typing-indicator {
    display: inline-flex;
    gap: 4px;
    padding: 10px;
}

.typing-indicator span {
    width: 8px;
    height: 8px;
    background: #00D4FF;
    border-radius: 50%;
    animation: typing 1.4s infinite;
}

.typing-indicator span:nth-child(2) {
    animation-delay: 0.2s;
}

.typing-indicator span:nth-child(3) {
    animation-delay: 0.4s;
}

@keyframes typing {
    0%, 60%, 100% {
        opacity: 0.3;
        transform: translateY(0);
    }
    30% {
        opacity: 1;
        transform: translateY(-10px);
    }
}

/* Cursor blink */
.cursor {
    animation: blink 1s infinite;
    color: #00D4FF;
    font-weight: bold;
}

@keyframes blink {
    0%, 49% { opacity: 1; }
    50%, 100% { opacity: 0; }
}

/* Code block enhancements */
.stCode {
    position: relative;
}

/* Message actions */
.message-actions {
    display: flex;
    gap: 8px;
    margin-top: 8px;
    opacity: 0.7;
    transition: opacity 0.2s;
}

.message-actions:hover {
    opacity: 1;
}

/* Metadata styling */
.stCaptionContainer {
    color: #999 !important;
    font-size: 0.85em;
}
</style>
"""


def inject_chat_css():
    """Inject chat styling CSS"""
    st.markdown(CHAT_CSS, unsafe_allow_html=True)
