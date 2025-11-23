"""
LLM Configuration Components
Provides UI components for model selection, parameters, and streaming controls
"""

import streamlit as st
from typing import Dict, List, Any, Optional, Callable
import time

class LLMConfigPanel:
    """
    Advanced LLM configuration panel with model selection,
    parameter controls, and streaming options
    """
    
    DEFAULT_MODELS = [
        {
            "name": "TinyLlama",
            "id": "tinyllama:latest",
            "description": "Fast, efficient 1.1B parameter model",
            "parameters": 1_100_000_000,
            "context_window": 2048,
            "recommended_temp": 0.7
        },
        {
            "name": "Llama 2 7B",
            "id": "llama2:7b",
            "description": "Balanced performance, 7B parameters",
            "parameters": 7_000_000_000,
            "context_window": 4096,
            "recommended_temp": 0.8
        },
        {
            "name": "Mistral 7B",
            "id": "mistral:7b",
            "description": "High quality, efficient 7B model",
            "parameters": 7_000_000_000,
            "context_window": 8192,
            "recommended_temp": 0.7
        },
        {
            "name": "CodeLlama 7B",
            "id": "codellama:7b",
            "description": "Specialized for code generation",
            "parameters": 7_000_000_000,
            "context_window": 16384,
            "recommended_temp": 0.5
        }
    ]
    
    @staticmethod
    def render_model_selector(available_models: Optional[List[Dict]] = None,
                             current_model: str = "tinyllama:latest") -> str:
        """
        Render model selection dropdown with details
        
        Returns:
            Selected model ID
        """
        models = available_models or LLMConfigPanel.DEFAULT_MODELS
        
        st.markdown("### 🤖 AI Model Selection")
        
        # Create model options
        model_names = [m["name"] for m in models]
        model_ids = [m["id"] for m in models]
        
        # Find current index
        current_idx = 0
        if current_model in model_ids:
            current_idx = model_ids.index(current_model)
        
        # Model selector
        selected_idx = st.selectbox(
            "Select Model",
            range(len(models)),
            format_func=lambda x: f"{model_names[x]} ({models[x]['parameters'] / 1_000_000_000:.1f}B)",
            index=current_idx,
            key="llm_model_selector",
            help="Choose the LLM model for generation"
        )
        
        selected_model = models[selected_idx]
        
        # Show model details
        with st.expander("📋 Model Details", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Parameters", f"{selected_model['parameters'] / 1_000_000_000:.1f}B")
                st.metric("Context Window", f"{selected_model['context_window']:,} tokens")
            
            with col2:
                st.metric("Recommended Temp", selected_model["recommended_temp"])
                if "speed" in selected_model:
                    st.metric("Speed", selected_model["speed"])
            
            st.caption(selected_model["description"])
        
        return selected_model["id"]
    
    @staticmethod
    def render_generation_params(recommended_temp: float = 0.7) -> Dict[str, Any]:
        """
        Render generation parameter controls
        
        Returns:
            Dictionary of generation parameters
        """
        st.markdown("### ⚙️ Generation Parameters")
        
        # Temperature
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=recommended_temp,
            step=0.1,
            help="Controls randomness. Lower = more focused, Higher = more creative",
            key="llm_temperature"
        )
        
        # Show temperature guidance
        if temperature < 0.3:
            temp_desc = "🎯 Very Focused (Deterministic)"
        elif temperature < 0.7:
            temp_desc = "📝 Balanced (Recommended)"
        elif temperature < 1.2:
            temp_desc = "🎨 Creative (Diverse)"
        else:
            temp_desc = "🌀 Very Creative (Chaotic)"
        
        st.caption(temp_desc)
        
        # Advanced parameters in expander
        with st.expander("🔧 Advanced Parameters", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                top_p = st.slider(
                    "Top P (Nucleus Sampling)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.9,
                    step=0.05,
                    help="Cumulative probability cutoff for token selection",
                    key="llm_top_p"
                )
                
                top_k = st.slider(
                    "Top K",
                    min_value=1,
                    max_value=100,
                    value=40,
                    step=1,
                    help="Number of top tokens to consider",
                    key="llm_top_k"
                )
            
            with col2:
                max_tokens = st.slider(
                    "Max Tokens",
                    min_value=50,
                    max_value=2000,
                    value=500,
                    step=50,
                    help="Maximum number of tokens to generate",
                    key="llm_max_tokens"
                )
                
                repeat_penalty = st.slider(
                    "Repeat Penalty",
                    min_value=1.0,
                    max_value=2.0,
                    value=1.1,
                    step=0.1,
                    help="Penalty for repeating tokens",
                    key="llm_repeat_penalty"
                )
        
        return {
            "temperature": temperature,
            "top_p": top_p if st.session_state.get("llm_top_p") is not None else 0.9,
            "top_k": top_k if st.session_state.get("llm_top_k") is not None else 40,
            "max_tokens": max_tokens if st.session_state.get("llm_max_tokens") is not None else 500,
            "repeat_penalty": repeat_penalty if st.session_state.get("llm_repeat_penalty") is not None else 1.1
        }
    
    @staticmethod
    def render_streaming_controls() -> Dict[str, bool]:
        """
        Render streaming and response controls
        
        Returns:
            Dictionary of streaming options
        """
        st.markdown("### 🔄 Response Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            stream_enabled = st.checkbox(
                "Enable Streaming",
                value=True,
                help="Show responses token-by-token as they're generated",
                key="llm_stream_enabled"
            )
        
        with col2:
            show_thinking = st.checkbox(
                "Show Thinking Process",
                value=False,
                help="Display model's reasoning steps (if supported)",
                key="llm_show_thinking"
            )
        
        return {
            "stream": stream_enabled,
            "show_thinking": show_thinking
        }
    
    @staticmethod
    def render_system_prompt() -> str:
        """
        Render system prompt configuration
        
        Returns:
            System prompt text
        """
        st.markdown("### 📝 System Prompt")
        
        default_prompt = "You are JARVIS, an advanced AI assistant created by SutazAI. You are helpful, accurate, and concise."
        
        # Preset prompts
        presets = {
            "JARVIS (Default)": default_prompt,
            "Code Assistant": "You are an expert programmer. Provide clear, well-commented code with explanations.",
            "Data Analyst": "You are a data science expert. Analyze data thoroughly and provide insights with visualizations.",
            "Technical Writer": "You are a technical documentation specialist. Write clear, comprehensive documentation.",
            "Creative Writer": "You are a creative writing assistant. Help with storytelling, dialogue, and narrative.",
            "Custom": ""
        }
        
        preset_selection = st.selectbox(
            "Prompt Preset",
            list(presets.keys()),
            key="llm_prompt_preset"
        )
        
        if preset_selection == "Custom":
            system_prompt = st.text_area(
                "System Prompt",
                value=default_prompt,
                height=100,
                help="Define the AI assistant's role and behavior",
                key="llm_system_prompt_custom"
            )
        else:
            system_prompt = presets[preset_selection]
            st.text_area(
                "System Prompt",
                value=system_prompt,
                height=100,
                disabled=True,
                key="llm_system_prompt_display"
            )
        
        return system_prompt
    
    @staticmethod
    def render_context_display(context_used: int, context_limit: int):
        """
        Render context window usage visualization
        
        Args:
            context_used: Number of tokens currently used
            context_limit: Maximum context window size
        """
        st.markdown("### 📊 Context Usage")
        
        usage_percent = (context_used / context_limit) * 100
        
        # Color coding
        if usage_percent < 50:
            color = "#4CAF50"  # Green
        elif usage_percent < 80:
            color = "#FF9800"  # Orange
        else:
            color = "#F44336"  # Red
        
        # Progress bar
        st.progress(usage_percent / 100, text=f"{context_used:,} / {context_limit:,} tokens ({usage_percent:.1f}%)")
        
        # Warning if approaching limit
        if usage_percent > 80:
            st.warning("⚠️ Context window nearly full. Consider clearing old messages.")
        
        # Show estimated cost (if applicable)
        if context_used > 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Used", f"{context_used:,}")
            with col2:
                st.metric("Available", f"{context_limit - context_used:,}")
            with col3:
                st.metric("Limit", f"{context_limit:,}")


class StreamingHandler:
    """Handle streaming response display"""
    
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.tokens = []
        self.complete = False
        self.start_time = time.time()
    
    def add_token(self, token: str):
        """Add token to streaming response"""
        self.tokens.append(token)
        self._update_display()
    
    def finalize(self, full_response: Optional[str] = None):
        """Finalize streaming response"""
        self.complete = True
        if full_response:
            self.tokens = [full_response]
        self._update_display()
    
    def _update_display(self):
        """Update the display with current tokens"""
        content = "".join(self.tokens)
        
        with self.placeholder.container():
            with st.chat_message("assistant"):
                st.markdown(content)
                
                if not self.complete:
                    # Show typing cursor
                    st.markdown('<span style="animation: blink 1s infinite;">▊</span>', 
                              unsafe_allow_html=True)
                else:
                    # Show completion time
                    elapsed = time.time() - self.start_time
                    st.caption(f"⚡ Generated in {elapsed:.2f}s ({len(self.tokens)} tokens)")
    
    def get_content(self) -> str:
        """Get complete content"""
        return "".join(self.tokens)
    
    def get_token_count(self) -> int:
        """Get number of tokens"""
        return len(self.tokens)
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds"""
        return time.time() - self.start_time


def create_llm_config_sidebar() -> Dict[str, Any]:
    """
    Create complete LLM configuration sidebar
    
    Returns:
        Dictionary with all LLM configuration
    """
    config = {}
    
    # Model selection
    config["model"] = LLMConfigPanel.render_model_selector()
    
    st.divider()
    
    # Generation parameters
    config["params"] = LLMConfigPanel.render_generation_params()
    
    st.divider()
    
    # Streaming controls
    config["streaming"] = LLMConfigPanel.render_streaming_controls()
    
    st.divider()
    
    # System prompt
    config["system_prompt"] = LLMConfigPanel.render_system_prompt()
    
    return config
