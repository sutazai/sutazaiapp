"""
Performance Optimized Components
Lazy loading, caching, and error boundary components for better performance
"""

import streamlit as st
import time
from typing import Callable, Any, Dict, Optional, List
import logging
from functools import wraps

logger = logging.getLogger(__name__)

# Performance monitoring
_component_metrics = {
    "render_times": {},
    "cache_hits": 0,
    "cache_misses": 0
}

def measure_performance(component_name: str):
    """Decorator to measure component rendering performance"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # Record performance metrics
                render_time = (time.time() - start_time) * 1000
                if component_name not in _component_metrics["render_times"]:
                    _component_metrics["render_times"][component_name] = []
                _component_metrics["render_times"][component_name].append(render_time)
                
                # Keep only last 10 measurements
                if len(_component_metrics["render_times"][component_name]) > 10:
                    _component_metrics["render_times"][component_name] = _component_metrics["render_times"][component_name][-10:]
                
                return result
                
            except Exception as e:
                logger.error(f"Error in {component_name}: {e}")
                render_error_boundary(component_name, str(e))
                return None
                
        return wrapper
    return decorator

def render_error_boundary(component_name: str, error_message: str):
    """Render error boundary UI"""
    with st.container():
        st.error(f"⚠️ Component Error: {component_name}")
        
        with st.expander("🔍 Error Details", expanded=False):
            st.code(error_message)
            st.info("This component is temporarily unavailable. Please refresh the page or try again later.")
            
            if st.button(f"🔄 Retry {component_name}", key=f"retry_{component_name}"):
                st.rerun()

class LazyLoadContainer:
    """Container that loads content only when visible or requested"""
    
    def __init__(self, container_id: str, load_function: Callable, threshold: int = 1000):
        self.container_id = container_id
        self.load_function = load_function
        self.threshold = threshold
        self.is_loaded = False
        
    def render(self, force_load: bool = False):
        """Render the lazy load container"""
        
        # Check if should load
        should_load = force_load or st.session_state.get(f"{self.container_id}_visible", False)
        
        if not should_load:
            # Show placeholder
            with st.container():
                st.info(f"📦 Click to load content")
                if st.button(f"🚀 Load Content", key=f"load_{self.container_id}"):
                    st.session_state[f"{self.container_id}_visible"] = True
                    self.is_loaded = True
                    st.rerun()
            return
        
        # Load and render content
        if not self.is_loaded:
            with st.spinner("Loading content..."):
                try:
                    self.load_function()
                    self.is_loaded = True
                except Exception as e:
                    render_error_boundary(f"LazyLoad-{self.container_id}", str(e))

@st.cache_data(ttl=300, show_spinner=False)  # 5 minute cache
def cached_expensive_computation(computation_id: str, *args, **kwargs) -> Any:
    """Cache expensive computations"""
    _component_metrics["cache_hits"] += 1
    # This would normally contain the actual computation
    # For now, return a placeholder
    return {"computation_id": computation_id, "cached": True, "timestamp": time.time()}

class ProgressiveLoader:
    """Load content progressively to improve perceived performance"""
    
    def __init__(self, items: List[Any], batch_size: int = 5):
        self.items = items
        self.batch_size = batch_size
        self.loaded_count = 0
        
    def render_batch(self, render_function: Callable):
        """Render next batch of items"""
        
        # Initialize session state
        if f"progressive_loader_{id(self)}" not in st.session_state:
            st.session_state[f"progressive_loader_{id(self)}"] = 0
        
        loaded_count = st.session_state[f"progressive_loader_{id(self)}"]
        
        # Render current batch
        end_index = min(loaded_count + self.batch_size, len(self.items))
        current_batch = self.items[loaded_count:end_index]
        
        for item in current_batch:
            render_function(item)
        
        # Show load more button if there are more items
        if end_index < len(self.items):
            remaining = len(self.items) - end_index
            if st.button(f"🔽 Load More ({remaining} remaining)", key=f"load_more_{id(self)}"):
                st.session_state[f"progressive_loader_{id(self)}"] = end_index
                st.rerun()
        else:
            st.success(f"✅ All {len(self.items)} items loaded")

@measure_performance("optimized_metric_card")
def render_optimized_metric_card(title: str, value: str, delta: str = None, help_text: str = None):
    """Render optimized metric card with caching"""
    
    with st.container():
        # Use columns for better layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.metric(
                label=title,
                value=value,
                delta=delta,
                help=help_text
            )
        
        with col2:
            # Add small loading indicator if needed
            if st.session_state.get(f"{title}_loading", False):
                st.spinner()

@st.cache_data(ttl=60, show_spinner=False)
def get_cached_dashboard_data(data_type: str) -> Dict:
    """Get cached dashboard data"""
    # This would normally fetch real data
    return {
        "data_type": data_type,
        "cached_at": time.time(),
        "sample_data": list(range(10))
    }

class VirtualizedList:
    """Virtualized list for handling large datasets efficiently"""
    
    def __init__(self, items: List[Any], item_height: int = 50, container_height: int = 400):
        self.items = items
        self.item_height = item_height
        self.container_height = container_height
        self.visible_count = container_height // item_height
        
    def render(self, render_function: Callable, start_index: int = 0):
        """Render visible items only"""
        
        # Calculate visible range
        end_index = min(start_index + self.visible_count, len(self.items))
        visible_items = self.items[start_index:end_index]
        
        # Render navigation
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if start_index > 0 and st.button("⬆️ Previous", key=f"prev_{id(self)}"):
                new_start = max(0, start_index - self.visible_count)
                st.session_state[f"virtualized_start_{id(self)}"] = new_start
                st.rerun()
        
        with col2:
            st.caption(f"Showing {start_index + 1}-{end_index} of {len(self.items)} items")
        
        with col3:
            if end_index < len(self.items) and st.button("⬇️ Next", key=f"next_{id(self)}"):
                new_start = min(len(self.items) - self.visible_count, start_index + self.visible_count)
                st.session_state[f"virtualized_start_{id(self)}"] = new_start
                st.rerun()
        
        # Render visible items
        with st.container():
            for item in visible_items:
                render_function(item)

def render_performance_monitor():
    """Render performance monitoring dashboard"""
    
    with st.expander("📊 Performance Monitor", expanded=False):
        
        # Component render times
        if _component_metrics["render_times"]:
            st.subheader("Component Render Times")
            
            for component, times in _component_metrics["render_times"].items():
                if times:
                    avg_time = sum(times) / len(times)
                    max_time = max(times)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"{component} (avg)", f"{avg_time:.1f}ms")
                    with col2:
                        st.metric("Max", f"{max_time:.1f}ms")
                    with col3:
                        # Performance indicator
                        if avg_time < 100:
                            st.success("⚡ Fast")
                        elif avg_time < 500:
                            st.warning("⚠️ Moderate")
                        else:
                            st.error("🐌 Slow")
        
        # Cache statistics
        st.subheader("Cache Performance")
        total_cache_requests = _component_metrics["cache_hits"] + _component_metrics["cache_misses"]
        
        if total_cache_requests > 0:
            hit_rate = _component_metrics["cache_hits"] / total_cache_requests * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Cache Hits", _component_metrics["cache_hits"])
            with col2:
                st.metric("Cache Misses", _component_metrics["cache_misses"])
            with col3:
                st.metric("Hit Rate", f"{hit_rate:.1f}%")
        
        # Reset button
        if st.button("🔄 Reset Metrics"):
            reset_performance_metrics()
            st.success("Performance metrics reset!")

def reset_performance_metrics():
    """Reset performance metrics"""
    global _component_metrics
    _component_metrics = {
        "render_times": {},
        "cache_hits": 0,
        "cache_misses": 0
    }

# Debounced input to reduce API calls
class DebouncedInput:
    """Input component with debouncing to reduce API calls"""
    
    def __init__(self, key: str, delay: float = 0.5):
        self.key = key
        self.delay = delay
        
    def render(self, label: str, placeholder: str = "", help_text: str = None) -> Optional[str]:
        """Render debounced input"""
        
        # Initialize session state
        if f"debounced_{self.key}" not in st.session_state:
            st.session_state[f"debounced_{self.key}"] = ""
            st.session_state[f"debounced_{self.key}_last_change"] = time.time()
        
        # Render input
        current_value = st.text_input(
            label,
            value=st.session_state[f"debounced_{self.key}"],
            placeholder=placeholder,
            help=help_text,
            key=f"raw_{self.key}"
        )
        
        # Check if value changed
        if current_value != st.session_state[f"debounced_{self.key}"]:
            st.session_state[f"debounced_{self.key}_last_change"] = time.time()
            
            # Return None if still within delay period
            if time.time() - st.session_state[f"debounced_{self.key}_last_change"] < self.delay:
                return None
            
            # Update debounced value
            st.session_state[f"debounced_{self.key}"] = current_value
            return current_value
        
        return st.session_state[f"debounced_{self.key}"] if st.session_state[f"debounced_{self.key}"] else None

# Skeleton loader for better perceived performance
def render_skeleton_loader(height: int = 100, count: int = 3):
    """Render skeleton loader while content is loading"""
    
    for i in range(count):
        with st.container():
            # Create skeleton effect with CSS
            st.markdown(f"""
            <div style="
                height: {height}px;
                background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
                background-size: 200% 100%;
                animation: loading 1.5s infinite;
                border-radius: 8px;
                margin: 8px 0;
            "></div>
            
            <style>
            @keyframes loading {{
                0% {{ background-position: 200% 0; }}
                100% {{ background-position: -200% 0; }}
            }}
            </style>
            """, unsafe_allow_html=True)