"""
Lazy Loading System for SutazAI Frontend Components
Implements intelligent component loading to reduce initial bundle size
"""

import streamlit as st
import asyncio
import importlib
import time
from functools import wraps
import logging
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)

class LazyComponentLoader:
    """Intelligent lazy loading system for heavy components"""
    
    def __init__(self):
        self._loaded_modules = {}
        self._loading_cache = {}
        self._component_registry = {}
        self._executor = ThreadPoolExecutor(max_workers=3)
        self._lock = threading.Lock()
    
    def register_component(self, name: str, import_path: str, 
                          load_condition: Optional[Callable] = None,
                          dependencies: Optional[List[str]] = None):
        """Register a component for lazy loading"""
        self._component_registry[name] = {
            'import_path': import_path,
            'load_condition': load_condition,
            'dependencies': dependencies or [],
            'loaded': False,
            'loading': False
        }
    
    def _should_load(self, component_name: str) -> bool:
        """Check if component should be loaded"""
        if component_name not in self._component_registry:
            return False
        
        component_info = self._component_registry[component_name]
        load_condition = component_info.get('load_condition')
        
        if load_condition and not load_condition():
            return False
        
        return True
    
    async def _load_module_async(self, import_path: str):
        """Load module asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, 
            lambda: importlib.import_module(import_path)
        )
    
    async def load_component(self, component_name: str):
        """Load component if not already loaded"""
        with self._lock:
            if component_name not in self._component_registry:
                logger.error(f"Component {component_name} not registered")
                return None
            
            component_info = self._component_registry[component_name]
            
            # Check if already loaded
            if component_info['loaded']:
                return self._loaded_modules.get(component_name)
            
            # Check if currently loading
            if component_info['loading']:
                # Wait for loading to complete
                while component_info['loading']:
                    await asyncio.sleep(0.1)
                return self._loaded_modules.get(component_name)
            
            # Check if should load
            if not self._should_load(component_name):
                return None
            
            # Start loading
            component_info['loading'] = True
        
        try:
            # Load dependencies first
            for dep in component_info['dependencies']:
                await self.load_component(dep)
            
            # Load the component
            start_time = time.time()
            module = await self._load_module_async(component_info['import_path'])
            load_time = time.time() - start_time
            
            with self._lock:
                self._loaded_modules[component_name] = module
                component_info['loaded'] = True
                component_info['loading'] = False
            
            logger.info(f"Loaded {component_name} in {load_time:.2f}s")
            return module
            
        except Exception as e:
            logger.error(f"Failed to load {component_name}: {e}")
            with self._lock:
                component_info['loading'] = False
            return None
    
    def sync_load_component(self, component_name: str):
        """Synchronous component loading"""
        return asyncio.run(self.load_component(component_name))
    
    def is_loaded(self, component_name: str) -> bool:
        """Check if component is loaded"""
        return self._component_registry.get(component_name, {}).get('loaded', False)
    
    def get_loaded_modules(self) -> Dict[str, Any]:
        """Get all loaded modules"""
        return self._loaded_modules.copy()
    
    def preload_components(self, component_names: List[str]):
        """Preload multiple components asynchronously"""
        async def preload_all():
            tasks = [self.load_component(name) for name in component_names]
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Run in background
        asyncio.create_task(preload_all())

# Global lazy loader instance
lazy_loader = LazyComponentLoader()

# Register common heavy components
lazy_loader.register_component(
    'plotly_charts',
    'plotly.express',
    load_condition=lambda: st.session_state.get('show_charts', False)
)

lazy_loader.register_component(
    'plotly_graph_objects',
    'plotly.graph_objects',
    dependencies=['plotly_charts']
)

lazy_loader.register_component(
    'pandas_heavy',
    'pandas',
    load_condition=lambda: st.session_state.get('need_dataframes', False)
)

lazy_loader.register_component(
    'numpy_compute',
    'numpy',
    load_condition=lambda: st.session_state.get('need_computation', False)
)

def lazy_import(component_name: str):
    """Decorator for lazy importing heavy modules"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check if component is loaded
            if not lazy_loader.is_loaded(component_name):
                with st.spinner(f"Loading {component_name}..."):
                    module = lazy_loader.sync_load_component(component_name)
                    if module is None:
                        st.error(f"Failed to load required component: {component_name}")
                        return None
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

class ConditionalRenderer:
    """Render components conditionally based on user interaction"""
    
    @staticmethod
    def render_on_demand(component_func: Callable, trigger_text: str, 
                        icon: str = "📊", help_text: str = None):
        """Render component only when user requests it"""
        
        button_key = f"render_{hash(trigger_text)}"
        
        if st.button(f"{icon} {trigger_text}", help=help_text, key=button_key):
            st.session_state[f"show_{button_key}"] = True
        
        if st.session_state.get(f"show_{button_key}", False):
            with st.container():
                component_func()
    
    @staticmethod
    def render_in_expander(component_func: Callable, title: str, 
                          expanded: bool = False, icon: str = "📋"):
        """Render component in expandable container"""
        
        with st.expander(f"{icon} {title}", expanded=expanded):
            # Only render when expanded
            if expanded or st.session_state.get(f"expanded_{hash(title)}", False):
                component_func()

class ProgressiveLoader:
    """Progressive loading with visual feedback"""
    
    @staticmethod
    def load_with_progress(components: List[tuple], progress_container=None):
        """
        Load components with progress indicator
        
        Args:
            components: List of (component_name, display_name) tuples
            progress_container: Streamlit container for progress bar
        """
        
        if progress_container is None:
            progress_container = st.empty()
        
        progress_bar = progress_container.progress(0)
        status_text = st.empty()
        
        total_components = len(components)
        loaded_modules = {}
        
        for i, (component_name, display_name) in enumerate(components):
            status_text.text(f"Loading {display_name}...")
            
            try:
                module = lazy_loader.sync_load_component(component_name)
                if module:
                    loaded_modules[component_name] = module
                    status_text.success(f"✅ {display_name} loaded")
                else:
                    status_text.warning(f"⚠️ {display_name} skipped")
            except Exception as e:
                status_text.error(f"❌ {display_name} failed: {str(e)}")
            
            # Update progress
            progress = (i + 1) / total_components
            progress_bar.progress(progress)
            
            # Brief pause for visual feedback
            time.sleep(0.1)
        
        # Clear progress indicators after a moment
        time.sleep(1)
        progress_container.empty()
        status_text.empty()
        
        return loaded_modules

class SmartPreloader:
    """Intelligent component preloading based on user behavior"""
    
    @staticmethod
    def predict_needed_components() -> List[str]:
        """Predict which components user will likely need"""
        current_page = st.session_state.get('current_page', 'Dashboard')
        
        # Define component dependencies by page
        page_components = {
            'Dashboard': ['plotly_charts', 'plotly_graph_objects'],
            'AI Chat': ['pandas_heavy'],
            'Agent Control': ['numpy_compute'],
            'Hardware Optimizer': ['plotly_charts', 'numpy_compute'],
            'Analytics': ['plotly_charts', 'plotly_graph_objects', 'pandas_heavy']
        }
        
        return page_components.get(current_page, [])
    
    @staticmethod
    def preload_for_page():
        """Preload components likely needed for current page"""
        needed_components = SmartPreloader.predict_needed_components()
        if needed_components:
            lazy_loader.preload_components(needed_components)

# Performance monitoring for lazy loading
class LazyLoadMetrics:
    """Monitor lazy loading performance"""
    
    @staticmethod
    def get_loading_stats() -> Dict[str, Any]:
        """Get lazy loading performance statistics"""
        total_registered = len(lazy_loader._component_registry)
        total_loaded = len(lazy_loader._loaded_modules)
        
        loading_ratio = (total_loaded / total_registered * 100) if total_registered > 0 else 0
        
        return {
            'total_registered': total_registered,
            'total_loaded': total_loaded,
            'loading_ratio': f"{loading_ratio:.1f}%",
            'loaded_components': list(lazy_loader._loaded_modules.keys())
        }
    
    @staticmethod
    def render_metrics_widget():
        """Render loading metrics widget"""
        stats = LazyLoadMetrics.get_loading_stats()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Registered Components", stats['total_registered'])
        
        with col2:
            st.metric("Loaded Components", stats['total_loaded'])
        
        with col3:
            st.metric("Loading Efficiency", stats['loading_ratio'])
        
        if stats['loaded_components']:
            with st.expander("Loaded Components Details"):
                for component in stats['loaded_components']:
                    st.text(f"✅ {component}")

# Export main components
__all__ = [
    'lazy_loader', 'lazy_import', 'ConditionalRenderer', 
    'ProgressiveLoader', 'SmartPreloader', 'LazyLoadMetrics'
]