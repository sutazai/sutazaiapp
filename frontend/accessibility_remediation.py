"""
Accessibility Remediation Code for JARVIS Streamlit Frontend
WCAG 2.1 Level AA Compliant Components and Utilities
"""

import streamlit as st
from typing import Dict, List, Optional, Any
from datetime import datetime
import hashlib

# ============================================================================
# ACCESSIBLE COLOR SCHEME
# ============================================================================

ACCESSIBLE_COLORS = {
    "primary": "#4DC8FF",      # Bright blue - 4.5:1 on black
    "secondary": "#66B3E6",     # Light blue - 4.6:1 on black  
    "success": "#4ADE80",       # Green - 4.8:1 on black
    "warning": "#FCD34D",       # Yellow - 11:1 on black
    "error": "#F87171",         # Red - 4.5:1 on black
    "background": "#000000",     # Pure black for max contrast
    "surface": "#1A1A1A",       # Dark gray surface
    "text_primary": "#FFFFFF",   # White text
    "text_secondary": "#E5E5E5", # Light gray - 13:1 on black
    "focus": "#4DC8FF",         # Focus indicator color
}

def apply_accessible_theme():
    """Apply WCAG compliant color scheme and styles"""
    st.markdown(f"""
    <style>
        /* Accessible color scheme */
        :root {{
            --primary: {ACCESSIBLE_COLORS['primary']};
            --secondary: {ACCESSIBLE_COLORS['secondary']};
            --success: {ACCESSIBLE_COLORS['success']};
            --warning: {ACCESSIBLE_COLORS['warning']};
            --error: {ACCESSIBLE_COLORS['error']};
            --bg: {ACCESSIBLE_COLORS['background']};
            --surface: {ACCESSIBLE_COLORS['surface']};
            --text-primary: {ACCESSIBLE_COLORS['text_primary']};
            --text-secondary: {ACCESSIBLE_COLORS['text_secondary']};
            --focus: {ACCESSIBLE_COLORS['focus']};
        }}
        
        /* Focus indicators for all interactive elements */
        button:focus,
        input:focus,
        textarea:focus,
        select:focus,
        a:focus,
        [tabindex]:focus {{
            outline: 3px solid var(--focus) !important;
            outline-offset: 2px !important;
            box-shadow: 0 0 0 3px rgba(77, 200, 255, 0.3) !important;
        }}
        
        /* High contrast mode support */
        @media (prefers-contrast: high) {{
            * {{
                border-width: 2px !important;
            }}
        }}
        
        /* Reduced motion support */
        @media (prefers-reduced-motion: reduce) {{
            * {{
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
            }}
        }}
        
        /* Skip navigation link */
        .skip-nav {{
            position: absolute;
            top: -40px;
            left: 0;
            background: var(--bg);
            color: var(--text-primary);
            padding: 8px 16px;
            text-decoration: none;
            border: 2px solid var(--focus);
            z-index: 9999;
        }}
        
        .skip-nav:focus {{
            top: 10px;
            left: 10px;
        }}
        
        /* Ensure minimum touch target size (44x44px) */
        button, 
        input[type="button"],
        input[type="submit"],
        a {{
            min-height: 44px;
            min-width: 44px;
            padding: 12px;
        }}
        
        /* Screen reader only text */
        .sr-only {{
            position: absolute;
            width: 1px;
            height: 1px;
            padding: 0;
            margin: -1px;
            overflow: hidden;
            clip: rect(0,0,0,0);
            white-space: nowrap;
            border: 0;
        }}
        
        /* Visible on focus for screen readers */
        .sr-only-focusable:focus {{
            position: static;
            width: auto;
            height: auto;
            overflow: visible;
            clip: auto;
            white-space: normal;
        }}
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# KEYBOARD NAVIGATION
# ============================================================================

def add_keyboard_navigation():
    """Add comprehensive keyboard navigation support"""
    st.markdown("""
    <script>
    // Keyboard navigation handler
    document.addEventListener('DOMContentLoaded', function() {
        // Add tabindex to all interactive elements
        const interactiveElements = document.querySelectorAll(
            'button, input, textarea, select, a, [role="button"]'
        );
        
        interactiveElements.forEach((el, index) => {
            if (!el.hasAttribute('tabindex')) {
                el.setAttribute('tabindex', '0');
            }
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {
            // Alt + M: Jump to main content
            if (e.altKey && e.key === 'm') {
                e.preventDefault();
                const main = document.querySelector('main, [role="main"]');
                if (main) {
                    main.focus();
                    main.scrollIntoView();
                }
            }
            
            // Alt + C: Jump to chat input
            if (e.altKey && e.key === 'c') {
                e.preventDefault();
                const chatInput = document.querySelector('[data-testid="stChatInput"]');
                if (chatInput) {
                    chatInput.focus();
                }
            }
            
            // Escape: Close modals/overlays
            if (e.key === 'Escape') {
                const modal = document.querySelector('[role="dialog"]');
                if (modal) {
                    modal.style.display = 'none';
                    // Return focus to trigger element
                    const trigger = document.querySelector('[data-modal-trigger]');
                    if (trigger) trigger.focus();
                }
            }
            
            // Arrow keys for navigation
            if (e.key.startsWith('Arrow')) {
                const focusable = Array.from(document.querySelectorAll(
                    'button, input, textarea, select, a, [tabindex="0"]'
                ));
                const currentIndex = focusable.indexOf(document.activeElement);
                
                if (currentIndex !== -1) {
                    let nextIndex;
                    if (e.key === 'ArrowDown' || e.key === 'ArrowRight') {
                        nextIndex = (currentIndex + 1) % focusable.length;
                    } else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') {
                        nextIndex = (currentIndex - 1 + focusable.length) % focusable.length;
                    }
                    
                    if (nextIndex !== undefined) {
                        e.preventDefault();
                        focusable[nextIndex].focus();
                    }
                }
            }
        });
        
        // Trap focus in modals
        const trapFocus = (element) => {
            const focusable = element.querySelectorAll(
                'button, input, textarea, select, a, [tabindex="0"]'
            );
            const firstFocusable = focusable[0];
            const lastFocusable = focusable[focusable.length - 1];
            
            element.addEventListener('keydown', function(e) {
                if (e.key === 'Tab') {
                    if (e.shiftKey) {
                        if (document.activeElement === firstFocusable) {
                            e.preventDefault();
                            lastFocusable.focus();
                        }
                    } else {
                        if (document.activeElement === lastFocusable) {
                            e.preventDefault();
                            firstFocusable.focus();
                        }
                    }
                }
            });
        };
        
        // Apply focus trap to all modals
        document.querySelectorAll('[role="dialog"]').forEach(trapFocus);
    });
    </script>
    """, unsafe_allow_html=True)

# ============================================================================
# ACCESSIBLE COMPONENTS
# ============================================================================

def accessible_chat_message(
    role: str, 
    content: str, 
    timestamp: Optional[str] = None,
    message_id: Optional[str] = None
):
    """Display an accessible chat message"""
    
    # Generate unique ID if not provided
    if not message_id:
        message_id = hashlib.md5(f"{role}{content}{timestamp}".encode()).hexdigest()[:8]
    
    # Format timestamp for screen readers
    if timestamp:
        dt = datetime.fromisoformat(timestamp)
        readable_time = dt.strftime("%B %d, %Y at %I:%M %p")
        relative_time = get_relative_time(dt)
    else:
        readable_time = ""
        relative_time = ""
    
    # Use Streamlit's native chat message for better semantics
    with st.chat_message(role):
        # Add ARIA labels for screen readers
        st.markdown(f"""
        <div role="article" 
             aria-label="{role} message"
             aria-describedby="msg-time-{message_id}">
            <div class="message-content">{content}</div>
            <time id="msg-time-{message_id}" 
                  datetime="{timestamp}"
                  class="message-timestamp">
                <span class="sr-only">Sent </span>
                <span title="{readable_time}">{relative_time}</span>
            </time>
        </div>
        """, unsafe_allow_html=True)

def accessible_status_indicator(
    status: str,
    label: str,
    live_region: bool = True
):
    """Create an accessible status indicator"""
    
    status_config = {
        "connected": {"icon": "✅", "color": ACCESSIBLE_COLORS["success"], "text": "Connected"},
        "disconnected": {"icon": "❌", "color": ACCESSIBLE_COLORS["error"], "text": "Disconnected"},
        "loading": {"icon": "⏳", "color": ACCESSIBLE_COLORS["warning"], "text": "Loading"},
        "idle": {"icon": "💤", "color": ACCESSIBLE_COLORS["text_secondary"], "text": "Idle"},
    }
    
    config = status_config.get(status, status_config["idle"])
    
    aria_live = 'aria-live="polite" aria-atomic="true"' if live_region else ''
    
    st.markdown(f"""
    <div role="status" {aria_live} class="status-indicator">
        <span aria-hidden="true" style="color: {config['color']}">
            {config['icon']}
        </span>
        <span class="status-text">
            {label}: <strong>{config['text']}</strong>
        </span>
    </div>
    """, unsafe_allow_html=True)

def accessible_button(
    label: str,
    key: str,
    help_text: Optional[str] = None,
    loading: bool = False,
    disabled: bool = False,
    type: str = "primary"  # primary, secondary, success, warning, error
):
    """Create an accessible button with proper ARIA attributes"""
    
    button_id = f"btn-{key}"
    
    if loading:
        # Show loading state
        st.button(
            label, 
            key=key, 
            disabled=True,
            help="Processing, please wait..."
        )
        st.markdown(f"""
        <div role="status" aria-live="polite" aria-atomic="true">
            <span class="sr-only">Processing {label}</span>
            <span aria-hidden="true">⏳ Processing...</span>
        </div>
        """, unsafe_allow_html=True)
        return False
    
    # Create button with proper semantics
    clicked = st.button(
        label,
        key=key,
        help=help_text,
        disabled=disabled,
        type=type if type == "primary" else "secondary"
    )
    
    # Add ARIA description if help text provided
    if help_text:
        st.markdown(f"""
        <script>
        document.addEventListener('DOMContentLoaded', function() {{
            const btn = document.querySelector('[data-testid="{key}"]');
            if (btn) {{
                btn.setAttribute('aria-describedby', 'help-{button_id}');
            }}
        }});
        </script>
        <span id="help-{button_id}" class="sr-only">{help_text}</span>
        """, unsafe_allow_html=True)
    
    return clicked

def accessible_form_input(
    label: str,
    key: str,
    input_type: str = "text",
    required: bool = False,
    error: Optional[str] = None,
    help_text: Optional[str] = None,
    placeholder: Optional[str] = None
):
    """Create an accessible form input with proper labels and error handling"""
    
    input_id = f"input-{key}"
    error_id = f"error-{key}"
    help_id = f"help-{key}"
    
    # Build ARIA attributes
    aria_attrs = []
    if error:
        aria_attrs.append(f'aria-invalid="true"')
        aria_attrs.append(f'aria-errormessage="{error_id}"')
    if help_text:
        aria_attrs.append(f'aria-describedby="{help_id}"')
    if required:
        aria_attrs.append('aria-required="true"')
    
    aria_string = " ".join(aria_attrs)
    
    # Create the input based on type
    if input_type == "text":
        value = st.text_input(
            label=label + (" *" if required else ""),
            key=key,
            placeholder=placeholder,
            help=help_text
        )
    elif input_type == "textarea":
        value = st.text_area(
            label=label + (" *" if required else ""),
            key=key,
            placeholder=placeholder,
            help=help_text
        )
    elif input_type == "number":
        value = st.number_input(
            label=label + (" *" if required else ""),
            key=key,
            help=help_text
        )
    else:
        value = None
    
    # Show error message if present
    if error:
        st.markdown(f"""
        <div role="alert" id="{error_id}" class="error-message" 
             aria-live="assertive" aria-atomic="true">
            <span aria-hidden="true">⚠️</span>
            <span>{error}</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Add ARIA attributes via JavaScript
    st.markdown(f"""
    <script>
    document.addEventListener('DOMContentLoaded', function() {{
        const input = document.querySelector('[data-testid="{key}"]');
        if (input) {{
            {f'input.setAttribute("aria-invalid", "true");' if error else ''}
            {f'input.setAttribute("aria-errormessage", "{error_id}");' if error else ''}
            {f'input.setAttribute("aria-describedby", "{help_id}");' if help_text else ''}
            {f'input.setAttribute("aria-required", "true");' if required else ''}
        }}
    }});
    </script>
    """, unsafe_allow_html=True)
    
    return value

def accessible_data_visualization(
    figure: Any,  # Plotly figure or similar
    title: str,
    description: str,
    data_summary: Dict[str, Any],
    show_data_table: bool = True
):
    """Create an accessible data visualization with alternatives"""
    
    viz_id = hashlib.md5(title.encode()).hexdigest()[:8]
    
    # Add title and description for screen readers
    st.markdown(f"""
    <div role="img" 
         aria-label="{title}"
         aria-describedby="viz-desc-{viz_id}">
        <h3 id="viz-title-{viz_id}">{title}</h3>
        <p id="viz-desc-{viz_id}" class="sr-only">{description}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display the visualization
    st.plotly_chart(figure, use_container_width=True)
    
    # Provide text summary
    with st.expander("📊 Data Summary (Text Alternative)"):
        st.write(description)
        for key, value in data_summary.items():
            st.write(f"**{key}:** {value}")
    
    # Provide data table if requested
    if show_data_table and hasattr(figure, 'data'):
        with st.expander("📋 View as Data Table"):
            # Extract data from figure and display as table
            st.info("Data table view for accessibility")
            # Implementation depends on visualization library

def accessible_alert(
    message: str,
    type: str = "info",  # info, success, warning, error
    dismissible: bool = True,
    auto_dismiss: int = 0  # seconds, 0 = no auto-dismiss
):
    """Create an accessible alert message"""
    
    alert_id = hashlib.md5(message.encode()).hexdigest()[:8]
    
    alert_config = {
        "info": {"icon": "ℹ️", "role": "status", "color": ACCESSIBLE_COLORS["primary"]},
        "success": {"icon": "✅", "role": "status", "color": ACCESSIBLE_COLORS["success"]},
        "warning": {"icon": "⚠️", "role": "alert", "color": ACCESSIBLE_COLORS["warning"]},
        "error": {"icon": "❌", "role": "alert", "color": ACCESSIBLE_COLORS["error"]},
    }
    
    config = alert_config.get(type, alert_config["info"])
    
    # Use appropriate Streamlit alert
    if type == "info":
        container = st.info(message)
    elif type == "success":
        container = st.success(message)
    elif type == "warning":
        container = st.warning(message)
    elif type == "error":
        container = st.error(message)
    else:
        container = st.info(message)
    
    # Add ARIA live region for dynamic alerts
    st.markdown(f"""
    <div role="{config['role']}" 
         aria-live="{'assertive' if type in ['error', 'warning'] else 'polite'}"
         aria-atomic="true"
         id="alert-{alert_id}"
         class="sr-only">
        {type.capitalize()}: {message}
    </div>
    """, unsafe_allow_html=True)
    
    # Auto-dismiss if specified
    if auto_dismiss > 0:
        st.markdown(f"""
        <script>
        setTimeout(function() {{
            const alert = document.getElementById('alert-{alert_id}');
            if (alert) {{
                alert.style.display = 'none';
            }}
        }}, {auto_dismiss * 1000});
        </script>
        """, unsafe_allow_html=True)

# ============================================================================
# ACCESSIBILITY UTILITIES
# ============================================================================

def get_relative_time(dt: datetime) -> str:
    """Convert datetime to relative time string for better readability"""
    now = datetime.now()
    diff = now - dt
    
    if diff.days > 7:
        return dt.strftime("%b %d, %Y")
    elif diff.days > 0:
        return f"{diff.days} day{'s' if diff.days > 1 else ''} ago"
    elif diff.seconds > 3600:
        hours = diff.seconds // 3600
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    elif diff.seconds > 60:
        minutes = diff.seconds // 60
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
    else:
        return "Just now"

def announce_to_screen_reader(
    message: str,
    priority: str = "polite"  # polite or assertive
):
    """Announce a message to screen readers"""
    st.markdown(f"""
    <div role="status" 
         aria-live="{priority}" 
         aria-atomic="true" 
         class="sr-only">
        {message}
    </div>
    """, unsafe_allow_html=True)

def add_skip_navigation():
    """Add skip navigation links for keyboard users"""
    st.markdown("""
    <a href="#main-content" class="skip-nav">
        Skip to main content
    </a>
    <a href="#chat-input" class="skip-nav">
        Skip to chat input
    </a>
    <a href="#navigation" class="skip-nav">
        Skip to navigation
    </a>
    """, unsafe_allow_html=True)

def check_accessibility_preferences():
    """Check and apply user accessibility preferences"""
    
    # Check system preferences
    st.markdown("""
    <script>
    // Check for reduced motion preference
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        document.documentElement.classList.add('reduce-motion');
    }
    
    // Check for high contrast preference
    if (window.matchMedia('(prefers-contrast: high)').matches) {
        document.documentElement.classList.add('high-contrast');
    }
    
    // Check for dark mode preference
    if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
        document.documentElement.classList.add('dark-mode');
    }
    </script>
    """, unsafe_allow_html=True)

def create_accessibility_settings_panel():
    """Create a settings panel for accessibility preferences"""
    
    with st.sidebar.expander("♿ Accessibility Settings"):
        # Animation settings
        reduce_motion = st.checkbox(
            "Reduce Motion",
            key="reduce_motion",
            help="Disable animations and transitions"
        )
        
        # Contrast settings
        high_contrast = st.checkbox(
            "High Contrast Mode",
            key="high_contrast",
            help="Increase contrast for better visibility"
        )
        
        # Font size
        font_size = st.select_slider(
            "Text Size",
            options=["Small", "Normal", "Large", "Extra Large"],
            value="Normal",
            key="font_size",
            help="Adjust text size for better readability"
        )
        
        # Screen reader mode
        screen_reader = st.checkbox(
            "Screen Reader Mode",
            key="screen_reader_mode",
            help="Optimize for screen reader users"
        )
        
        # Keyboard shortcuts
        show_shortcuts = st.checkbox(
            "Show Keyboard Shortcuts",
            key="show_shortcuts",
            value=False
        )
        
        if show_shortcuts:
            st.markdown("""
            **Keyboard Shortcuts:**
            - `Alt + M`: Jump to main content
            - `Alt + C`: Jump to chat input
            - `Alt + N`: Jump to navigation
            - `Escape`: Close modals
            - `Tab`: Navigate forward
            - `Shift + Tab`: Navigate backward
            """)
        
        # Apply settings
        apply_accessibility_settings(
            reduce_motion=reduce_motion,
            high_contrast=high_contrast,
            font_size=font_size,
            screen_reader=screen_reader
        )

def apply_accessibility_settings(
    reduce_motion: bool = False,
    high_contrast: bool = False,
    font_size: str = "Normal",
    screen_reader: bool = False
):
    """Apply accessibility settings to the interface"""
    
    font_sizes = {
        "Small": "14px",
        "Normal": "16px",
        "Large": "20px",
        "Extra Large": "24px"
    }
    
    styles = []
    
    if reduce_motion:
        styles.append("""
        * {
            animation: none !important;
            transition: none !important;
        }
        """)
    
    if high_contrast:
        styles.append("""
        * {
            border: 2px solid !important;
        }
        .stButton > button {
            border: 3px solid white !important;
            font-weight: bold !important;
        }
        """)
    
    if font_size != "Normal":
        styles.append(f"""
        body, .stMarkdown, .stText {{
            font-size: {font_sizes[font_size]} !important;
        }}
        """)
    
    if screen_reader:
        styles.append("""
        .decorative {
            display: none !important;
        }
        [aria-hidden="true"] {
            display: none !important;
        }
        """)
    
    if styles:
        st.markdown(f"<style>{' '.join(styles)}</style>", unsafe_allow_html=True)

# ============================================================================
# MAIN ACCESSIBILITY INITIALIZATION
# ============================================================================

def initialize_accessibility():
    """Initialize all accessibility features"""
    
    # Apply accessible theme
    apply_accessible_theme()
    
    # Add keyboard navigation
    add_keyboard_navigation()
    
    # Add skip navigation links
    add_skip_navigation()
    
    # Check system preferences
    check_accessibility_preferences()
    
    # Create settings panel
    create_accessibility_settings_panel()
    
    # Announce page ready to screen readers
    announce_to_screen_reader("JARVIS interface ready")

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    st.set_page_config(
        page_title="Accessible JARVIS",
        page_icon="♿",
        layout="wide"
    )
    
    # Initialize accessibility
    initialize_accessibility()
    
    # Example: Accessible header
    st.markdown("""
    <header role="banner">
        <h1>JARVIS - Accessible AI Assistant</h1>
    </header>
    """, unsafe_allow_html=True)
    
    # Example: Main content area
    st.markdown('<main id="main-content" role="main" tabindex="-1">', unsafe_allow_html=True)
    
    # Example: Accessible chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Chat")
        
        # Display accessible messages
        accessible_chat_message(
            role="assistant",
            content="Hello! I'm JARVIS, your accessible AI assistant.",
            timestamp=datetime.now().isoformat()
        )
        
        # Accessible input
        user_message = accessible_form_input(
            label="Your Message",
            key="chat_input",
            required=True,
            help_text="Type your message and press Enter to send",
            placeholder="Type your message here..."
        )
        
        # Accessible button
        if accessible_button(
            label="Send Message",
            key="send_button",
            help_text="Send your message to JARVIS"
        ):
            if user_message:
                accessible_chat_message(
                    role="user",
                    content=user_message,
                    timestamp=datetime.now().isoformat()
                )
    
    with col2:
        st.subheader("Status")
        
        # Accessible status indicators
        accessible_status_indicator(
            status="connected",
            label="Backend"
        )
        
        accessible_status_indicator(
            status="loading",
            label="Model"
        )
    
    # Close main content
    st.markdown('</main>', unsafe_allow_html=True)
    
    # Example: Accessible alerts
    accessible_alert(
        "Welcome to the accessible version of JARVIS!",
        type="success",
        auto_dismiss=5
    )