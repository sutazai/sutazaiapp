import chat_interface
import config_dashboard
import document_processor
import streamlit as st
import system_monitor

# -----------------------------
# SutazAi UI Core
# -----------------------------


def main():
    st.set_page_config(
        page_title="SutazAi Console",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Apply custom theme
    apply_sutazai_theme()

    # Main navigation
    with st.sidebar:
        display_sutazai_header()
        menu_choice = st.selectbox(
            "SutazAi Navigation",
            [
                " Live Dashboard",
                " SutazAi Studio",
                " Configuration",
                " Documents",
            ],
        )

    # Router
    if menu_choice == " Live Dashboard":
        system_monitor.show_dashboard()
    elif menu_choice == " SutazAi Studio":
        chat_interface.show_sutazai_chat()
    elif menu_choice == " Configuration":
        config_dashboard.show_config_console()
    elif menu_choice == " Documents":
        document_processor.show_document_center()


def apply_sutazai_theme():
    st.markdown(
        f"""
        <style>
            {open('frontend/assets/sutazai_theme.css').read()}
        </style>
        """,
        unsafe_allow_html=True,
    )


def display_sutazai_header():
    st.markdown(
        """
        <div class="sutazai-header">
            <div class="logo">SutazAi</div>
            <div class="status">
                <span class="pulse"></span> SutazAi Core Active
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Main Application
# -----------------------------


def display_sutazai_header():
    st.markdown(
        """
        <style>
            .sutazai-header {
                color: #FF6F00;
                font-family: 'Courier New', monospace;
                font-size: 2.5em;
                text-align: center;
                margin-bottom: 30px;
            }
            .sutazai-logo {
                text-align: center;
                font-family: 'Courier New', monospace;
                color: #FF6F00;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sutazai-logo">'
        + """
     _________         __                      _____   .__
    /   _____/ __ __ _/  |_ _____   ________  /  _  \\  |__|
    \\_____  \\ |  |  \\   __\\__  \\  \\___   / /  /_\\  \\ |  |
    /        \\|  |  / |  |   / __ \\_ /    / /    |    \\|  |
   /_______  /|____/  |__|  (____  //_____ \\____|__  /|__|
           \\/                    \\/       \\/        \\/
        """
        + "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
