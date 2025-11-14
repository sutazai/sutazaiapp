#!/bin/bash

# JARVIS Frontend Startup Script
# This script starts the Streamlit frontend application

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting JARVIS Frontend...${NC}"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo -e "${RED}Virtual environment not found!${NC}"
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Verify all Streamlit components are properly installed
echo -e "${YELLOW}Verifying all Streamlit components...${NC}"

# Run comprehensive component verification if script exists
if [ -f "fix_all_streamlit_components.py" ]; then
    python fix_all_streamlit_components.py
    if [ $? -ne 0 ]; then
        echo -e "${RED}Some components failed verification!${NC}"
        echo -e "${YELLOW}Please check the components manually or run:${NC}"
        echo -e "${YELLOW}  python fix_all_streamlit_components.py${NC}"
        exit 1
    fi
else
    # Fallback: Check critical components manually
    echo -e "${YELLOW}Running fallback component checks...${NC}"
    
    # Check streamlit_chat
    if [ ! -d "venv/lib/python3.12/site-packages/streamlit_chat/frontend/dist" ]; then
        echo -e "${RED}streamlit_chat frontend assets missing! Reinstalling...${NC}"
        pip uninstall streamlit-chat -y
        pip install streamlit-chat==0.1.1 --no-cache-dir
    fi
    
    # Check streamlit_lottie
    if [ ! -d "venv/lib/python3.12/site-packages/streamlit_lottie/frontend/build" ]; then
        echo -e "${RED}streamlit_lottie frontend assets missing! Reinstalling...${NC}"
        pip uninstall streamlit-lottie -y
        pip install streamlit-lottie==0.0.5 --no-cache-dir
    fi
    
    # Check streamlit_option_menu
    if [ ! -d "venv/lib/python3.12/site-packages/streamlit_option_menu/frontend/dist" ]; then
        echo -e "${RED}streamlit_option_menu frontend assets missing! Reinstalling...${NC}"
        pip uninstall streamlit-option-menu -y
        pip install streamlit-option-menu==0.4.0 --no-cache-dir
    fi
fi

# Export environment variables
export BACKEND_URL=${BACKEND_URL:-"http://localhost:10200"}
export STREAMLIT_SERVER_PORT=${STREAMLIT_SERVER_PORT:-11000}
export STREAMLIT_SERVER_ADDRESS=${STREAMLIT_SERVER_ADDRESS:-"0.0.0.0"}
export STREAMLIT_THEME_BASE=${STREAMLIT_THEME_BASE:-"dark"}
export STREAMLIT_THEME_PRIMARY_COLOR=${STREAMLIT_THEME_PRIMARY_COLOR:-"#00D4FF"}
export WAKE_WORD=${WAKE_WORD:-"jarvis"}
export SPEECH_RECOGNITION_ENGINE=${SPEECH_RECOGNITION_ENGINE:-"google"}
export TTS_ENGINE=${TTS_ENGINE:-"pyttsx3"}
export VOICE_LANGUAGE=${VOICE_LANGUAGE:-"en-US"}
export ENABLE_VOICE_COMMANDS=${ENABLE_VOICE_COMMANDS:-"false"}
export ENABLE_TYPING_ANIMATION=${ENABLE_TYPING_ANIMATION:-"true"}
export SHOW_SYSTEM_METRICS=${SHOW_SYSTEM_METRICS:-"true"}
export SHOW_DOCKER_STATS=${SHOW_DOCKER_STATS:-"false"}
export PYTHONUNBUFFERED=1
export LOG_LEVEL=${LOG_LEVEL:-"INFO"}

echo -e "${GREEN}Starting Streamlit application on port $STREAMLIT_SERVER_PORT...${NC}"

# Start streamlit
streamlit run app.py \
    --server.port $STREAMLIT_SERVER_PORT \
    --server.address $STREAMLIT_SERVER_ADDRESS \
    --server.headless true \
    --theme.base $STREAMLIT_THEME_BASE \
    --theme.primaryColor "$STREAMLIT_THEME_PRIMARY_COLOR"