#!/bin/bash
# Unlimited OCR - Start Script v1.2.1
# This script updates the project from GitHub, cleans cache, and starts the application

set -e

# Configuration
REPO_URL="https://github.com/viktorbrojs-sys/unlimited-ocr"
BRANCH="main"
VENV_DIR=".venv"
APP_FILE="app.py"
PORT=7860

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     Unlimited OCR - Start Script v1.2.1               ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to get current version
get_version() {
    if [ -f "VERSION" ]; then
        cat VERSION
    else
        echo "unknown"
    fi
}

# Display current version
CURRENT_VERSION=$(get_version)
echo -e "${GREEN}Current version:${NC} $CURRENT_VERSION"
echo ""

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo -e "${RED}Error: Not a git repository. Please clone the repository first.${NC}"
    echo "Usage: git clone $REPO_URL"
    exit 1
fi

# Step 1: Update from GitHub
echo -e "${YELLOW}Step 1: Updating from GitHub...${NC}"
git fetch origin $BRANCH

# Check if there are changes to pull
if ! git diff --quiet HEAD@{u} 2>/dev/null; then
    echo -e "${YELLOW}Changes detected. Pulling updates...${NC}"
    # Stash any local changes
    git stash push -m "local-changes-before-update" >/dev/null 2>&1 || true
    git pull origin $BRANCH
    echo -e "${GREEN}✓ Repository updated successfully${NC}"
else
    echo -e "${GREEN}✓ Repository is up to date${NC}"
fi
echo ""

# Step 2: Create/activate virtual environment
echo -e "${YELLOW}Step 2: Setting up virtual environment...${NC}"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv $VENV_DIR
fi

# Activate virtual environment
source $VENV_DIR/bin/activate
echo -e "${GREEN}✓ Virtual environment ready${NC}"
echo ""

# Step 3: Install dependencies
echo -e "${YELLOW}Step 3: Installing dependencies...${NC}"
pip install -q --upgrade pip

# Two-step install (see install.sh / README.md): transformers==4.57.1 pins
# huggingface-hub<1.0 while gradio 6 needs >=1.16, so it must go in separately.
if [ -f "requirements.txt" ]; then
    pip install -q -r requirements.txt
fi
pip install -q "transformers==4.57.1"

echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Step 4: Clear Gradio cache (safe — this is only UI/temp-upload cache, not model weights)
echo -e "${YELLOW}Step 4: Clearing Gradio cache...${NC}"
rm -rf /tmp/gradio/*/ 2>/dev/null || true
rm -rf .gradio_cache/ 2>/dev/null || true
echo -e "${GREEN}✓ Cache cleared${NC}"
echo ""

# Step 6: Start the application
echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          Starting Unlimited OCR Server               ║${NC}"
echo -e "${BLUE}║          Version: $CURRENT_VERSION                    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Server will start on http://127.0.0.1:$PORT${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo ""

# app.py already auto-increments PORT internally on OSError (busy port),
# so we just run it once and let it handle that.
export PORT
python $APP_FILE
