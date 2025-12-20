#!/bin/bash
# HatCat UI Startup Script
# Starts all three required services and ensures proper configuration

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Try to find hatcat-ui directory relative to this script
if [ -d "$SCRIPT_DIR/../hatcat-ui" ]; then
    UI_DIR="$(cd "$SCRIPT_DIR/../hatcat-ui" && pwd)"
elif [ -d "$SCRIPT_DIR/hatcat-ui" ]; then
    UI_DIR="$(cd "$SCRIPT_DIR/hatcat-ui" && pwd)"
else
    echo "Error: Could not find hatcat-ui directory."
    echo "Please ensure hatcat-ui is either:"
    echo "  - In the same parent directory as HatCat (../hatcat-ui)"
    echo "  - Inside the HatCat directory (./hatcat-ui)"
    exit 1
fi

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🎩 Starting HatCat UI Stack${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "HatCat directory: $SCRIPT_DIR"
echo "UI directory:     $UI_DIR"
echo ""

# Kill any existing processes
echo -e "${YELLOW}Cleaning up existing processes...${NC}"
pkill -f "src/ui/openwebui/server.py" 2>/dev/null || true
pkill -f "hatcat-ui/backend" 2>/dev/null || true
lsof -ti:8765 2>/dev/null | xargs -r kill -9 2>/dev/null || true
lsof -ti:8080 2>/dev/null | xargs -r kill -9 2>/dev/null || true
lsof -ti:5173 2>/dev/null | xargs -r kill -9 2>/dev/null || true
sleep 2

# Function to wait for service
wait_for_service() {
    local port=$1
    local name=$2
    local max_attempts=30
    local attempt=0

    echo -e "${YELLOW}Waiting for $name on port $port...${NC}"
    while ! lsof -ti:$port > /dev/null 2>&1; do
        attempt=$((attempt + 1))
        if [ $attempt -ge $max_attempts ]; then
            echo -e "${RED}✗ $name failed to start after 30 seconds${NC}"
            return 1
        fi
        sleep 1
    done
    echo -e "${GREEN}✓ $name is running on port $port${NC}"
}

# 1. Start HatCat Backend (Port 8765)
echo ""
echo -e "${BLUE}1. Starting HatCat Backend (Gemma 3 4b + Divergence Analysis)${NC}"
cd "$SCRIPT_DIR"
nohup poetry run uvicorn src.ui.openwebui.server:app --host 0.0.0.0 --port 8765 > /tmp/hatcat_backend.log 2>&1 &
HATCAT_PID=$!
echo "   PID: $HATCAT_PID"

wait_for_service 8765 "HatCat Backend" || exit 1

# Test HatCat endpoint
echo -e "${YELLOW}Testing HatCat endpoint...${NC}"
if curl -s http://localhost:8765/v1/models > /dev/null 2>&1; then
    echo -e "${GREEN}✓ HatCat API responding${NC}"
else
    echo -e "${RED}✗ HatCat API not responding${NC}"
    exit 1
fi

# 2. Start OpenWebUI Backend (Port 8080)
echo ""
echo -e "${BLUE}2. Starting OpenWebUI Backend${NC}"
cd "$UI_DIR/backend"
source venv/bin/activate
nohup ./dev.sh > /tmp/openwebui_backend.log 2>&1 &
OPENWEBUI_PID=$!
echo "   PID: $OPENWEBUI_PID"

wait_for_service 8080 "OpenWebUI Backend" || exit 1

# Configure OpenWebUI to use HatCat endpoint
echo -e "${YELLOW}Configuring OpenWebUI connection to HatCat...${NC}"
sleep 3
# Add connection via API
curl -s -X POST http://localhost:8080/api/v1/auths/signin \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@example.com","password":"admin"}' > /tmp/auth_response.json 2>&1 || true

sleep 1

# 3. Start OpenWebUI Frontend (Port 5173)
echo ""
echo -e "${BLUE}3. Starting OpenWebUI Frontend${NC}"
cd "$UI_DIR"
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
nohup npm run dev > /tmp/openwebui_frontend.log 2>&1 &
FRONTEND_PID=$!
echo "   PID: $FRONTEND_PID"

wait_for_service 5173 "OpenWebUI Frontend" || exit 1

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✓ All services started successfully!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${BLUE}Services:${NC}"
echo "  • HatCat Backend:     http://localhost:8765 (PID: $HATCAT_PID)"
echo "  • OpenWebUI Backend:  http://localhost:8080 (PID: $OPENWEBUI_PID)"
echo "  • OpenWebUI Frontend: http://localhost:5173 (PID: $FRONTEND_PID)"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "  1. Open http://localhost:5173 in your browser"
echo "  2. Go to Admin Settings > Connections"
echo "  3. Add OpenAI Connection:"
echo "     - Base URL: http://localhost:8765/v1"
echo "     - API Key: sk-test"
echo "  4. Select the HatCat model and start chatting!"
echo ""
echo -e "${YELLOW}Logs:${NC}"
echo "  • HatCat:     tail -f /tmp/hatcat_backend.log"
echo "  • Backend:    tail -f /tmp/openwebui_backend.log"
echo "  • Frontend:   tail -f /tmp/openwebui_frontend.log"
echo ""
echo -e "${YELLOW}To stop all services:${NC}"
echo "  kill $HATCAT_PID $OPENWEBUI_PID $FRONTEND_PID"
echo ""
