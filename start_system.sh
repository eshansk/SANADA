#!/usr/bin/env bash
# ==============================================================================
# start_system.sh
# SANADA Machine Listening Full-Stack Launcher (Member 5)
# Starts both the FastAPI Backend API (Port 8000) and Frontend UI (Port 3000)
# ==============================================================================

echo "=================================================================="
echo " 🎧 Starting SANADA Full-Stack Machine Listening System"
echo "=================================================================="

# Function to clean up background processes upon exit
cleanup() {
    echo ""
    echo "Shutting down SANADA services..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

# 1. Start Backend API Server
echo " [1/2] Starting Backend API (FastAPI) on http://127.0.0.1:8000 ..."
python3 backend/app.py &
BACKEND_PID=$!

# Wait briefly for model checkpoints and engine to initialize
sleep 2

# 2. Start Unified Frontend & API Proxy Server
echo " [2/2] Starting Unified Server (Frontend & API Proxy) on http://0.0.0.0:3000 ..."
python3 dev_server.py &
FRONTEND_PID=$!

echo "------------------------------------------------------------------"
echo " 🚀 System is LIVE and ready:"
echo "    • Unified Web Client  : http://localhost:3000 (serves UI & proxies /api)"
echo "    • Backend API Direct  : http://localhost:8000"
echo "    • Interactive API Docs: http://localhost:8000/docs"
echo "    • Public Tunnel Ready : Run '~/cloudflared tunnel --url http://localhost:3000'"
echo "------------------------------------------------------------------"
echo " Press [CTRL + C] anytime to stop both servers."
echo "=================================================================="

wait
