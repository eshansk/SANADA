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

# 2. Start Frontend Web Client
echo " [2/2] Starting Frontend UI on http://127.0.0.1:3000 ..."
python3 -m http.server 3000 --directory frontend &
FRONTEND_PID=$!

echo "------------------------------------------------------------------"
echo " 🚀 System is LIVE and ready:"
echo "    • Frontend Web Client : http://localhost:3000"
echo "    • Backend API Service : http://localhost:8000"
echo "    • Interactive API Docs: http://localhost:8000/docs"
echo "------------------------------------------------------------------"
echo " Press [CTRL + C] anytime to stop both servers."
echo "=================================================================="

wait
