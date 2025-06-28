#!/bin/bash
# Shell script to start the Custom LLM Chatbot system on Linux/Mac

set -e  # Exit on any error

echo "========================================"
echo "Custom LLM Chatbot Startup Script"
echo "========================================"
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

# Check if virtual environment exists
if [ ! -f "venv/bin/activate" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment"
        exit 1
    fi
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Warning: Some dependencies may not have installed correctly"
fi

echo
echo "========================================"
echo "Starting LLM Server and UI"
echo "========================================"
echo

# Get model path from user
read -p "Enter the path to your trained model (or press Enter for default): " MODEL_PATH
if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="microsoft/DialoGPT-medium"
    echo "Using default model: $MODEL_PATH"
fi

# Get server type
echo
echo "Available server types:"
echo "1. pytorch (default, compatible with all systems)"
echo "2. vllm (high performance, requires compatible GPU)"
echo "3. onnx (optimized inference)"
echo
read -p "Choose server type (1-3, default=1): " SERVER_CHOICE

case $SERVER_CHOICE in
    2)
        SERVER_TYPE="vllm"
        ;;
    3)
        SERVER_TYPE="onnx"
        ;;
    *)
        SERVER_TYPE="pytorch"
        ;;
esac

echo
echo "Starting LLM server with $SERVER_TYPE backend..."
echo "Model: $MODEL_PATH"
echo
echo "The server will start on http://localhost:8000"
echo "The UI will start on http://localhost:8501"
echo
echo "Press Ctrl+C to stop the servers"
echo

# Function to cleanup background processes
cleanup() {
    echo
    echo "Stopping servers..."
    if [ ! -z "$SERVER_PID" ]; then
        kill $SERVER_PID 2>/dev/null || true
    fi
    if [ ! -z "$UI_PID" ]; then
        kill $UI_PID 2>/dev/null || true
    fi
    echo "Chatbot system stopped."
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start the LLM server in background
echo "Starting LLM server..."
python serve.py --server-type "$SERVER_TYPE" --model-path "$MODEL_PATH" --host localhost --port 8000 &
SERVER_PID=$!

# Wait a moment for server to start
echo "Waiting for server to initialize..."
sleep 10

# Check if server is still running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "Error: LLM server failed to start"
    exit 1
fi

# Start the UI
echo "Starting Streamlit UI..."
python run_ui.py --host localhost --port 8501 --server-url http://localhost:8000 --browser &
UI_PID=$!

# Wait for both processes
wait $SERVER_PID $UI_PID

echo
echo "Chatbot system stopped."