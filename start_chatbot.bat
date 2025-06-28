@echo off
REM Batch script to start the Custom LLM Chatbot system on Windows

echo ========================================
echo Custom LLM Chatbot Startup Script
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment not found. Creating one...
    python -m venv venv
    if errorlevel 1 (
        echo Error: Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/update dependencies
echo Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo Warning: Some dependencies may not have installed correctly
)

echo.
echo ========================================
echo Starting LLM Server and UI
echo ========================================
echo.

REM Get model path from user
set /p MODEL_PATH="Enter the path to your trained model (or press Enter for default): "
if "%MODEL_PATH%"=="" (
    set MODEL_PATH="microsoft/DialoGPT-medium"
    echo Using default model: %MODEL_PATH%
)

REM Get server type
echo.
echo Available server types:
echo 1. pytorch (default, compatible with all systems)
echo 2. vllm (high performance, requires compatible GPU)
echo 3. onnx (optimized inference)
echo.
set /p SERVER_CHOICE="Choose server type (1-3, default=1): "

if "%SERVER_CHOICE%"=="2" (
    set SERVER_TYPE=vllm
) else if "%SERVER_CHOICE%"=="3" (
    set SERVER_TYPE=onnx
) else (
    set SERVER_TYPE=pytorch
)

echo.
echo Starting LLM server with %SERVER_TYPE% backend...
echo Model: %MODEL_PATH%
echo.
echo The server will start on http://localhost:8000
echo The UI will start on http://localhost:8501
echo.
echo Press Ctrl+C to stop the servers
echo.

REM Start the LLM server in background
start "LLM Server" cmd /k "python serve.py --server-type %SERVER_TYPE% --model-path %MODEL_PATH% --host localhost --port 8000"

REM Wait a moment for server to start
echo Waiting for server to initialize...
timeout /t 10 /nobreak >nul

REM Start the UI
echo Starting Streamlit UI...
python run_ui.py --host localhost --port 8501 --server-url http://localhost:8000 --browser

echo.
echo Chatbot system stopped.
pause