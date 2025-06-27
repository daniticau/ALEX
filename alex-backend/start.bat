@echo off
echo 🤖 Starting ALEX Backend...

REM Create necessary directories
if not exist "data" mkdir data
if not exist "logs" mkdir logs

REM Check if .env exists
if not exist ".env" (
    echo ⚠️  No .env file found. Creating from .env.example...
    copy .env.example .env
    echo ✅ Created .env file. Please edit it with your API keys before continuing.
    echo 📝 Edit .env file with your actual API keys and settings.
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

REM Start the application
echo 🚀 Starting ALEX Backend on http://localhost:8000
echo 📖 API docs available at http://localhost:8000/docs
echo 🏥 Health check at http://localhost:8000/health
echo.
echo Press Ctrl+C to stop

uvicorn main:app --host 0.0.0.0 --port 8000 --reload