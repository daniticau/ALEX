#!/bin/bash

# ALEX Backend Startup Script

echo "🤖 Starting ALEX Backend..."

# Create necessary directories
mkdir -p data logs

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  No .env file found. Creating from .env.example..."
    cp .env.example .env
    echo "✅ Created .env file. Please edit it with your API keys before continuing."
    echo "📝 Edit .env file with your actual API keys and settings."
    exit 1
fi

# Install dependencies if needed
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Start the application
echo "🚀 Starting ALEX Backend on http://localhost:8000"
echo "📖 API docs available at http://localhost:8000/docs"
echo "🏥 Health check at http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop"

uvicorn main:app --host 0.0.0.0 --port 8000 --reload