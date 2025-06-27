# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ALEX is a multi-provider AI assistant backend built with FastAPI and Python. It supports multiple LLM providers (Ollama, OpenAI, Anthropic, DeepSeek) and provides a unified API for chat interactions with conversation history persistence.

## Core Architecture

- **FastAPI Application**: Main API server in `main.py` (currently empty - needs implementation)
- **Multi-Provider System**: Unified LLM manager that handles different AI providers
- **Database Layer**: SQLite with async support for conversation history and stats
- **Configuration Management**: Environment-based settings with model configurations
- **Docker Support**: Full containerization with Ollama integration

### Key Components

- `core/config.py`: Centralized configuration with model definitions and API key management
- `providers/llmmanager.py`: Main orchestrator that manages all LLM providers
- `database/db_manager.py`: Async SQLite operations for conversations and statistics
- `models/schemas.py`: Pydantic models for API requests/responses and data validation
- `providers/`: Individual provider implementations (Ollama, OpenAI, Anthropic, DeepSeek)

## Quick Setup

### Prerequisites
- Python 3.11+
- (Optional) Ollama for local models: https://ollama.ai

### Quick Start
```bash
cd alex-backend

# Copy and edit environment file
cp .env.example .env
# Edit .env with your API keys

# Option 1: Use startup scripts
./start.sh          # Linux/Mac
start.bat           # Windows

# Option 2: Manual setup
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### API Access
- Main API: http://localhost:8000
- Interactive docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

## Development Commands

### Local Development
```bash
# Install dependencies
pip install -r alex-backend/requirements.txt

# Run development server
cd alex-backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest tests/

# Code formatting and linting
black .
flake8 .
isort .
```

### Docker Development
```bash
# Build and run full stack with Ollama
cd alex-backend
docker-compose up --build

# Run backend only
docker build -t alex-backend .
docker run -p 8000:8000 alex-backend
```

## Model Configuration

Models are configured in `core/config.py` with the following providers:
- **Ollama**: Local models (llama3.1:8b, deepseek-coder)
- **OpenAI**: GPT-4, GPT-3.5-turbo
- **Anthropic**: Claude-3-sonnet
- **DeepSeek**: DeepSeek-chat

Model selection format: `provider/model-name` (e.g., `ollama/llama3.1:8b`)

## Environment Variables

Required environment variables (create `.env` file):
```
DATABASE_URL=sqlite:///./data/alex.db
OLLAMA_HOST=http://localhost:11434
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
DEEPSEEK_API_KEY=your_key_here
DEFAULT_MODEL=ollama/llama3.1:8b
```

## Database Schema

The system uses SQLite with three main tables:
- `conversations`: Conversation metadata and titles
- `messages`: Individual messages with role, content, and model info
- `model_stats`: Usage statistics and performance metrics

## Important Implementation Notes

- **Main FastAPI App**: `main.py` is currently empty and needs the FastAPI application implementation
- **Async Patterns**: All database and provider operations use async/await
- **Error Handling**: Providers have fallback mechanisms and health checks
- **Conversation Context**: Messages are linked by `conversation_id` for context preservation
- **Model Statistics**: Automatic tracking of usage, tokens, and response times

## Testing

Tests are organized by component:
- `test_database.py`: Database operations and schema validation
- `test_providers.py`: LLM provider functionality
- `test_main.py`: API endpoint testing (needs main.py implementation)

Run specific test categories:
```bash
pytest tests/test_database.py
pytest tests/test_providers.py -v
```