"""
ALEX Backend - Multi-Provider AI Assistant API
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.config import get_settings, Settings
from database.db_manager import DatabaseManager
from providers.llmmanager import LLMManager
from models.schemas import (
    ChatRequest, ChatResponse, ConversationSummary, 
    HealthCheck, ErrorResponse
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
db_manager: Optional[DatabaseManager] = None
llm_manager: Optional[LLMManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global db_manager, llm_manager
    
    settings = get_settings()
    
    try:
        # Initialize database
        db_manager = DatabaseManager(settings.database_url)
        await db_manager.initialize()
        logger.info("Database initialized")
        
        # Initialize LLM manager
        llm_manager = LLMManager(settings)
        await llm_manager.initialize()
        logger.info("LLM Manager initialized")
        
        app.state.start_time = time.time()
        logger.info("ALEX Backend started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    finally:
        # Cleanup
        if llm_manager:
            await llm_manager.cleanup()
        if db_manager:
            await db_manager.close()
        logger.info("Application shutdown complete")


def create_app() -> FastAPI:
    """Create FastAPI application"""
    settings = get_settings()
    
    app = FastAPI(
        title="ALEX Backend",
        description="Multi-Provider AI Assistant API",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app


app = create_app()


def get_db() -> DatabaseManager:
    """Dependency to get database manager"""
    if db_manager is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    return db_manager


def get_llm() -> LLMManager:
    """Dependency to get LLM manager"""
    if llm_manager is None:
        raise HTTPException(status_code=503, detail="LLM Manager not initialized")
    return llm_manager


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_server_error",
            message="An unexpected error occurred"
        ).dict()
    )


@app.get("/health", response_model=HealthCheck)
async def health_check(
    db: DatabaseManager = Depends(get_db),
    llm: LLMManager = Depends(get_llm)
):
    """Health check endpoint"""
    try:
        # Check database connectivity
        db_healthy = await db.is_connected()
        
        # Check LLM providers
        provider_health = await llm.health_check()
        
        # Calculate uptime
        uptime = time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
        
        services = {
            "database": db_healthy,
            "llm_manager": llm.is_ready(),
            **provider_health
        }
        
        overall_status = "healthy" if all(services.values()) else "degraded"
        
        return HealthCheck(
            status=overall_status,
            timestamp=time.time(),
            services=services,
            uptime=uptime
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheck(
            status="unhealthy",
            timestamp=time.time(),
            services={"error": str(e)},
            uptime=0
        )


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: DatabaseManager = Depends(get_db),
    llm: LLMManager = Depends(get_llm)
):
    """Main chat endpoint"""
    try:
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or f"conv_{uuid.uuid4().hex[:8]}"
        
        # Get conversation history if requested
        history = []
        if request.use_history and request.conversation_id:
            history = await db.get_conversation_history(
                conversation_id, 
                limit=request.max_history or 10
            )
        
        # Generate response
        response = await llm.generate_response(
            message=request.message,
            model=request.model,
            conversation_id=conversation_id,
            history=history,
            system_prompt=request.system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        # Save conversation if requested
        if request.save_history:
            # Save user message
            await db.save_message(
                conversation_id=conversation_id,
                role="user",
                content=request.message
            )
            
            # Save assistant response
            await db.save_message(
                conversation_id=conversation_id,
                role="assistant",
                content=response["content"],
                model_used=response["model"],
                tokens_used=response["tokens_used"]
            )
        
        return ChatResponse(
            message=response["content"],
            conversation_id=conversation_id,
            model_used=response["model"],
            tokens_used=response["tokens_used"],
            response_time=response["response_time"]
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate response")


@app.get("/conversations", response_model=List[ConversationSummary])
async def list_conversations(
    limit: int = 20,
    db: DatabaseManager = Depends(get_db)
):
    """List recent conversations"""
    try:
        conversations = await db.list_conversations(limit=limit)
        return conversations
    except Exception as e:
        logger.error(f"Failed to list conversations: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve conversations")


@app.get("/conversations/{conversation_id}/history")
async def get_conversation_history(
    conversation_id: str,
    limit: int = 50,
    db: DatabaseManager = Depends(get_db)
):
    """Get conversation history"""
    try:
        history = await db.get_conversation_history(conversation_id, limit=limit)
        return history
    except Exception as e:
        logger.error(f"Failed to get conversation history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve conversation history")


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    db: DatabaseManager = Depends(get_db)
):
    """Delete a conversation"""
    try:
        await db.delete_conversation(conversation_id)
        return {"message": "Conversation deleted successfully"}
    except Exception as e:
        logger.error(f"Failed to delete conversation: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete conversation")


@app.put("/conversations/{conversation_id}/title")
async def update_conversation_title(
    conversation_id: str,
    title: str,
    db: DatabaseManager = Depends(get_db)
):
    """Update conversation title"""
    try:
        await db.update_conversation_title(conversation_id, title)
        return {"message": "Conversation title updated successfully"}
    except Exception as e:
        logger.error(f"Failed to update conversation title: {e}")
        raise HTTPException(status_code=500, detail="Failed to update conversation title")


@app.get("/models")
async def list_models(llm: LLMManager = Depends(get_llm)):
    """List available models"""
    try:
        models = llm.get_available_models()
        loaded_models = llm.get_loaded_models()
        
        model_list = []
        for model in models:
            model_list.append({
                "name": model,
                "loaded": model in loaded_models,
                "type": model.split('/')[0] if '/' in model else 'ollama'
            })
        
        return {"models": model_list}
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model list")


@app.post("/models/{model_name}/load")
async def load_model(
    model_name: str,
    llm: LLMManager = Depends(get_llm)
):
    """Load a specific model"""
    try:
        success = await llm.load_model(model_name)
        if success:
            return {"message": f"Model {model_name} loaded successfully"}
        else:
            raise HTTPException(status_code=400, detail=f"Failed to load model {model_name}")
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model {model_name}")


@app.post("/models/{model_name}/unload")
async def unload_model(
    model_name: str,
    llm: LLMManager = Depends(get_llm)
):
    """Unload a specific model"""
    try:
        success = await llm.unload_model(model_name)
        if success:
            return {"message": f"Model {model_name} unloaded successfully"}
        else:
            raise HTTPException(status_code=400, detail=f"Failed to unload model {model_name}")
    except Exception as e:
        logger.error(f"Failed to unload model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to unload model {model_name}")


@app.get("/stats")
async def get_stats(
    db: DatabaseManager = Depends(get_db),
    llm: LLMManager = Depends(get_llm)
):
    """Get system statistics"""
    try:
        db_stats = await db.get_stats()
        model_stats = llm.get_model_stats()
        
        return {
            "database": db_stats,
            "models": model_stats,
            "uptime": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
        }
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "ALEX Backend",
        "version": "1.0.0",
        "description": "Multi-Provider AI Assistant API",
        "status": "running"
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )