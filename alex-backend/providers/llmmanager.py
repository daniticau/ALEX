"""
LLM Manager for ALEX - Handles multiple providers
"""

import asyncio
import time
import logging
from typing import List, Dict, Any, Optional
import aiohttp
import json

from core.config import Settings, get_model_config, get_api_key
from models.schemas import LLMResponse, ConversationHistory, ModelType
from providers.ollama_provider import OllamaProvider
from providers.openai_provider import OpenAIProvider
from providers.anthropic_provider import AnthropicProvider
from providers.deepseek_provider import DeepSeekProvider

logger = logging.getLogger(__name__)

class LLMManager:
    """Manages multiple LLM providers and model switching"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.providers = {}
        self.loaded_models = set()
        self.model_stats = {}
        self.ready = False
        
    async def initialize(self):
        """Initialize all available providers"""
        try:
            # Initialize Ollama provider
            self.providers['ollama'] = OllamaProvider(
                host=self.settings.ollama_host,
                timeout=self.settings.ollama_timeout
            )
            
            # Initialize API providers if keys are available
            if self.settings.openai_api_key:
                self.providers['openai'] = OpenAIProvider(
                    api_key=self.settings.openai_api_key
                )
                
            if self.settings.anthropic_api_key:
                self.providers['anthropic'] = AnthropicProvider(
                    api_key=self.settings.anthropic_api_key
                )
                
            if self.settings.deepseek_api_key:
                self.providers['deepseek'] = DeepSeekProvider(
                    api_key=self.settings.deepseek_api_key
                )
            
            # Initialize all providers
            for provider_name, provider in self.providers.items():
                try:
                    await provider.initialize()
                    logger.info(f"Initialized {provider_name} provider")
                except Exception as e:
                    logger.warning(f"Failed to initialize {provider_name}: {e}")
            
            self.ready = True
            logger.info("LLM Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM Manager: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup all providers"""
        for provider in self.providers.values():
            try:
                await provider.cleanup()
            except Exception as e:
                logger.warning(f"Error during provider cleanup: {e}")
        
        self.ready = False
        logger.info("LLM Manager cleanup completed")
    
    def is_ready(self) -> bool:
        """Check if manager is ready"""
        return self.ready
    
    def get_available_models(self) -> List[str]:
        """Get list of all available models"""
        return list(self.settings.model_configs.keys())
    
    def _parse_model_name(self, model_name: str) -> tuple:
        """Parse model name into provider and model"""
        if '/' in model_name:
            provider, model = model_name.split('/', 1)
            return provider, model
        else:
            # Default to ollama if no provider specified
            return 'ollama', model_name
    
    def _get_provider(self, provider_name: str):
        """Get provider instance"""
        provider = self.providers.get(provider_name)
        if not provider:
            raise ValueError(f"Provider {provider_name} not available")
        return provider
    
    def _build_messages(self, message: str, history: List[ConversationHistory], 
                       system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        """Build message list for LLM"""
        messages = []
        
        # Add system message if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        for hist in history:
            messages.append({
                "role": hist.role.value,
                "content": hist.content
            })
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        return messages
    
    async def generate_response(
        self,
        message: str,
        model: Optional[str] = None,
        conversation_id: Optional[str] = None,
        history: List[ConversationHistory] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate response using specified model"""
        
        if not self.ready:
            raise RuntimeError("LLM Manager not initialized")
        
        # Use default model if none specified
        if not model:
            model = self.settings.default_model
        
        # Get model configuration
        model_config = get_model_config(model)
        if not model_config:
            raise ValueError(f"Model {model} not configured")
        
        # Parse provider and model name
        provider_name, model_name = self._parse_model_name(model)
        provider = self._get_provider(provider_name)
        
        # Build parameters
        params = {
            "model": model_name,
            "temperature": temperature or model_config.get("temperature", self.settings.default_temperature),
            "max_tokens": max_tokens or model_config.get("max_tokens", self.settings.default_max_tokens),
            "system_prompt": system_prompt or model_config.get("system_prompt")
        }
        
        # Build message history
        history = history or []
        messages = self._build_messages(message, history, params["system_prompt"])
        
        # Generate response with timing
        start_time = time.time()
        
        try:
            response = await provider.generate_response(messages, params)
            response_time = time.time() - start_time
            
            # Update model stats
            self._update_model_stats(model, response_time, response.get("tokens_used", 0))
            
            return {
                "content": response["content"],
                "model": model,
                "tokens_used": response.get("tokens_used", 0),
                "response_time": response_time,
                "finish_reason": response.get("finish_reason")
            }
            
        except Exception as e:
            logger.error(f"Error generating response with {model}: {e}")
            raise RuntimeError(f"Failed to generate response: {str(e)}")
    
    def _update_model_stats(self, model: str, response_time: float, tokens: int):
        """Update model usage statistics"""
        if model not in self.model_stats:
            self.model_stats[model] = {
                "usage_count": 0,
                "total_tokens": 0,
                "total_time": 0.0,
                "avg_response_time": 0.0,
                "last_used": time.time()
            }
        
        stats = self.model_stats[model]
        stats["usage_count"] += 1
        stats["total_tokens"] += tokens
        stats["total_time"] += response_time
        stats["avg_response_time"] = stats["total_time"] / stats["usage_count"]
        stats["last_used"] = time.time()
    
    async def load_model(self, model_name: str) -> bool:
        """Load a specific model (mainly for local providers)"""
        provider_name, model = self._parse_model_name(model_name)
        
        if provider_name == 'ollama':
            provider = self._get_provider(provider_name)
            success = await provider.load_model(model)
            if success:
                self.loaded_models.add(model_name)
            return success
        
        # API models don't need explicit loading
        return True
    
    async def unload_model(self, model_name: str) -> bool:
        """Unload a specific model (mainly for local providers)"""
        provider_name, model = self._parse_model_name(model_name)
        
        if provider_name == 'ollama':
            provider = self._get_provider(provider_name)
            success = await provider.unload_model(model)
            if success:
                self.loaded_models.discard(model_name)
            return success
        
        # API models don't need explicit unloading
        return True
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get model usage statistics"""
        return self.model_stats.copy()
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models"""
        return list(self.loaded_models)
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all providers"""
        health = {}
        
        for provider_name, provider in self.providers.items():
            try:
                health[provider_name] = await provider.health_check()
            except Exception as e:
                logger.warning(f"Health check failed for {provider_name}: {e}")
                health[provider_name] = False
        
        return health