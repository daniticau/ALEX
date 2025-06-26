"""
Ollama Provider for ALEX - Local LLM inference
"""

import asyncio
import aiohttp
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class OllamaProvider:
    """Provider for Ollama local LLM inference"""
    
    def __init__(self, host: str = "http://localhost:11434", timeout: int = 120):
        self.host = host.rstrip('/')
        self.timeout = timeout
        self.session = None
        self.available_models = []
        
    async def initialize(self):
        """Initialize the Ollama provider"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        
        # Check if Ollama is running and get available models
        try:
            await self.health_check()
            await self._refresh_models()
            logger.info(f"Ollama provider initialized with {len(self.available_models)} models")
        except Exception as e:
            logger.warning(f"Ollama initialization failed: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
    
    async def health_check(self) -> bool:
        """Check if Ollama is running"""
        try:
            async with self.session.get(f"{self.host}/api/tags") as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False
    
    async def _refresh_models(self):
        """Refresh list of available models"""
        try:
            async with self.session.get(f"{self.host}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    self.available_models = [model["name"] for model in data.get("models", [])]
                else:
                    logger.warning(f"Failed to fetch models: {response.status}")
        except Exception as e:
            logger.error(f"Error refreshing models: {e}")
    
    async def generate_response(self, messages: List[Dict[str, str]], params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response using Ollama"""
        
        # Convert messages to Ollama format
        prompt = self._format_messages(messages)
        
        payload = {
            "model": params["model"],
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": params.get("temperature", 0.7),
                "num_predict": params.get("max_tokens", 2048),
            }
        }
        
        try:
            async with self.session.post(
                f"{self.host}/api/generate",
                json=payload
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Ollama API error ({response.status}): {error_text}")
                
                data = await response.json()
                
                return {
                    "content": data.get("response", ""),
                    "tokens_used": self._estimate_tokens(data.get("response", "")),
                    "finish_reason": "stop" if data.get("done", False) else "incomplete"
                }
                
        except asyncio.TimeoutError:
            raise RuntimeError("Ollama request timed out")
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise RuntimeError(f"Ollama generation failed: {str(e)}")
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to a single prompt string"""
        prompt_parts = []
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars ≈ 1 token)"""
        return len(text) // 4
    
    async def load_model(self, model_name: str) -> bool:
        """Load a model in Ollama"""
        try:
            # First try to pull the model if it doesn't exist
            if model_name not in self.available_models:
                await self._pull_model(model_name)
            
            # Test the model by making a simple request
            test_payload = {
                "model": model_name,
                "prompt": "Hello",
                "stream": False,
                "options": {"num_predict": 1}
            }
            
            async with self.session.post(
                f"{self.host}/api/generate",
                json=test_payload
            ) as response:
                if response.status == 200:
                    await self._refresh_models()
                    return True
                else:
                    logger.error(f"Failed to load model {model_name}: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return False
    
    async def _pull_model(self, model_name: str):
        """Pull a model from Ollama registry"""
        logger.info(f"Pulling model {model_name}...")
        
        payload = {"name": model_name}
        
        async with self.session.post(
            f"{self.host}/api/pull",
            json=payload
        ) as response:
            if response.status != 200:
                raise RuntimeError(f"Failed to pull model {model_name}")
            
            # Stream the pull progress
            async for line in response.content:
                if line:
                    try:
                        data = json.loads(line.decode())
                        if "status" in data:
                            logger.info(f"Pull progress: {data['status']}")
                    except json.JSONDecodeError:
                        continue
    
    async def unload_model(self, model_name: str) -> bool:
        """Unload a model from memory (Ollama manages this automatically)"""
        # Ollama automatically manages model loading/unloading
        # We can't explicitly unload, but we can verify the model exists
        await self._refresh_models()
        return model_name in self.available_models
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return self.available_models.copy()
    
    async def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a model"""
        try:
            payload = {"name": model_name}
            async with self.session.post(
                f"{self.host}/api/show",
                json=payload
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {}
        except Exception as e:
            logger.error(f"Error getting model info for {model_name}: {e}")
            return {}