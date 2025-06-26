"""
DeepSeek Provider for ALEX
"""

import aiohttp
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class DeepSeekProvider:
    """Provider for DeepSeek API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1"
        self.session = None
        
    async def initialize(self):
        """Initialize the DeepSeek provider"""
        self.session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            timeout=aiohttp.ClientTimeout(total=60)
        )
        
        # Test the API key
        try:
            await self.health_check()
            logger.info("DeepSeek provider initialized successfully")
        except Exception as e:
            logger.error(f"DeepSeek initialization failed: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
    
    async def health_check(self) -> bool:
        """Check if DeepSeek API is accessible"""
        try:
            async with self.session.get(f"{self.base_url}/models") as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"DeepSeek health check failed: {e}")
            return False
    
    async def generate_response(self, messages: List[Dict[str, str]], params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response using DeepSeek API"""
        
        # DeepSeek uses OpenAI-compatible format
        formatted_messages = []
        system_prompt = params.get("system_prompt")
        
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})
        
        # Add non-system messages
        for msg in messages:
            if msg["role"] != "system":
                formatted_messages.append(msg)
        
        payload = {
            "model": params["model"],
            "messages": formatted_messages,
            "temperature": params.get("temperature", 0.7),
            "max_tokens": params.get("max_tokens", 2048),
            "stream": False
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload
            ) as response:
                
                if response.status != 200:
                    error_data = await response.json()
                    error_msg = error_data.get("error", {}).get("message", "Unknown error")
                    raise RuntimeError(f"DeepSeek API error ({response.status}): {error_msg}")
                
                data = await response.json()
                choice = data["choices"][0]
                usage = data.get("usage", {})
                
                return {
                    "content": choice["message"]["content"],
                    "tokens_used": usage.get("total_tokens", 0),
                    "finish_reason": choice.get("finish_reason", "stop")
                }
                
        except Exception as e:
            logger.error(f"DeepSeek generation error: {e}")
            raise RuntimeError(f"DeepSeek generation failed: {str(e)}")
    
    async def load_model(self, model_name: str) -> bool:
        """DeepSeek models don't need explicit loading"""
        return True
    
    async def unload_model(self, model_name: str) -> bool:
        """DeepSeek models don't need explicit unloading"""
        return True