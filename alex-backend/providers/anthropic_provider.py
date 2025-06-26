"""
Anthropic Provider for ALEX
"""

import aiohttp
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class AnthropicProvider:
    """Provider for Anthropic Claude API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.anthropic.com/v1"
        self.session = None
        
    async def initialize(self):
        """Initialize the Anthropic provider"""
        self.session = aiohttp.ClientSession(
            headers={
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            },
            timeout=aiohttp.ClientTimeout(total=60)
        )
        
        # Test the API key
        try:
            await self.health_check()
            logger.info("Anthropic provider initialized successfully")
        except Exception as e:
            logger.error(f"Anthropic initialization failed: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
    
    async def health_check(self) -> bool:
        """Check if Anthropic API is accessible"""
        try:
            # Anthropic doesn't have a simple health endpoint, so we'll do a minimal completion
            test_payload = {
                "model": "claude-3-haiku-20240307",
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "Hi"}]
            }
            
            async with self.session.post(
                f"{self.base_url}/messages",
                json=test_payload
            ) as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"Anthropic health check failed: {e}")
            return False
    
    async def generate_response(self, messages: List[Dict[str, str]], params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response using Anthropic API"""
        
        # Anthropic expects messages without system role mixed in
        formatted_messages = []
        system_prompt = params.get("system_prompt")
        
        # Filter out system messages and format for Anthropic
        for msg in messages:
            if msg["role"] != "system":
                formatted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        payload = {
            "model": params["model"],
            "messages": formatted_messages,
            "max_tokens": params.get("max_tokens", 2048),
            "temperature": params.get("temperature", 0.7),
        }
        
        # Add system prompt if provided
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            async with self.session.post(
                f"{self.base_url}/messages",
                json=payload
            ) as response:
                
                if response.status != 200:
                    error_data = await response.json()
                    error_msg = error_data.get("error", {}).get("message", "Unknown error")
                    raise RuntimeError(f"Anthropic API error ({response.status}): {error_msg}")
                
                data = await response.json()
                usage = data.get("usage", {})
                
                # Extract content from response
                content = ""
                if data.get("content") and len(data["content"]) > 0:
                    content = data["content"][0].get("text", "")
                
                return {
                    "content": content,
                    "tokens_used": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                    "finish_reason": data.get("stop_reason", "stop")
                }
                
        except Exception as e:
            logger.error(f"Anthropic generation error: {e}")
            raise RuntimeError(f"Anthropic generation failed: {str(e)}")
    
    async def load_model(self, model_name: str) -> bool:
        """Anthropic models don't need explicit loading"""
        return True
    
    async def unload_model(self, model_name: str) -> bool:
        """Anthropic models don't need explicit unloading"""
        return True