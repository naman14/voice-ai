from ..base_llm import BaseLLM
import aiohttp
import os
from typing import Optional, AsyncGenerator
import json

class GroqLLM(BaseLLM):
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = None
        self.model = "llama3-70b-8192"  # Default model

    def setup(self):
        if self.is_setup:
            return
        
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
            
        self.is_setup = True

    async def generate(self, prompt: str, **kwargs) -> str:
        if not self.is_setup:
            raise RuntimeError("LLM not initialized")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json={
                    "model": self.model,
                    "messages": prompt,
                    "max_tokens": kwargs.get("max_tokens", 100),
                    "temperature": kwargs.get("temperature", 0.7),
                    "top_p": kwargs.get("top_p", 0.9),
                    "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
                    "presence_penalty": kwargs.get("presence_penalty", 0.6)
                }
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Groq API error: {error_text}")
                
                response_data = await response.json()
                return response_data["choices"][0]["message"]["content"]

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        if not self.is_setup:
            raise RuntimeError("LLM not initialized")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        final_response = ""

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json={
                    "model": self.model,
                    "messages": prompt,
                    "max_tokens": kwargs.get("max_tokens", 100),
                    "temperature": kwargs.get("temperature", 0.7),
                    "top_p": kwargs.get("top_p", 0.9),
                    "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
                    "presence_penalty": kwargs.get("presence_penalty", 0.6),
                    "stream": True  # Enable streaming
                }
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Groq API error: {error_text}")
                

                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line:
                        if line.startswith('data: '):
                            if line.startswith('data: [DONE]'):
                                break
                            try:
                                data = json.loads(line[6:])  # Remove 'data: ' prefix
                                if data["choices"][0]["finish_reason"] is not None:
                                    break
                                if "content" in data["choices"][0]["delta"]:
                                    chunk = data["choices"][0]["delta"]["content"]
                                    final_response += chunk
                                    yield final_response
                            except json.JSONDecodeError:
                                continue

    async def cleanup(self):
        self.is_setup = False
