from typing import Generator, Optional, AsyncGenerator
import requests
import aiohttp
from ..base_tts import BaseTTS
import base64
from ..base_tts import TTSChunk

class ElevenLabsTTS(BaseTTS):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.elevenlabs.io/v1"
        self.chunk_size = 1024
        
    def setup(self):
        """Initialize the TTS system"""
        pass  # No setup needed for ElevenLabs
        
    async def cleanup(self):
        """Clean up resources"""
        pass  # No cleanup needed for ElevenLabs
        
    async def generate_speech(self, text: str, language: str, voice_id: Optional[str] = None, voice_samples: Optional[str] = None, speed: float = 1.0) -> TTSChunk:
        """Generate speech using ElevenLabs API"""
        url = f"{self.base_url}/text-to-speech/{voice_id}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }
        data = {
            "text": text,
            "model_id": "eleven_flash_v2",
            "output_format": "pcm_32",
            "sample_rate": 24000
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"ElevenLabs API error: {error_text}")
                
                audio = await response.read()
                
                return TTSChunk(audio, "pcm", 24000)
            
        
    async def generate_speech_stream(self, text: str, language: str, voice_id: Optional[str] = None, voice_samples: Optional[str] = None, speed: float = 1.0) -> AsyncGenerator[TTSChunk, None]:
        """Generate speech in streaming mode using ElevenLabs API"""
        url = f"{self.base_url}/text-to-speech/{voice_id}/stream?output_format=pcm_24000"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }
        data = {
            "text": text,
            "model_id": "eleven_flash_v2"
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"ElevenLabs API error: {error_text}")
                
                async for chunk in response.content.iter_chunked(self.chunk_size):
                    yield TTSChunk(
                        chunk=chunk,
                        sample_rate=24000,
                        format="pcm_f32le"
                    ) 