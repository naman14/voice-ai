from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from fastapi.responses import JSONResponse
import os
import time
from typing import Dict, List, Optional
from fastapi.background import BackgroundTasks
from voiceai.config.agents_config import agent_manager
from typing import AsyncGenerator
from dotenv import load_dotenv
from .base_tts import BaseTTS
from .base_tts import TTSChunk
from fastapi import APIRouter
import base64

load_dotenv()

class TTS:
    def __init__(self):
        self.tts: Optional[BaseTTS] = None
        self._setup_tts()

    def _setup_tts(self):
        self.use_external = os.getenv("USE_EXTERNAL_TTS", "False").lower() == "true"
        self.provider = os.getenv("EXTERNAL_TTS_PROVIDER", "").lower()

        print(f"Setting up TTS: use_external: {self.use_external}")

        if self.use_external:
            if self.provider == "cartesia":
                from .external.cartesia_tts import CartesiaTTS
                self.tts = CartesiaTTS(api_key=os.getenv("CARTESIA_API_KEY"))
            elif self.provider == "elevenlabs":
                from .external.elevenlabs_tts import ElevenLabsTTS
                self.tts = ElevenLabsTTS(api_key=os.getenv("ELEVENLABS_API_KEY"))
            else:
                raise ValueError(f"Unsupported external TTS provider: {self.provider}")
        else:
            # from .local_tts_pool import LocalTTSPool
            from .local_tts import LocalTTS
            self.tts = LocalTTS()

        self.setup()

        print(f"TTS setup complete")

    def setup(self):
        """Initialize the TTS"""
        if self.tts:
            self.tts.setup()

    async def generate_speech(self, text: str, language: str, voice_id: Optional[str] = None, voice_samples: Optional[str] = None, speed: float = 1.0) -> TTSChunk:
        """Generate speech for the given prompt"""
        if not self.tts:
            raise RuntimeError("TTS not initialized")
        
        return await self.tts.generate_speech(text, language, voice_id, voice_samples, speed)

    async def generate_speech_stream(self, text: str, language: str, voice_id: Optional[str] = None, voice_samples: Optional[str] = None, speed: float = 1.0) -> AsyncGenerator[TTSChunk, None]:
        if not self.tts:
            raise RuntimeError("TTS not initialized")
        
        return self.tts.generate_speech_stream(text, language, voice_id, voice_samples, speed)
                

    async def cleanup(self):
        """Cleanup resources"""
        if self.tts:
            await self.tts.cleanup()

    def get_voice_id(self, config_id: str) -> Optional[str]:
        config = agent_manager.get_agent_config(config_id)
        if not config:
            raise HTTPException(status_code=404, detail="Config not found")

        return config.cartesia_voice_id if self.provider == "cartesia" else config.elevenlabs_voice_id

# Initialize tts instance
tts_instance = TTS()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def stream_audio_chunks(websocket: WebSocket, text: str, config_id: str, speed: float = 1.0):
    """TTS streaming implementation with voice conversion and raw PCM output"""
    try:
        await websocket.send_json({
            "type": "stream_start",
            "timestamp": time.time()
        })

        config = agent_manager.get_agent_config(config_id)
        if not config:
            raise HTTPException(status_code=404, detail="Config not found")

        voice_samples = config.voice_samples
        language = config.language
        
        voice_id = tts_instance.get_voice_id(config_id)
        
        async for chunk in await tts_instance.generate_speech_stream(text, language, voice_id, voice_samples, speed):
            base64_chunk = base64.b64encode(chunk.chunk).decode("utf-8")
            await websocket.send_json({
                "type": "audio_chunk",
                "chunk": base64_chunk,
                "format": chunk.format,
                "sample_rate": chunk.sample_rate,
                "timestamp": time.time()
            })

        await websocket.send_json({
            "type": "stream_end",
            "timestamp": time.time()
        })

    except WebSocketDisconnect:
        print("WebSocket disconnected during streaming")
    except Exception as e:
        print(f"Error in stream_audio_chunks: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "error": str(e)
            })
        except:
            pass

@app.websocket("/tts/stream")
async def tts_stream(websocket: WebSocket):
   
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if not isinstance(data, dict):
                continue
                
            text = data.get("text")
            config_id = data.get("config_id", "")
            session_id = data.get("session_id", "")
            speed = data.get("speed", 1.0)
            
            if text:
                await stream_audio_chunks(websocket, text, config_id, speed)
            
    except WebSocketDisconnect:
        print("WebSocket connection closed normally")
    except Exception as e:
        print(f"WebSocket error: {e}")

tts_router = APIRouter()

@tts_router.post("/tts")
async def generate_tts(
    data: dict
):        
    try:
        text = data.get("text", "")

        # send either config_id or config
        # if sending config_id, then make sure config has been added to the agent manager before calling this
        config_id = data.get("config_id", "")
        config = data.get("config", {})

        if config:
             await agent_manager.add_agent_config(config)
        
        # get latest config from agent manager
        config = agent_manager.get_agent_config(config_id)

        if not config:
            raise HTTPException(status_code=404, detail="Config not found")

        voice_samples = config.voice_samples
        language = config.language
        voice_id = tts_instance.get_voice_id(config_id)
        speed = config.speed

        tts_chunk = await tts_instance.generate_speech(text, language, voice_id, voice_samples, speed)

        return JSONResponse({
            "audio": base64.b64encode(tts_chunk.chunk).decode("utf-8"),
            "format": tts_chunk.format,
            "sample_rate": tts_chunk.sample_rate
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
