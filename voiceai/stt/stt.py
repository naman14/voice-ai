from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import time
from voiceai.config.agents_config import agent_manager
import os
from .base_stt import BaseSTT
from typing import Optional
import torch
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class STT:
    def __init__(self):
        self.stt: Optional[BaseSTT] = None
        self._setup_stt()

    def _setup_stt(self):
        # from .whisper_stt import LocalWhisperSTT
        # from .whisper_vllm import WhisperVLLM
        # from .stt_llm_combined import STTLLMCombined
        # from .whisper_hf import WhisperHF
        # from .sensevoice_stt import SenseVoiceSTT
        
        from .whisper_jax import WhisperJax
        self.stt = WhisperJax()
        self.setup()
        

    def setup(self):
        """Initialize the STT"""
        if self.stt:
            self.stt.setup()
    
    async def transcribe(self, audio_data: bytes, language: str) -> str:
        """
        Transcribe audio data to text using the configured STT provider
        
        Args:
            audio_data (bytes): Raw audio data to transcribe
            
        Returns:
            str: Transcribed text
        """
        start_time = time.time()

        transcribed_text = await self.stt.transcribe(audio_data, language)

        end_time = time.time()
        print(f"Transcription completed in {end_time - start_time:.2f} seconds: {transcribed_text}")

        return transcribed_text

stt_instance = STT()

@app.post("/transcribe")
async def transcribe_audio(
    audio_file: UploadFile = File(..., description="The audio file to transcribe"),
    config_id: str = Form(..., description="The configuration ID for the agent")
):
    try:
        # Validate config_id and get agent configuration
        if not config_id:
            return {"error": "config_id is required"}
            
        config = agent_manager.get_agent_config(config_id)
        if not config:
            return {"error": f"No configuration found for config_id: {config_id}"}
            
        
        content = await audio_file.read()
        
        transcribed_text = await stt_instance.transcribe(content, config.language)
        
        return {
            "text": transcribed_text,
        }
        
    except Exception as e:
        import traceback
        print(f"Error in transcribe_audio: {str(e)}")
        print(traceback.format_exc())
        return {"error": str(e)}