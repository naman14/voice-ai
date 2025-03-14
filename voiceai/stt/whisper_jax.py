from faster_whisper import WhisperModel
import torch
from .base_stt import BaseSTT
import numpy as np
import io
from typing import Union
from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp
import jax

class WhisperJax(BaseSTT):
    def __init__(self):
        self.device = None
        self.model = None

    def setup(self) -> None:
        """Initialize the Whisper model"""
        print("Loading Whisper model...")
        self.pipeline = FlaxWhisperPipline("openai/whisper-small.en",  
            dtype=jnp.bfloat16,
            max_length=70)
        # jit the pipeline with empty audio bytes
        print("Pipeline JIT..")
        self.pipeline({
                "array": np.zeros(1),
                "sampling_rate": 16000
            })
        print("Pipeline setup complete")
        self.pipelineHi = FlaxWhisperPipline("sanchit-gandhi/whisper-small-hi-flax",  
            dtype=jnp.bfloat16,
            max_length=70)
        # jit the pipeline with empty audio bytes
        print("Pipeline JIT..")
        self.pipelineHi({
                "array": np.zeros(1),
                "sampling_rate": 16000
            })
        print("Pipeline setup complete")
        
    async def transcribe(self, audio_data: Union[bytes, np.ndarray], language: str) -> str:
        if self.pipeline is None:
            raise Exception("Whisper model not initialized")
            
        try:
            # If input is bytes, convert to numpy array
            if isinstance(audio_data, bytes):
                audio_data = np.frombuffer(audio_data, dtype=np.int16)
            
            # Convert to float32 and normalize
            audio_np = audio_data.astype(np.float32) / 32768.0
            
            if language == "hi":
                text = self.pipelineHi({
                    "array": audio_np,
                    "sampling_rate": 16000,
                })
            else:
                text = self.pipeline({
                    "array": audio_np,
                    "sampling_rate": 16000,
                })
            print("Text: ", text)
            return text["text"]

        except Exception as e:
            raise Exception(f"Local Whisper transcription failed: {str(e)}") 