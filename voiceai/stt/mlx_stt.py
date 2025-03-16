from lightning_whisper_mlx import LightningWhisperMLX
import torch
from .base_stt import BaseSTT
import numpy as np
import io
from typing import Union

class MLXSTT(BaseSTT):
    def __init__(self):
        self.device = None
        self.model = None

    def setup(self) -> None:
        """Initialize the Whisper model"""
        print("Loading Whisper model...")
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built() else "cpu"
        self.model = LightningWhisperMLX(model="distil-small.en", batch_size=4, quant=None)


    async def transcribe(self, audio_data: Union[bytes, np.ndarray], language: str) -> str:
        if self.model is None:
            raise Exception("Whisper model not initialized")
            
        try:
            # If input is bytes, convert to numpy array
            if isinstance(audio_data, bytes):
                audio_data = np.frombuffer(audio_data, dtype=np.int16)
            
            # Convert to float32 and normalize
            audio_np = audio_data.astype(np.float32) / 32768.0
            
            # Transcribe using local model
            result = self.model.transcribe(
                audio_np,
            )
            
            transcribed_text = result["text"]
            return transcribed_text

        except Exception as e:
            raise Exception(f"Local Whisper transcription failed: {str(e)}") 