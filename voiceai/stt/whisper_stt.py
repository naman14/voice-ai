from faster_whisper import WhisperModel
import torch
from .base_stt import BaseSTT
import numpy as np
import io
from typing import Union

class LocalWhisperSTT(BaseSTT):
    def __init__(self):
        self.device = None
        self.model = None

    def setup(self) -> None:
        """Initialize the Whisper model"""
        print("Loading Whisper model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = WhisperModel(
            "small",
            device="auto",
            compute_type="int8",
            download_root="./models"
        )

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
            segments, info = self.model.transcribe(
                audio_np,
                beam_size=5
            )
            
            transcribed_text = " ".join([segment.text for segment in segments])
            return transcribed_text

        except Exception as e:
            raise Exception(f"Local Whisper transcription failed: {str(e)}") 