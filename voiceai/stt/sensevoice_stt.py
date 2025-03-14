from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import torch
from .base_stt import BaseSTT
import numpy as np
import io
from typing import Union

class SenseVoiceSTT(BaseSTT):
    def __init__(self):
        self.device = None
        self.model = None

    def setup(self) -> None:
        """Initialize the SenseVoice model"""
        print("Loading SenseVoice model...")
        model_dir = "iic/SenseVoiceSmall"

        self.model = AutoModel(
            model=model_dir,
            trust_remote_code=True,
            remote_code="./model.py",    
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device="cuda:0",
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
            
            res = self.model.generate(
                input=audio_np,
                cache={},
                language="auto", # "zh", "en", "yue", "ja", "ko", "nospeech"
                use_itn=False,
                batch_size=4, 
            )
            text = rich_transcription_postprocess(res[0]["text"])
            print(text)
            return text

        except Exception as e:
            raise Exception(f"Local Whisper transcription failed: {str(e)}") 