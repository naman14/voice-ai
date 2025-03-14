from faster_whisper import WhisperModel
import torch
from .base_stt import BaseSTT
import numpy as np
import io
from typing import Union
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


class WhisperHF(BaseSTT):
    def __init__(self):
        self.device = None
        self.model = None

    def setup(self) -> None:
        """Initialize the Whisper model"""
        print("Loading Whisper model...")
    
        model_id = "openai/whisper-small"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True,
            attn_implementation="sdpa"
        )
        self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=128,
            torch_dtype=torch_dtype,
            device="cuda",
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
            
            # forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=language, task="transcribe")
            # print(f"Transcribing audio with language: {language}")
            
            # result = self.pipe(audio_np, generate_kwargs = {"forced_decoder_ids":forced_decoder_ids})

            result = self.pipe(audio_np)
            transcribed_text = result["text"]
            return transcribed_text

        except Exception as e:
            raise Exception(f"Local Whisper transcription failed: {str(e)}") 