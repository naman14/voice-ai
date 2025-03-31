import io
import torch
import numpy as np
import os
from typing import Generator, Optional, Dict, AsyncGenerator
from TTS.api import TTS
from TTS.utils.manage import ModelManager
from TTS.tts.models.xtts import Xtts
from TTS.tts.configs.xtts_config import XttsConfig, XttsAudioConfig, XttsArgs
import asyncio
from .base_tts import BaseTTS
from .base_tts import TTSChunk
import time
import base64
import torchaudio
import threading

class LocalTTS(BaseTTS):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built() else "cpu"
        self.model = None
        self.tts_lock = threading.Lock()
        self.speaker_latents_cache: Dict[str, tuple] = {}
        
    def setup(self):
        """Initialize the TTS system"""
       
       # check if custom xtts model
        xttsPath = os.getenv('XTTS_MODEL_PATH')
        print("XTTS_MODEL_PATH: " + xttsPath)
        if xttsPath is None or not os.path.exists(xttsPath):
            # else download default xtts model
            print("Downloading default XTTS model...")
            manager = ModelManager(models_file=TTS.get_models_file_path(), progress_bar=True)
            model_path, config_path, _ = manager.download_model(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
            print("Downloaded default XTTS model to: " + str(model_path))
            xttsPath = str(model_path)
        print("Loading XTTS model...")
        config = XttsConfig(
            model_args=XttsArgs(
                input_sample_rate=24000,
            ),
            audio=XttsAudioConfig(
                sample_rate=24000,
                output_sample_rate=24000
            )
        )
        config.load_json(xttsPath + "/config.json")
        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(config, checkpoint_dir=xttsPath, use_deepspeed=self.device == "cuda")
        if self.device == "cuda":
            self.model.cuda()
                
    async def cleanup(self):
        """Clean up resources"""
        self.speaker_latents_cache.clear()
        
    async def get_speaker_latents(self, voice_samples: str) -> tuple:
        """Get or compute speaker latents"""
        # Convert list to tuple if voice_samples is a list
        cache_key = voice_samples if isinstance(voice_samples, str) else tuple(voice_samples)
        
        if cache_key not in self.speaker_latents_cache:
            print("Computing speaker latents...")
            gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(audio_path=voice_samples)
            self.speaker_latents_cache[cache_key] = (gpt_cond_latent, speaker_embedding)
        else:
            gpt_cond_latent, speaker_embedding = self.speaker_latents_cache[cache_key]
        return gpt_cond_latent, speaker_embedding
        
    
    async def generate_speech(self, text: str, language: str, voice_id: Optional[str] = None, voice_samples: Optional[str] = None, speed: float = 1.0) -> TTSChunk:
        """Generate speech using local TTS model"""

        gpt_cond_latent, speaker_embedding = await self.get_speaker_latents(voice_samples)

        print("Starting full audio generation...")
        t0 = time.time()
        
        with self.tts_lock:
            # Generate the complete audio
            audio = self.model.inference(
                text,
                language,
                gpt_cond_latent,
                speaker_embedding,
                temperature=0.7,
                speed=speed
            )

            # Convert audio tensor to WAV format in memory
            buffer = io.BytesIO()
            torchaudio.save(buffer, torch.tensor(audio["wav"]).unsqueeze(0), 24000, format="wav")
            buffer.seek(0)
            
            audio_bytes = buffer.read()
            
            print(f"Audio generation completed in {time.time() - t0:.2f} seconds")
            
            return TTSChunk(audio_bytes, "pcm", 24000)
        
    async def generate_speech_stream(self, text: str, language: str, voice_id: Optional[str] = None, voice_samples: Optional[str] = None, speed: float = 1.0) -> AsyncGenerator[TTSChunk, None]:
        """Generate speech in streaming mode using local TTS model"""
        gpt_cond_latent, speaker_embedding = await self.get_speaker_latents(voice_samples)

        print("Starting streaming inference, language: " + language + ", speed: " + str(speed))
        t0 = time.time()

        chunk_counter = 0
        
        with self.tts_lock:
            for chunk in self.model.inference_stream(
                text,
                language,
                gpt_cond_latent,
                speaker_embedding,
                temperature=0.7,
                enable_text_splitting=True,
                speed=speed
            ):
                if chunk_counter == 0:
                    print(f"Time to first chunk: {time.time() - t0}")
                
                # Convert tensor to raw PCM bytes
                chunk_bytes = chunk.squeeze().cpu().numpy().tobytes()

                # print('sending chunk: ' + str(chunk_counter))
                yield TTSChunk(
                    chunk=chunk_bytes,
                    sample_rate=24000,
                    format="pcm"
                )
                chunk_counter += 1 