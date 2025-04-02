import io
import torch
import numpy as np
import os
from typing import Generator, Optional, Dict, AsyncGenerator, List, Tuple
from TTS.api import TTS
from TTS.tts.models.xtts import Xtts
from TTS.tts.configs.xtts_config import XttsConfig, XttsAudioConfig, XttsArgs
import asyncio
from .base_tts import BaseTTS
from .base_tts import TTSChunk
import time
import base64
import torchaudio
import threading

class XTTSPool(BaseTTS):
    def __init__(self, max_gpu_utilization: float = 0.2):
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built() else "cpu"
        self.models: List[Tuple[Xtts, threading.Lock, int]] = []
        self.speaker_latents_cache: Dict[str, tuple] = {}
        self.max_gpu_utilization = max_gpu_utilization
        self.current_model_index = 0
        self.pool_lock = threading.Lock()
    
    def setup(self):
        """Initialize the TTS system"""
        self.setup_model_pool()
    
    def setup_model_pool(self):
        """Initialize the TTS model pool based on GPU memory constraints."""
        if self.device == "cuda":
            free_memory, total_memory = torch.cuda.mem_get_info()
            max_memory_usage = total_memory * self.max_gpu_utilization
            model_memory_estimate = self.estimate_model_memory()
            max_models = int(max_memory_usage // model_memory_estimate)
            print("Max memory usage: " + str(max_memory_usage))
            print("Max TTS models: " + str(max_models))

            for index in range(max_models):
                model = self.load_model()
                model_lock = threading.Lock()
                self.models.append((model, model_lock, index))
        else:
            # For CPU, load a single model
            model = self.load_model()
            model_lock = threading.Lock()
            self.models.append((model, model_lock, 0))

    def estimate_model_memory(self) -> float:
        """Estimate the memory usage of a single model in bytes."""
        return 1_800_000_000  # Example: 1.8 GB per model

    def load_model(self):
        """Load a single TTS model instance."""
        print("Loading TTS model...")
        xttsPath = os.getenv('XTTS_MODEL_PATH')
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
        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, checkpoint_dir=xttsPath, use_deepspeed=self.device == "cuda")
        if self.device == "cuda":
            model.cuda()
        return model

    async def get_model(self) -> Tuple[Xtts, threading.Lock, int]:
        """Get a model, its lock, and index from the pool using a round-robin strategy."""
        with self.pool_lock:
            model, model_lock, model_index = self.models[self.current_model_index]
            self.current_model_index = (self.current_model_index + 1) % len(self.models)
        print("Returning TTS model index: " + str(model_index))
        return model, model_lock, model_index

    async def cleanup(self):
        """Clean up resources"""
        self.speaker_latents_cache.clear()
        
    async def get_speaker_latents(self, voice_samples: str, model: Xtts, model_lock: threading.Lock) -> tuple:
        """Get or compute speaker latents using a specific model."""
        # Convert list to tuple if voice_samples is a list
        cache_key = voice_samples if isinstance(voice_samples, str) else tuple(voice_samples)
        
        if cache_key not in self.speaker_latents_cache:
            print("Computing speaker latents...")
            gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=voice_samples)
            self.speaker_latents_cache[cache_key] = (gpt_cond_latent, speaker_embedding)
        else:
            gpt_cond_latent, speaker_embedding = self.speaker_latents_cache[cache_key]
        return gpt_cond_latent, speaker_embedding
    
    async def generate_speech(self, text: str, language: str, voice_id: Optional[str] = None, voice_samples: Optional[str] = None, speed: float = 1.0) -> TTSChunk:
        """Generate speech using local TTS model"""
        model, model_lock, _ = await self.get_model()
        
        with model_lock:
            try:
                gpt_cond_latent, speaker_embedding = await self.get_speaker_latents(voice_samples, model, model_lock)

                print("Starting full audio generation...")
                t0 = time.time()
                
                # Generate the complete audio
                audio = model.inference(
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
            except Exception as e:
                print(f"Error during speech generation: {e}")
        
    async def generate_speech_stream(self, text: str, language: str, voice_id: Optional[str] = None, voice_samples: Optional[str] = None, speed: float = 1.0) -> AsyncGenerator[TTSChunk, None]:
        """Generate speech in streaming mode using local TTS model"""
        model, model_lock, model_index = await self.get_model()
        
        with model_lock:
            try:
                gpt_cond_latent, speaker_embedding = await self.get_speaker_latents(voice_samples, model, model_lock)

                print("Starting streaming inference with model index: " + str(model_index))
                t0 = time.time()

                chunk_counter = 0
                for chunk in model.inference_stream(
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

                    print('sending chunk: ' + str(chunk_counter) + ' with model index: ' + str(model_index))
                    yield TTSChunk(
                        chunk=chunk_bytes,
                        sample_rate=24000,
                        format="pcm"
                    )
                    chunk_counter += 1
            except Exception as e:
                print(f"Error during streaming speech generation: {e}") 