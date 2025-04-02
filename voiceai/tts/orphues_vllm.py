import io
import torch
import numpy as np
import os
from typing import Generator, Optional, Dict, AsyncGenerator
from .base_tts import BaseTTS
from .base_tts import TTSChunk
import time
import base64
import torchaudio
import threading
import asyncio
import queue
from transformers import AutoTokenizer
from vllm import AsyncLLMEngine, AsyncEngineArgs
from vllm.sampling_params import SamplingParams
import uuid

# pip install snac
from snac import SNAC

# we use vLLM backend for orphues, 
# similar to https://github.com/canopyai/Orpheus-TTS/blob/main/orpheus_tts_pypi/orpheus_tts/engine_class.py

class OrpheusTTS(BaseTTS):
    def __init__(self):
        self.model_name = "canopylabs/orpheus-tts-0.1-finetune-prod"
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built() else "cpu"
        self.tts = None
        self.engine = None
        self.is_setup = False
        self.is_async = True
        self.gpu_memory_utilization = 0.3
        self.tokens_decoder = OrphuesTokensDecoder()
        
    def setup(self):
        """Initialize the TTS system"""
        if (self.is_setup):
            return
       
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True
        )
    
        self.engine = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(
                model=self.model_name,
                tensor_parallel_size=1, 
                gpu_memory_utilization=self.gpu_memory_utilization,
                dtype="float16",
                max_model_len=1024,
                max_num_seqs=20,
                quantization="fp8",
                enable_prefix_caching=True
            )
        )

        self.is_setup = True
                
    async def cleanup(self):
        """Clean up resources"""
        if self.tts:
            del self.tts
        if self.engine:
            del self.engine
        if self.tokenizer:
            del self.tokenizer
        self.is_setup = False
    
    async def generate_speech(self, text: str, language: str, voice_id: Optional[str] = None, voice_samples: Optional[str] = None, speed: float = 1.0) -> TTSChunk:
        """Generate speech using local TTS model"""
        # Collect all audio chunks from the stream
        audio_chunks = []
        async for chunk in self.generate_speech_stream(text, language, voice_id, voice_samples, speed):
            if chunk and chunk.audio:
                audio_chunks.append(chunk.audio)
        
        # Combine all audio chunks
        if not audio_chunks:
            raise RuntimeError("No audio was generated")
        
        combined_audio = b''.join(audio_chunks)
        
        return TTSChunk(combined_audio, "pcm", 24000)
        
    async def generate_speech_stream(self, text: str, language: str, voice_id: Optional[str] = None, voice_samples: Optional[str] = None, speed: float = 1.0) -> AsyncGenerator[TTSChunk, None]:
        """Generate speech in streaming mode using local TTS model"""
        if not self.is_setup:
            raise RuntimeError("LLM not initialized")
        
        if not self.is_async:
            raise RuntimeError("Async generation is not enabled. Please set is_async to True.")

        start_time = time.time()
        first_token_time = None
        token_count = 0

        # vLLM sampling parameters
        stop_token_ids = [49158]
        sampling_params = SamplingParams(
            max_tokens=1200,
            temperature=0.6,
            top_p=0.8,
            top_k=50,
            repetition_penalty=1.2,
            stop_token_ids=stop_token_ids
        )

        request_id = str(uuid.uuid4())
        prompt_string = self._format_prompt(text, voice_id)
        
        # Reset the token decoder for a new generation
        self.tokens_decoder.reset()

        async for request_output in self.engine.generate(prompt_string, sampling_params, request_id=request_id):
            for output in request_output.outputs:
                text = output.text

                # Log first token time
                if first_token_time is None:
                    first_token_time = time.time() - start_time
                    print(f"Time to first token: {first_token_time:.3f}s")

                # Process the token and get audio if available
                chunk = self.tokens_decoder.decode(text)
                if chunk:
                    yield chunk
                
                token_count += 1

        # Log performance
        total_time = time.time() - start_time
        tokens_per_second = token_count / total_time if total_time > 0 else 0
        print(f"\nTotal generation time: {total_time:.3f}s")
        print(f"Tokens per second: {tokens_per_second:.2f}")
    
    def _format_prompt(self, prompt, voice="tara", model_type="larger"):
        if model_type == "smaller":
            if voice:
                return f"<custom_token_3>{prompt}[{voice}]<custom_token_4><custom_token_5>"
            else:
                return f"<custom_token_3>{prompt}<custom_token_4><custom_token_5>"
        else:
            if voice:
                adapted_prompt = f"{voice}: {prompt}"
                prompt_tokens = self.tokenizer(adapted_prompt, return_tensors="pt")
                start_token = torch.tensor([[ 128259]], dtype=torch.int64)
                end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
                all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
                prompt_string = self.tokenizer.decode(all_input_ids[0])
                return prompt_string
            else:
                prompt_tokens = self.tokenizer(prompt, return_tensors="pt")
                start_token = torch.tensor([[ 128259]], dtype=torch.int64)
                end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
                all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
                prompt_string = self.tokenizer.decode(all_input_ids[0])
                return prompt_string

# based on https://github.com/canopyai/Orpheus-TTS/blob/main/orpheus_tts_pypi/orpheus_tts/decoder.py
class OrphuesTokensDecoder:
    def __init__(self):
        print("Initializing OrphuesTokensDecoder")
        self.model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
        self.snac_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built() else "cpu"
        self.model = self.model.to(self.snac_device)
        self.buffer = []
        self.count = 0
    
    def reset(self):
        """Reset the decoder state for a new generation"""
        self.buffer = []
        self.count = 0

    def decode(self, token_string):
        """Process a token string and return audio if available"""
        token = self._turn_token_into_id(token_string, self.count)
        
        if token is not None and token > 0:
            self.buffer.append(token)
            self.count += 1

            if self.count % 7 == 0 and self.count > 27:
                buffer_to_proc = self.buffer[-28:]
                audio_bytes = self._convert_to_audio(buffer_to_proc, self.count)
                if audio_bytes is not None:
                    # Create a TTSChunk with the audio data
                    return TTSChunk(audio_bytes, "pcm", 24000)
        
        return None  # Return None if no audio is ready yet
    
    def _convert_to_audio(self, multiframe, count):
        frames = []
        if len(multiframe) < 7:
            return None
        
        codes_0 = torch.tensor([], device=self.snac_device, dtype=torch.int32)
        codes_1 = torch.tensor([], device=self.snac_device, dtype=torch.int32)
        codes_2 = torch.tensor([], device=self.snac_device, dtype=torch.int32)

        num_frames = len(multiframe) // 7
        frame = multiframe[:num_frames*7]

        for j in range(num_frames):
            i = 7*j
            if codes_0.shape[0] == 0:
                codes_0 = torch.tensor([frame[i]], device=self.snac_device, dtype=torch.int32)
            else:
                codes_0 = torch.cat([codes_0, torch.tensor([frame[i]], device=self.snac_device, dtype=torch.int32)])

            if codes_1.shape[0] == 0:
                codes_1 = torch.tensor([frame[i+1]], device=self.snac_device, dtype=torch.int32)
                codes_1 = torch.cat([codes_1, torch.tensor([frame[i+4]], device=self.snac_device, dtype=torch.int32)])
            else:
                codes_1 = torch.cat([codes_1, torch.tensor([frame[i+1]], device=self.snac_device, dtype=torch.int32)])
                codes_1 = torch.cat([codes_1, torch.tensor([frame[i+4]], device=self.snac_device, dtype=torch.int32)])
            
            if codes_2.shape[0] == 0:
                codes_2 = torch.tensor([frame[i+2]], device=self.snac_device, dtype=torch.int32)
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+3]], device=self.snac_device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+5]], device=self.snac_device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+6]], device=self.snac_device, dtype=torch.int32)])
            else:
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+2]], device=self.snac_device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+3]], device=self.snac_device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+5]], device=self.snac_device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+6]], device=self.snac_device, dtype=torch.int32)])

        codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]
        # check that all tokens are between 0 and 4096 otherwise return None
        if torch.any(codes[0] < 0) or torch.any(codes[0] > 4096) or torch.any(codes[1] < 0) or torch.any(codes[1] > 4096) or torch.any(codes[2] < 0) or torch.any(codes[2] > 4096):
            return None

        with torch.inference_mode():
            audio_hat = self.model.decode(codes)
        
        audio_slice = audio_hat[:, :, 2048:4096]
        detached_audio = audio_slice.detach().cpu()
        audio_np = detached_audio.numpy()
        audio_int16 = (audio_np * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        return audio_bytes

    def _turn_token_into_id(self, token_string, index):
        # Strip whitespace
        token_string = token_string.strip()
        
        # Find the last token in the string
        last_token_start = token_string.rfind("<custom_token_")
        
        if last_token_start == -1:
            # No token found in the string
            return None
        
        # Extract the last token
        last_token = token_string[last_token_start:]
        
        # Process the last token
        if last_token.startswith("<custom_token_") and last_token.endswith(">"):
            try:
                number_str = last_token[14:-1]
                return int(number_str) - 10 - ((index % 7) * 4096)
            except ValueError:
                return None
        else:
            return None
