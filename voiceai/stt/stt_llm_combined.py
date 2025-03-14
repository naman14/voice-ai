from faster_whisper import WhisperModel
import torch
from .base_stt import BaseSTT
import numpy as np
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.inputs.data import TextPrompt
from typing import Union
import uuid
import time
from transformers import AutoTokenizer
import re
from vllm import LLM


class STTLLMCombined(BaseSTT):
    def __init__(self):
        self.device = None
        self.engine = None

    def setup(self) -> None:
        """Initialize the Whisper model"""
        print("Loading Whisper model VLLM...")
        self.model_name = "fixie-ai/ultravox-v0_5-llama-3_2-1b"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True
        )
        self.llm = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(
                model="fixie-ai/ultravox-v0_5-llama-3_2-1b",
                tensor_parallel_size=1,  # Adjust based on your setup
                gpu_memory_utilization=0.5,  # Optimize GPU usage
                dtype="float16",
                max_num_seqs=20,
                quantization="fp8",
                max_model_len=2048,
                trust_remote_code=True,
                enable_prefix_caching=True
            )
        )

    async def transcribe(self, audio_data: Union[bytes, np.ndarray], language: str) -> str:
        if self.llm is None:
            raise Exception("Whisper model not initialized")
            
        try:
            # If input is bytes, convert to numpy array
            if isinstance(audio_data, bytes):
                audio_data = np.frombuffer(audio_data, dtype=np.int16)
            
            # Convert to float32 and normalize
            audio_np = audio_data.astype(np.float32) / 32768.0

            audio_asset = (audio_np, 16000)

            messages = [
                {
                    "role": "system",
                    "content": "You are Donald Trump."
                },
                {
                    'role': 'user',
                    'content': "<|audio|>\n" * 1
                }]
            
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            sampling_params = SamplingParams(
                temperature=0,
                top_p=1.0,
                max_tokens=90,
            )

            start_time = time.time()
            first_token_time = None
            token_count = 0

    
            request_id = str(uuid.uuid4())

            text_prompt = TextPrompt(
                prompt=prompt,
                multi_modal_data={
                    "audio": audio_asset
                }
            )

            async for request_output in self.llm.generate(text_prompt, sampling_params, request_id=request_id):
                for output in request_output.outputs:
                    text = output.text

                    # Log first token time
                    if first_token_time is None:
                        first_token_time = time.time() - start_time
                        print(f"Time to first token: {first_token_time:.3f}s")

                    token_count += 1
                    response = text  # S
            
            duration = time.time() - start_time

            print(f"vllm whisper completed in {duration:.2f} seconds: {response}")

            return response  # Return only final response

        except Exception as e:
            raise Exception(f"Local Whisper transcription failed: {str(e)}") 