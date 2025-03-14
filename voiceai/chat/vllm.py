from .base_llm import BaseLLM
import asyncio
import os
import time
import re
from typing import Optional, AsyncGenerator
import torch
from vllm import LLM, SamplingParams, AsyncLLMEngine, AsyncEngineArgs
from transformers import AutoTokenizer
import uuid


class VLLM(BaseLLM):
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B-Instruct"):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.llm = None
        self.engine = None
        self.is_setup = False
        self.is_async = False
        self.gpu_memory_utilization = 0.5

    def setup(self):
        if self.is_setup:
            return

        # Hugging Face login
        from huggingface_hub import login
        login(token=os.getenv('HUGGINGFACE_API_KEY'))

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True
        )

        if self.is_async:
            self.engine = AsyncLLMEngine.from_engine_args(
                AsyncEngineArgs(
                    model=self.model_name,
                    tensor_parallel_size=1,  # Adjust based on your setup
                    gpu_memory_utilization=self.gpu_memory_utilization,  # Optimize GPU usage
                    dtype="float16",
                    max_model_len=1024,
                    max_num_seqs=20,
                    quantization="fp8",
                    enable_prefix_caching=True
                )
            )
        else:
            # Initialize vLLM engine
            self.engine = LLM(
                model=self.model_name,
                tensor_parallel_size=1,  # Adjust based on your setup
                gpu_memory_utilization=self.gpu_memory_utilization,  # Optimize GPU usage
                dtype="float16",
                max_model_len=1024,
                max_num_seqs=20,
                quantization="fp8",
                enable_prefix_caching=True
            )

        self.is_setup = True

    async def generate(self, prompt: str, **kwargs) -> str:
        if not self.is_setup:
            raise RuntimeError("LLM not initialized")

        start_time = time.time()
        formatted_chat = self.tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )

        sampling_params = SamplingParams(
            max_tokens=kwargs.get("max_new_tokens", 120),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            top_k=kwargs.get("top_k", 50),
            repetition_penalty=kwargs.get("repetition_penalty", 1.2),
            stop=["User:", "USER:", "Human:", "HUMAN:"],  # Custom stopping criteria
        )

        response = self.engine.generate(formatted_chat, sampling_params)


        generated_text = response[0].outputs[0].text
        token_count = len(response[0].outputs[0].token_ids)

        total_time = time.time() - start_time

        tokens_per_second = token_count / total_time if total_time > 0 else 0
        print(f"\nTotal generation time: {total_time:.3f}s")
        print(f"Tokens per second: {tokens_per_second:.2f}")
           
        return generated_text

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        if not self.is_setup:
            raise RuntimeError("LLM not initialized")
        
        if not self.is_async:
            raise RuntimeError("Async generation is not enabled. Please set is_async to True.")

        start_time = time.time()
        first_token_time = None
        token_count = 0
        response = ""

        # Format prompt using tokenizer
        formatted_chat = self.tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )

        # vLLM sampling parameters
        sampling_params = SamplingParams(
            max_tokens=kwargs.get("max_new_tokens", 120),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            top_k=kwargs.get("top_k", 50),
            repetition_penalty=kwargs.get("repetition_penalty", 1.2),
            stop=["User:", "USER:", "Human:", "HUMAN:"],  # Custom stopping criteria
        )

        request_id = str(uuid.uuid4())
        async for request_output in self.engine.generate(formatted_chat, sampling_params, request_id=request_id):
            for output in request_output.outputs:
                text = output.text

                # Log first token time
                if first_token_time is None:
                    first_token_time = time.time() - start_time
                    print(f"Time to first token: {first_token_time:.3f}s")

                yield text
                
                token_count += 1
                response = text  # Store final response            

        # Log performance
        total_time = time.time() - start_time
        tokens_per_second = token_count / total_time if total_time > 0 else 0
        print(f"\nTotal generation time: {total_time:.3f}s")
        print(f"Tokens per second: {tokens_per_second:.2f}")


    async def cleanup(self):
        if self.llm:
            del self.llm
        if self.engine:
            del self.engine
        if self.tokenizer:
            del self.tokenizer
        self.is_setup = False
