from .base_llm import BaseLLM
from typing import Optional
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    StoppingCriteriaList, 
    StoppingCriteria,
    BitsAndBytesConfig
)
from typing import AsyncGenerator
import re
import os
import time
from threading import Thread
from transformers import TextIteratorStreamer

class ChatStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stops=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.stops = stops or ["User:", "USER:", "Human:", "HUMAN:"]
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_tokens = input_ids[0, -10:].cpu()
        decoded = self.tokenizer.decode(last_tokens)
        return any(stop in decoded for stop in self.stops)

class LocalLLM(BaseLLM):
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B-Instruct"):
        super().__init__()
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.stopping_criteria = None
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built() else "cpu"

    def setup(self):
        if self.is_setup:
            return
            
        hf_token = os.getenv('HUGGINGFACE_API_KEY')
        if hf_token and hf_token != "":
            # Hugging Face login
            from huggingface_hub import login
            login(token=hf_token)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True
        )

        ds_config = {
            "tensor_parallel": {"tp_size": 1},
            "dtype": "fp16",
            "replace_with_kernel_inject": True,
            "replace_method": "auto"
        }

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            # quantization_config=bnb_config
        )

        # self.model = deepspeed.init_inference(
        #     self.model,
        #     config=ds_config
        # )
        
        self.stopping_criteria = StoppingCriteriaList([
            ChatStoppingCriteria(self.tokenizer)
        ])
        
        self.is_setup = True

    async def generate(self, prompt: str, **kwargs) -> str:
        if not self.is_setup:
            raise RuntimeError("LLM not initialized")

        chat = prompt
        
        formatted_chat = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(formatted_chat, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=kwargs.get("max_new_tokens", 40),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            top_k=kwargs.get("top_k", 50),
            repetition_penalty=kwargs.get("repetition_penalty", 1.2),
            no_repeat_ngram_size=kwargs.get("no_repeat_ngram_size", 3),
            use_cache=True,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
            stopping_criteria=self.stopping_criteria
        )
        
        decoded_output = self.tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
        response = decoded_output
        
        # Clean up response
        response = re.sub(r'^.*?:', '', response).strip()
        
        return response
    
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        if not self.is_setup:
            raise RuntimeError("LLM not initialized")

        start_time = time.time()
        first_token_time = None
        token_count = 0
        response = ""

        # Prepare input
        formatted_chat = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(formatted_chat, return_tensors="pt").to(self.device)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        # Run generation in a separate thread
        thread = Thread(target=self.model.generate, kwargs={
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "max_new_tokens": kwargs.get("max_new_tokens", 40),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "top_k": kwargs.get("top_k", 50),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.2),
            "no_repeat_ngram_size": kwargs.get("no_repeat_ngram_size", 3),
            "use_cache": True,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
            "stopping_criteria": self.stopping_criteria,
            "streamer": streamer,
        })
        thread.start()

        # Collect tokens and log timings
        for new_text in streamer:
            if first_token_time is None:
                first_token_time = time.time() - start_time
                print(f"Time to first token: {first_token_time:.3f}s")

            yield new_text
            
            token_count += 1
            response += new_text  # Accumulate final response

        thread.join()

        # Final timing logs
        total_time = time.time() - start_time
        tokens_per_second = token_count / total_time if total_time > 0 else 0
        print(f"\nTotal generation time: {total_time:.3f}s")
        print(f"Tokens per second: {tokens_per_second:.2f}")
        
    
    async def cleanup(self):
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        if self.stopping_criteria:
            del self.stopping_criteria
        self.is_setup = False 