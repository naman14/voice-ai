from .base_llm import BaseLLM
from mlx_lm import load, generate
from typing import AsyncGenerator

class MLXLLM(BaseLLM):
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B-Instruct"):
        super().__init__()
        self.model_name = model_name
        self.is_setup = False

    def setup(self):
        self.is_setup = True
        self.model, self.tokenizer = load(self.model_name)

    async def generate(self, prompt: str, **kwargs) -> str:
        if not self.is_setup:
            raise RuntimeError("LLM not initialized")
        
        chat = prompt

        formatted_chat = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True)

        text = generate(self.model, self.tokenizer, prompt=formatted_chat, verbose=True)

        print(text)

        return text
    
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        if not self.is_setup:
            raise RuntimeError("LLM not initialized")

        raise NotImplementedError("MLX LLM does not support streaming")
        
        
    async def cleanup(self):
        self.is_setup = False
