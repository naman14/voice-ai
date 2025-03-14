from .base_llm import BaseLLM


class EmptyLLM(BaseLLM):
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-3B-Instruct"):
        super().__init__()
        self.model_name = model_name
        self.is_setup = False

    def setup(self):
        self.is_setup = True

    async def generate(self, prompt: str, **kwargs) -> str:
        if not self.is_setup:
            raise RuntimeError("LLM not initialized")

        return prompt[-1]['content']
       

    async def cleanup(self):
        self.is_setup = False
