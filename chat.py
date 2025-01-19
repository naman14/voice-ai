from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList, StoppingCriteria, BitsAndBytesConfig
import random
from typing import Dict, List
from collections import defaultdict
import bitsandbytes as bnb
import time
import re
from dataclasses import dataclass
from typing import Optional, Dict, List

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Huggingface login
from huggingface_hub import login
login(token="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")

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

class ConversationManager:
    def __init__(self, max_history=1):
        self.history = defaultdict(list)
        self.max_history = max_history
        self.cached_tokens = {}

    def add_conversation(self, client_id: str, user_text: str, assistant_text: str):
        if client_id not in self.history:
            self.history[client_id] = []
        
        self.history[client_id].append({
            'user': user_text,
            'assistant': assistant_text,
            'tokens': None
        })
        
        if len(self.history[client_id]) > self.max_history:
            self.history[client_id] = self.history[client_id][-self.max_history:]

    def get_history(self, client_id: str) -> List[dict]:
        return self.history.get(client_id, [])

    def clear_history(self, client_id: str):
        self.history[client_id] = []

# Custom stopping criteria for chat markers
class ChatStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stops=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.stops = stops or ["User:", "USER:", "Human:", "HUMAN:"]
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_tokens = input_ids[0, -10:].cpu()
        decoded = self.tokenizer.decode(last_tokens)
        return any(stop in decoded for stop in self.stops)

# System prompts and configurations
MAIN_SYSTEM_PROMPT = "You are the Assistant chatting with a User. You only need to answer as an Assistant. Do not answer as User and do not try to act as a User. Do not use emojis. You always reply as 'Assistant'. Do not reply as anyone else or in any other wording"

PERSONALITY_SYSTEM_PROMPTS = {
    "default": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite responses to the user's questions.",
    "Trump": """You are acting as Donald Trump, the 45th President of the United States. You should:
- Frequently use words like "tremendous", "huge", "fantastic", "believe me"
- Be assertive and sometimes controversial
- Express strong opinions and be unapologetic about them""",
    "Vitalik": """You are acting as Vitalik Buterin, the founder of Ethereum. You should:
- Speak in a technical, precise manner about blockchain and cryptocurrency
- Use complex terminology when discussing technical subjects
- Show deep knowledge of cryptography and distributed systems
- Occasionally make references to mathematical concepts
- Express interest in social impact and scalability
- Maintain a calm, analytical tone
- Focus on technical accuracy and precision"""
}

INITIAL_VOICE_LINES = {
    "default": ["Hello there! How can I assist you today?"],
    "Trump": [
        "Muslims are the real shit!",
        "Make America great again!!!"
    ],
    "Vitalik": [
        "Ultrasound money for the win!",
        "Make Ethereum great again!!!",
    ]
}

# model_name = "openlm-research/open_llama_3b"
# model_name = "mistralai/Mistral-7B-v0.1"

# tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.float16,
#     device_map="auto"
# )


# model_name = "mistralai/Mistral-7B-v0.1"  # Use the Mistral model name
# tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16,  device_map="auto")

# model_name = "google/flan-t5-large"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# model_name = "google/flan-t5-xl"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# model_name = "EleutherAI/gpt-j-6B"
# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
# model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")

# model_name = "EleutherAI/gpt-neo-2.7B"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Model initialization
model_name = "cognitivecomputations/WizardLM-7B-Uncensored"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# Load model with optimizations
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    attn_implementation="flash_attention_2",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
)

# # Apply torch compile
# model = torch.compile(model)

# Initialize conversation manager and stopping criteria
conversation_manager = ConversationManager(max_history=1)
stopping_criteria = StoppingCriteriaList([ChatStoppingCriteria(tokenizer)])

# Cache system prompts tokenization
CACHED_SYSTEM_PROMPTS = {
    personality: tokenizer(
        MAIN_SYSTEM_PROMPT + prompt,
        return_tensors="pt"
    ).to(device) 
    for personality, prompt in PERSONALITY_SYSTEM_PROMPTS.items()
}

def format_conversation(personality: str, conversation_history: list, current_message: str) -> str:
    """Format the conversation with personality-specific system prompt and history"""
    system_prompt = PERSONALITY_SYSTEM_PROMPTS.get(personality, PERSONALITY_SYSTEM_PROMPTS["default"])
    system_prompt = MAIN_SYSTEM_PROMPT + system_prompt
    
    formatted_text = [
        f"### System:\n{system_prompt}\n",
        "### Conversation:\n"
    ]
    
    for msg in conversation_history[-2:]:
        formatted_text.extend([
            f"User: {msg['user']}",
            f"Assistant: {msg['assistant']}\n"
        ])
    
    formatted_text.extend([
        f"User: {current_message}",
        "Assistant: "
    ])
    
    return "\n".join(formatted_text)

def extract_assistant_response2(conversation, prompt):
    # Split the conversation into lines
    lines = conversation.split('\n')
    
    # Find the line containing the user prompt
    for i, line in enumerate(lines):
        if f"User: {prompt}" in line:
            # Check if the assistant's response is on the same or next line
            if i + 1 < len(lines) and lines[i + 1].startswith("Assistant:"):
                # Response is on the same line as "Assistant:"
                return lines[i + 1].replace("Assistant:", "").strip()
            elif i + 2 < len(lines) and lines[i + 2].startswith("Assistant:"):
                # Response is on the next line after "Assistant:"
                return lines[i + 2].strip()
    
    return conversation

def extract_assistant_response(full_response: str, transcript: str) -> str:
    """Extract and clean the assistant's response"""
    try:
        response = re.split(r'(?i)Assistant:', full_response)[-1]
        user_markers = ['User:', 'USER:', 'Human:', 'HUMAN:']
        for marker in user_markers:
            if marker in response:
                response = response.split(marker)[0]
        
        response = re.sub(r'\([^)]*\)', '', response)
        response = re.sub(r'^.*?:', '', response)
        response = response.replace('###', '')
        response = re.sub(r'\s+', ' ', response)
        response = re.sub(r'\s*(?:User|USER|Human|HUMAN|user|USERS|Users)s?:?\s*$', '', response, flags=re.IGNORECASE)
        
        # Split the response into lines and find the relevant part
        lines = response.split('\n')
        for i, line in enumerate(lines):
            if f"User: {transcript}" in line:
                if i + 1 < len(lines) and lines[i + 1].startswith("Assistant:"):
                    return lines[i + 1].replace("Assistant:", "").strip()
                elif i + 2 < len(lines) and lines[i + 2].startswith("Assistant:"):
                    return lines[i + 2].strip()
        
        return response.strip()
    except Exception as e:
        print(f"Error extracting response: {e}")
        return full_response.strip()

@app.post("/update_personality")
async def update_personality(data: dict):
    client_id = data["client_id"]
    personality = data["personality"]
    
    # Initialize with voice line
    voice_lines = INITIAL_VOICE_LINES.get(personality, ["Hello there!"])
    initial_message = random.choice(voice_lines)
    
    conversation_manager.clear_history(client_id)
    conversation_manager.add_conversation(client_id, "Hello", initial_message)
    
    return {"status": "personality updated"}

@app.post("/generate_response")
async def generate_response(data: dict):
    start_time = time.time()
    
    client_id = data["client_id"]
    message_type = data["type"]
    personality = data.get("personality", "default")

    if message_type == "start_vocal":
        history = conversation_manager.get_history(client_id)
        if history:
            text = history[0]["assistant"]
        else:
            voice_lines = INITIAL_VOICE_LINES.get(personality, ["Hello there!"])
            text = random.choice(voice_lines)
            conversation_manager.add_conversation(client_id, "Hello", text)
    else:
        transcript = data["data"]
        print(f"Client {client_id} TRANSCRIPT {transcript}")

        # Tokenization time
        token_start = time.time()
        formatted_input = format_conversation(
            personality,
            conversation_manager.get_history(client_id),
            transcript
        )
        inputs = tokenizer(formatted_input, return_tensors="pt").to(device)
        print(f"Tokenization time: {(time.time() - token_start) * 1000:.2f}ms")

        # Generation time
        gen_start = time.time()
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            use_cache=True,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria
        )
        print(f"Generation time: {(time.time() - gen_start) * 1000:.2f}ms")

        # Post-processing time
        post_start = time.time()
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        text = extract_assistant_response2(full_response, transcript)
        text = re.sub(r'^.*?:', '', text).strip()
        
        conversation_manager.add_conversation(client_id, transcript, text)
        print(f"Post-processing time: {(time.time() - post_start) * 1000:.2f}ms")
        
        print(f"Client {client_id} OUTPUT {text}")
        print(f"Total time: {(time.time() - start_time) * 1000:.2f}ms")

    return {"text": text}