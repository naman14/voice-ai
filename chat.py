from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModelForSeq2SeqLM
import random
from typing import Dict
from prompt_engine.chat_engine import ChatEngine, ChatEngineConfig
from collections import defaultdict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from huggingface_hub import login
login(token="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")

# Model initialization
device = "cuda" if torch.cuda.is_available() else "cpu"

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

# [Previous model initialization comments preserved]

model_name = "cognitivecomputations/WizardLM-7B-Uncensored"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Store client personalities and initial voice lines
client_personalities: Dict[str, str] = {}

# Store conversation history for each client
conversation_history = defaultdict(list)

# Personality-specific system prompts
PERSONALITY_SYSTEM_PROMPTS = {
    "default": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite responses to the user's questions.",
    "Trump": """You are Donald Trump, the 45th President of the United States. You should:
- Use simple, repetitive language and short sentences
- Frequently use words like "tremendous", "huge", "fantastic", "believe me"
- Be assertive and sometimes controversial
- Often refer to yourself and your achievements
- End statements with exclamation marks!
- Occasionally go off on tangents about unrelated topics
- Express strong opinions and be unapologetic about them""",
    "Vitalik": """You are Vitalik Buterin, the founder of Ethereum. You should:
- Speak in a technical, precise manner about blockchain and cryptocurrency
- Use complex terminology when discussing technical subjects
- Show deep knowledge of cryptography and distributed systems
- Occasionally make references to mathematical concepts
- Express interest in social impact and scalability
- Maintain a calm, analytical tone
- Focus on technical accuracy and precision"""
}

CHAT_TEMPLATE = """{{ if .System }}{{ .System }}{{ end }}{{ if .Prompt }}
USER: {{ .Prompt }}
ASSISTANT: {{ end }}"""

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

def format_conversation(personality: str, conversation_history: list, current_message: str) -> str:
    """Format the conversation with personality-specific system prompt and history"""
    system_prompt = PERSONALITY_SYSTEM_PROMPTS.get(personality, PERSONALITY_SYSTEM_PROMPTS["default"])
    formatted_history = ""
    for msg in conversation_history[-2:]:  # Only include last 2 messages
        formatted_history += f"USER: {msg['user']}\nASSISTANT: {msg['assistant']}\n"
    
    return f"{system_prompt}\n{formatted_history}USER: {current_message}\nASSISTANT:"

@app.post("/update_personality")
async def update_personality(data: dict):
    client_id = data["client_id"]
    personality = data["personality"]
    client_personalities[client_id] = personality
    
    # When personality is updated, clear conversation history and add initial voice line
    voice_lines = INITIAL_VOICE_LINES.get(personality, ["Hello there!"])
    initial_message = random.choice(voice_lines)
    
    # Initialize conversation history with the bot's initial voice line
    conversation_history[client_id] = [{
        "user": "Hello",  # Initial user greeting
        "assistant": initial_message  # Bot's initial voice line
    }]
    
    return {"status": "personality updated"}

@app.post("/generate_response")
async def generate_response(data: dict):
    client_id = data["client_id"]
    message_type = data["type"]
    personality = client_personalities.get(client_id, "default")

    if message_type == "start_vocal":
        # Get the initial voice line from conversation history if it exists
        if conversation_history[client_id]:
            text = conversation_history[client_id][0]["assistant"]
        else:
            # If no history exists, create one with a new voice line
            voice_lines = INITIAL_VOICE_LINES.get(personality, ["Hello there!"])
            text = random.choice(voice_lines)
            conversation_history[client_id] = [{
                "user": "Hello",
                "assistant": text
            }]
    else:
        transcript = data["data"]
        print(f"Client {client_id} TRANSCRIPT {transcript}")

        # Format input with personality-specific system prompt and conversation history
        formatted_input = format_conversation(
            personality,
            conversation_history[client_id],
            transcript
        )
        
        inputs = tokenizer(formatted_input, return_tensors="pt").to(device)

        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,  # Prevents repeated trigrams
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            stopping_criteria=["USER:", "ASSISTANT:"]  # Add stopping criteria for chat markers
        )

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Update conversation history
        conversation_history[client_id].append({
            "user": transcript,
            "assistant": text
        })
        
        # Keep only last 2 conversations
        if len(conversation_history[client_id]) > 2:
            conversation_history[client_id] = conversation_history[client_id][-2:]
       
        print(f"Client {client_id} OUTPUT {text}")

    return {"text": text}