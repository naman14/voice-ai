from collections import defaultdict
from typing import List, Dict
from voiceai.config.agents_config import agent_manager
import re
import random

# System prompts and configurations
MAIN_SYSTEM_PROMPT = "You are a Voice agent on voiceai.fun platform. Detailed instructions to follow. Please reply in max 30 words."

class ConversationManager:
    def __init__(self, max_history=1):
        self.history = defaultdict(list)
        self.max_history = max_history
        self.cached_tokens = {}

    def add_conversation(self, session_id: str, user_text: str, assistant_text: str):
        if session_id not in self.history:
            self.history[session_id] = []
        
        self.history[session_id].append({
            'user': user_text,
            'assistant': assistant_text,
            'tokens': None
        })
        
        if len(self.history[session_id]) > self.max_history:
            self.history[session_id] = self.history[session_id][-self.max_history:]

    def get_history(self, session_id: str) -> List[dict]:
        return self.history.get(session_id, [])

    def clear_history(self, session_id: str):
        self.history[session_id] = []


class ConversationFormatter:
    def __init__(self):
        pass

    def format_messages(self, config_id: str, conversation_history: list, current_message: str) -> List[Dict[str, str]]:
        """Format conversation history into Groq API message format"""
        config = agent_manager.get_agent_config(config_id)
        if not config:
            raise ValueError("Config not found for config_id: " + config_id)

        system_prompt = config.system_prompt
        if not system_prompt:
            raise ValueError("System prompt not found for config_id: " + config_id)
        
        system_prompt = MAIN_SYSTEM_PROMPT + system_prompt

        if config.language == "hi":
            # system_prompt = system_prompt.replace("Please reply in no more than 30 words. ", "")
            system_prompt = system_prompt + " आपका जवाब हिंदी में होना चाहिए।"
        
        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]
        
        # Add conversation history
        for msg in conversation_history[-2:]:
            messages.extend([
                {"role": "user", "content": msg["user"]},
                {"role": "assistant", "content": msg["assistant"]}
            ])
        
        messages.append({"role": "user", "content": current_message})
        
        return messages
    
    def check_uncensored(self, text):
        triggers = [
            "hate speech or discriminatory language",
            "I can't engage",
            "I won't engage"
        ]
        
        responses = [
            "How about you shut the fuck up",
            "Yeah, what else you got on your filthy mind",
            "You should go and ask your mom about that"
        ]

        if any(trigger in text.lower() for trigger in triggers):
            return random.choice(responses)

        return text
    
    def remove_emotions(self, text: str) -> str:
        # Remove text between * * (including the * characters)
        text = re.sub(r'\*[^*]*\*', '', text)
        # Remove text between ( ) (including the () characters)
        text = re.sub(r'\([^)]*\)', '', text)
        # Return the cleaned text
        return text.strip()
    
    def cleanup_response(self, response: str) -> str:
        if not response:
            fallback_responses = [
                "I understand what you're saying. Could you rephrase that?",
                "That's an interesting point. Could you elaborate?",
                "I see what you mean. Let's explore that further.",
            ]
            response = random.choice(fallback_responses)
            print("Using fallback response:", response)
        
        response = self.check_uncensored(response)
        response = self.remove_emotions(response)
        return response