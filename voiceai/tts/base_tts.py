from abc import ABC, abstractmethod
from typing import Generator, Optional, AsyncGenerator

class TTSChunk:
    def __init__(self, chunk: bytes, format: str, sample_rate: int):
        self.chunk = chunk
        self.format = format
        self.sample_rate = sample_rate

class BaseTTS(ABC):
    """Base class for all TTS implementations"""
    
    @abstractmethod
    def setup(self):
        """Initialize the TTS system"""
        pass
        
    @abstractmethod
    async def cleanup(self):
        """Clean up resources"""
        pass
    
    @abstractmethod
    async def generate_speech(self, text: str, language: str, voice_id: Optional[str] = None, voice_samples: Optional[str] = None, speed: float = 1.0) -> TTSChunk:
        """Generate speech from text synchronously
        
        Args:
            text: Text to convert to speech
            language: Language of the text
            voice_id: Optional voice identifier
            voice_samples: Optional voice samples
            
        Returns:
            TTSChunk object containing audio bytes, sample rate, and format
        """
        pass
        
    @abstractmethod
    async def generate_speech_stream(self, text: str, language: str, voice_id: Optional[str] = None, voice_samples: Optional[str] = None, speed: float = 1.0) -> AsyncGenerator[TTSChunk, None]:
        """Generate speech from text in streaming mode
        
        Args:
            text: Text to convert to speech
            language: Language of the text
            voice_id: Optional voice identifier
            voice_samples: Optional voice samples
            
        Returns:
            Generator yielding TTSChunk objects
        """
        pass 