from abc import ABC, abstractmethod

class BaseSTT(ABC):
    """Base class for Speech-to-Text implementations"""
    
    @abstractmethod
    def setup(self) -> None:
        """
        Setup function to initialize any resources needed by the STT implementation.
        Should be called after instance creation.
        """
        pass
    
    @abstractmethod
    async def transcribe(self, audio_data: bytes, language: str) -> str:
        """
        Transcribe audio data to text
        
        Args:
            audio_data (bytes): Raw audio data to transcribe
            
        Returns:
            str: Transcribed text
        """
        pass 