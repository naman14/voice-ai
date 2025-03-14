import time
from dataclasses import dataclass
from typing import Optional
from contextlib import contextmanager

@dataclass
class LatencyMetrics:
    """Stores latency metrics for a single conversation turn."""
    conversation_id: str
    turn_id: int
    
    # Timestamps for different stages
    silence_detected_time: Optional[float] = None
    transcription_start_time: Optional[float] = None
    transcription_end_time: Optional[float] = None
    llm_start_time: Optional[float] = None
    llm_first_chunk_time: Optional[float] = None
    llm_first_sentence_time: Optional[float] = None
    llm_end_time: Optional[float] = None
    tts_start_time: Optional[float] = None
    tts_first_chunk_time: Optional[float] = None
    
    def get_total_latency(self) -> Optional[float]:
        """Calculate end-to-end latency from silence detection to first TTS chunk."""
        if self.silence_detected_time and self.tts_first_chunk_time:
            return self.tts_first_chunk_time - self.silence_detected_time
        return None
    
    def get_transcription_latency(self) -> Optional[float]:
        """Calculate transcription phase latency."""
        if self.transcription_start_time and self.transcription_end_time:
            return self.transcription_end_time - self.transcription_start_time
        return None
    
    def get_llm_latency(self) -> Optional[float]:
        """Calculate LLM inference latency."""
        if self.llm_start_time and self.llm_end_time:
            return self.llm_end_time - self.llm_start_time
        return None
    
    def get_tts_first_chunk_latency(self) -> Optional[float]:
        """Calculate time to first TTS chunk."""
        if self.tts_start_time and self.tts_first_chunk_time:
            return self.tts_first_chunk_time - self.tts_start_time
        return None
    
    def log_metrics(self):
        """Log metrics for a specific conversation turn."""
        print("--------------------------------")
        print(f"VoiceChat: Metrics for conversation {self.conversation_id}, turn {self.turn_id}:")
        print("VoiceChat: Total latency: ", self.get_total_latency())
        print("VoiceChat: Transcription latency: ", self.get_transcription_latency())
        if self.llm_first_sentence_time:
            print("VoiceChat: LLM first sentence latency: ", self.llm_first_sentence_time - self.llm_start_time)
        if self.llm_first_chunk_time:
            print("VoiceChat: LLM first chunk latency: ", self.llm_first_chunk_time - self.llm_start_time)
        print("VoiceChat: LLM latency: ", self.get_llm_latency())
        print("VoiceChat: TTS first chunk latency: ", self.get_tts_first_chunk_latency())
        print("--------------------------------")

class MetricsManager:
    """Manages latency metrics collection across the system."""
    
    def __init__(self):
        self.metrics = {}
    
    def create_metrics(self, conversation_id: str, turn_id: int) -> LatencyMetrics:
        """Create a new metrics object for a conversation turn."""
        metrics = LatencyMetrics(conversation_id=conversation_id, turn_id=turn_id)
        self.metrics[f"{conversation_id}_{turn_id}"] = metrics
        return metrics
    
    def get_metrics(self, conversation_id: str, turn_id: int) -> Optional[LatencyMetrics]:
        """Retrieve metrics for a specific conversation turn."""
        return self.metrics.get(f"{conversation_id}_{turn_id}")
    

    @contextmanager
    def measure_time(self, conversation_id: str, turn_id: int, stage: str):
        """Context manager to measure time for a specific stage."""
        metrics = self.get_metrics(conversation_id, turn_id)
        if not metrics:
            metrics = self.create_metrics(conversation_id, turn_id)
        
        start_attr = f"{stage}_start_time"
        end_attr = f"{stage}_end_time"
        
        setattr(metrics, start_attr, time.time())
        try:
            yield
        finally:
            setattr(metrics, end_attr, time.time())

# Global metrics manager instance
metrics_manager = MetricsManager()
