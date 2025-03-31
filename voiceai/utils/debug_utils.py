import wave
import os
from datetime import datetime
import numpy as np

def save_audio_chunks(audio_chunks: list[np.ndarray], session_id: str, turn_id: int):
    """
    Save audio chunks to a WAV file for debugging purposes.
    
    Args:
        audio_chunks: List of numpy arrays containing audio data
        session_id: Session identifier
        turn_id: Turn number in the conversation
    """
    if not audio_chunks:
        return
        
    # Create debug_audio directory if it doesn't exist
    debug_dir = "debug_audio"
    os.makedirs(debug_dir, exist_ok=True)
    
    # Create filename with timestamp, session_id and turn_id
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{debug_dir}/debug_{session_id}_{turn_id}_{timestamp}.wav"
    
    # Combine all chunks
    combined_audio = np.concatenate(audio_chunks)
    
    # Save as WAV file
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(16000)  # 16kHz
        wav_file.writeframes(combined_audio.tobytes())
