import numpy as np
import time

class AudioSpeechDetector:
    def __init__(self, 
                 sample_rate=16000, 
                 energy_threshold=0.1,  # Percentage of max possible energy
                 min_speech_duration=0.3,  # seconds
                 max_silence_duration=0.4,  # seconds
                 max_recording_duration=10.0,
                 debug=True):
        """
        Initialize speech detector with configurable parameters
        
        :param sample_rate: Audio sample rate (Hz)
        :param energy_threshold: Percentage of max possible energy for speech detection
        :param min_speech_duration: Minimum duration of speech to be considered valid (seconds)
        :param max_silence_duration: Maximum duration of silence within speech (seconds)
        :param max_recording_duration: Maximum duration of continuous recording
        :param debug: Enable debug logging
        """
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.min_speech_duration = min_speech_duration
        self.max_silence_duration = max_silence_duration
        self.max_recording_duration = max_recording_duration
        self.debug = debug
        
        # Reset internal state
        self.reset()
    
    def reset(self):
        """
        Reset the internal state of the speech detector
        """
        self.current_audio_chunks = []
        self.last_speech_time = None
        self.start_recording_time = None
        self.consecutive_silence_chunks = 0
    
    def _log(self, message):
        """
        Debug logging
        """
        if self.debug:
            print(f"[SpeechDetector] {message}")
    
    def _calculate_normalized_energy(self, audio_chunk):
        """
        Calculate normalized energy of the audio chunk
        
        :param audio_chunk: NumPy array of audio samples
        :return: Normalized energy (0-1 range)
        """
        # Assuming 16-bit audio, max possible value is 32768
        max_possible_amplitude = 32768.0
        
        # Calculate peak amplitude
        peak_amplitude = np.max(np.abs(audio_chunk))
        
        # Calculate normalized energy as percentage of max possible
        normalized_energy = peak_amplitude / max_possible_amplitude
        
        return normalized_energy
    
    def add_audio_chunk(self, audio_chunk):
        """
        Add an audio chunk and check for speech detection conditions
        
        :param audio_chunk: NumPy array of audio samples
        :return: Dict with detection status and potential actions
        """
        # Validate input
        if audio_chunk is None or len(audio_chunk) == 0:
            return {"action": "continue"}
        
        current_time = time.time()
        
        # Initialize recording start time if not set
        if self.start_recording_time is None:
            self.start_recording_time = current_time
        
        # Calculate normalized energy
        chunk_energy = self._calculate_normalized_energy(audio_chunk)
        
        # Detailed logging
        self._log(f"Chunk Analysis: " +
                  f"Total Samples: {len(audio_chunk)}, " +
                  f"Normalized Energy: {chunk_energy:.6f}, " +
                  f"Threshold: {self.energy_threshold}")
        
        # Check if chunk energy is above threshold (indicating speech)
        if chunk_energy > self.energy_threshold:
            # Store the chunk
            self.current_audio_chunks.append(audio_chunk)
            
            self.last_speech_time = current_time
            self.consecutive_silence_chunks = 0
            self._log("Speech detected in chunk")
            return {"action": "continue"}
        
        # Increment consecutive silence chunks
        self.consecutive_silence_chunks += 1
        
        # Check max recording duration
        # if current_time - self.start_recording_time > self.max_recording_duration:
        #     self._log("Max recording duration reached")
        #     return {
        #         "action": "process",
        #         "reason": "max_duration_reached"
        #     }
        
        # Check for prolonged silence
        silence_duration = self.consecutive_silence_chunks * (len(audio_chunk) / self.sample_rate)
        
        self._log(f"Silence tracking: " +
                  f"Consecutive Silent Chunks: {self.consecutive_silence_chunks}, " +
                  f"Total Silence Duration: {silence_duration:.3f}s")
        
        # If silence has persisted beyond max_silence_duration
        if silence_duration > self.max_silence_duration:
            # Combine audio chunks
            full_audio = np.concatenate(self.current_audio_chunks) if self.current_audio_chunks else np.array([])
            
            # Calculate speech duration based on energy
            speech_chunks = [chunk for chunk in self.current_audio_chunks 
                             if self._calculate_normalized_energy(chunk) > self.energy_threshold]
            full_speech_audio = np.concatenate(speech_chunks) if speech_chunks else np.array([])
            
            speech_duration_full = len(full_speech_audio) / self.sample_rate
            
            self._log(f"Full Audio Speech Duration: {speech_duration_full:.3f}s")
            
            # Check if speech duration meets minimum requirement
            if speech_duration_full >= self.min_speech_duration:
                self._log("Speech end detected - processing audio")
                processed_chunks = self.current_audio_chunks.copy()
                
                # Reset state
                self.reset()
                
                return {
                    "action": "process",
                    "reason": "speech_end_detected",
                    "audio_chunks": processed_chunks
                }
            else:
                self._log("Not enough speech duration to process")
                # Reset if not enough speech detected
                self.reset()
        
        return {"action": "continue"}