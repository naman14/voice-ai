import numpy as np
import opuslib

class OpusEncoder:
    def __init__(self, sample_rate=24000, channels=1, application=opuslib.APPLICATION_AUDIO):
        self.encoder = opuslib.Encoder(sample_rate, channels, application)
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_size = int(sample_rate * 0.08)  # 80ms frame size (1920 samples at 24kHz)
        
    def encode(self, pcm_data):
        """Encode PCM data to Opus frames"""
        if len(pcm_data) == 0:
            return b''
        opus_frame = self.encoder.encode_float(pcm_data.tobytes(), self.frame_size)
        return opus_frame

class OpusDecoder:
    def __init__(self, sample_rate=24000, channels=1):
        self.decoder = opuslib.Decoder(sample_rate, channels)
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_size = int(sample_rate * 0.08)  # 80ms frame size (1920 samples at 24kHz)
        
    def decode(self, opus_data):
        """Decode Opus frames to PCM data"""
        if len(opus_data) == 0:
            return np.array([], dtype=np.int16)
            
        pcm_chunk = self.decoder.decode(bytes(opus_data), self.frame_size)
        if(type(pcm_chunk) == bytes):
            pcm_chunk = np.frombuffer(pcm_chunk, dtype=np.int16)
            return pcm_chunk
        else:
            return None