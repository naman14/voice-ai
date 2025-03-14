import json
import random
import os
import aiohttp
import asyncio
from collections import OrderedDict
from typing import Dict, Optional
import time
from pydub import AudioSegment
import tempfile
from dotenv import load_dotenv
import threading
from dataclasses import dataclass
import ffmpeg

load_dotenv()

@dataclass
class AgentConfig:
    voice_samples: str
    system_prompt: str
    language: str
    cartesia_voice_id: str
    elevenlabs_voice_id: str
    speed: float
    
    def to_dict(self) -> dict:
        return {
            "voiceSamples": self.voice_samples,
            "systemPrompt": self.system_prompt,
            "language": self.language,
            "cartesiaVoiceId": self.cartesia_voice_id,
            "elevenlabsVoiceId": self.elevenlabs_voice_id,
            "speed": self.speed
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AgentConfig':
        return cls(
            voice_samples=data.get("voiceSamples", ""),
            system_prompt=data.get("systemPrompt", ""),
            language=data.get("language", "en"),
            cartesia_voice_id=data.get("cartesiaVoiceId", ""),
            elevenlabs_voice_id=data.get("elevenlabsVoiceId", ""),
            speed=data.get("speed", 1.0)
        )

class AgentManager:
    def __init__(self):
        self.fast_mode = os.getenv("FAST_MODE", "False").lower() == "true"
        self.config_dir = "configs"
        self.voice_samples_dir = "voice_samples"
        self.configs: Dict[str, AgentConfig] = {}
        self.lock = threading.Lock()
        self.max_audio_duration = 15000  # 15 seconds in milliseconds
        self.max_samples = 100
        self.voice_sample_access_times: OrderedDict[str, float] = OrderedDict()
        
        os.makedirs(self.voice_samples_dir, exist_ok=True)

        if not self.fast_mode:
            # Only create config directories if we're in normal mode
            os.makedirs(self.config_dir, exist_ok=True)

    def cleanup_old_samples(self):
        """Remove oldest voice samples if we exceed max_samples"""
        while len(self.voice_sample_access_times) > self.max_samples:
            oldest_file = next(iter(self.voice_sample_access_times))
            file_path = os.path.join(self.voice_samples_dir, oldest_file)
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                del self.voice_sample_access_times[oldest_file]
            except Exception as e:
                print(f"Error removing old sample {oldest_file}: {e}")

    async def download_voice_sample(self, url: str, config_id: str) -> Optional[str]:
        """Download voice sample, convert to wav, limit duration and return local file path"""
        if not url or not (url.startswith('http://') or url.startswith('https://')):
            return url  # Return original path if not a HTTP(S) URL
            
        final_path = os.path.join(self.voice_samples_dir, f"{config_id}.wav")
        
        # Check if file already exists
        if os.path.exists(final_path):
            self.voice_sample_access_times[f"{config_id}.wav"] = time.time()
            print(f"Voice sample already exists for config_id: {config_id}")
            return final_path
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(url)[1]) as temp_file:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            content = await response.read()
                            if not content:
                                print(f"Downloaded empty content from URL: {url}")
                                return None
                                
                            temp_file.write(content)
                            temp_file.flush()
                            
                            try:
                                # Check if the downloaded file is already a WAV file
                                if url.lower().endswith('.wav'):
                                    # If it's already WAV, just move the file
                                    os.rename(temp_file.name, final_path)
                                    print(f"File is already WAV, moved to: {final_path}")
                                else:
                                    # Convert non-WAV files
                                    print(f"Converting non-WAV file using ffmpeg: {temp_file.name}")
                                    stream = (
                                        ffmpeg
                                        .input(temp_file.name)
                                        .output(
                                            final_path,
                                            f='wav',
                                            acodec='pcm_s16le',
                                            ar=44100,
                                            ac=1,
                                            vn=None,
                                            loglevel='error'
                                        )
                                    )
                                    
                                    # Print the ffmpeg command for debugging
                                    print(f"FFmpeg command: {' '.join(stream.compile())}")
                                    
                                    ffmpeg.run(stream, overwrite_output=True, capture_stderr=True)
                                
                                if os.path.exists(final_path):
                                    # Add to access times and cleanup if needed
                                    self.voice_sample_access_times[f"{config_id}.wav"] = time.time()
                                    self.cleanup_old_samples()
                                    print(f"Successfully processed voice sample to: {final_path}")
                                    return final_path
                                else:
                                    print("File processing failed - output file not created")
                                    return None
                                    
                            except Exception as e:
                                print(f"Error processing audio: {str(e)}")
                                return None
                            finally:
                                if os.path.exists(temp_file.name):
                                    os.unlink(temp_file.name)
                        else:
                            print(f"Failed to download voice sample. Status code: {response.status}")
                            return None
        except Exception as e:
            print(f"Error downloading voice sample: {str(e)}")
            return None

    async def add_agent_config(self, config: dict) -> None:
        """Add or update agent configuration"""
        config_id = config.get("configId")
        if not config_id:
            raise ValueError("Config ID is required")

        agent_config = AgentConfig(
            voice_samples=config.get("voiceSampleUrl", ""),
            system_prompt=config.get("systemPrompt", ""),
            language=config.get("language", "en"),
            cartesia_voice_id=config.get("cartesiaVoiceId", ""),
            elevenlabs_voice_id=config.get("elevenlabsVoiceId", ""),
            speed=config.get("speed", 1.0)
        )

        voice_sample_path = await self.download_voice_sample(
            agent_config.voice_samples, 
            config_id
        )

        if voice_sample_path:
            self.voice_sample_access_times[f"{config_id}.wav"] = time.time()
            self.cleanup_old_samples()
            agent_config.voice_samples = voice_sample_path
        
        with self.lock:
            self.configs[config_id] = agent_config
            
            if not self.fast_mode:
                # In normal mode, also write to file
                config_path = os.path.join(self.config_dir, f"{config_id}.json")
                with open(config_path, 'w') as f:
                    json.dump(agent_config.to_dict(), f)

    def get_agent_config(self, config_id: str) -> Optional[AgentConfig]:
        """Get agent configuration"""
        with self.lock:
            if self.fast_mode:
                # In fast mode, return directly from memory
                return self.configs.get(config_id)
            
            # In normal mode, try memory first, then file
            if config_id in self.configs:
                return self.configs[config_id]
            
            # Try loading from file
            config_path = os.path.join(self.config_dir, f"{config_id}.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)
                        agent_config = AgentConfig.from_dict(config_data)
                        # Cache in memory
                        self.configs[config_id] = agent_config
                        return agent_config
                except Exception as e:
                    print(f"Error loading config {config_id}: {e}")
                    return None
            
            return None

    def remove_agent_config(self, config_id: str) -> bool:
        """Remove agent configuration"""
        with self.lock:
            # Always remove from memory
            self.configs.pop(config_id, None)
            
            if not self.fast_mode:
                # In normal mode, also remove file
                config_path = os.path.join(self.config_dir, f"{config_id}.json")
                try:
                    if os.path.exists(config_path):
                        os.remove(config_path)
                except Exception as e:
                    print(f"Error removing config file {config_id}: {e}")
                    return False
            return True

    def clear_configs(self) -> None:
        """Clear all configurations"""
        with self.lock:
            self.configs.clear()
            
            if not self.fast_mode:
                # In normal mode, also clear files
                for filename in os.listdir(self.config_dir):
                    if filename.endswith('.json'):
                        try:
                            os.remove(os.path.join(self.config_dir, filename))
                        except Exception as e:
                            print(f"Error removing config file {filename}: {e}")


agent_manager = AgentManager()