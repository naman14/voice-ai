import asyncio
import aiohttp
import requests
from typing import Generator, Optional, List, Dict, AsyncGenerator
from ..base_tts import BaseTTS
import time
from cartesia import AsyncCartesia
import base64
import numpy as np
from ..base_tts import TTSChunk

class CartesiaWebSocketManager:
    def __init__(self, api_key: str, pool_size: int = 1):
        self.api_key = api_key
        self.pool_size = pool_size
        self.client = AsyncCartesia(api_key=api_key)
        self.connections: List[Dict] = []
        self.lock = asyncio.Lock()
        
    async def initialize_pool(self):
        """Initialize the WebSocket connection pool"""
        print("Initializing WebSocket pool...")
        for _ in range(self.pool_size):
            try:
                ws = await self.client.tts.websocket()
                self.connections.append({
                    "websocket": ws,
                    "in_use": False,
                    "last_used": time.time()
                })
                print(f"Added connection to pool. Total: {len(self.connections)}")
            except Exception as e:
                print(f"Error creating WebSocket connection: {e}")
        
    async def get_connection(self):
        """Get an available WebSocket connection from the pool"""
        async with self.lock:
            # First try to find an unused connection
            for conn in self.connections:
                if not conn["in_use"]:
                    conn["in_use"] = True
                    conn["last_used"] = time.time()
                    return conn["websocket"]
            
            # If no available connections, create a new one
            try:
                ws = await self.client.tts.websocket()
                conn = {
                    "websocket": ws,
                    "in_use": True,
                    "last_used": time.time()
                }
                self.connections.append(conn)
                return ws
            except Exception as e:
                print(f"Error creating new WebSocket connection: {e}")
                raise
    
    async def release_connection(self, ws):
        """Release a WebSocket connection back to the pool"""
        async with self.lock:
            for conn in self.connections:
                if conn["websocket"] == ws:
                    conn["in_use"] = False
                    conn["last_used"] = time.time()
                    break
    
    async def maintain_pool(self):
        """Periodically check and maintain the WebSocket pool"""
        while True:
            try:
                async with self.lock:
                    current_time = time.time()
                    # Check each connection
                    for i in range(len(self.connections) - 1, -1, -1):
                        conn = self.connections[i]
                        # If connection is old (1 hour) and not in use, close it
                        if not conn["in_use"] and (current_time - conn["last_used"]) > 3600:
                            try:
                                await conn["websocket"].close()
                            except:
                                pass
                            self.connections.pop(i)
                    
                    # Ensure we maintain minimum pool size
                    while len(self.connections) < self.pool_size:
                        try:
                            ws = await self.client.tts.websocket()
                            self.connections.append({
                                "websocket": ws,
                                "in_use": False,
                                "last_used": time.time()
                            })
                        except Exception as e:
                            print(f"Error maintaining pool: {e}")
                            break
                            
            except Exception as e:
                print(f"Error in maintain_pool: {e}")
            
            await asyncio.sleep(300)  # Check every 5 minutes

class CartesiaTTS(BaseTTS):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.ws_manager = CartesiaWebSocketManager(api_key)
        self.cartesia_client = AsyncCartesia(api_key=api_key)
        self.cartesia_stream_format = {
            "container": "raw",
            "encoding": "pcm_s16le",
            "sample_rate": 24000,
        }

        self.cartesia_bytes_format = {
            "container": "wav",
            "encoding": "pcm_f32le",
            "sample_rate": 24000,
        }
        
    def setup(self):
        """Initialize the TTS system"""
        self.ws_manager.initialize_pool()
        asyncio.create_task(self.ws_manager.maintain_pool())

    async def cleanup(self):
        """Clean up resources"""
        await self.ws_manager.cleanup()

    async def generate_speech(self, text: str, language: str, voice_id: Optional[str] = None, voice_samples: Optional[str] = None, speed: float = 1.0) -> TTSChunk:
        """Cartesia endpoint for single audio generation"""

        if not self.cartesia_client:
            raise ValueError("Cartesia client not initialized")
        
        # Generate audio using Cartesia's REST API
        response = await self.cartesia_client.tts.bytes(
            model_id="sonic",
            transcript=text,
            voice_id=voice_id,
            output_format=self.cartesia_bytes_format
        )
        
        # Convert to base64
        audio_base64 = base64.b64encode(response).decode('utf-8')
        
        # Return base64 encoded audio data
        return TTSChunk(audio_base64, self.cartesia_bytes_format["sample_rate"], "wav")
        
        
    async def generate_speech_stream(self, text: str, language: str, voice_id: Optional[str] = None, voice_samples: Optional[str] = None, speed: float = 1.0) -> AsyncGenerator[TTSChunk, None]:
        """Generate speech in streaming mode using Cartesia TTS API"""

        ws = await self.ws_manager.get_connection()
        try:
            t0 = time.time()
            stream = await ws.send(
                model_id="sonic",
                transcript=text,
                voice_id=voice_id,
                stream=True,
                output_format=self.cartesia_stream_format
            )

            # Buffer for accumulating audio data
            buffer = np.array([], dtype=np.float32)
            chunk_size = 4800  # 0.2 seconds at 24kHz
            chunk_counter = 0
            
            async for output in stream:
                if chunk_counter == 0:
                    print(f"Time to first chunk: {time.time() - t0}")
                
                # Convert bytes to numpy array
                audio_chunk = np.frombuffer(output["audio"], dtype=np.float32)
                
                # Add to buffer
                buffer = np.append(buffer, audio_chunk)
                
                # Process complete chunks
                while len(buffer) >= chunk_size:
                    # Extract chunk with overlap
                    chunk = buffer[:chunk_size]
                    buffer = buffer[chunk_size:]  # Remove processed data
                    
                    # Apply fade in/out to reduce artifacts
                    if chunk_counter > 0:  # Apply fade-in
                        fade_samples = 240  # 10ms fade
                        fade_in = np.linspace(0, 1, fade_samples)
                        chunk[:fade_samples] *= fade_in
                    
                    if len(buffer) < chunk_size:  # Apply fade-out to last chunk
                        fade_samples = 240
                        fade_out = np.linspace(1, 0, fade_samples)
                        chunk[-fade_samples:] *= fade_out
                    
                    # Convert to bytes and send
                    chunk_bytes = chunk.tobytes()
                    chunk_base64 = base64.b64encode(chunk_bytes).decode("utf-8")

                    yield TTSChunk(
                        chunk=chunk_base64,
                        sample_rate=self.cartesia_stream_format["sample_rate"],
                        format=self.cartesia_stream_format["encoding"]
                    )
                
                    chunk_counter += 1
            
            # Send any remaining buffer
            if len(buffer) > 0:
                chunk_bytes = buffer.tobytes()
                chunk_base64 = base64.b64encode(chunk_bytes).decode("utf-8")
                
                yield TTSChunk(
                    chunk=chunk_base64,
                    sample_rate=self.cartesia_stream_format["sample_rate"],
                    format=self.cartesia_stream_format["encoding"]
                )

        finally:
            await self.ws_manager.release_connection(ws)

