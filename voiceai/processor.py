import wave
import numpy as np
import aiohttp
import json
import os
from datetime import datetime
from typing import Optional, List
from voiceai.utils.speechdetector import AudioSpeechDetector
import asyncio
from fastapi import WebSocket
from voiceai.server import AudioSession
class AudioProcessor:
    def __init__(self, session_id: str, session: AudioSession, config_id: str):
        self.session_id = session_id
        self.config_id = config_id
        self.audio_chunks: List[np.ndarray] = []
        self.is_speaking = False
        self.is_responding = False
        self.tts_lock = asyncio.Lock()
        self.session = session

        self.speech_detector = AudioSpeechDetector(
            sample_rate=16000,
            energy_threshold=0.15,
            min_speech_duration=0.4,
            max_silence_duration=0.5,
            max_recording_duration=10.0,
            debug=False
        )
        
        # Service URLs
        self.STT_SERVICE_URL = "http://localhost:8001/transcribe"
        self.CHAT_SERVICE_URL = "http://localhost:8002/chat"
        self.TTS_SERVICE_URL = "ws://localhost:8003/tts/stream"
        self.tts_websocket: Optional[aiohttp.ClientWebSocketResponse] = None

    async def process_audio_chunk(self, binary_data: bytes, websocket: WebSocket) -> None:
        """Process incoming audio chunks and handle VAD internally"""
        if len(binary_data) > 0:
            audio_data = np.frombuffer(binary_data, dtype=np.int16)
            detection_result = self.speech_detector.add_audio_chunk(audio_data)
            
            if detection_result['action'] == 'process':
                self.audio_chunks = detection_result.get('audio_chunks', [])
                await self.process_speech(websocket)

    async def process_speech(self, websocket: WebSocket) -> None:
        """Handle speech processing and response generation"""
        print("Processing speech...")
        self.is_responding = True
        try:
            audio_file = await self.save_audio()
            if not audio_file:
                self.is_responding = False
                return

            transcript = await self.transcribe_audio(audio_file)

            if not transcript.strip() or "thank you" in transcript.lower():
                self.is_responding = False
                return
                
            await self.process_text(transcript, websocket)

        except Exception as e:
            self.is_responding = False
            print(f"Error processing audio: {e}")
            await self.send_error(websocket, str(e))
        finally:
            if os.path.exists(audio_file):
                os.remove(audio_file)

    async def process_text(self, text: str, websocket: WebSocket) -> None:
        """Process text input and generate response"""
        print(f"Processing text: {text}")
        async with self.tts_lock:
            self.is_responding = True
            try:
                chat_response = await self.get_chat_response(text)
                await self.stream_tts(chat_response, websocket)
            except Exception as e:
                self.is_responding = False
                print(f"Error processing text: {e}")
                await self.send_error(websocket, str(e))

    async def handle_client_message(self, message_type: str, websocket: WebSocket) -> None:
        """Handle client control messages"""
        if message_type == "speech_start":
            self.is_speaking = True
            self.audio_chunks = []
        elif message_type == "speech_end":
            self.is_speaking = False
            if self.audio_chunks:
                await self.process_speech(websocket)

    async def send_error(self, websocket: WebSocket, error: str) -> None:
        """Send error message to client"""
        try:
            await websocket.send_json({
                "type": "error",
                "error": error,
                "session_id": self.session_id
            })
        except RuntimeError:
            pass  # WebSocket is closed

    async def save_audio(self) -> str:
        if not self.audio_chunks:
            return ""
            
        os.makedirs("audio_files", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"audio_files/{self.session_id}_{timestamp}.wav"
        
        audio_data = np.concatenate(self.audio_chunks)
        
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(audio_data.tobytes())
        
        self.audio_chunks = []
        return filename

    async def transcribe_audio(self, audio_file: str) -> str:
        async with aiohttp.ClientSession() as session:
            form_data = aiohttp.FormData()
            form_data.add_field('audio_file', open(audio_file, 'rb'))
            form_data.add_field('config_id', self.config_id)
            
            async with session.post(self.STT_SERVICE_URL, data=form_data) as response:
                transcript_result = await response.json()
                return transcript_result['text']

    async def get_chat_response(self, text: str) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.CHAT_SERVICE_URL,
                json={
                    "text": text, 
                    "session_id": self.session_id, 
                    "config_id": self.config_id
                }
            ) as response:
                chat_result = await response.json()
                return chat_result['response']

    async def cleanup(self):
        """Clean up any open connections and resources"""
        if self.tts_websocket:
            try:
                await self.tts_websocket.close()
            except Exception as e:
                print(f"Error closing TTS websocket: {e}")
            self.tts_websocket = None

    async def stream_tts(self, text: str, websocket) -> None:
        # Close any existing TTS connection
        await self.cleanup()
        
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(self.TTS_SERVICE_URL) as ws:
                self.tts_websocket = ws  # Store the websocket connection
                await ws.send_json({
                    "text": text,
                    "config_id": self.config_id,
                    "session_id": self.session_id
                })

                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        await websocket.send_json({
                            "type": "audio_stream",
                            "data": data,
                            "session_id": self.session_id
                        })
                        
                        if data.get("type") == "stream_end":
                            await websocket.send_json({
                                "type": "audio_stream_end",
                                "session_id": self.session_id
                            })
                            break
                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break 