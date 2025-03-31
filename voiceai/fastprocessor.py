import wave
import numpy as np
import json
import os
from datetime import datetime
from typing import Optional, List
import asyncio
from fastapi import WebSocket
from voiceai.utils.speechdetector import AudioSpeechDetector
from voiceai.utils.debug_utils import save_audio_chunks
from voiceai.config.agents_config import agent_manager
import time
import io
from voiceai.utils.metrics import metrics_manager
from voiceai.server import AudioSession
import base64

class FastProcessor:
    def __init__(self, session_id: str, session: AudioSession, config_id: str, allow_interruptions: bool = False):
        self.session_id = session_id
        self.config_id = config_id
        self.audio_chunks: List[np.ndarray] = []
        self.is_speaking = False
        self.is_responding = False
        self.tts_lock = asyncio.Lock()
        self.current_turn_id = 0
        self.current_metrics = None
        self.metrics = []
        self.allow_interruptions = allow_interruptions

        self.session = session

        # Add interruption handling
        self.current_task = None
        self.should_interrupt = False

        # Lazy import services only when FastProcessor is actually used
        from voiceai.stt.stt import stt_instance
        from voiceai.chat.chat import chat_instance
        from voiceai.tts.tts import tts_instance
        
        # Initialize services directly
        self.stt_service = stt_instance
        self.chat_service = chat_instance
        self.tts_service = tts_instance
        self.config = agent_manager.get_agent_config(config_id)
        self.voice_samples = self.config.voice_samples
        self.language = self.config.language
        self.speed = self.config.speed
        self.system_prompt = self.config.system_prompt

        self.chat_tts_stream = False
        
        self.speech_detector = AudioSpeechDetector(
            sample_rate=16000,
            energy_threshold=0.15,
            min_speech_duration=0.4,
            max_silence_duration=0.5,
            max_recording_duration=10.0,
            debug=False
        )

        # create metrics
        self.current_metrics = metrics_manager.create_metrics(self.session_id, self.current_turn_id)


    async def process_audio_chunk(self, binary_data: bytes, websocket: WebSocket) -> None:
        """Process incoming audio chunks and handle VAD internally"""
        if len(binary_data) > 0:
            audio_data = np.frombuffer(binary_data, dtype=np.int16)
            detection_result = self.speech_detector.add_audio_chunk(audio_data)
            
            if detection_result['action'] == 'process':
                print("New conversation turn started")
                
                self.audio_chunks = detection_result.get('audio_chunks', [])

                # Debug: Save audio chunks to file
                # save_audio_chunks(self.audio_chunks, self.session_id, self.current_turn_id)
                
                # notify client that a new conversation turn has started
                # useful for client to discard any existing queued audio chunks from previous turn

                if self.allow_interruptions:
                    await websocket.send_json({
                        "type": "new_conversation_turn",
                        "session_id": self.session_id
                    })
                    # This is where we've detected a complete speech segment
                    # If we're currently responding, we should interrupt
                    if self.is_responding:
                        print("Interrupting current response...")
                        self.should_interrupt = True
                        if self.current_task and not self.current_task.done():
                            self.current_task.cancel()

                # Create new task for processing
                self.should_interrupt = False
                self.current_task = asyncio.create_task(self.process_speech(websocket))

    async def process_speech(self, websocket: WebSocket) -> None:
        """Handle speech processing and response generation"""
        print("Processing speech...")
        self.current_turn_id += 1
        self.current_metrics = metrics_manager.create_metrics(self.session_id, self.current_turn_id)
        self.current_metrics.silence_detected_time = time.time()
        self.metrics.append(self.current_metrics)
        self.is_responding = True
        try:
            # Convert audio chunks list to a single numpy array
            if self.audio_chunks:
                combined_audio = np.concatenate(self.audio_chunks)
                self.audio_chunks = []
                self.current_metrics.transcription_start_time = time.time()
                
                # Check for interruption
                if self.should_interrupt:
                    self.is_responding = False
                    return

                # Pass the numpy array directly to STT service
                transcript = await self.stt_service.transcribe(combined_audio, self.language)
                self.current_metrics.transcription_end_time = time.time()

                if not transcript.strip() or "thank you" in transcript.lower():
                    self.is_responding = False
                    return
                    
                await self.process_text(transcript, websocket)
            else:
                self.is_responding = False

        except asyncio.CancelledError:
            self.is_responding = False
            print("Speech processing was interrupted")
        except Exception as e:
            self.is_responding = False
            print(f"Error processing audio: {e}")
            await self.send_error(websocket, str(e))

    async def process_text(self, text: str, websocket: WebSocket) -> None:
        """Process text input and generate response"""
        print(f"Processing text: {text}")
        async with self.tts_lock:
            self.is_responding = True
            self.current_metrics.llm_start_time = time.time()
            try:
                data = {    
                    "text": text,
                    "session_id": self.session_id,
                    "config_id": self.config_id
                }
                
                if self.chat_tts_stream:
                    # Buffer to accumulate text
                    previous_text = ""
                    current_sentence = ""
                    is_first_chunk = False
                    sentence_index = 0
                    # Stream chat response and process sentences as they come
                    async for chunk in self.chat_service.generate_stream(data):
                        # Get only the new content by removing the previous text
                        if not is_first_chunk:
                            is_first_chunk = True
                            self.current_metrics.llm_first_chunk_time = time.time()
                            
                        new_content = chunk[len(previous_text):]
                        current_sentence += new_content
                        previous_text = chunk
                        
                        # Check if we have a complete sentence
                        if any(current_sentence.rstrip().endswith(p) for p in ['.', '!', '?', '|']):
                            self.current_metrics.llm_first_sentence_time = time.time()
                            # Process complete sentence with TTS
                            await self.stream_tts(current_sentence, websocket, sentence_index=sentence_index)
                            current_sentence = ""
                            sentence_index += 1
        
                    # Process any remaining text
                    if current_sentence.strip():
                        await self.stream_tts(current_sentence, websocket)
                    
                    self.current_metrics.llm_end_time = time.time()

                else:
                    chat_response = await self.chat_service.generate_response(data)
                    self.current_metrics.llm_end_time = time.time()
                    await self.stream_tts(chat_response, websocket)
                
            except Exception as e:
                self.is_responding = False
                print(f"Error processing text: {e}")
                await self.send_error(websocket, str(e))

    async def stream_tts(self, text: str, websocket: WebSocket, sentence_index: int = 0) -> None:
        """Stream TTS audio directly"""
        try:
            self.current_metrics.tts_start_time = time.time()
            voice_id = self.tts_service.get_voice_id(self.config_id)
            is_first_chunk = False
            async for chunk in await self.tts_service.generate_speech_stream(text, self.language, voice_id, self.voice_samples, self.speed):
                # Check for interruption
                if self.should_interrupt:
                    break

                if not is_first_chunk and sentence_index == 0:
                    self.current_metrics.tts_first_chunk_time = time.time()
                    is_first_chunk = True
                    self.current_metrics.log_metrics()

                await self.send_audio_chunk(websocket, chunk.chunk, chunk.format, chunk.sample_rate)
                
            if not self.should_interrupt:
                await websocket.send_json({
                    "type": "tts_stream_end",
                    "session_id": self.session_id
                })
        except asyncio.CancelledError:
            print("TTS streaming was interrupted")
        except Exception as e:
            print(f"Error in TTS streaming: {e}")
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

    async def cleanup(self):
        """Clean up any resources"""
        pass  # No cleanup needed for direct calls 


    async def send_audio_chunk(self, websocket, chunk, format, sample_rate):
        session = self.session
        if not session:
            print("No session found, returning")
            return

        if session.audio_format == "opus" and session.opus_stream_outbound:
            try:
                # Convert to float32 numpy array if needed
                if not isinstance(chunk, np.ndarray):
                    chunk = np.frombuffer(chunk, dtype=np.float32)
                
                # Ensure the chunk is in the correct format for Opus encoding
                if chunk.dtype != np.float32:
                    chunk = chunk.astype(np.float32) / 32768.0  # Convert from int16 to float32
                
                # Process the audio in 80ms frames (1920 samples at 24kHz)
                FRAME_SIZE = 1920  # 80ms at 24kHz
                
                for i in range(0, len(chunk), FRAME_SIZE):
                    frame = chunk[i:i + FRAME_SIZE]
                    
                    # Pad the last frame if necessary
                    if len(frame) < FRAME_SIZE:
                        frame = np.pad(frame, (0, FRAME_SIZE - len(frame)), 'constant')
                    
                    # Append PCM frame to the Opus stream
                    session.opus_stream_outbound.append_pcm(frame)
                
                # Read encoded Opus data
                opus_data = session.opus_stream_outbound.read_bytes()
                print(f"Opus data size: {len(opus_data) if opus_data else 0} bytes")
                
                if opus_data:
                    # Send as binary WebSocket message
                    print("Sending Opus data to client")
                    await websocket.send_bytes(opus_data)
                
            except Exception as e:
                print(f"Error in Opus processing: {str(e)}")
        else:
            # Original PCM handling
            data = {
                "type": "audio_chunk",
                "chunk": base64.b64encode(chunk).decode("utf-8"),
                "format": format,
                "sample_rate": sample_rate,
                "timestamp": time.time()
            }
            await websocket.send_json({
                "type": "tts_stream",
                "data": data,
                "session_id": self.session_id
            })