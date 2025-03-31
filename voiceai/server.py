from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, APIRouter, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

import json
from typing import Dict, Optional, Union
from voiceai.config.agents_config import agent_manager
from dotenv import load_dotenv
import os
import numpy as np
from scipy import signal

# Add imports for Opus codec handling using sphn
try:
    import sphn
    OPUS_AVAILABLE = True
except ImportError:
    OPUS_AVAILABLE = False

load_dotenv()

fast_mode = os.getenv("FAST_MODE", "False").lower() == "true"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AudioSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.config_id = ""
        self.websocket: Optional[WebSocket] = None
        self.processor = None
        self.allow_interruptions = False
        self.audio_format = "pcm"  # Default to PCM
        
        # Initialize Opus streams if available
        self.opus_stream_inbound = None
        self.opus_stream_outbound = None
        if OPUS_AVAILABLE:
            self.opus_stream_inbound = sphn.OpusStreamReader(24000) 
            self.opus_stream_outbound = sphn.OpusStreamWriter(24000)

class ConnectionManager:
    def __init__(self):
        self.active_sessions: Dict[str, AudioSession] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = AudioSession(session_id)
        self.active_sessions[session_id].websocket = websocket
        return self.active_sessions[session_id]

    async def disconnect(self, session_id: str):
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            if session.processor:
                await session.processor.cleanup()
            del self.active_sessions[session_id]

manager = ConnectionManager()

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    session = await manager.connect(websocket, session_id)
    print(f"Client connected: {session_id}")

    try:
        # Handle initial configuration
        data = await websocket.receive_text()
        config = json.loads(data)

        if not isinstance(config, dict):
            print("Error: Config is not a dictionary")
            return
            
        required_fields = ["configId"]
        if missing_fields := [field for field in required_fields if field not in config]:
            print(f"Error: Missing required fields in config: {missing_fields}")
            return

        if "allowInterruptions" in config and config["allowInterruptions"]:
            session.allow_interruptions = True
            
        # Set audio format if specified
        if "format" in config and config["format"] == "opus":
            if OPUS_AVAILABLE:
                session.audio_format = "opus"
                print(f"Using Opus codec for session {session_id}")
            else:
                print("Opus requested but not available, falling back to PCM")
                session.audio_format = "pcm"
        else:
            session.audio_format = "pcm"
            print(f"Using PCM format for session {session_id}")

        await agent_manager.add_agent_config(config)
        
        session.config_id = config["configId"]
        
        # Create appropriate processor based on mode
        from voiceai.processor import AudioProcessor
        from voiceai.fastprocessor import FastProcessor
        ProcessorClass = FastProcessor if fast_mode else AudioProcessor
        session.processor = ProcessorClass(session_id, session, config["configId"], session.allow_interruptions)

        await websocket.send_json({
            "type": "call_ready",
            "session_id": session_id
        })

        print(f"Received configId: {session.config_id}")
        
        # Main message handling loop
        while True:
            message = await websocket.receive()
            
            if message["type"] == "websocket.disconnect":
                break
                
            if message["type"] == "websocket.receive":
                if "bytes" in message:
                    try:
                        audio_data = message["bytes"]
                        
                        # Handle Opus decoding if needed
                        if session.audio_format == "opus" and session.opus_stream_inbound:
                            # Append bytes to the Opus stream
                            session.opus_stream_inbound.append_bytes(audio_data)
                            
                            # Read decoded PCM data (24kHz)
                            pcm_data = session.opus_stream_inbound.read_pcm()
                            if pcm_data is not None and len(pcm_data) > 0:
                                # Convert to int16
                                pcm_int16 = (pcm_data * 32767).astype(np.int16)
                                
                                # we receive 24Khz audio when using opus
                                # Resample from 24kHz to 16kHz
                                resampled_audio = signal.resample_poly(pcm_int16, 2, 3)  # 24000 * (2/3) = 16000
                                
                                # Convert back to bytes
                                pcm_bytes = resampled_audio.astype(np.int16).tobytes()
                                await session.processor.process_audio_chunk(pcm_bytes, websocket)
                        else:
                            # Process raw PCM data
                            await session.processor.process_audio_chunk(audio_data, websocket)
                    except Exception as e:
                        print(f"Error processing audio data: {e}")
                
                elif "text" in message:
                    try:
                        data = json.loads(message["text"])
                        if data["type"] in ["speech_start", "speech_end"]:
                            await session.processor.handle_client_message(data["type"], websocket)
                        elif data["type"] == "transcript":
                            await session.processor.process_text(data["text"], websocket)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")
                        
    except WebSocketDisconnect:
        print(f"Client disconnected: {session_id}")
    except Exception as e:
        print(f"Error in websocket connection: {e}")
    finally:
        print(f"Cleaning up session: {session_id}")
        await manager.disconnect(session_id)

if fast_mode:
    # initialize all services
    print("Initializing services")

    print("Initializing Chat")
    import voiceai.chat.chat
    print("Initializing TTS")
    import voiceai.tts.tts
    print("Initializing STT")
    import voiceai.stt.stt

    stt_instance = voiceai.stt.stt.stt_instance
    chat_instance = voiceai.chat.chat.chat_instance
    tts_instance = voiceai.tts.tts.tts_instance

    print("Services initialized")

    # start individual tts and chat endpoints
    # if not in fast mode, the tts server will be started as a separate process

    from voiceai.chat.chat import chat_router
    from voiceai.tts.tts import tts_router

    app.include_router(chat_router)
    app.include_router(tts_router)

    print("Chat and TTS routers added")

# Serve voiceai.html at root
@app.get("/voiceai")
async def read_root():
    return FileResponse("voiceai.html")

        
