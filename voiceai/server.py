from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, APIRouter, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

import json
from typing import Dict, Optional, Union, List, Any
from voiceai.config.agents_config import agent_manager
from dotenv import load_dotenv
import os
import numpy as np
import asyncio
import queue  # Add this import for simple queues

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
        
         # Processing queue for speech segments
        self.speech_queue = queue.Queue()
        # Outbound queue for messages to the client
        self.outbound_queue = queue.Queue()
        
        # Initialize Opus streams if available
        self.opus_stream_inbound = None
        self.opus_stream_outbound = None
        if OPUS_AVAILABLE:
            self.opus_stream_inbound = sphn.OpusStreamReader(24000) 
            self.opus_stream_outbound = sphn.OpusStreamWriter(24000)
        
        # Tasks for the parallel loops
        self.receive_task = None
        self.process_task = None
        self.send_task = None
        
        # Control flags
        self.running = False

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
            
            # Mark session as not running to stop all loops
            session.running = False
            
            # Cancel all tasks with proper error handling
            for task_name, task in [
                ("receive", session.receive_task), 
                ("process", session.process_task), 
                ("send", session.send_task)
            ]:
                if task and not task.done():
                    print(f"Cancelling {task_name} task for session {session_id}")
                    task.cancel()
                    try:
                        # Wait for the task to be cancelled with a timeout
                        await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
                    except asyncio.TimeoutError:
                        print(f"Timeout waiting for {task_name} task to cancel")
                    except asyncio.CancelledError:
                        print(f"{task_name.capitalize()} task cancelled successfully")
                    except Exception as e:
                        print(f"Error cancelling {task_name} task: {e}")
            
            # Clean up processor resources
            if session.processor:
                try:
                    await session.processor.cleanup()
                except Exception as e:
                    print(f"Error cleaning up processor: {e}")
            
            # Clean up Opus streams if needed
            if OPUS_AVAILABLE:
                if session.opus_stream_inbound:
                    session.opus_stream_inbound = None
                if session.opus_stream_outbound:
                    session.opus_stream_outbound = None
            
            # Remove the session
            del self.active_sessions[session_id]
            print(f"Session {session_id} successfully removed")

manager = ConnectionManager()

async def receive_loop(session: AudioSession, websocket: WebSocket):
    """Loop that receives audio data from the client"""
    try:
        print(f"Starting receive loop for session {session.session_id}")
        while session.running:
            try:
                message = await websocket.receive()
                
                if message["type"] == "websocket.disconnect":
                    print(f"Received disconnect message in receive loop for session {session.session_id}")
                    session.running = False
                    break
                    
                if message["type"] == "websocket.receive":
                    if "bytes" in message:
                        audio_data = message["bytes"]
                        
                        # For both Opus and PCM, just pass the raw data to the processor
                        # The processor will handle it appropriately based on the format
                        await session.processor.receive_raw_audio(audio_data)
                    
                    elif "text" in message:
                        try:
                            data = json.loads(message["text"])
                            if data["type"] in ["speech_start", "speech_end"]:
                                await session.processor.handle_client_message(data["type"])
                            elif data["type"] == "transcript":
                                await session.processor.receive_transcript(data["text"])
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON in receive loop: {e}")
            except asyncio.TimeoutError:
                # This is expected, just continue the loop
                continue
            except WebSocketDisconnect:
                print(f"WebSocket disconnected in receive loop for session {session.session_id}")
                session.running = False
                break
            except Exception as e:
                print(f"Error in receive loop message handling: {e}")
                # Continue the loop instead of breaking to maintain connection
                # unless it's a critical error
                if isinstance(e, (RuntimeError, ConnectionError)):
                    print("Critical connection error, breaking receive loop")
                    session.running = False
                    break
    except asyncio.CancelledError:
        print(f"Receive loop cancelled for session {session.session_id}")
    except Exception as e:
        print(f"Unhandled error in receive loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"Receive loop ended for session {session.session_id}")
        session.running = False

async def process_loop(session: AudioSession):
    """Loop that processes audio data and generates responses"""
    try:
        while session.running:
            await asyncio.sleep(0.02)

            if not session.speech_queue.empty():
                await session.processor.process_pending()
            
    except asyncio.CancelledError:
        print(f"Process loop cancelled for session {session.session_id}")
    except Exception as e:
        print(f"Error in process loop: {e}")
        import traceback
        traceback.print_exc()
        session.running = False

async def send_loop(session: AudioSession, websocket: WebSocket):
    """Loop that sends responses back to the client"""
    try:
        while session.running:
            await asyncio.sleep(0.02)
            try:
                # Get the next message to send without blocking
                if not session.outbound_queue.empty():
                    message = session.outbound_queue.get(block=False)
                    if message:
                        if isinstance(message, dict):
                            # Send JSON message
                            await websocket.send_json(message)
                        elif isinstance(message, bytes):
                            # Send binary data
                            await websocket.send_bytes(message)
                        else:
                            print(f"Unknown message type: {type(message)}")
                        
                        session.outbound_queue.task_done()
            except Exception as e:
                print(f"Error processing message in send loop: {e}")
                if isinstance(e, (RuntimeError, ConnectionError)):
                    print("Critical connection error, breaking send loop")
                    session.running = False
                    break
    except asyncio.CancelledError:
        print(f"Send loop cancelled for session {session.session_id}")
    except Exception as e:
        print(f"Error in send loop: {e}")
        session.running = False

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
        
        # Create appropriate processor
        from voiceai.processor import AudioProcessor
        from voiceai.fastprocessor import FastProcessor
        ProcessorClass = FastProcessor if fast_mode else AudioProcessor
        session.processor = ProcessorClass(session_id, session, config["configId"], session.allow_interruptions)
        
        # Start all tasks in parallel
        session.running = True
        tasks = [
            asyncio.create_task(receive_loop(session, websocket), name="receive"),
            asyncio.create_task(process_loop(session), name="process"),
            asyncio.create_task(send_loop(session, websocket), name="send")
        ]
        
        # Store tasks in session
        session.receive_task, session.process_task, session.send_task = tasks
        
        # Send ready message
        await websocket.send_json({
            "type": "call_ready",
            "session_id": session_id
        })

        print(f"Received configId: {session.config_id}")
        
        await asyncio.gather(*tasks, return_exceptions=True)
            
    except WebSocketDisconnect:
        print(f"Client disconnected: {session_id}")
    except Exception as e:
        print(f"Error in websocket connection: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure the session is marked as not running
        if session_id in manager.active_sessions:
            manager.active_sessions[session_id].running = False
        
        # Cancel all tasks
        if hasattr(session, 'receive_task'):
            for task in [session.receive_task, session.process_task, session.send_task]:
                if task and not task.done():
                    task.cancel()
        
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

        
