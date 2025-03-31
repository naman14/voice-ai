import wave
import numpy as np
import json
import os
from datetime import datetime
from typing import Optional, List, Any, Dict
import asyncio
import queue
from voiceai.utils.speechdetector import AudioSpeechDetector
from voiceai.utils.debug_utils import save_audio_chunks
from voiceai.config.agents_config import agent_manager
import time
import io
from voiceai.utils.metrics import metrics_manager
from voiceai.server import AudioSession
import base64
from scipy import signal

class FastProcessor:
    def __init__(self, session_id: str, session: AudioSession, config_id: str, allow_interruptions: bool = False):
        self.session_id = session_id
        self.config_id = config_id
        self.audio_chunks: List[np.ndarray] = []
        self.is_speaking = False
        self.is_responding = False
        self.current_turn_id = 0
        self.current_metrics = None
        self.use_vad = session.use_vad
        self.metrics = []
        self.allow_interruptions = allow_interruptions

        self.session = session

        # Add interruption handling
        self.current_task = None
        self.should_interrupt = False
    
        # For non-VAD mode, we need to store audio chunks
        self.current_audio_chunks = []

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
        
        if self.use_vad:
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

    async def receive_raw_audio(self, binary_data: bytes) -> None:
        """Process incoming raw audio data"""
        if len(binary_data) == 0:
            return
        
        session = self.session
        
        # Handle Opus decoding if needed
        if session.audio_format == "opus" and session.opus_stream_inbound:
            # Append bytes to the Opus stream
            session.opus_stream_inbound.append_bytes(binary_data)
            
            # Read decoded PCM data
            pcm_data = session.opus_stream_inbound.read_pcm()
            if pcm_data is not None and len(pcm_data) > 0:
                # Convert to int16
                pcm_int16 = (pcm_data * 32767).astype(np.int16)
                
                # Resample from 24kHz to 16kHz
                resampled_audio = signal.resample_poly(pcm_int16, 2, 3)  # 24000 * (2/3) = 16000
                
                # Convert back to bytes
                pcm_bytes = resampled_audio.astype(np.int16).tobytes()
                
                # Process the PCM data
                await self.receive_audio_data(pcm_bytes)
        else:
            # For PCM, pass directly to the audio data processor
            await self.receive_audio_data(binary_data)

    async def receive_audio_data(self, binary_data: bytes) -> None:
        """Process incoming audio data and add to speech detector"""
        if len(binary_data) > 0:
            audio_data = np.frombuffer(binary_data, dtype=np.int16)
            
            if self.use_vad:
                # Use VAD to detect speech segments
                detection_result = self.speech_detector.add_audio_chunk(audio_data)
                
                if detection_result['action'] == 'process':
                    print("New speech segment detected")
                    
                    audio_chunks = detection_result.get('audio_chunks', [])
                    
                    # Put the detected speech in the queue for processing
                    self.session.input_speech_queue.put(audio_chunks)
            else:
                # In non-VAD mode, just collect audio chunks when speaking
                if self.is_speaking:
                    self.current_audio_chunks.append(audio_data)

    async def receive_transcript(self, text: str) -> None:
        """Handle direct transcript input"""
        if text.strip():
            self.current_turn_id += 1
            self.current_metrics = metrics_manager.create_metrics(self.session_id, self.current_turn_id)
            self.current_metrics.transcription_start_time = time.time()
            self.current_metrics.transcription_end_time = time.time()
            self.metrics.append(self.current_metrics)
            
            # Put the transcript in the queue for processing
            self.session.input_text_queue.put(text)

    async def process_pending(self) -> None:
        """Process any pending speech segments"""
        print("Processing pending speech segments")
        try:
            # Get the next speech segment to process
            audio_chunks = self.session.input_speech_queue.get(block=False)
        except queue.Empty:
            audio_chunks = None
        try:
            transcript = self.session.input_text_queue.get(block=False)
        except queue.Empty:
            transcript = None
        
        if not audio_chunks and not transcript:
            return
        
        # Notify client that a new conversation turn has started
        if self.allow_interruptions and self.is_responding:
            print("Interruption allowed, sending new conversation turn")
            await self.send_message({
                "type": "new_conversation_turn",
                "session_id": self.session_id
            })
            
            # if interruptions allowed, we should interrupt the current task and cancel it
            self.should_interrupt = True
            if self.current_task and not self.current_task.done():
                print(f"Cancelling existing task for session {self.session_id}")
                self.current_task.cancel()
                try:
                    await asyncio.wait_for(self.current_task, timeout=0.5)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
                
        else:
            # if interruptions not allowed, we should ignore the new audio chunks if we are currently responding
            if self.is_responding:
                return
        
        processing_id = time.time()  # Use timestamp as unique ID
        self.current_processing_id = processing_id
        self.is_responding = True
        self.should_interrupt = False
        
        # Create a new task for the thread processing
        self.current_task = asyncio.create_task(
            asyncio.to_thread(
                self._process_input_in_thread, 
                audio_chunks, 
                transcript,
                processing_id
            ),
            name=f"process_thread_{self.session_id}"
        )
        
        # Mark the queue task as done
        if audio_chunks:
            self.session.input_speech_queue.task_done()
        elif transcript:
            self.session.input_text_queue.task_done()

    def _process_input_in_thread(self, audio_chunks: List[np.ndarray] | None, transcript: str | None, processing_id: float) -> None:
        """Process speech in a separate thread to avoid blocking the event loop"""
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Check if we should interrupt before starting
            if self.should_interrupt or self.current_processing_id != processing_id:
                print("Thread interrupted before processing started")
                return
            
            # Run the async process_speech method in this thread's event loop
            if audio_chunks:
                loop.run_until_complete(self.process_speech(audio_chunks, processing_id))
            elif transcript:
                loop.run_until_complete(self.process_text(transcript, processing_id))
        except Exception as e:
            print(f"Error in thread processing: {e}")
            import traceback
            traceback.print_exc()
            
            # Send error message if not interrupted
            if not self.should_interrupt and self.current_processing_id == processing_id:
                loop.run_until_complete(self.send_error(str(e)))
        finally:
            # Ensure the responding flag is reset if this is still the current task
            # This check prevents a cancelled task from resetting the flag for a new task
            if not self.should_interrupt and self.current_processing_id == processing_id:
                self.is_responding = False
            loop.close()

    async def process_speech(self, audio_chunks: List[np.ndarray], processing_id: float) -> None:
        """Handle speech processing and response generation"""
        print("Processing speech...")
        self.current_turn_id += 1
        self.current_metrics = metrics_manager.create_metrics(self.session_id, self.current_turn_id)
        self.current_metrics.silence_detected_time = time.time()
        self.metrics.append(self.current_metrics)
        
        try:
            # Convert audio chunks list to a single numpy array
            if audio_chunks:
                combined_audio = np.concatenate(audio_chunks)
                self.current_metrics.transcription_start_time = time.time()
                
                # Check for interruption
                if self.should_interrupt or self.current_processing_id != processing_id:
                    print("Speech processing interrupted before transcription")
                    return

                # Pass the numpy array directly to STT service
                transcript = await self.stt_service.transcribe(combined_audio, self.language)
                self.current_metrics.transcription_end_time = time.time()

                # Check for interruption again
                if self.should_interrupt or self.current_processing_id != processing_id:
                    print("Speech processing interrupted after transcription")
                    return

                if not transcript.strip() or "thank you" in transcript.lower():
                    return
                    
                # Check for interruption again
                if self.should_interrupt or self.current_processing_id != processing_id:
                    print("Speech processing interrupted before text processing")
                    return
                    
                await self.process_text(transcript, processing_id=processing_id)
            else:
                print("No audio chunks to process")

        except asyncio.CancelledError:
            print("Speech processing was cancelled")
        except Exception as e:
            print(f"Error processing audio: {e}")
            if not self.should_interrupt and self.current_processing_id == processing_id:
                await self.send_error(str(e))

    async def process_text(self, text: str, processing_id: float = None) -> None:
        """Process text input and generate response"""
        print(f"Processing text: {text}")
        self.current_metrics.llm_start_time = time.time()
        
        # If no processing_id was provided, use the current one
        if processing_id is None:
            processing_id = self.current_processing_id
        
        try:
            data = {    
                "text": text,
                "session_id": self.session_id,
                "config_id": self.config_id
            }
            
            # Check for interruption before starting
            if self.should_interrupt or self.current_processing_id != processing_id:
                print("Text processing interrupted before starting")
                return
            
            if self.chat_tts_stream:
                # Buffer to accumulate text
                previous_text = ""
                current_sentence = ""
                is_first_chunk = False
                sentence_index = 0
                
                async for chunk in self.chat_service.generate_stream(data):
                    # Check for interruption during streaming
                    if self.should_interrupt or self.current_processing_id != processing_id:
                        print("Text processing interrupted during streaming")
                        break
                        
                    if not is_first_chunk:
                        is_first_chunk = True
                        self.current_metrics.llm_first_chunk_time = time.time()
                        
                    new_content = chunk[len(previous_text):]
                    current_sentence += new_content
                    previous_text = chunk
                    
                    # Check if we have a complete sentence
                    if any(current_sentence.rstrip().endswith(p) for p in ['.', '!', '?', '|']):
                        self.current_metrics.llm_first_sentence_time = time.time()
                        # Process sentence directly instead of creating a task
                        if not self.should_interrupt and self.current_processing_id == processing_id:
                            await self.stream_tts(current_sentence, sentence_index=sentence_index, processing_id=processing_id)
                        current_sentence = ""
                        sentence_index += 1

                # Process any remaining text
                if current_sentence.strip() and not self.should_interrupt and self.current_processing_id == processing_id:
                    await self.stream_tts(current_sentence, processing_id=processing_id)
                
                self.current_metrics.llm_end_time = time.time()

            else:
                # Check for interruption before generating response
                if self.should_interrupt or self.current_processing_id != processing_id:
                    print("Text processing interrupted before response generation")
                    return
                    
                chat_response = await self.chat_service.generate_response(data)
                self.current_metrics.llm_end_time = time.time()
                
                # Check for interruption after generating response
                if self.should_interrupt or self.current_processing_id != processing_id:
                    print("Text processing interrupted after response generation")
                    return
                
                # Process TTS directly instead of creating a task
                await self.stream_tts(chat_response, processing_id=processing_id)
            
        except Exception as e:
            if not self.should_interrupt and self.current_processing_id == processing_id:
                self.is_responding = False
                print(f"Error processing text: {e}")
                await self.send_error(str(e))

    async def stream_tts(self, text: str, sentence_index: int = 0, processing_id: float = None) -> None:
        """Stream TTS audio directly"""
        # If no processing_id was provided, use the current one
        if processing_id is None:
            processing_id = self.current_processing_id
        
        try:
            # Check for interruption before starting
            if self.should_interrupt or self.current_processing_id != processing_id:
                print("TTS streaming interrupted before starting")
                return
            
            self.current_metrics.tts_start_time = time.time()
            voice_id = self.tts_service.get_voice_id(self.config_id)
            is_first_chunk = False
            
            async for chunk in await self.tts_service.generate_speech_stream(text, self.language, voice_id, self.voice_samples, self.speed):
                # Check for interruption during streaming
                if self.should_interrupt or self.current_processing_id != processing_id:
                    print("TTS streaming interrupted during generation")
                    break

                if not is_first_chunk and sentence_index == 0:
                    self.current_metrics.tts_first_chunk_time = time.time()
                    is_first_chunk = True
                    self.current_metrics.log_metrics()

                await self.send_audio_chunk(chunk.chunk, chunk.format, chunk.sample_rate)
                
            # Only send end message if not interrupted
            if not self.should_interrupt and self.current_processing_id == processing_id:
                await self.send_message({
                    "type": "tts_stream_end",
                    "session_id": self.session_id
                })
        except asyncio.CancelledError:
            print("TTS streaming was cancelled")
        except Exception as e:
            if not self.should_interrupt and self.current_processing_id == processing_id:
                print(f"Error in TTS streaming: {e}")
                await self.send_error(str(e))

    async def handle_client_message(self, message_type: str) -> None:
        """Handle client control messages"""
        if message_type == "speech_start":
            self.is_speaking = True
            # Clear audio chunks when starting a new speech segment in non-VAD mode
            if not self.use_vad:
                self.current_audio_chunks = []
        elif message_type == "speech_end":
            self.is_speaking = False
            
            # In non-VAD mode, process collected audio chunks when speech ends
            if not self.use_vad and self.current_audio_chunks:
                print("Processing collected audio chunks in non-VAD mode")
                # Put the collected chunks in the queue for processing
                self.session.input_speech_queue.put(self.current_audio_chunks)
                # Clear the chunks after queuing
                self.current_audio_chunks = []

    async def send_message(self, message: Any) -> None:
        """Send a message to the client via the outbound queue"""
        if hasattr(self.session, 'outbound_queue'):
            try:
                self.session.outbound_queue.put(message)
            except Exception as e:
                print(f"Error sending message to queue: {e}")

    async def send_error(self, error: str) -> None:
        """Send error message to client"""
        await self.send_message({
            "type": "error",
            "error": error,
            "session_id": self.session_id
        })

    async def cleanup(self):
        """Clean up any resources"""
        self.should_interrupt = True
        if self.current_task and not self.current_task.done():
            self.current_task.cancel()
            try:
                await self.current_task
            except asyncio.CancelledError:
                pass

    async def send_audio_chunk(self, chunk, format, sample_rate):
        """Send audio chunk to the client"""
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
                
                if opus_data:
                    # Send as binary WebSocket message via the outbound queue
                    await self.send_message(opus_data)
                
            except Exception as e:
                print(f"Error in Opus processing: {str(e)}")
        else:
            # PCM handling
            data = {
                "type": "tts_stream",
                "data": {
                    "type": "audio_chunk",
                    "chunk": base64.b64encode(chunk).decode("utf-8"),
                    "format": format,
                    "sample_rate": sample_rate,
                    "timestamp": time.time()
                },
                "session_id": self.session_id
            }
            await self.send_message(data)