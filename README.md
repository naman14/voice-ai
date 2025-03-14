# voiceai.fun AI Node

A high-performance, real-time AI speech-to-speech system with voice cloning capabilities. Features modular components for speech recognition, text generation, and speech synthesis. Designed for sub-500ms latency and unrestricted conversations through self-hosted models.

## Features

- **Real-time Processing**: End-to-end latency under 500ms for natural conversations
- **Modular Architecture**: Easily swap components with custom implementations
- **Self-hosted Models**: Complete control over the conversation flow without external restrictions
- **WebSocket Streaming**: Real-time audio streaming for instant responses
- **Real-time Voice Cloning**: Clone voices from short audio samples with customizable personalities
- **Custom XTTS Model**: Finetuned XTTS_2 model optimized for certain agents. Use your own voice models with XTTS, ElevenLabs, or Cartesia

## Architecture

### Core Components

The system consists of four main services that work together to provide real-time speech-to-speech conversion:

1. **Server** (Port 8000)
   - Main WebSocket endpoint for client connections
   - Handles raw PCM audio byte streams (16-bit, 16kHz)
   - Manages session state and current personality management
   - Coordinates communication between services

2. **Speech-to-Text** (Port 8001)
   - Audio transcription service
   - Processes PCM audio chunks
   - Uses Whisper for transcription

3. **Chat Generation** (Port 8002)
   - Text response generation
   - Supports LLM models and cloud (Groq)
   - Maintains conversation context

4. **TTS Streaming** (Port 8003)
   - Voice synthesis and cloning service
   - Sub 100ms first audio chunk streaming with XTTS and DeepSpeed enabled
   - Streams raw PCM audio bytes back to client
   - Supports XTTS, ElevenLabs, and Cartesia

### Audio Processing Pipeline

```
Client                    Server                  Services
  |                         |                        |
  |-- PCM Audio Bytes -->   |                        |
  |                         |-- Speech Detection --> |
  |                         |                        |
  |                         |-- STT Processing --->  |
  |                         |                        |
  |                         |-- Chat Generation -->  |
  |                         |                        |
  |                         |<-- TTS Streaming ---   |
  | <-- PCM Audio Bytes --  |                        |
```

### Audio Format Specifications

- **Input Audio**: Raw PCM bytes
  - Sample Rate: 16kHz
  - Bit Depth: 16-bit
  - Channels: Mono
  - No headers or containers

- **Output Audio**: Raw PCM bytes
  - Sample Rate: 24kHz
  - Bit Depth: 32-bit float
  - Channels: Mono
  - Streamed in chunks for real-time playback

## Quick Start

### Prerequisites

- Python 3.11.11
- CUDA-compatible GPU
- Required models:
  - Whisper (small.en)
  - Any conversation model from huggingface (default Llama 8B)
  - XTTS-v2 or API keys for ElevenLabs/Cartesia

### Installation

```bash
# Clone the repository
git clone 'https://github.com/voiceai-fun/voiceai-fun-ai-node'
cd voiceai-fun-ai-node

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys and model paths
```

Download the [XTTS-v2](https://huggingface.co/coqui/XTTS-v2) model for tts and update env path

### Running the Services

```bash
# Start all services with combined logging
python launcher.py --combined-logs

# Start specific services
python launcher.py --services server chat tts
```

## Component Customization

### Speech-to-Text Options

- Default: Whisper (uses faster-whisper for sub 100ms transcription)
- Can be changed with any other model

### Chat Model Options

- Default: Self hosted Llama 70B (`ENABLE_LOCAL_MODEL=True`)
- Change with any other open source model
- Cloud:
    - Supports Groq for much faster chat inferences

### Text-to-Speech Options

1. XTTS-V2 (Default)
   - Pre-trained model optimized for voice cloning
   - Real-time streaming and voice adaptation
   - Low latency (100-200ms)
   - [Download Model](https://github.com/voiceai-fun/voiceai-fun-xtts)

3. ElevenLabs
   - Set `ELEVENLABS_API_KEY`

4. Cartesia
   - Set `CARTESIA_API_KEY`

## API Specification

### Main WebSocket Endpoint

```
ws://localhost:8000/ws/{session_id}
```

Initial connection message:
```json
{
  "agentId": "string",
  "userId": "string",
  "agentName": "string"
}
```

Send audio bytes from client
```
ws.send(audioBuffer);
```

Server sends audio bytes to socket
```json
{
    "type": "tts_stream",
    "data": data
}
```

End of tts stream message from server to client
```json
{
    "type": "tts_stream_end"
}
```

### Speech-to-Text API

```http
POST /transcribe
Content-Type: multipart/form-data

file: audio_file
```

### Chat API

```
Local self hosted model - POST localhost:8002/chat 
Groq - POST localhost:8002/chat/groq 
```

```http
POST /chat
Content-Type: application/json

{
  "text": "string",
  "session_id": "string",
  "personality": "string"
}
```

### Text-to-Speech WebSocket Endpoints

```
ws://localhost:8003/tts/stream
ws://localhost:8003/tts/stream/cartesia
ws://localhost:8003/tts/stream/elevenlabs
```

Message format:
```json
{
  "text": "string",
  "personality": "string"
}
```

### Speech Detection

The system includes a highly customizable speech detector (`AudioSpeechDetector`) that manages real-time audio processing:

```python
detector = AudioSpeechDetector(
    sample_rate=16000,
    energy_threshold=0.1,      # Adjust for ambient noise
    min_speech_duration=0.3,   # Minimum speech duration
    max_silence_duration=0.4,  # Silence before processing
    max_recording_duration=10.0
)
```

Features:
- Real-time energy-based speech detection
- Configurable thresholds and timings
- Automatic silence detection
- Efficient audio chunk processing
- Built-in state management


## Performance Optimization

- Uses DeepSpeed for inference optimization
- Streaming audio chunks for minimal latency
- Efficient WebSocket connection pooling
- Caches speaker latents for faster TTS
- Optimized voice cloning pipeline for real-time adaptation
- Tested on cloud L4 GPU (24gb VRAM), with DeepSpeed - sub 500ms end to end latency

## Environment Variables

```env
XTTS_MODEL_PATH=path/to/xtts/model
ELEVENLABS_API_KEY=your_key_here
CARTESIA_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
HUGGINGFACE_API_KEY=your_key_here
```
