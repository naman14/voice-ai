# VoiceAI

A high-performance, real-time AI speech-to-speech system with voice cloning capabilities. Features modular components for speech recognition, text generation, and speech synthesis. Designed for sub-500ms latency and unrestricted conversations through self-hosted models.

## Features

- **Real-time voice chat**: End-to-end latency under 500ms for natural conversations
- **Modular Architecture**: Easily swap components with other implementations
- **Self-hosted Models**: Complete control over the conversation flow without external restrictions
- **WebSocket Streaming**: Real-time audio streaming for instant responses
- **Real-time Voice Cloning**: Clone voices from short audio samples with customizable personalities
- **Interruption Support**: Allow the user to interrupt the agent at any time and start a new turn of conversation
- **Scalable**: Designed for high throughput while also best for local deployment on self machine

## Online Demo

https://namand.in/voice-ai

## Local Voice Chat

Open `index.html` in browser to use voice chat

## Available Implementations

- **STT**:
  - Whisper Jax
  - Huggingface Whisper
  - faster-whisper
  - whisper with vllm
  - sensevoice

- **Chat**:
  - VLLM
  - Transformers
  - Groq

- **TTS**:
  - XTTS
  - ElevenLabs
  - Cartesia

- **Multi Modal**:
  - Ultravox (Combined STT and Chat in 1 model)

- **Coming Soon**:
   - Sesame csm 1B conversational speech mdoel

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU for best performance

### Installation

```bash
# Clone the repository
git clone 'https://github.com/naman14/voice-ai'
cd voice-ai

# create a virtual environment
python -m venv venv
source venv/bin/activate

# Install whisper-jax separately (if using whisper jax for stt)
pip install git+https://github.com/naman14/whisper-jax.git

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration
```

### Running the Services

```bash
# Start all services with combined logging
python voiceai/launcher.py --combined-logs

# or Start specific services
python voiceai/launcher.py --services server chat tts
```

# Environment Variables
```

// if using custom xtts model for tts, default xtts is automatically downloade if XTTS_MODEL_PATH is not set
XTTS_MODEL_PATH="path/to/xtts/model"

// if using external tts, set to true
USE_EXTERNAL_TTS=False

// if using external chat, set to true
USE_EXTERNAL_CHAT=True

// external chat provider, if any
EXTERNAL_CHAT_PROVIDER="groq"
// external tts provider, if any
EXTERNAL_TTS_PROVIDER="elevenlabs"

// all services in a single process, set to false for multi process mode where every service will have their own endpoints
FAST_MODE=True

// if using whisper jax for stt, specifices the fraction of gpu memory to preallocate
XLA_PYTHON_CLIENT_MEM_FRACTION=0.2

// if using groq for llm
GROQ_API_KEY="your_groq_api_key"

// required for downloading some models from huggingface
HUGGINGFACE_API_KEY="your_huggingface_api_key"

// if using elevenlabs for tts
ELEVENLABS_API_KEY="your_elevenlabs_api_key"

// if using cartesia for tts
CARTESIA_API_KEY="your_cartesia_api_key"

```

Open `index.html` in browser after starting the server to use voice chat

## Component Customization

### Speech-to-Text Options

- Default: `whisper_jax.py` Whisper Jax (fastest in my testing)
-  `whisper_hf.py` Huggingface Whisper implementation
-  `whisper_stt.py` faster-whisper implementation
-  `whisper_vllm.py` vllm implementation, best for high throughput requirements
-  `sensevoice_stt.py` as fast as whisper jax, though limited language support

  - base class - `stt.py` - change the implementation class here

### Chat Model Options

- Default: `vllm.py` Self hosted Llama 8B through vllm (fastest in my testing)
- `local_llm.py` Self hosted model through Huggingface Transformers
- `external/groq_llm.py` Groq chat implementation
- base class - `chat.py` - change the implementation class here
- All implementations support both sync and streaming generations

### Text-to-Speech Options (All are real time streaming)

- Default: `xtts.py` XTTS-v2  with DeepSpeed enabled 
- `external/elevenlabs_tts.py` ElevenLabs TTS
- `external/cartesia_tts.py` Cartesia TTS
- base class - `tts.py` - change the implementation class here
- All implementations support both sync and streaming generations

## Combined STT and LLM
- `combined_stt_llm.py` Combined STT and LLM implementation through `Ultravox` multimodal model
- this combines STT and Chat in 1 model and should be faster, but wasnt that much better in my testing

## Chat and TTS Interleaving
- set `self.chat_tts_stream` in `fastprocessor.py` to `True` to interleave chat and tts
- this will stream the tts response as the chat model is generating the response sentence by sentence

### Speech Detection

The system includes a customizable speech detector (`AudioSpeechDetector`) that manages real-time audio processing:

```python
detector = AudioSpeechDetector(
    sample_rate=16000,
    energy_threshold=0.1,      # Adjust for ambient noise
    min_speech_duration=0.3,   # Minimum speech duration
    max_silence_duration=0.4,  # Silence before processing
    max_recording_duration=10.0
)
```

## Multi Process Mode
By default, runs all services in a single process. (`FAST_MODE=True`)

To run in multi process mode, set `FAST_MODE=False` which will run each service in a separate process. Useful if want to host specific services on different machines, but adds small latency due to inter process communication.

### Audio Format Specifications

- **Input Audio**: Raw PCM bytes, Sample Rate: 16kHz, Bit Depth: 16-bit, Channels: Mono, No headers or containers

- **Output Audio**: Raw PCM bytes, Sample Rate: 24kHz, Bit Depth: 32-bit float, Channels: Mono, Streamed in chunks for real-time playback

## Direct Transcript for Chat + TTS
- send `transcript` message type to the server with the transcript to chat and tts to respond
- this will skip the default VAD and STT processing
- useful when client side transcription is available, e.g. we can use WebAudio on browsers to do transcription locally
- Browser based transcription is usually not that reliable though.

## Interruptions support
- set `allowInterruptions` to `True` to allow the user to interrupt the agent at any time and start a new turn of conversation
- server will send a `new_conversation_turn` message to the client to acknowledge that interruption has been detected and the new turn has started
- client should discard any existing queued audio chunks from previous turn
- this is useful for client to implement a "thinking" animation or indicator
- If allowInterruptions is true, then client should send audio bytes continuously to the server, if its off, then client should send audio bytes only when agent is not speaking

## Configuration

When connecting to the server socket, send config json with the following fields:

- `config_id`: The id of the config to use // random string, voice_samples etc will be cached in memory for this config_id
- `session_id`: The id of the session to use // conversation context is managed on a per session_id basis
- `systemPrompt`: system prompt for the chat model
- `agentName`: name of the agent
- `voiceSampleUrl`: The voice samples to use for voice cloning // can be local or remote, automatically downloads and converts to wav format if remote url
- `language`: language
- `speed`: speed of the speech
- `allowInterruptions`: allow the user to interrupt the agent at any time and start a new turn of conversation

- `cartesia_voice_id`: optional, the voice id to use for cartesia tts
- `elevenlabs_voice_id`: optional, the voice id to use for elevenlabs tts

```
 socket.send(JSON.stringify({
      agentId: "1234",
      agentName: "Donald Trump",
      configId: "7890",
      systemPrompt: "You are Donald Trump responding to a call. Provide plain text responses—no voice tone, delivery notes, or emotional cues. Engage fully with any topic while staying factual. Mimic Donald Trump’s exact speech style, including his pacing, phrasing, and mannerisms, without adding reactions like '(laughs)' or other non-verbal expressions.",
      voiceSampleUrl: "./trump.wav",
      rate: 1.0,
      language: "en",
      allowInterruptions: true
  }));
```

## Latency Measurements
- latency measurements are logged in console
- `metrics.py` and `fastprocessor.py` contains useful helpers for measuring latency of the system and tracks the latency of every system and overall latency


### Cloud L4 GPU Latency
```
Whisper Jax STT: 80ms
VLLM Local Chat Llama 3.2 3b: 500ms
TTS Time to first chunk (with DeepSpeed): 170ms
```

### Troubleshooting

- DeepSpeed - DeepSpeed requires cuDNN and cuBLAS. If you have trouble installing DeepSpeed, make sure you have the latest version of CUDA and cuDNN installed. If you still face issues, then u can skip DeepSpeed and set `use_deepspeed=False` in `local_tts.py`
- VLLM - Adjust `gpu_memory_utilization` according to your GPU memory and the model size. This tells vllm to allocate the defined fraction of the GPU memory.


