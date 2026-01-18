# Dual-Agent Streaming Architecture

## Overview

The dual-agent bot combines two LLMs running in parallel within a single conversation:

1. **Conversational Agent (Nemotron)** - Front-facing, speaks to user
   - Local llama.cpp with buffered sentence-boundary streaming
   - Magpie TTS with adaptive mode (streaming → batch)
   - ~500-700ms voice-to-voice latency
   - Optimized for low latency and natural conversation

2. **Background Agent (Gemini)** - Silent worker, executes tools/analysis
   - Cloud-based Gemini 2.5 Flash
   - Tool/function calling support
   - Runs in parallel without interrupting user conversation
   - Results fed back into shared conversation context

## Architecture

```text
User (Voice Input)
        │
        ▼
┌──────────────────────────────────┐
│    STT (NVIDIA Parakeet)         │
│   (Nemotron Speech ASR)          │
└──────────────┬───────────────────┘
               │ Transcription
        ┌──────▼──────────────────────────────┐
        │   Shared Turn Processor             │
        │   (SmartTurn V3 analyzer)           │
        │   + Shared LLMContext               │
        └──────┬──────────────────┬───────────┘
               │                  │
      ┌────────▼────────┐  ┌──────▼──────────────┐
      │  Conversational │  │   Background Agent │
      │  Pipeline       │  │   Pipeline         │
      │                 │  │                    │
      │ ┌─────────────┐ │  │ ┌────────────────┐│
      │ │ Nemotron    │ │  │ │ Gemini 2.5     ││
      │ │ Buffered    │ │  │ │ (Tool Support) ││
      │ │ LLM         │ │  │ │                ││
      │ └──────┬──────┘ │  │ └────────────────┘│
      │        │        │  │        │           │
      │ ┌──────▼──────┐ │  │ (No TTS, silent) │
      │ │ Magpie TTS  │ │  │                    │
      │ │ (Adaptive)  │ │  └────────────────────┘
      │ └──────┬──────┘ │
      └────────┼────────┘
               │ Audio (only conversational)
        ┌──────▼──────────────────┐
        │ Transport Output        │
        │ (WebRTC/Daily/Twilio)   │
        └─────────────────────────┘
```

## Key Design Principles

### 1. **Shared Context Management**

Both agents see the same conversation history:

```python
context = LLMContext(messages)

conversational_aggregator = LLMContextAggregatorPair(context)
background_aggregator = LLMContextAggregatorPair(
    context,
    user_params=LLMUserAggregatorParams(
        user_turn_strategies=ExternalUserTurnStrategies()
    ),
)
```

- Single `LLMContext` ensures both agents share message history
- Background agent uses `ExternalUserTurnStrategies` to respect shared turn management

### 2. **Shared Turn Processing**

One `UserTurnProcessor` manages turn boundaries for both agents:

```python
user_turn_processor = UserTurnProcessor(
    user_turn_strategies=UserTurnStrategies(
        stop=[TurnAnalyzerUserTurnStopStrategy(
            turn_analyzer=LocalSmartTurnAnalyzerV3()
        )]
    ),
)
```

Both agents respect the same turn detection:

- VAD silence detection (200ms default)
- SmartTurn analyzer for intelligent turn boundaries
- No race conditions or turn conflicts

### 3. **Parallel Execution**

`ParallelPipeline` runs both agents simultaneously:

```python
ParallelPipeline(
    [
        # Conversational pipeline
        conversational_aggregator.user(),
        conversational_llm,
        tts,
        transport.output(),
        conversational_aggregator.assistant(),
    ],
    [
        # Background pipeline (silent)
        background_aggregator.user(),
        background_llm,
        background_aggregator.assistant(),
    ],
)
```

- Both pipelines process the same input frames
- Conversational agent outputs audio → user hears
- Background agent processes tools/analysis → context update only

### 4. **No Blocking**

User never waits for background work:

- Conversational agent responds with low latency (~500-700ms)
- Background agent work happens asynchronously
- Tools can execute while user listens and responds
- Results added to conversation context for future turns

## Tool Support

Tools are defined in `tools.py`:

```python
GEMINI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get current date and time",
            ...
        }
    },
    # More tools...
]
```

Gemini can call these tools during background processing:

1. User says something
2. Conversational agent responds immediately (Nemotron)
3. Background agent analyzes, calls tools if needed (Gemini)
4. Tool results added to context
5. Next conversational turn incorporates tool insights

## Environment Variables

```bash
# Local inference (Nemotron)
NVIDIA_ASR_URL=ws://localhost:8080
NVIDIA_LLAMA_CPP_URL=http://localhost:8000
NVIDIA_TTS_URL=http://localhost:8001

# Cloud inference (Gemini)
GOOGLE_API_KEY=your-api-key
```

## Running the Bot

```bash
# Local development
uv run pipecat_bots/bot_dual_interleaved_streaming.py

# With Daily transport
uv run pipecat_bots/bot_dual_interleaved_streaming.py -t daily

# With Twilio transport
uv run pipecat_bots/bot_dual_interleaved_streaming.py -t twilio

# Enable audio recording
ENABLE_RECORDING=true uv run pipecat_bots/bot_dual_interleaved_streaming.py
```

## Latency Budget

| Stage | Agent | Latency |
|-------|-------|---------|
| VAD silence | Shared | 200ms |
| STT final | Shared | 30-50ms |
| LLM context cache | Conversational | ~0ms |
| LLM first segment | Conversational | 100-150ms |
| TTS first audio | Conversational | 370ms |
| **Total V2V (user hears)** | **~500-700ms** |
| Tool execution | Background | No impact on user |
| Background LLM | Background | Asynchronous |

## Comparison: Single vs Dual Agent

| Aspect | Single Agent (bot_interleaved_streaming.py) | Dual Agent (bot_dual_interleaved_streaming.py) |
|--------|---|---|
| Latency | ~500-700ms | ~500-700ms (same!) |
| Local LLM | Nemotron | Nemotron (unchanged) |
| Cloud LLM | None | Gemini (background) |
| Tools | No | Yes (Gemini) |
| Complexity | Simpler | More powerful |
| Cost | Lower | Higher (Gemini calls) |

## ★ Insight ─────────────────────────────────────

The dual-agent pattern is powerful because:

1. **No latency penalty** - User doesn't wait for background work
2. **Shared context** - Both agents see same conversation
3. **Task separation** - Conversational focus vs tool/analysis work
4. **Scalability** - Can add more workers to ParallelPipeline as needed
─────────────────────────────────────────────────

## Example Usage Flow

```yaml
User: "What's the weather in San Francisco and tell me a joke?"

Timeline:
├─ 0ms      User finishes speaking
├─ 200ms    VAD detects silence
├─ 250ms    STT final transcription arrives
├─ 250ms    Both agents receive transcription
│
├─ Conversational Path:
│  ├─ 250ms  Nemotron starts generating
│  ├─ 350ms  First segment ready (24 tokens)
│  ├─ 720ms  Magpie TTS first audio starts
│  └─ 720ms  User hears: "Sure! Let me tell you a funny one..."
│
├─ Background Path (Parallel):
│  ├─ 250ms  Gemini starts processing
│  ├─ 300ms  Gemini calls "get_weather" tool
│  ├─ 400ms  Weather data returned
│  ├─ 500ms  Gemini generates analysis
│  └─ 700ms  Results added to context
│
└─ Next Turn:
   Conversation context now includes:
   - User's request
   - Conversational response
   - Weather analysis
   - Available for next turn
```

Both agents work simultaneously, but user only waits for conversational agent!
