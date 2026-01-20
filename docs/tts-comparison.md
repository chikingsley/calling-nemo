# TTS/Voice AI Comparison: Evaluating Alternatives to Magpie TTS

This document compares four TTS/voice AI solutions for potential integration with our Pipecat-based voice agent.

## Quick Reference Table

| Aspect | Pocket TTS | Chatterbox Turbo | Magpie (Current) | Moshi/DSM |
|--------|------------|------------------|------------------|-----------|
| **Vendor** | Kyutai Labs | Resemble AI | NVIDIA | Kyutai Labs |
| **Model Size** | 100M | 350M | 357M | 7B (Temporal) |
| **Hardware** | CPU only | GPU | GPU | GPU |
| **TTFB** | ~200ms | Low (unspec) | ~750-1000ms | ~200ms |
| **Speed** | 6x RT (M4 CPU) | Optimized | 3.7x RT | Real-time streaming |
| **Languages** | English | EN / 23+ (Multilingual) | 7 languages | EN/FR |
| **Voice Cloning** | Yes | Yes (zero-shot) | No | No |
| **Paralinguistics** | No | Yes (`[laugh]`, etc.) | No | 70+ emotions native |
| **Architecture** | TTS only | TTS only | TTS only | Full-duplex speech-to-speech |
| **Tool Calling** | N/A | N/A | N/A | Via inner monologue (custom) |
| **License** | MIT | MIT | NVIDIA | MIT |

---

## 1. Pocket TTS (Kyutai Labs)

**Repository:** <https://github.com/kyutai-labs/pocket-tts>

### Overview

Lightweight TTS designed for CPU inference. Built on Kyutai's Delayed Streams Modeling research.

### Key Features

- 100M parameters - runs efficiently on CPU
- ~200ms latency to first audio
- 6x real-time on MacBook Air M4 (2 cores)
- 8 preset voices: alba, marius, javert, jean, fantine, cosette, eponine, azelma
- Voice cloning from audio samples
- Handles infinitely long text

### Deployment

```bash
# Simple server
uvx pocket-tts serve  # localhost:8000

# CLI generation
uvx pocket-tts generate --voice alba --text "Hello world"
```

### Python API

```python
from pocket_tts import PocketTTS

model = PocketTTS()
audio = model.generate("Hello world", voice="alba")
```

### Pros

- **Frees GPU entirely** for LLM inference
- Lowest latency (~200ms TTFB)
- Simplest deployment
- Voice cloning included

### Cons

- English only
- No paralinguistic control
- Less mature than alternatives

---

## 2. Chatterbox (Resemble AI)

**Repository:** <https://github.com/resemble-ai/chatterbox>

### Overview

Three-model family optimized for different use cases. Turbo variant designed for low-latency voice agents.

### Model Variants

| Model | Params | Languages | Features |
|-------|--------|-----------|----------|
| Chatterbox-Turbo | 350M | English | Low-latency, paralinguistics |
| Chatterbox-Multilingual | 500M | 23+ | Zero-shot cloning, global deployment |
| Chatterbox (Original) | 500M | English | CFG, exaggeration tuning |

### Key Features

- Zero-shot voice cloning (10s reference audio recommended)
- Paralinguistic tags: `[laugh]`, `[cough]`, `[chuckle]`, `[sigh]`
- Single-step mel decoder (vs 10-step diffusion)
- Built-in neural watermarking (cannot be disabled)

### Deployment

```bash
pip install chatterbox-tts
```

### Python API

```python
from chatterbox.tts_turbo import ChatterboxTurboTTS

model = ChatterboxTurboTTS.from_pretrained(device="cuda")

# Basic generation
wav = model.generate("Hello world", audio_prompt_path="voice_ref.wav")

# With paralinguistics
wav = model.generate(
    "That's hilarious [laugh] I can't believe it!",
    audio_prompt_path="voice_ref.wav",
    exaggeration=0.7,
    cfg_weight=0.3
)
```

### Pros

- Voice cloning with emotional control
- Paralinguistics make responses more natural
- 23-language support (Multilingual model)
- Well-documented tuning parameters

### Cons

- Requires GPU
- No built-in server (need to wrap in FastAPI)
- Mandatory watermarking
- VRAM sharing with LLM

---

## 3. NVIDIA Magpie TTS (Current Implementation)

**Model:** `nvidia/magpie_tts_multilingual_357m`

### Overview

Our current production TTS. GPU-based with sophisticated streaming and artifact handling.

### Key Features

- 357M parameters
- 7 languages: English, Spanish, German, French, Vietnamese, Italian, Mandarin
- 5 voices: aria (default), john, sofia, jason, leo
- Three streaming modes: batch, frame-by-frame, sentence-level
- Extensive audio artifact handling (overlap-add, crossfade, fade-out)

### Current Architecture

```text
Pipecat Bot
    │
    ▼ WebSocket /ws/tts/stream
┌─────────────────────────────────┐
│ Magpie TTS Server (Port 8001)   │
│ - StreamManager                 │
│ - MagpieTTSModel (GPU)          │
│ - HiFi-GAN Vocoder              │
└─────────────────────────────────┘
```

### Performance Metrics (Measured)

- TTFB: 746ms - 1,061ms
- RTF: ~0.27x (3.7x faster than real-time)
- Warmup: ~45-60 seconds

### Pros

- Multilingual production-ready
- Sophisticated streaming with artifact handling
- Well-integrated with Pipecat
- Battle-tested in our setup

### Cons

- High latency (~750ms+ TTFB)
- Complex deployment (NeMo container)
- No voice cloning
- GPU memory pressure

---

## 4. Moshi / Delayed Streams Modeling (Kyutai Labs)

**Repositories:**

- <https://github.com/kyutai-labs/moshi>
- <https://github.com/kyutai-labs/delayed-streams-modeling>

### Overview

**Fundamentally different architecture.** Moshi is a full-duplex speech-to-speech foundation model, not just TTS. It replaces the entire STT→LLM→TTS pipeline with a single end-to-end model.

### Architecture

```text
Traditional Pipeline:
  Audio In → STT → Text → LLM → Text → TTS → Audio Out
  (Sequential, ~2-3s latency)

Moshi Architecture:
  Audio In ──┐
             ├──► 7B Temporal Transformer ──► Audio Out
  Audio Out ─┘    (processes both streams)

  + Inner Monologue (text tokens for reasoning)
  (Full-duplex, ~200ms latency)
```

### Key Components

**Mimi Codec:**

- 24kHz audio → 12.5Hz tokens
- 1.1 kbps bandwidth
- 80ms streaming latency

**Temporal Transformer (7B):**

- Handles both speaker streams simultaneously
- Processes text + audio tokens together
- Inner monologue for coherent reasoning

### Key Features

- True full-duplex (listen while speaking)
- 70+ emotions and speaking styles
- ~200ms end-to-end latency
- Inner monologue enhances reasoning
- Runs on L4 GPU at 200ms latency

### DSM-based Models

| Model | Params | Latency | Use Case |
|-------|--------|---------|----------|
| STT 1B (EN/FR) | ~1B | 0.5s | Real-time transcription |
| STT 2.6B (EN) | ~2.6B | 2.5s | High accuracy |
| Moshi | 7B | 200ms | Full dialogue |

### Tool Calling Integration

Moshi does **not have native tool calling**. However, integration is possible via the inner monologue:

```text
┌─────────────────────────────────────────────────────────┐
│                    Moshi Model                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Inner Monologue Stream (Text Tokens)            │   │
│  │ "The user wants the weather. I should call..."  │   │
│  └──────────────────────┬──────────────────────────┘   │
│                         │                               │
│                         ▼                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Tool Call Detector (Custom Layer)               │   │
│  │ - Pattern match on inner monologue              │   │
│  │ - Pause audio generation                        │   │
│  │ - Execute tool                                  │   │
│  │ - Inject result back into context               │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

**Proposed Integration Pattern:**

```python
class MoshiToolHandler:
    """Intercept inner monologue for tool calls."""

    TOOL_PATTERNS = [
        r"I should call (\w+)",
        r"Let me check (\w+)",
        r"I'll use the (\w+) tool",
    ]

    async def process_monologue(self, text_tokens: list[str]):
        text = self.tokenizer.decode(text_tokens)

        for pattern in self.TOOL_PATTERNS:
            if match := re.search(pattern, text):
                tool_name = match.group(1)

                # Pause audio generation
                await self.moshi.pause_generation()

                # Execute tool
                result = await self.execute_tool(tool_name)

                # Inject result and resume
                await self.moshi.inject_context(f"Tool result: {result}")
                await self.moshi.resume_generation()
```

**Challenges:**

1. Inner monologue text is not structured JSON
2. Need to train/fine-tune for consistent tool call syntax
3. Pausing mid-generation affects speech naturalness
4. Tool execution latency breaks real-time flow

**Alternative: Hybrid Architecture**

```text
┌──────────────────────────────────────────────────────────┐
│                   Hybrid Voice Agent                      │
│                                                          │
│  Simple queries ──► Moshi (fast, natural)                │
│                                                          │
│  Complex queries ──► Route to:                           │
│    ├── STT (Moshi/DSM)                                   │
│    ├── Text LLM (tool calling)                           │
│    └── TTS (Pocket/Chatterbox/Magpie)                    │
│                                                          │
│  Detection: Keywords, intent classification, or          │
│             inner monologue pattern matching             │
└──────────────────────────────────────────────────────────┘
```

### Pros

- Lowest latency for conversational AI (~200ms e2e)
- Most natural conversation flow
- True full-duplex (no turn-taking)
- Emotional expressiveness built-in

### Cons

- 7B model requires significant GPU
- No native tool calling
- Replaces entire pipeline (big change)
- Limited language support (EN/FR)

---

## Integration Comparison

### Effort to Integrate

| System | Integration Effort | Changes Required |
|--------|-------------------|------------------|
| Pocket TTS | **Low** | New Pipecat service, HTTP client |
| Chatterbox | **Medium** | FastAPI wrapper + Pipecat service |
| Magpie | **Already done** | N/A |
| Moshi | **High** | Replace entire pipeline, custom tool handling |

### GPU Memory Impact (RTX 5070 - 12GB)

| Configuration | TTS VRAM | LLM Headroom |
|--------------|----------|--------------|
| Pocket TTS (CPU) | 0 GB | **12 GB** |
| Chatterbox Turbo | ~1.4 GB | ~10.6 GB |
| Magpie | ~1.4 GB | ~10.6 GB |
| Moshi 7B | ~14 GB+ | **Won't fit** |

### Pipecat Integration Pattern

```python
# Unified interface for swappable TTS backends
class TTSBackend(Protocol):
    async def synthesize(
        self,
        text: str,
        voice: str = "default",
        language: str = "en"
    ) -> AsyncIterator[AudioFrame]: ...

class PocketTTSBackend(TTSBackend):
    """CPU-based, lowest latency"""
    base_url = "http://localhost:8000"

class ChatterboxBackend(TTSBackend):
    """GPU-based, voice cloning + paralinguistics"""
    base_url = "http://localhost:8002"

class MagpieBackend(TTSBackend):
    """GPU-based, multilingual"""
    base_url = "http://localhost:8001"

# Smart routing
class AdaptiveTTSService(TTSService):
    def select_backend(self, text: str, context: dict) -> TTSBackend:
        if context.get("language") != "en":
            return self.magpie  # Multilingual
        if context.get("voice_clone_ref"):
            return self.chatterbox  # Voice cloning
        if "[laugh]" in text or "[sigh]" in text:
            return self.chatterbox  # Paralinguistics
        return self.pocket  # Default: fastest
```

---

## Recommendations

### For Lowest Latency (English Only)

**Use Pocket TTS**

- 200ms TTFB vs 750ms+ current
- Frees GPU for larger LLM
- Simple deployment

### For Voice Cloning / Expressiveness

**Use Chatterbox Turbo**

- Zero-shot cloning
- Paralinguistic tags
- Good balance of speed/quality

### For Multilingual Production

**Keep Magpie** (or consider Chatterbox Multilingual)

- Proven stability
- 7 languages

### For Next-Gen Conversational AI

**Explore Moshi** (research/experimentation)

- True full-duplex changes the game
- Tool calling needs custom work
- May not fit in 12GB with LLM

### Hybrid Recommendation

```yaml
Primary:     Pocket TTS (CPU) - default for speed
Fallback 1:  Chatterbox - voice cloning, emotions
Fallback 2:  Magpie - non-English languages
Future:      Moshi - full-duplex exploration
```

---

## Next Steps

1. **Benchmark Pocket TTS** - Test TTFB and quality vs Magpie
2. **Prototype Chatterbox wrapper** - FastAPI server with WebSocket support
3. **Implement adaptive routing** - Switch backends based on context
4. **Moshi exploration** - Standalone testing for tool call patterns

---

## References

- [Pocket TTS](https://github.com/kyutai-labs/pocket-tts)
- [Chatterbox](https://github.com/resemble-ai/chatterbox)
- [NVIDIA Magpie TTS](https://huggingface.co/nvidia/magpie_tts_multilingual_357m)
- [Moshi](https://github.com/kyutai-labs/moshi)
- [Delayed Streams Modeling](https://github.com/kyutai-labs/delayed-streams-modeling)
- [DSM Paper](https://arxiv.org/abs/2509.08753)
- [Moshi Paper](https://arxiv.org/abs/2410.00037)
