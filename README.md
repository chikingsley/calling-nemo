# Calling Nemo

Voice-controlled agent for Linux/Mac workstation control and Claude Code session management.

Built on [Pipecat](https://github.com/pipecat-ai/pipecat) with cloud services (swappable to local via nemotron).

## What it does

- **Voice commands** → Execute shell commands on Linux
- **Claude Code sessions** → Start, send messages, check output via tmux
- **Mac control** → SSH to Mac for opening apps, running scripts

## Architecture

```
[Microphone] → STT → LLM → TTS → [Speaker]
                      ↓
               [Tool Execution]
               - run_shell_command
               - run_mac_command
               - start_claude_session
               - send_to_claude
               - get_claude_output
```

## Quick Start

```bash
uv sync
cp .env.example .env  # Add your API keys
uv run agent_gemini_live.py -t webrtc --host 0.0.0.0
```

## Testing

```bash
uv run pytest -v                              # 33 real integration tests
uv run pytest --cov=. --cov-report=term-missing  # 85% coverage
```

## Cloud Services (Current)

| Component | Service |
|-----------|---------|
| STT | Deepgram |
| LLM | Gemini 2.5 Flash Lite |
| TTS | Rime (arcana model) |

## Local Services (nemotron-january-2026/)

Fully local alternative with NVIDIA stack:
- **STT**: NeMo Parakeet (CUDA)
- **LLM**: llama.cpp with buffered streaming (100% KV cache reuse)
- **TTS**: Magpie/FastPitch (CUDA)

See `nemotron-january-2026/README.md` for container setup.

## Requirements

- Python 3.12+
- API keys: `GOOGLE_API_KEY`, `DEEPGRAM_API_KEY`, `RIME_API_KEY`
- For Mac control: SSH access configured for `work-mac`
