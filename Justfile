set dotenv-load

default:
    @just --list

# =============================================================================
# Bot Commands
# =============================================================================

# Run the Gemini tools voice agent (cloud STT/TTS)
voice:
    -fuser -k 7860/tcp 2>/dev/null
    sleep 0.3
    uv run pipecat_bots/bot_gemini_tools.py -t webrtc --host 0.0.0.0

# Run the dual-agent bot (local Nemotron + cloud Gemini tools)
dual:
    -fuser -k 7860/tcp 2>/dev/null
    sleep 0.3
    uv run pipecat_bots/bot_dual_interleaved_streaming.py -t webrtc --host 0.0.0.0

# Run the interleaved streaming bot (local only, lowest latency)
interleaved:
    -fuser -k 7860/tcp 2>/dev/null
    sleep 0.3
    uv run pipecat_bots/bot_interleaved_streaming.py -t webrtc --host 0.0.0.0

# Kill any running bot on port 7860
kill:
    -fuser -k 7860/tcp 2>/dev/null

# =============================================================================
# Nemotron Container
# =============================================================================

# Start the nemotron container (ASR + LLM + TTS)
container-start:
    ./scripts/nemotron.sh start

# Stop the nemotron container
container-stop:
    ./scripts/nemotron.sh stop

# Restart the nemotron container
container-restart:
    ./scripts/nemotron.sh restart

# Check nemotron container status
container-status:
    ./scripts/nemotron.sh status

# View nemotron container logs (all services)
container-logs:
    ./scripts/nemotron.sh logs

# View ASR logs only
logs-asr:
    ./scripts/nemotron.sh logs asr

# View TTS logs only
logs-tts:
    ./scripts/nemotron.sh logs tts

# View LLM logs only
logs-llm:
    ./scripts/nemotron.sh logs llm

# Open shell in nemotron container
container-shell:
    ./scripts/nemotron.sh shell

# =============================================================================
# Development
# =============================================================================

# Sync dependencies
sync:
    uv sync

# Lint and format
lint:
    uv run ruff check --fix .
    uv run ruff format .

# Type check
check:
    uv run ty check

# Run tests
test:
    uv run pytest

# Full check: lint + type check + test
ci: lint check test
