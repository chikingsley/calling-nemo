set dotenv-load

default:
    @just --list

# Run the voice agent (kills any existing instance first)
voice:
    -fuser -k 7860/tcp 2>/dev/null
    sleep 0.3
    uv run agent_gemini_live.py -t webrtc --host 0.0.0.0

# Kill the running voice agent
kill:
    -fuser -k 7860/tcp 2>/dev/null

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
