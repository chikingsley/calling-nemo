#!/usr/bin/env python3
"""
Voice Control Agent with Gemini Tools - Deepgram STT + Gemini LLM + Rime TTS

Single-agent bot with tool calling for computer control and Claude Code integration.
Uses GoogleLLMService (REST API) with registered function handlers.

Requires: GOOGLE_API_KEY, DEEPGRAM_API_KEY, RIME_API_KEY

Run:
    just voice
    # or
    uv run pipecat_bots/bot_gemini_tools.py -t webrtc --host 0.0.0.0
"""

import os
import subprocess

from dotenv import load_dotenv
from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frameworks.rtvi import RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.rime.tts import RimeNonJsonTTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams

load_dotenv(override=True)

# =============================================================================
# Tool Functions
# =============================================================================


async def run_shell_command(params: FunctionCallParams):
    """Execute a shell command on the local machine."""
    cmd = params.arguments.get("command", "")
    logger.info(f"Executing: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        output = result.stdout or result.stderr or "Command completed"
        await params.result_callback(
            {
                "success": result.returncode == 0,
                "output": output[:500],
                "return_code": result.returncode,
            }
        )
    except subprocess.TimeoutExpired:
        await params.result_callback({"success": False, "output": "Command timed out"})
    except Exception as e:
        await params.result_callback({"success": False, "output": str(e)})


async def start_claude_session(params: FunctionCallParams):
    """Start a Claude Code session in a tmux window."""
    project_path = params.arguments.get("project_path", "~")
    session_name = params.arguments.get("session_name", "claude")
    cmd = (
        f'tmux new-session -d -s {session_name} "cd {project_path} && claude" 2>/dev/null || '
        f'tmux send-keys -t {session_name} "cd {project_path} && claude" Enter'
    )
    try:
        subprocess.run(cmd, shell=True, timeout=5)
        await params.result_callback(
            {
                "success": True,
                "message": f"Claude session '{session_name}' started in {project_path}",
            }
        )
    except Exception as e:
        await params.result_callback({"success": False, "message": str(e)})


async def send_to_claude(params: FunctionCallParams):
    """Send a message to an active Claude Code session."""
    message = params.arguments.get("message", "")
    session_name = params.arguments.get("session_name", "claude")
    escaped = message.replace("'", "'\"'\"'")
    cmd = f"tmux send-keys -t {session_name} '{escaped}' Enter"
    try:
        subprocess.run(cmd, shell=True, timeout=5)
        await params.result_callback(
            {"success": True, "message": f"Sent to Claude: {message[:100]}..."}
        )
    except Exception as e:
        await params.result_callback({"success": False, "message": str(e)})


async def get_claude_output(params: FunctionCallParams):
    """Get recent output from a Claude Code session."""
    session_name = params.arguments.get("session_name", "claude")
    lines = params.arguments.get("lines", 50)
    cmd = f"tmux capture-pane -t {session_name} -p -S -{lines}"
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        await params.result_callback({"success": True, "output": result.stdout[-2000:]})
    except Exception as e:
        await params.result_callback({"success": False, "message": str(e)})


async def run_mac_command(params: FunctionCallParams):
    """Run a command on a Mac via SSH over Tailscale."""
    cmd = params.arguments.get("command", "")
    host = params.arguments.get("host", "work-mac")
    logger.info(f"Executing on {host}: {cmd}")
    ssh_cmd = f'ssh -o ConnectTimeout=5 {host} "{cmd}"'
    try:
        result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True, timeout=30)
        output = result.stdout or result.stderr or "Command completed"
        await params.result_callback(
            {
                "success": result.returncode == 0,
                "output": output[:500],
                "return_code": result.returncode,
            }
        )
    except subprocess.TimeoutExpired:
        await params.result_callback({"success": False, "output": "SSH command timed out"})
    except Exception as e:
        await params.result_callback({"success": False, "output": str(e)})


# =============================================================================
# Tool Definitions
# =============================================================================

tools_list = [
    FunctionSchema(
        name="run_shell_command",
        description="Execute a shell command on the Linux machine",
        properties={"command": {"type": "string", "description": "The bash command to execute"}},
        required=["command"],
    ),
    FunctionSchema(
        name="start_claude_session",
        description="Start a Claude Code AI session in a tmux window for a project",
        properties={
            "project_path": {
                "type": "string",
                "description": "Path to the project directory",
            },
            "session_name": {
                "type": "string",
                "description": "Name for the tmux session (default: claude)",
            },
        },
        required=["project_path"],
    ),
    FunctionSchema(
        name="send_to_claude",
        description="Send a message or prompt to an active Claude Code session",
        properties={
            "message": {
                "type": "string",
                "description": "The message to send to Claude",
            },
            "session_name": {
                "type": "string",
                "description": "Tmux session name (default: claude)",
            },
        },
        required=["message"],
    ),
    FunctionSchema(
        name="get_claude_output",
        description="Get recent output from a Claude Code session to check progress",
        properties={
            "session_name": {
                "type": "string",
                "description": "Tmux session name (default: claude)",
            },
            "lines": {
                "type": "integer",
                "description": "Number of lines to retrieve (default: 50)",
            },
        },
        required=[],
    ),
    FunctionSchema(
        name="run_mac_command",
        description=(
            "Run a command on a Mac computer via SSH. Use for opening apps, "
            "controlling the Mac, running scripts. Examples: 'open -a ChatGPT', "
            "'open -a Safari https://google.com', 'osascript -e ...'"
        ),
        properties={
            "command": {
                "type": "string",
                "description": "The command to run on the Mac (e.g., 'open -a ChatGPT')",
            },
            "host": {
                "type": "string",
                "description": "Which Mac to target: 'work-mac' or 'home-mac' (default: work-mac)",
            },
        },
        required=["command"],
    ),
]

tools = ToolsSchema(standard_tools=tools_list)

# =============================================================================
# Transport Configuration
# =============================================================================

transport_params = {
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
    ),
}

# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """You are a voice-controlled assistant running on a Linux workstation.
You help the user control their computers and manage Claude Code sessions.

Your capabilities:
- Execute shell commands on this Linux machine (run_shell_command)
- Control Mac computers via SSH (run_mac_command) - can open apps, run scripts
- Start and interact with Claude Code sessions
  (start_claude_session, send_to_claude, get_claude_output)

Available Macs: work-mac (default), home-mac

Guidelines:
- Keep responses SHORT and conversational (they're spoken aloud)
- For dangerous commands, confirm before executing
- Report command results concisely
- Use plain text, no special characters or emojis

The user may say things like:
- "Open ChatGPT on my Mac"
- "Open a new Claude session on the nemotron project"
- "Run git status"
- "Check on the Docker build"
- "Open Safari on home Mac"
"""

# =============================================================================
# Main Bot
# =============================================================================


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting Voice Control Agent (Deepgram STT + Gemini LLM + Deepgram TTS)")

    # Validate API keys
    google_api_key = os.getenv("GOOGLE_API_KEY")
    deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")

    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is required")
    if not deepgram_api_key:
        raise ValueError("DEEPGRAM_API_KEY environment variable is required")

    # STT: Deepgram
    stt = DeepgramSTTService(api_key=deepgram_api_key)

    # LLM: Google Gemini
    llm = GoogleLLMService(
        api_key=google_api_key,
        model="gemini-2.5-flash-lite",
    )

    # Register tool handlers
    llm.register_function("run_shell_command", run_shell_command)
    llm.register_function("start_claude_session", start_claude_session)
    llm.register_function("send_to_claude", send_to_claude)
    llm.register_function("get_claude_output", get_claude_output)
    llm.register_function("run_mac_command", run_mac_command)

    # TTS: Rime (arcana model uses non-JSON websocket)
    tts = RimeNonJsonTTSService(
        api_key=os.getenv("RIME_API_KEY", "qhzsV6D8OhhwHFeuYBhD0lSgmHIwfyqfYnxdLlOAcDs"),
        voice_id="celeste",
        model="arcana",
    )

    # Context for conversation - pass tools to context, not LLM
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    context = LLMContext(messages, tools)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

    # RTVI processor handles client messages
    rtvi = RTVIProcessor()
    # RTVI observer monitors frames and sends metrics/transcriptions to client
    rtvi_observer = RTVIObserver(rtvi)

    # Event handler: when client is ready, mark bot as ready
    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi_processor):
        await rtvi_processor.set_bot_ready()

    pipeline = Pipeline(
        [
            transport.input(),
            rtvi,
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[rtvi_observer],
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        # Add greeting prompt and kick off conversation
        messages.append({
            "role": "user",
            "content": "Say hello briefly and tell me you're ready to help.",
        })
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main entry point."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
