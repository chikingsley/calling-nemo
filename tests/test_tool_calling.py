"""Test tool calling with Gemini API - verify correct tool selection."""

import os

import pytest
from dotenv import load_dotenv

load_dotenv(override=True)
from google import genai
from google.genai.types import FunctionDeclaration, GenerateContentConfig, Tool


@pytest.fixture
def api_key():
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        pytest.skip("GOOGLE_API_KEY not set")
    return key


@pytest.fixture
def agent_tools():
    """All the tools from our voice agent."""
    return Tool(
        function_declarations=[
            FunctionDeclaration(
                name="run_shell_command",
                description="Execute a shell command on the Linux machine",
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The bash command to execute",
                        }
                    },
                    "required": ["command"],
                },
            ),
            FunctionDeclaration(
                name="start_claude_session",
                description="Start a Claude Code AI session in a tmux window for a project",
                parameters={
                    "type": "object",
                    "properties": {
                        "project_path": {
                            "type": "string",
                            "description": "Path to the project directory",
                        },
                        "session_name": {
                            "type": "string",
                            "description": "Name for the tmux session (default: claude)",
                        },
                    },
                    "required": ["project_path"],
                },
            ),
            FunctionDeclaration(
                name="send_to_claude",
                description="Send a message or prompt to an active Claude Code session",
                parameters={
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "The message to send to Claude",
                        },
                        "session_name": {
                            "type": "string",
                            "description": "Tmux session name (default: claude)",
                        },
                    },
                    "required": ["message"],
                },
            ),
            FunctionDeclaration(
                name="get_claude_output",
                description="Get recent output from a Claude Code session to check progress",
                parameters={
                    "type": "object",
                    "properties": {
                        "session_name": {
                            "type": "string",
                            "description": "Tmux session name (default: claude)",
                        },
                        "lines": {
                            "type": "integer",
                            "description": "Number of lines to retrieve (default: 50)",
                        },
                    },
                    "required": [],
                },
            ),
            FunctionDeclaration(
                name="run_mac_command",
                description="Run a command on a Mac computer via SSH. Use for opening apps, controlling the Mac, running scripts.",
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The command to run on the Mac",
                        },
                        "host": {
                            "type": "string",
                            "description": "Which Mac to target: 'work-mac' or 'home-mac'",
                        },
                    },
                    "required": ["command"],
                },
            ),
        ]
    )


SYSTEM_PROMPT = """You are a voice-controlled assistant running on a Linux workstation.
You help the user control their computer and manage Claude Code sessions.

Your capabilities:
- Execute shell commands on Linux (run_shell_command)
- Start and interact with Claude Code sessions (start_claude_session, send_to_claude, get_claude_output)
- Control Mac computers via SSH (run_mac_command)

Guidelines:
- Use the appropriate tool for each request
- For Linux commands use run_shell_command
- For Mac commands use run_mac_command
- For Claude Code interactions use the claude tools
"""


# =============================================================================
# Test run_shell_command triggers
# =============================================================================


async def test_triggers_shell_for_git_status(api_key, agent_tools):
    """Test that 'git status' triggers run_shell_command."""
    client = genai.Client(api_key=api_key)
    config = GenerateContentConfig(system_instruction=SYSTEM_PROMPT, tools=[agent_tools])

    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents="Run git status",
        config=config,
    )

    part = response.candidates[0].content.parts[0]
    assert part.function_call is not None
    assert part.function_call.name == "run_shell_command"
    assert "git status" in part.function_call.args.get("command", "").lower()


async def test_triggers_shell_for_docker(api_key, agent_tools):
    """Test that docker commands trigger run_shell_command."""
    client = genai.Client(api_key=api_key)
    config = GenerateContentConfig(system_instruction=SYSTEM_PROMPT, tools=[agent_tools])

    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents="Show me running docker containers",
        config=config,
    )

    part = response.candidates[0].content.parts[0]
    assert part.function_call is not None
    assert part.function_call.name == "run_shell_command"
    assert "docker" in part.function_call.args.get("command", "").lower()


async def test_triggers_shell_for_ports(api_key, agent_tools):
    """Test that checking ports triggers run_shell_command."""
    client = genai.Client(api_key=api_key)
    config = GenerateContentConfig(system_instruction=SYSTEM_PROMPT, tools=[agent_tools])

    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents="What's running on port 8080?",
        config=config,
    )

    part = response.candidates[0].content.parts[0]
    assert part.function_call is not None
    assert part.function_call.name == "run_shell_command"


# =============================================================================
# Test Claude session triggers
# =============================================================================


async def test_triggers_start_claude(api_key, agent_tools):
    """Test that 'open claude on project X' triggers start_claude_session."""
    client = genai.Client(api_key=api_key)
    config = GenerateContentConfig(system_instruction=SYSTEM_PROMPT, tools=[agent_tools])

    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents="Start a Claude session on the voice-control-agent project",
        config=config,
    )

    part = response.candidates[0].content.parts[0]
    assert part.function_call is not None
    assert part.function_call.name == "start_claude_session"
    assert "project_path" in part.function_call.args


async def test_triggers_send_to_claude(api_key, agent_tools):
    """Test that sending a message to Claude triggers send_to_claude."""
    client = genai.Client(api_key=api_key)
    config = GenerateContentConfig(system_instruction=SYSTEM_PROMPT, tools=[agent_tools])

    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents="Send a message to Claude saying: please run the tests",
        config=config,
    )

    part = response.candidates[0].content.parts[0]
    assert part.function_call is not None
    assert part.function_call.name == "send_to_claude"
    assert "message" in part.function_call.args


async def test_triggers_get_claude_output(api_key, agent_tools):
    """Test that checking Claude's progress triggers get_claude_output."""
    client = genai.Client(api_key=api_key)
    config = GenerateContentConfig(system_instruction=SYSTEM_PROMPT, tools=[agent_tools])

    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents="What's Claude working on? Check the output.",
        config=config,
    )

    part = response.candidates[0].content.parts[0]
    assert part.function_call is not None
    assert part.function_call.name == "get_claude_output"


# =============================================================================
# Test Mac command triggers
# =============================================================================


async def test_triggers_mac_for_open_safari(api_key, agent_tools):
    """Test that 'open Safari' triggers run_mac_command."""
    client = genai.Client(api_key=api_key)
    config = GenerateContentConfig(system_instruction=SYSTEM_PROMPT, tools=[agent_tools])

    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents="Open Safari on my Mac",
        config=config,
    )

    part = response.candidates[0].content.parts[0]
    assert part.function_call is not None
    assert part.function_call.name == "run_mac_command"
    assert "safari" in part.function_call.args.get("command", "").lower()


async def test_triggers_mac_for_open_app(api_key, agent_tools):
    """Test that opening an app on Mac triggers run_mac_command."""
    client = genai.Client(api_key=api_key)
    config = GenerateContentConfig(system_instruction=SYSTEM_PROMPT, tools=[agent_tools])

    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents="Open Calculator on work-mac",
        config=config,
    )

    part = response.candidates[0].content.parts[0]
    assert part.function_call is not None
    assert part.function_call.name == "run_mac_command"


async def test_triggers_mac_for_volume(api_key, agent_tools):
    """Test that volume control on Mac triggers run_mac_command."""
    client = genai.Client(api_key=api_key)
    config = GenerateContentConfig(system_instruction=SYSTEM_PROMPT, tools=[agent_tools])

    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents="Set the Mac volume to 50%",
        config=config,
    )

    part = response.candidates[0].content.parts[0]
    assert part.function_call is not None
    assert part.function_call.name == "run_mac_command"


# =============================================================================
# Test correct tool discrimination
# =============================================================================


async def test_linux_vs_mac_discrimination(api_key, agent_tools):
    """Test that Linux commands go to shell, Mac commands go to mac."""
    client = genai.Client(api_key=api_key)
    config = GenerateContentConfig(system_instruction=SYSTEM_PROMPT, tools=[agent_tools])

    # Linux command
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents="Run htop on the Linux machine",
        config=config,
    )
    part = response.candidates[0].content.parts[0]
    assert part.function_call.name == "run_shell_command"

    # Mac command
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents="Open Finder on my Mac",
        config=config,
    )
    part = response.candidates[0].content.parts[0]
    assert part.function_call.name == "run_mac_command"
