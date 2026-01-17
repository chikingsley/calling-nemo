"""Pipeline integration tests - testing function calling through the actual pipeline.

Based on pipecat's test patterns from:
/home/simon/github/pipecat/tests/integration/test_integration_unified_function_calling.py
"""

import os

import pytest
from dotenv import load_dotenv

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import LLMContextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.tests.utils import run_test

load_dotenv(override=True)


def agent_tools() -> ToolsSchema:
    """Create the tools schema matching our voice agent."""
    return ToolsSchema(
        standard_tools=[
            FunctionSchema(
                name="run_shell_command",
                description="Execute a shell command on the Linux machine",
                properties={
                    "command": {"type": "string", "description": "The bash command to execute"}
                },
                required=["command"],
            ),
            FunctionSchema(
                name="run_mac_command",
                description="Run a command on a Mac computer via SSH",
                properties={
                    "command": {"type": "string", "description": "The command to run on the Mac"},
                    "host": {"type": "string", "description": "Which Mac to target"},
                },
                required=["command"],
            ),
        ]
    )


@pytest.fixture
def api_key():
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        pytest.skip("GOOGLE_API_KEY not set")
    return key


async def test_pipeline_function_call_triggered(api_key):
    """Test that function calls are triggered through the full pipeline."""
    call_count = 0
    called_function = None
    called_args = None

    async def mock_handler(params: FunctionCallParams):
        nonlocal call_count, called_function, called_args
        call_count += 1
        called_function = params.function_name
        called_args = params.arguments

    llm = GoogleLLMService(api_key=api_key, model="gemini-2.5-flash-lite")
    llm.register_function(None, mock_handler)  # Register for all functions

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use run_shell_command for Linux commands.",
        },
        {"role": "user", "content": "Run git status"},
    ]
    context = LLMContext(messages, agent_tools())

    pipeline = Pipeline([llm])
    frames_to_send = [LLMContextFrame(context)]

    await run_test(pipeline, frames_to_send=frames_to_send, expected_down_frames=None)

    assert call_count == 1
    assert called_function == "run_shell_command"
    assert "git" in called_args.get("command", "").lower()


async def test_pipeline_correct_function_selected(api_key):
    """Test that the correct function is selected based on the request."""
    called_functions = []

    async def mock_handler(params: FunctionCallParams):
        called_functions.append(params.function_name)

    llm = GoogleLLMService(api_key=api_key, model="gemini-2.5-flash-lite")
    llm.register_function(None, mock_handler)

    # Test Linux command
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use run_shell_command for Linux, run_mac_command for Mac.",
        },
        {"role": "user", "content": "List files in /tmp on this Linux machine"},
    ]
    context = LLMContext(messages, agent_tools())
    pipeline = Pipeline([llm])

    await run_test(
        pipeline, frames_to_send=[LLMContextFrame(context)], expected_down_frames=None
    )

    assert "run_shell_command" in called_functions


async def test_pipeline_mac_command_selected(api_key):
    """Test that Mac commands trigger run_mac_command."""
    called_functions = []

    async def mock_handler(params: FunctionCallParams):
        called_functions.append(params.function_name)

    llm = GoogleLLMService(api_key=api_key, model="gemini-2.5-flash-lite")
    llm.register_function(None, mock_handler)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use run_shell_command for Linux, run_mac_command for Mac. Default to work-mac.",
        },
        {"role": "user", "content": "Open Safari on work-mac"},
    ]
    context = LLMContext(messages, agent_tools())
    pipeline = Pipeline([llm])

    await run_test(
        pipeline, frames_to_send=[LLMContextFrame(context)], expected_down_frames=None
    )

    assert "run_mac_command" in called_functions
