"""Test the agent tools - real integration tests."""

import subprocess
from dataclasses import dataclass, field
from typing import Any

import pytest
from dotenv import load_dotenv

load_dotenv(override=True)

# Import the actual tool functions from the agent
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent_gemini_live import (
    get_claude_output,
    run_mac_command,
    run_shell_command,
    send_to_claude,
    start_claude_session,
)


@dataclass
class MockFunctionCallParams:
    """Mock FunctionCallParams for testing tools."""

    arguments: dict = field(default_factory=dict)
    result: Any = None

    async def result_callback(self, result: Any):
        self.result = result


# =============================================================================
# run_shell_command tests
# =============================================================================


async def test_run_shell_command_echo():
    """Test running a simple echo command."""
    params = MockFunctionCallParams(arguments={"command": "echo hello"})
    await run_shell_command(params)

    assert params.result["success"] is True
    assert "hello" in params.result["output"]
    assert params.result["return_code"] == 0


async def test_run_shell_command_ls():
    """Test running ls command."""
    params = MockFunctionCallParams(arguments={"command": "ls -la /tmp"})
    await run_shell_command(params)

    assert params.result["success"] is True
    assert params.result["return_code"] == 0


async def test_run_shell_command_pwd():
    """Test running pwd command."""
    params = MockFunctionCallParams(arguments={"command": "pwd"})
    await run_shell_command(params)

    assert params.result["success"] is True
    assert "/" in params.result["output"]


async def test_run_shell_command_failing_command():
    """Test handling a command that fails."""
    params = MockFunctionCallParams(arguments={"command": "ls /nonexistent_path_12345"})
    await run_shell_command(params)

    assert params.result["success"] is False
    assert params.result["return_code"] != 0


async def test_run_shell_command_output_truncation():
    """Test that long output is truncated to 500 chars."""
    params = MockFunctionCallParams(arguments={"command": "seq 1 1000"})
    await run_shell_command(params)

    assert params.result["success"] is True
    assert len(params.result["output"]) <= 500


async def test_run_shell_command_pipe():
    """Test running piped commands."""
    params = MockFunctionCallParams(arguments={"command": "echo 'hello world' | grep hello"})
    await run_shell_command(params)

    assert params.result["success"] is True
    assert "hello" in params.result["output"]


async def test_run_shell_command_env_var():
    """Test accessing environment variables."""
    params = MockFunctionCallParams(arguments={"command": "echo $HOME"})
    await run_shell_command(params)

    assert params.result["success"] is True
    assert "/" in params.result["output"]


# =============================================================================
# tmux session tests (start_claude_session, send_to_claude, get_claude_output)
# =============================================================================


@pytest.fixture
def tmux_test_session():
    """Create a test tmux session and clean it up after."""
    session_name = "pytest_test_session"
    # Kill any existing test session
    subprocess.run(f"tmux kill-session -t {session_name} 2>/dev/null", shell=True)
    yield session_name
    # Cleanup
    subprocess.run(f"tmux kill-session -t {session_name} 2>/dev/null", shell=True)


async def test_start_claude_session(tmux_test_session):
    """Test starting a tmux session (without actually running claude)."""
    # We'll start a session with just bash, not claude
    session_name = tmux_test_session

    # Manually create the session for testing
    subprocess.run(f"tmux new-session -d -s {session_name} bash", shell=True)

    # Verify session exists
    result = subprocess.run(
        f"tmux has-session -t {session_name} 2>/dev/null",
        shell=True,
    )
    assert result.returncode == 0


async def test_send_to_tmux_session(tmux_test_session):
    """Test sending commands to a tmux session."""
    session_name = tmux_test_session

    # Create the session
    subprocess.run(f"tmux new-session -d -s {session_name} bash", shell=True)

    # Send a command using our tool
    params = MockFunctionCallParams(
        arguments={"message": "echo PYTEST_MARKER_12345", "session_name": session_name}
    )
    await send_to_claude(params)

    assert params.result["success"] is True


async def test_get_tmux_output(tmux_test_session):
    """Test capturing output from a tmux session."""
    session_name = tmux_test_session

    # Create session and run a command
    subprocess.run(f"tmux new-session -d -s {session_name} bash", shell=True)
    subprocess.run(
        f"tmux send-keys -t {session_name} 'echo UNIQUE_TEST_OUTPUT_67890' Enter",
        shell=True,
    )

    # Wait a moment for command to execute
    import asyncio

    await asyncio.sleep(0.5)

    # Capture output using our tool
    params = MockFunctionCallParams(arguments={"session_name": session_name, "lines": 50})
    await get_claude_output(params)

    assert params.result["success"] is True
    assert "UNIQUE_TEST_OUTPUT_67890" in params.result["output"]


async def test_get_output_nonexistent_session():
    """Test getting output from a session that doesn't exist."""
    params = MockFunctionCallParams(
        arguments={"session_name": "nonexistent_session_xyz123", "lines": 50}
    )
    await get_claude_output(params)

    # Should still return success=True but empty output (tmux returns empty)
    # or the output indicates no session
    assert params.result is not None


# =============================================================================
# run_mac_command tests (SSH)
# =============================================================================


@pytest.fixture
def ssh_available():
    """Check if SSH to work-mac is available."""
    result = subprocess.run(
        "ssh -o ConnectTimeout=2 -o BatchMode=yes work-mac echo ok 2>/dev/null",
        shell=True,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip("SSH to work-mac not available")
    return True


async def test_run_mac_command(ssh_available):
    """Test running a command on Mac via SSH."""
    params = MockFunctionCallParams(arguments={"command": "echo hello_from_mac", "host": "work-mac"})
    await run_mac_command(params)

    assert params.result["success"] is True
    assert "hello_from_mac" in params.result["output"]


async def test_run_mac_command_hostname(ssh_available):
    """Test getting hostname from Mac."""
    params = MockFunctionCallParams(arguments={"command": "hostname", "host": "work-mac"})
    await run_mac_command(params)

    assert params.result["success"] is True
    assert params.result["output"].strip() != ""


async def test_run_mac_command_unavailable_host():
    """Test handling unavailable SSH host."""
    params = MockFunctionCallParams(
        arguments={"command": "echo test", "host": "nonexistent-host-12345"}
    )
    await run_mac_command(params)

    assert params.result["success"] is False


async def test_run_mac_command_open_app(ssh_available):
    """Test opening an app on Mac (Calculator - harmless)."""
    params = MockFunctionCallParams(
        arguments={"command": "open -a Calculator", "host": "work-mac"}
    )
    await run_mac_command(params)

    assert params.result["success"] is True
    # Close it after
    subprocess.run(
        'ssh work-mac "osascript -e \'quit app \"Calculator\"\'"',
        shell=True,
        capture_output=True,
    )


async def test_run_mac_command_pwd(ssh_available):
    """Test running pwd on Mac."""
    params = MockFunctionCallParams(
        arguments={"command": "pwd", "host": "work-mac"}
    )
    await run_mac_command(params)

    assert params.result["success"] is True
    assert "/" in params.result["output"]


async def test_run_mac_command_whoami(ssh_available):
    """Test running whoami on Mac."""
    params = MockFunctionCallParams(
        arguments={"command": "whoami", "host": "work-mac"}
    )
    await run_mac_command(params)

    assert params.result["success"] is True
    assert params.result["output"].strip() != ""


# =============================================================================
# Full Claude session workflow test
# =============================================================================


async def test_full_tmux_workflow(tmux_test_session):
    """Test complete workflow: create session -> send command -> get output."""
    import asyncio

    session_name = tmux_test_session

    # Step 1: Create session manually (simulating start_claude_session)
    subprocess.run(f"tmux new-session -d -s {session_name} bash", shell=True)

    # Step 2: Send a command using send_to_claude
    send_params = MockFunctionCallParams(
        arguments={
            "message": "echo 'WORKFLOW_TEST_MARKER_99999' && echo 'SECOND_LINE'",
            "session_name": session_name,
        }
    )
    await send_to_claude(send_params)
    assert send_params.result["success"] is True

    # Wait for command to execute
    await asyncio.sleep(0.5)

    # Step 3: Get output using get_claude_output
    get_params = MockFunctionCallParams(
        arguments={"session_name": session_name, "lines": 100}
    )
    await get_claude_output(get_params)

    assert get_params.result["success"] is True
    assert "WORKFLOW_TEST_MARKER_99999" in get_params.result["output"]
    assert "SECOND_LINE" in get_params.result["output"]


async def test_tmux_multiple_commands(tmux_test_session):
    """Test sending multiple commands and getting combined output."""
    import asyncio

    session_name = tmux_test_session
    subprocess.run(f"tmux new-session -d -s {session_name} bash", shell=True)

    # Send multiple commands
    for i in range(3):
        params = MockFunctionCallParams(
            arguments={"message": f"echo 'COMMAND_{i}'", "session_name": session_name}
        )
        await send_to_claude(params)
        assert params.result["success"] is True

    await asyncio.sleep(0.5)

    # Get all output
    get_params = MockFunctionCallParams(
        arguments={"session_name": session_name, "lines": 100}
    )
    await get_claude_output(get_params)

    # All commands should be in output
    for i in range(3):
        assert f"COMMAND_{i}" in get_params.result["output"]


async def test_tmux_special_characters(tmux_test_session):
    """Test sending messages with special characters."""
    import asyncio

    session_name = tmux_test_session
    subprocess.run(f"tmux new-session -d -s {session_name} bash", shell=True)

    # Send message with quotes and special chars
    params = MockFunctionCallParams(
        arguments={
            "message": "echo \"hello 'world' with $pecial chars\"",
            "session_name": session_name,
        }
    )
    await send_to_claude(params)
    assert params.result["success"] is True

    await asyncio.sleep(0.5)

    get_params = MockFunctionCallParams(
        arguments={"session_name": session_name, "lines": 50}
    )
    await get_claude_output(get_params)
    assert "hello" in get_params.result["output"]
