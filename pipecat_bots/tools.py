"""
Tool definitions for dual-agent bot.

Tools are defined here to keep the bot file clean and organized.
These are the actual tools from agent_gemini_live.py.
"""

import subprocess

from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams

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

# Map of function names to handlers for registration
TOOL_HANDLERS = {
    "run_shell_command": run_shell_command,
    "start_claude_session": start_claude_session,
    "send_to_claude": send_to_claude,
    "get_claude_output": get_claude_output,
    "run_mac_command": run_mac_command,
}
