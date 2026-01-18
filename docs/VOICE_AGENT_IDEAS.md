# Voice Agent Control System - Ideas

## Architecture

```
Mac (you) ──SSH──► Linux (RTX 5070 + Voice Agent)
                      │
                      ├── Nemotron ASR (speech → text)
                      ├── LLM (Gemini Flash / local Nemotron)
                      ├── Magpie TTS (text → speech)
                      └── Tool execution layer
                            ├── Local shell commands
                            ├── SSH to Mac for Mac control
                            ├── Claude Code spawning
                            └── Todo/notification queue
```

## Modes

### 1. Command Mode (default)
- Listen for voice input
- LLM interprets intent
- Execute tools if needed
- Respond with TTS

### 2. Think-Out-Loud Mode
- Trigger: "I'm going to think out loud" / "transcribe mode"
- Just transcribes, no LLM responses
- Buffers everything said
- Exit triggers:
  - "send" / "execute" → send buffer to Claude Code
  - "cancel" → discard buffer
  - "save as note" → save to file

### 3. Monitoring Mode
- Background: watch for build completion, test results, etc.
- Proactively notify: "Hey, the build finished" or "Tests failed"

## Core Tools to Implement

```python
tools = [
    # System control
    "run_shell_command",      # Execute on Linux
    "run_on_mac",             # SSH + execute on Mac

    # Window/app management
    "open_warp_window",       # New terminal
    "open_project",           # cd + open in editor
    "start_claude_session",   # Launch Claude Code on project

    # Claude interaction
    "send_to_claude",         # Send prompt to active session
    "get_claude_status",      # Check if Claude is working

    # Productivity
    "get_next_todo",          # What should I work on?
    "add_todo",               # Add item to list
    "complete_todo",          # Mark done

    # Notifications
    "watch_process",          # Notify when PID exits
    "watch_file",             # Notify on file change
]
```

## Example Interactions

### Opening a project
> "Open the nemotron project in a new Warp window and start Claude"

Agent:
1. Calls `open_warp_window()`
2. Calls `run_shell_command("cd /home/simon/github/nemotron-january-2026")`
3. Calls `start_claude_session()`
4. Says "Done, Claude is starting on the nemotron project"

### Think-out-loud workflow
> "I'm going to think out loud"

Agent: "Ok, I'm listening. Say 'send' when you're ready."

> "So I want to refactor the authentication system to use JWT tokens instead of sessions, and we need to update the middleware to validate tokens, and also add refresh token support..."

Agent: [just transcribes, no response]

> "send"

Agent:
1. Takes transcript buffer
2. Calls `send_to_claude(buffer)`
3. Says "Sent to Claude. I'll let you know when there's progress."

### Build monitoring
> "Let me know when the Docker build finishes"

Agent:
1. Calls `watch_process(pid_of_docker_build)`
2. Says "Ok, I'll notify you when it's done"

[Later, build finishes]

Agent: "Hey, the Docker build finished successfully."

## Technical Notes

### Controlling Mac from Linux
```bash
# SSH with key auth (no password prompt)
ssh simon@macbook "open -a Warp"

# AppleScript via SSH for complex actions
ssh simon@macbook "osascript -e 'tell application \"Warp\" to activate'"
```

### Claude Code integration
```bash
# Start Claude Code in a tmux session
tmux new-session -d -s claude "cd /project && claude"

# Send input to Claude session
tmux send-keys -t claude "your prompt here" Enter

# Capture output (poll or watch)
tmux capture-pane -t claude -p
```

### State persistence
Simple JSON file for todos and state:
```json
{
  "todos": [
    {"id": 1, "text": "Fix auth bug", "done": false},
    {"id": 2, "text": "Update docs", "done": false}
  ],
  "watches": [
    {"type": "process", "pid": 12345, "notify": "build complete"}
  ]
}
```

## Questions to Resolve

1. **Audio routing**: If you're on Mac SSH'd to Linux, where does audio come from?
   - Option A: Run ASR/TTS on Linux, stream audio from Mac microphone
   - Option B: Run full voice agent on Mac, call Linux for GPU inference
   - Option C: WebRTC - open browser UI on Mac that connects to Linux agent

2. **LLM choice**:
   - Gemini Flash 2.5 (fast, good tool calling, API cost)
   - Local Nemotron (free, but less capable for complex tool orchestration)
   - Hybrid: local for simple, Gemini for complex

3. **Latency requirements**:
   - Real-time conversation: need fast TTS/ASR
   - Command-and-control: can tolerate more latency
