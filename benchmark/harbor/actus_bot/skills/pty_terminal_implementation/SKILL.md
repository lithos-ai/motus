---
name: pty_terminal_implementation
description: How to implement headless terminals and PTY-based process management correctly using high-level libraries, timeout protection, and safe subprocess testing patterns.
version: 1.0.1
required_tools: ['sandbox_sh']
---

# PTY and Terminal Implementation

Correctly implement headless terminals, PTY-based process management, and terminal emulation. This skill covers the critical pitfalls of low-level PTY APIs and the safe, reliable patterns that replace them.

## When to Use This Skill

- You need to implement a headless terminal or terminal emulator.
- You are building PTY-based process management (spawning shells, interactive programs).
- A task involves `pty`, `os.fork()`, `os.openpty()`, or terminal I/O in Python.
- You need to test code that spawns child processes or forks.
- You are implementing anything that wraps an interactive CLI program.

---

## Avoid Unnecessary Dependencies (YAGNI)

- **Do NOT use `pyte` for screen emulation unless the task interface explicitly requires screen reading functionality** (e.g., `get_screen_text()` method). pyte 0.8.2 has a known bug with private CSI sequences (from programs like vim, less, top) that causes `TypeError: Screen.select_graphic_rendition() got an unexpected keyword argument 'private'`.
- If you MUST use pyte, wrap ALL `stream.feed()` calls in `try/except TypeError` for graceful degradation.
- **Implement exactly what the interface requires — no more.** Read the abstract base class or interface definition carefully and implement the minimum viable solution. Don't add speculative features like screen parsing unless explicitly needed.
- When testing terminal implementations, test with complex programs (vim, less) not just simple commands, since these generate the widest variety of escape sequences.

---

## Rule 1: Prefer High-Level Libraries Over Raw System Calls

When implementing terminal emulation or headless terminal functionality, **ALWAYS** use `pexpect` (Python) rather than manual `pty.openpty()` + `os.fork()` combinations.

### Why the Manual Approach Fails

The raw PTY + fork pattern is extremely error-prone:

```python
# ❌ DANGEROUS — Do NOT use this pattern
import pty, os

master_fd, slave_fd = pty.openpty()  # Easy to swap these two!
pid = os.fork()                       # Causes execute_bash to hang
if pid == 0:
    # child process
    os.setsid()
    os.dup2(slave_fd, 0)
    os.dup2(slave_fd, 1)
    os.dup2(slave_fd, 2)
    os.execvp('/bin/bash', ['/bin/bash', '-i'])
else:
    # parent process
    os.close(slave_fd)
    # ... read/write master_fd ...
```

**Specific failure modes:**

| Problem | Consequence |
|---------|-------------|
| `pty.openpty()` returns `(master_fd, slave_fd)` — easy to swap | Reads/writes go to the wrong end; silent data loss or hangs |
| `os.fork()` creates a child process | `sandbox_sh` waits for ALL child processes to exit — tool hangs indefinitely |
| Signal handling must be done manually | Zombie processes, orphaned PTYs, resource leaks |
| Cleanup requires closing both FDs + reaping child | Missed cleanup leaves dangling processes and file descriptors |
| Terminal size, modes, echo settings are manual | Incorrect terminal behavior, garbled output |

### The Core Problem with `os.fork()` and `sandbox_sh`

This is the most critical issue: **`sandbox_sh` waits for all child processes to complete before returning**. If your code calls `os.fork()` and the child process runs an interactive shell or long-lived process, the tool call will hang for the entire step timeout (14+ minutes wasted), consuming your budget with zero useful output.

---

## Rule 2: Use `pexpect` — The Recommended Pattern

`pexpect` handles PTY creation, process lifecycle, and signal handling correctly in a single, battle-tested library.

### Installation

```bash
pip install pexpect
```

### Basic Pattern

```python
# ✅ CORRECT — Use pexpect for PTY management
import pexpect

# Spawn an interactive bash shell
child = pexpect.spawn('/bin/bash', args=['-i'], encoding=None)

# Set terminal size (optional but recommended)
child.setwinsize(24, 80)

# Send a command
child.sendline('echo "Hello from PTY"')

# Wait for expected output
child.expect(r'\$')  # Wait for the shell prompt

# Read output
output = child.before  # Bytes of output before the match

# Clean shutdown
child.sendline('exit')
child.close()
```

### Why `pexpect` Is Better

| Feature | `pexpect` | Manual `pty`+`fork` |
|---------|-----------|---------------------|
| PTY creation | Automatic | Manual `openpty()` |
| Process spawning | No `fork()` — uses `pty.fork()` internally with proper isolation | Raw `os.fork()` — hangs `sandbox_sh` |
| FD management | Automatic | Manual open/close/dup2 |
| Signal handling | Built-in | Must implement yourself |
| Timeout support | `child.expect(..., timeout=10)` | Must implement yourself |
| Cleanup on error | Automatic via destructor | Must handle in try/finally |
| Terminal sizing | `child.setwinsize(rows, cols)` | Manual `ioctl` calls |

### Common `pexpect` Operations

```python
import pexpect

child = pexpect.spawn('/bin/bash', args=['-i'], encoding='utf-8')
child.setwinsize(24, 80)

# Send command and capture output
child.sendline('ls -la /tmp')
child.expect(r'\$', timeout=10)
output = child.before

# Handle multiple possible outcomes
child.sendline('some_command')
index = child.expect(['Success', 'Error', pexpect.TIMEOUT], timeout=15)
if index == 0:
    print("Command succeeded")
elif index == 1:
    print("Command failed:", child.before)
elif index == 2:
    print("Command timed out")

# Run something interactive
child.sendline('python3')
child.expect('>>>', timeout=10)
child.sendline('print(1+1)')
child.expect('>>>', timeout=10)
print(child.before)  # Contains "2"
child.sendline('exit()')

# Always clean up
child.close()
```

---

## Rule 3: Timeout Protection for All Subprocess-Spawning Tests

When running or testing code that spawns child processes, forks, or starts servers, **ALWAYS** wrap execution with a timeout. A single hung process can consume the entire step budget.

### ✅ Safe Test Execution Patterns

**Pattern A: `timeout` wrapper (preferred)**

```bash
# Always use timeout when running code that might fork or spawn processes
timeout 30 python3 my_pty_script.py

# With a specific signal for cleanup
timeout --signal=KILL 30 python3 my_pty_script.py
```

**Pattern B: Background process with explicit kill**

```bash
# For server-like processes that need a client test
python3 server.py &
PID=$!
sleep 2
curl -s http://localhost:8080/health
kill $PID 2>/dev/null
wait $PID 2>/dev/null
```

**Pattern C: Subshell with timeout**

```bash
# Run in a subshell so it can be killed cleanly
(timeout 30 python3 test_interactive.py) 2>&1
echo "Exit code: $?"
```

### ❌ Dangerous Patterns — Never Do These

```bash
# NEVER run fork-heavy code without timeout protection
python3 pty_server.py              # May hang forever

# NEVER run interactive process tests without a kill mechanism
python3 -c "
import pty, os
pid = os.fork()
if pid == 0:
    os.execvp('/bin/bash', ['/bin/bash'])
"                                   # Will hang sandbox_sh

# NEVER start a server without a way to stop it
python3 -m http.server 8080        # Runs forever, wastes entire step
```

---

## Rule 4: Testing Subprocess-Heavy Code Safely

When you need to verify that PTY/terminal code works correctly, structure your tests so they **cannot** block the tool.

### Self-Terminating Test Script

```python
#!/usr/bin/env python3
"""Test script that is guaranteed to terminate."""
import signal
import sys

# Hard timeout — kill ourselves after 20 seconds no matter what
signal.alarm(20)

import pexpect

try:
    child = pexpect.spawn('/bin/bash', args=['-i'], encoding='utf-8', timeout=10)
    child.setwinsize(24, 80)

    # Test: send a command and verify output
    child.sendline('echo HELLO_WORLD')
    child.expect('HELLO_WORLD', timeout=5)
    print("TEST PASSED: Got expected output")

    child.sendline('exit')
    child.close()

except pexpect.TIMEOUT:
    print("TEST FAILED: Timeout waiting for output")
    sys.exit(1)
except Exception as e:
    print(f"TEST FAILED: {e}")
    sys.exit(1)
finally:
    # Ensure child is cleaned up
    try:
        child.close(force=True)
    except:
        pass

print("All tests passed")
```

Run it safely:

```bash
timeout 30 python3 test_pty.py
echo "Test exit code: $?"
```

### Testing a Long-Running PTY Server

```bash
# Write the test
cat > /tmp/test_server.py << 'PYEOF'
import pexpect, sys, time

# Start the server under test
server = pexpect.spawn('python3 my_server.py', timeout=10)

try:
    server.expect('Listening on', timeout=5)
    print("Server started successfully")

    # Run client tests here...

    print("All tests passed")
except pexpect.TIMEOUT:
    print("Server failed to start")
    sys.exit(1)
finally:
    server.terminate(force=True)
PYEOF

# Run with timeout protection
timeout 30 python3 /tmp/test_server.py
```

---

## Step-by-Step Workflow

### Phase 1: Set Up Environment

```bash
# Install pexpect if not already available
pip install pexpect 2>/dev/null || pip3 install pexpect 2>/dev/null
python3 -c "import pexpect; print('pexpect version:', pexpect.__version__)"
```

### Phase 2: Implement Using `pexpect`

1. Import `pexpect` — never import `pty` and `os.fork` for process spawning.
2. Use `pexpect.spawn()` to create the PTY-managed process.
3. Use `child.expect()` with timeouts for all I/O operations.
4. Use `child.close()` or `child.terminate()` for cleanup.

### Phase 3: Test Safely

1. Write a self-contained test script with `signal.alarm()` as a hard deadline.
2. Run with `timeout 30 python3 test_script.py`.
3. Verify exit code: `echo $?` — should be 0 for success.
4. If the test hangs despite timeout, check for background processes: `ps aux | grep python`.

### Phase 4: Clean Up

1. Kill any leftover processes: `pkill -f "my_server.py"` (if applicable).
2. Remove temp test files: `rm -f /tmp/test_*.py`.

---

## Error Handling

### Test Hangs Despite Timeout

```bash
# Find and kill stuck processes
ps aux | grep -E "(python|bash)" | grep -v grep
kill -9 <PID>

# Nuclear option: kill all python children
pkill -9 -f "my_script.py"
```

### `pexpect.TIMEOUT` During Normal Operation

- Increase the timeout: `child.expect('pattern', timeout=30)`
- Check if the process is still alive: `child.isalive()`
- Look at what was received: `print(child.before)`

### `pexpect` Not Available

```bash
# Try multiple install methods
pip install pexpect || pip3 install pexpect || python3 -m pip install pexpect

# If all fail, use the vendored pure-Python version
python3 -c "
import urllib.request, zipfile, io
url = 'https://files.pythonhosted.org/packages/source/p/pexpect/pexpect-4.9.0.tar.gz'
# ... fallback installation
"
```

### Process Won't Die

```bash
# Escalate signal strength
kill $PID          # SIGTERM — polite
sleep 2
kill -9 $PID       # SIGKILL — force
wait $PID 2>/dev/null
```

---

## Quick Reference

| Task | ✅ Do This | ❌ Not This |
|------|-----------|------------|
| Spawn interactive process | `pexpect.spawn('/bin/bash')` | `pty.openpty()` + `os.fork()` |
| Wait for output | `child.expect('pattern', timeout=10)` | `os.read(master_fd, 1024)` in a loop |
| Set terminal size | `child.setwinsize(24, 80)` | `fcntl.ioctl(fd, termios.TIOCSWINSZ, ...)` |
| Clean up process | `child.close()` | Manual `os.kill()` + `os.waitpid()` + `os.close()` |
| Run PTY test | `timeout 30 python3 test.py` | `python3 test.py` (no timeout) |
| Test a server | Background + sleep + kill pattern | Direct execution (hangs forever) |

---

## Examples

### Example 1: Headless Terminal That Runs Commands

```python
import pexpect

class HeadlessTerminal:
    def __init__(self, rows=24, cols=80):
        self.child = pexpect.spawn('/bin/bash', args=['-i'], encoding='utf-8', timeout=30)
        self.child.setwinsize(rows, cols)
        # Wait for initial prompt
        self.child.expect(r'[\$#]', timeout=10)

    def run_command(self, cmd, timeout=10):
        """Send a command and return its output."""
        self.child.sendline(cmd)
        self.child.expect(r'[\$#]', timeout=timeout)
        return self.child.before.strip()

    def close(self):
        """Clean shutdown."""
        self.child.sendline('exit')
        self.child.close()

# Usage
term = HeadlessTerminal()
output = term.run_command('whoami')
print(f"User: {output}")
term.close()
```

### Example 2: Testing the Headless Terminal

```bash
cat > /tmp/test_headless.py << 'PYEOF'
import signal, sys
signal.alarm(20)  # Hard 20-second deadline

import pexpect

child = pexpect.spawn('/bin/bash', args=['-i'], encoding='utf-8', timeout=10)
child.setwinsize(24, 80)
child.expect(r'[\$#]', timeout=5)

# Test 1: Simple command
child.sendline('echo TEST_OUTPUT_12345')
child.expect('TEST_OUTPUT_12345', timeout=5)
print("PASS: echo command works")

# Test 2: Command with exit code
child.sendline('ls /nonexistent 2>&1; echo "EXIT:$?"')
child.expect(r'EXIT:\d+', timeout=5)
print("PASS: exit code capture works")

child.sendline('exit')
child.close()
print("All tests passed")
PYEOF

timeout 30 python3 /tmp/test_headless.py
echo "Exit: $?"
```

### Example 3: Safe Server Testing Pattern

```bash
# Start server in background with a PID file
timeout 30 python3 my_server.py &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server to be ready
for i in $(seq 1 10); do
    if curl -s http://localhost:8080/health > /dev/null 2>&1; then
        echo "Server is ready"
        break
    fi
    sleep 1
done

# Run tests
curl -s http://localhost:8080/api/test
echo "Test exit code: $?"

# Always clean up
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
echo "Server stopped"
```
