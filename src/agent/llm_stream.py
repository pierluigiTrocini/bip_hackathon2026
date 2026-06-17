"""
Shared LLM streaming utility.

Every ollama.generate() call in the project routes through `generate()` here.
It streams tokens in dark gray, then erases them after a 1-second pause —
identical to the discovery-phase UX. A module-level lock prevents two
concurrent callers from interleaving output on the terminal.

Verbosity:
  LOOP_VERBOSE=True  → show streaming output during the agent loop
  LOOP_VERBOSE=False → loop-phase calls run silently (discovery always visible)
  Set via main.py based on the -v / --verbose CLI flag.
"""
import os
import sys
import threading

import ollama

# Controlled by main.py: True when -v flag is passed, False (default) otherwise.
LOOP_VERBOSE: bool = False

_print_lock = threading.Lock()


def generate(
    model: str,
    prompt: str,
    format,
    options: dict,
    keep_alive: str = "30s",
    show_output: bool = True,
) -> str:
    """
    Drop-in replacement for ollama.generate() with optional gray streaming UX.

    show_output=True  → stream tokens in dark gray; box stays and scrolls naturally
    show_output=False → run silently, no terminal output

    Thread-safe: serialises terminal writes via _print_lock.
    """
    with _print_lock:
        if not show_output:
            return _call_silent(model, prompt, format, options, keep_alive)

        text, _ = _stream_tokens(model, prompt, format, options, keep_alive)

    return text


# ── Internal helpers ──────────────────────────────────────────────────────────

def _call_silent(model: str, prompt: str, format, options: dict, keep_alive: str) -> str:
    """Run ollama.generate() consuming all chunks without writing to stdout."""
    full_text = ""
    for chunk in ollama.generate(
        model=model,
        prompt=prompt,
        format=format,
        options=options,
        stream=True,
        keep_alive=keep_alive,
    ):
        token = (
            chunk.get("response", "")
            if isinstance(chunk, dict)
            else getattr(chunk, "response", "")
        )
        full_text += token
    return full_text


def _stream_tokens(
    model: str,
    prompt: str,
    format,
    options: dict,
    keep_alive: str,
) -> tuple[str, int]:
    """Stream tokens to stdout in dark gray. Returns (full_text, newline_count)."""
    try:
        cols = os.get_terminal_size().columns
    except OSError:
        cols = 80

    header = f"\033[90m  ╭─ {model} {'─' * max(0, cols - len(model) - 6)}\n  │ \033[0m"
    sys.stdout.write(header)
    sys.stdout.flush()

    full_text = ""
    newline_count = 1
    current_col = 4  # length of "  │ " prefix

    sys.stdout.write("\033[90m")
    sys.stdout.flush()

    try:
        for chunk in ollama.generate(
            model=model,
            prompt=prompt,
            format=format,
            options=options,
            stream=True,
            keep_alive=keep_alive,
        ):
            token = (
                chunk.get("response", "")
                if isinstance(chunk, dict)
                else getattr(chunk, "response", "")
            )
            if not token:
                continue
            full_text += token
            sys.stdout.write(token)
            sys.stdout.flush()

            for ch in token:
                if ch == "\n":
                    newline_count += 1
                    current_col = 0
                else:
                    current_col += 1
                    if current_col >= cols:
                        newline_count += 1
                        current_col = 0
    finally:
        sys.stdout.write("\033[0m")
        if current_col > 0:
            sys.stdout.write("\n")
            newline_count += 1
        sys.stdout.flush()

    return full_text, newline_count
