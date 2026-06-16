"""
Shared LLM streaming utility.

Every ollama.generate() call in the project routes through `generate()` here.
It streams tokens in dark gray, then erases them with ANSI escape codes —
identical to the discovery-phase UX. A module-level lock prevents two
concurrent callers from interleaving output on the terminal.
"""
import os
import sys
import threading

import ollama

_print_lock = threading.Lock()


def generate(
    model: str,
    prompt: str,
    format,
    options: dict,
    keep_alive: str = "30s",
) -> str:
    """
    Drop-in replacement for ollama.generate() with gray streaming UX.
    Returns the full response string (equivalent to resp.response).
    Thread-safe: serialises terminal output via _print_lock.
    """
    with _print_lock:
        return _stream_and_erase(model, prompt, format, options, keep_alive)


def _stream_and_erase(
    model: str,
    prompt: str,
    format,
    options: dict,
    keep_alive: str,
) -> str:
    """Stream to stdout in gray, erase when done, return accumulated text."""
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
            token = chunk.get("response", "") if isinstance(chunk, dict) else getattr(chunk, "response", "")
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

    sys.stdout.write(f"\033[{newline_count}A\033[J")
    sys.stdout.flush()

    return full_text
