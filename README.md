# BIP Hackathon 2026 — Autonomous Trading Agent

## Quick start

```bash
# 1. Install dependencies
uv sync

# 2. Pull Ollama models
ollama pull gemma4:12b
ollama pull qwen2.5:3b

# 3. Copy and fill env
cp .env.example .env
# edit .env with your Alpaca Paper Trading keys

# 4. Run infrastructure tests
uv run python tests/test_connections.py

# 5. Start the agent
uv run python main.py
```

## Architecture

Dual-model pipeline (Gemma4:12b + qwen2.5:3b), HOT/WARM/COLD memory,
adaptive timeout, imitative layer, persistent JSONL journal, rich TUI.
See CLAUDE_CODE_SPEC.md for full specification.

## Stack

- Broker: Alpaca Paper Trading (paper=True, always)
- Reasoning LLM: Gemma4:12b via Ollama
- Sentiment LLM: qwen2.5:3b via Ollama
- Journal: JSONL append-only, outcome tracking
- UI: rich terminal dashboard
- Python 3.12+, uv package manager