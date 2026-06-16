import os
from dotenv import load_dotenv

load_dotenv()

# Alpaca
ALPACA_API_KEY: str = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET_KEY: str = os.environ["ALPACA_SECRET_KEY"]
ALPACA_PAPER: bool = True  # HARDCODED — never change

# Ollama
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_REASONING_MODEL: str = os.getenv("OLLAMA_REASONING_MODEL", "gemma4:12b")
OLLAMA_SENTIMENT_MODEL: str = os.getenv("OLLAMA_SENTIMENT_MODEL", "qwen2.5:3b")

# Agent behaviour
TICKERS: list[str] = os.getenv("TICKERS", "AAPL,TSLA,NVDA,MSFT").split(",")
CONFIDENCE_THRESHOLD_NORMAL: float = float(os.getenv("CONFIDENCE_THRESHOLD_NORMAL", "0.65"))
CONFIDENCE_THRESHOLD_CONSERVATIVE: float = float(os.getenv("CONFIDENCE_THRESHOLD_CONSERVATIVE", "0.80"))
MAX_POSITION_PCT_NORMAL: float = float(os.getenv("MAX_POSITION_PCT_NORMAL", "0.10"))
MAX_POSITION_PCT_CONSERVATIVE: float = float(os.getenv("MAX_POSITION_PCT_CONSERVATIVE", "0.05"))
DRAWDOWN_THRESHOLD: float = float(os.getenv("DRAWDOWN_THRESHOLD", "0.05"))

# Adaptive timeout
T_WAIT_MULTIPLIER: float = float(os.getenv("T_WAIT_MULTIPLIER", "3.0"))
T_BEHAVIOR_MULTIPLIER: float = float(os.getenv("T_BEHAVIOR_MULTIPLIER", "5.0"))
T_WAIT_MIN: int = int(os.getenv("T_WAIT_MIN", "15"))
T_WAIT_MAX: int = int(os.getenv("T_WAIT_MAX", "120"))
T_BEHAVIOR_MIN: int = int(os.getenv("T_BEHAVIOR_MIN", "20"))
T_BEHAVIOR_MAX: int = int(os.getenv("T_BEHAVIOR_MAX", "180"))

# Memory
HOT_WINDOW_SIZE: int = int(os.getenv("HOT_WINDOW_SIZE", "5"))
WARM_COMPACTION_TRIGGER: int = int(os.getenv("WARM_COMPACTION_TRIGGER", "15"))

# Paths
JOURNAL_PATH: str = os.getenv("JOURNAL_PATH", "data/journal.jsonl")
ERROR_LOG_PATH: str = os.getenv("ERROR_LOG_PATH", "data/error_log.jsonl")
SESSION_PATH: str = os.getenv("SESSION_PATH", "data/session.json")
IMITATIVE_DATASET_PATH: str = os.getenv("IMITATIVE_DATASET_PATH", "data/strategies/imitative_dataset.json")
