import os
from dotenv import load_dotenv

load_dotenv()

# Alpaca
ALPACA_API_KEY: str = os.environ["ALPACA_API_KEY"]
ALPACA_SECRET_KEY: str = os.environ["ALPACA_SECRET_KEY"]
ALPACA_PAPER: bool = True  # HARDCODED — never change

# External data
NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
POLYGON_API_KEY: str = os.getenv("POLYGON_API_KEY", "")

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

# News log
NEWS_LOG_PATH:             str = os.getenv("NEWS_LOG_PATH", "data/news_log.jsonl")
NEWS_LOG_COMPACT_EVERY:    int = int(os.getenv("NEWS_LOG_COMPACT_EVERY", "100"))
NEWS_LOG_MAX_PER_TICKER:   int = int(os.getenv("NEWS_LOG_MAX_PER_TICKER", "50"))
NEWS_DISPLAY_MAX_ARTICLES: int = int(os.getenv("NEWS_DISPLAY_MAX_ARTICLES", "3"))

# Correlation Engine (NCCI)
NCCI_REBUILD_EVERY:       int   = int(os.getenv("NCCI_REBUILD_EVERY", "10"))
NCCI_THRESHOLD_DISPLAY:   float = float(os.getenv("NCCI_THRESHOLD_DISPLAY", "0.20"))
NCCI_KEYWORD_MIN_WEIGHT:  float = float(os.getenv("NCCI_KEYWORD_MIN_WEIGHT", "0.15"))
NCCI_HALF_LIFE_DAYS:      float = float(os.getenv("NCCI_HALF_LIFE_DAYS", "7.0"))

# Market Disruptor
DISRUPTOR_NEWS_PATH: str = os.getenv("DISRUPTOR_NEWS_PATH", "data/disruptor_news.jsonl")

# ── F1: News Context Layer ────────────────────────────────────────────────────
NEWS_CONTEXT_HISTORY_CYCLES: int   = int(os.getenv("NEWS_CONTEXT_HISTORY_CYCLES",   "5"))
NEWS_CONTEXT_MAX_ARTICLES:   int   = int(os.getenv("NEWS_CONTEXT_MAX_ARTICLES",     "6"))
NEWS_CONTEXT_MIN_RELEVANCE:  float = float(os.getenv("NEWS_CONTEXT_MIN_RELEVANCE",  "0.50"))

# ── F2: Position Manager ──────────────────────────────────────────────────────
POSITION_MIN_STOP_LOSS_PCT:     float = float(os.getenv("POSITION_MIN_STOP_LOSS_PCT",     "2.0"))
POSITION_MAX_STOP_LOSS_PCT:     float = float(os.getenv("POSITION_MAX_STOP_LOSS_PCT",     "8.0"))
POSITION_MIN_TAKE_PROFIT_PCT:   float = float(os.getenv("POSITION_MIN_TAKE_PROFIT_PCT",   "3.0"))
POSITION_MAX_TAKE_PROFIT_PCT:   float = float(os.getenv("POSITION_MAX_TAKE_PROFIT_PCT",   "15.0"))
POSITION_VOLATILITY_MULTIPLIER: float = float(os.getenv("POSITION_VOLATILITY_MULTIPLIER", "3.0"))
POSITION_HISTORY_CYCLES:        int   = int(os.getenv("POSITION_HISTORY_CYCLES",          "8"))
POSITION_SENTIMENT_TREND_WINDOW:int   = int(os.getenv("POSITION_SENTIMENT_TREND_WINDOW",  "3"))

# ── F3: Technical Indicators ──────────────────────────────────────────────────
TECHNICAL_BARS_LOOKBACK:  int   = int(os.getenv("TECHNICAL_BARS_LOOKBACK",    "20"))
TECHNICAL_RSI_PERIOD:     int   = int(os.getenv("TECHNICAL_RSI_PERIOD",       "14"))
TECHNICAL_BB_PERIOD:      int   = int(os.getenv("TECHNICAL_BB_PERIOD",        "20"))
TECHNICAL_BB_STD:         float = float(os.getenv("TECHNICAL_BB_STD",         "2.0"))
TECHNICAL_RSI_OVERBOUGHT: float = float(os.getenv("TECHNICAL_RSI_OVERBOUGHT", "70.0"))
TECHNICAL_RSI_OVERSOLD:   float = float(os.getenv("TECHNICAL_RSI_OVERSOLD",   "30.0"))
TECHNICAL_BB_SQUEEZE_PCT: float = float(os.getenv("TECHNICAL_BB_SQUEEZE_PCT", "1.5"))

# ── F4: User Preference Engine ────────────────────────────────────────────────
PREFERENCE_UPDATE_EVERY:       int   = int(os.getenv("PREFERENCE_UPDATE_EVERY",        "5"))
PREFERENCE_WAIT_HISTORY:       int   = int(os.getenv("PREFERENCE_WAIT_HISTORY",        "10"))
PREFERENCE_CONFLICT_THRESHOLD: float = float(os.getenv("PREFERENCE_CONFLICT_THRESHOLD","0.05"))
PREFERENCE_EMOTION_WEIGHT:     float = float(os.getenv("PREFERENCE_EMOTION_WEIGHT",    "0.3"))
PREFERENCE_STYLE_WEIGHT:       float = float(os.getenv("PREFERENCE_STYLE_WEIGHT",      "0.4"))

# Telegram Bot
TELEGRAM_BOT_TOKEN:     str = os.getenv("TELEGRAM_BOT_TOKEN", "DISABLED")
TELEGRAM_CHAT_ID:       str = os.getenv("TELEGRAM_CHAT_ID",   "0")
# Persisted snapshot of the last completed cycle — lets /resume and /nerd answer
# even before the first cycle of a fresh run completes (or right after a restart).
TELEGRAM_SNAPSHOT_PATH: str = os.getenv("TELEGRAM_SNAPSHOT_PATH", "data/telegram_snapshot.json")
