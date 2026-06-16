import threading
import time
from collections import deque

import requests

from src.agent import config


class AdaptiveTimeout:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._api_window: deque[float] = deque(maxlen=10)
        self._ollama_window: deque[float] = deque(maxlen=10)

    def record_api_latency(self, latency_seconds: float) -> None:
        with self._lock:
            self._api_window.append(latency_seconds)

    def record_ollama_latency(self, latency_seconds: float) -> None:
        with self._lock:
            self._ollama_window.append(latency_seconds)

    def _api_avg(self) -> float:
        with self._lock:
            if not self._api_window:
                return 1.0
            return sum(self._api_window) / len(self._api_window)

    def _ollama_avg(self) -> float:
        with self._lock:
            if not self._ollama_window:
                return 1.0
            return sum(self._ollama_window) / len(self._ollama_window)

    def t_wait(self) -> int:
        raw = self._api_avg() * config.T_WAIT_MULTIPLIER
        return max(config.T_WAIT_MIN, min(config.T_WAIT_MAX, int(raw)))

    def t_behavior(self) -> int:
        raw = self._ollama_avg() * config.T_BEHAVIOR_MULTIPLIER
        return max(config.T_BEHAVIOR_MIN, min(config.T_BEHAVIOR_MAX, int(raw)))

    def ping_api(self) -> float:
        avg = self._api_avg()
        try:
            url = "https://paper-api.alpaca.markets/v2/clock"
            headers = {
                "APCA-API-KEY-ID": config.ALPACA_API_KEY,
                "APCA-API-SECRET-KEY": config.ALPACA_SECRET_KEY,
            }
            start = time.monotonic()
            requests.get(url, headers=headers, timeout=10)
            latency = time.monotonic() - start
            self.record_api_latency(latency)
            return latency
        except Exception:
            return avg

    def ping_ollama(self) -> float:
        avg = self._ollama_avg()
        try:
            start = time.monotonic()
            requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
            latency = time.monotonic() - start
            self.record_ollama_latency(latency)
            return latency
        except Exception:
            return avg

    def calibrate(self) -> None:
        for _ in range(3):
            self.ping_api()
            self.ping_ollama()

    def summary(self) -> dict:
        return {
            "api_avg": round(self._api_avg(), 3),
            "ollama_avg": round(self._ollama_avg(), 3),
            "t_wait": self.t_wait(),
            "t_behavior": self.t_behavior(),
        }
