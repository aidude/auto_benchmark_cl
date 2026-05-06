"""
Logging for auto_benchmark_cl.

Two outputs per run (keyed by RUN_ID = HHMMSS at import time):
  logs/YYYY-MM-DD/run_<RUN_ID>.log         — human-readable, all levels
  logs/YYYY-MM-DD/llm_calls_<RUN_ID>.jsonl — one JSON line per LLM API call

Usage:
  from src.utils.logger import get_logger, log_llm_call
  log = get_logger(__name__)
  log.info("started")
"""
import json
import logging
from datetime import datetime
from pathlib import Path

_NOW = datetime.now()
RUN_ID: str = _NOW.strftime("%H%M%S")
_DATE: str  = _NOW.strftime("%Y-%m-%d")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_LOG_DIR: Path = _PROJECT_ROOT / "logs" / _DATE
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_LOG_FILE: Path = _LOG_DIR / f"run_{RUN_ID}.log"
_LLM_FILE: Path = _LOG_DIR / f"llm_calls_{RUN_ID}.jsonl"

_FMT = logging.Formatter(
    "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
_FILE_FMT = logging.Formatter(
    "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)


def get_logger(name: str) -> logging.Logger:
    """Return a logger writing to console (INFO) and daily log file (DEBUG)."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(_FMT)

    fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_FILE_FMT)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def log_llm_call(
    *,
    model: str,
    provider: str,
    endpoint: str,
    messages: list[dict],
    response_text: str,
    usage: dict,
    latency_ms: float,
    cost_usd: float | None = None,
) -> None:
    """Append one structured record to llm_calls_<RUN_ID>.jsonl."""
    entry = {
        "ts":                datetime.now().isoformat(timespec="milliseconds"),
        "run_id":            RUN_ID,
        "model":             model,
        "provider":          provider,
        "endpoint":          endpoint,
        "prompt_tokens":     usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens":      usage.get("total_tokens"),
        "latency_ms":        round(latency_ms, 1),
        "cost_usd":          round(cost_usd, 6) if cost_usd is not None else None,
        "messages":          messages,
        "response":          response_text,
    }
    with _LLM_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
