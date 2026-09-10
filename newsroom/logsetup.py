"""Structured JSON logging (architecture §11: logs carry item/event ids).

Named `logsetup` rather than `logging` on purpose: a module called `logging.py`
inside the package shadows the standard library's `logging` whenever the package
directory lands on sys.path[0] (e.g. running a file directly), which breaks every
`import logging` in the codebase.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    """One JSON object per line. Extra fields (item_id, event_id, source_id, …)
    passed via ``logger.info(msg, extra={...})`` are merged into the record."""

    _RESERVED = set(
        logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
    ) | {"message", "asctime", "taskName"}

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in self._RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(level: str | None = None) -> None:
    lvl = (level or os.getenv("NEWSROOM_LOG_LEVEL") or "INFO").upper()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(lvl)
