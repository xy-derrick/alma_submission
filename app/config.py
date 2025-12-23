from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG: Dict[str, Any] = {
    "gemini": {
        "enabled": False,
        "api_key": "AIzaSyA7aICMXXQupYlz6hgv1zfn4i0d14Z3g5g",
        "model": "gemini-2.5-flash",
        "summary_enabled": True,
        "structured_enabled": True,
        "validation_enabled": True,
        "fill_enabled": True,
        "max_chars": 12000,
    },
    "fill": {
        "provider": "browser_use"
    },
}


def load_config() -> Dict[str, Any]:
    base_dir = Path(__file__).resolve().parents[1]
    config_path = Path(os.getenv("CONFIG_PATH", base_dir / "config.json"))
    if not config_path.exists():
        return DEFAULT_CONFIG.copy()

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        LOGGER.warning("Config load failed: %s", exc)
        return DEFAULT_CONFIG.copy()

    return deep_merge(DEFAULT_CONFIG, data)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in base.items():
        result[key] = value
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result
