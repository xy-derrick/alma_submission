from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict

from dotenv import load_dotenv

from .schema import ExtractionResult

LOGGER = logging.getLogger(__name__)

FORM_URL = "https://mendrika-alma.github.io/form-submission/"

RULES = """
- Only use values from the provided JSON. Never guess.
- If a required field exists but JSON is missing, leave it blank and report it.
- Do NOT submit the form. After filling just stay the page.
""".strip()


async def fill_form_with_browser_use(data: ExtractionResult, url: str) -> Dict[str, Any]:
    if sys.platform.startswith("win"):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except Exception:
            pass
    load_dotenv()
    if os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY") or ""

    payload = {"passport": data.passport.model_dump(), "g28": data.g28.model_dump()}
    task = f"Open the form URL: {url or FORM_URL}\nFill the form using this JSON.\n\n{RULES}\n\nJSON:\n{json.dumps(payload, ensure_ascii=False)}"

    try:
        from browser_use import Agent, Browser, ChatGoogle  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        LOGGER.warning("Browser-use dependencies missing: %s", exc)
        return {"status": "error", "method": "browser_use", "error": "missing dependencies", "filled_fields": []}

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    apikey="AIzaSyA7aICMXXQupYlz6hgv1zfn4i0d14Z3g5g"
    llm = ChatGoogle(model='gemini-3-pro-preview', api_key=apikey)
    agent = Agent(task=task, llm=llm, browser=Browser())

    result = await agent.run()
    return {"status": "ok", "method": "browser_use", "result": str(result), "filled_fields": []}
