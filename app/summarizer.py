from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from google import genai  # type: ignore

from .config import load_config

LOGGER = logging.getLogger(__name__)


def summarize_text(text: str) -> Optional[str]:
    config = load_config().get("gemini", {})
    if not config.get("enabled"):
        LOGGER.info("Stage: Gemini summary | skipped (disabled)")
        return None
    if not config.get("summary_enabled", True):
        LOGGER.info("Stage: Gemini summary | skipped (summary disabled)")
        return None

    client = build_client(config)
    if client is None:
        return None

    model = config.get("model", "gemini-2.5-flash")
    trimmed_text = trim_text(text, config)
    prompt = (
        "Summarize the extracted document text into a concise list of key facts. "
        "Include names, dates, IDs, addresses, firm details, and contact info if present. "
        "Keep it factual and avoid speculation.\n\n"
        f"Text:\n{trimmed_text}"
    )
    return generate_text(client, model, prompt, stage="Gemini summary")


def extract_structured_fields(text: str) -> Optional[Dict[str, Any]]:
    config = load_config().get("gemini", {})
    if not config.get("enabled"):
        LOGGER.info("Stage: Gemini JSON | skipped (disabled)")
        return None
    if not config.get("structured_enabled", True):
        LOGGER.info("Stage: Gemini JSON | skipped (structured disabled)")
        return None

    client = build_client(config)
    if client is None:
        return None

    model = config.get("model", "gemini-2.5-flash")
    trimmed_text = trim_text(text, config)
    prompt = (
        "Extract the document into JSON using this schema exactly. "
        "Return ONLY valid JSON, no markdown or commentary. "
        "Use null for missing values. Dates should be ISO (YYYY-MM-DD) when possible.\n\n"
        "{\n"
        '  "passport": {\n'
        '    "full_name": null,\n'
        '    "given_name": null,\n'
        '    "surname": null,\n'
        '    "dob": null,\n'
        '    "country": null,\n'
        '    "sex": null,\n'
        '    "number": null,\n'
        '    "issue_country": null,\n'
        '    "expiry_date": null\n'
        "  },\n"
        '  "g28": {\n'
        '    "client": {\n'
        '      "full_name": null,\n'
        '      "given_name": null,\n'
        '      "surname": null,\n'
        '      "dob": null,\n'
        '      "country": null,\n'
        '      "address": null,\n'
        '      "city": null,\n'
        '      "state": null,\n'
        '      "zip": null,\n'
        '      "email": null,\n'
        '      "phone": null\n'
        "    },\n"
        '    "attorney": {\n'
        '      "name": null,\n'
        '      "firm": null,\n'
        '      "address": null,\n'
        '      "city": null,\n'
        '      "state": null,\n'
        '      "zip": null,\n'
        '      "email": null,\n'
        '      "phone": null\n'
        "    }\n"
        "  }\n"
        "}\n\n"
        f"Text:\n{trimmed_text}"
    )

    raw = generate_text(client, model, prompt, stage="Gemini JSON")
    if not raw:
        return None
    payload = extract_json_from_text(raw)
    if isinstance(payload, dict):
        LOGGER.info("Stage: Gemini JSON | result=ok")
        return payload
    LOGGER.warning("Stage: Gemini JSON | result=invalid")
    return None


def validate_extraction(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    config = load_config().get("gemini", {})
    if not config.get("enabled"):
        LOGGER.info("Stage: Gemini validation | skipped (disabled)")
        return None
    if not config.get("validation_enabled", True):
        LOGGER.info("Stage: Gemini validation | skipped (validation disabled)")
        return None

    client = build_client(config)
    if client is None:
        return None

    model = config.get("model", "gemini-2.5-flash")
    payload_text = json.dumps(payload, ensure_ascii=True)
    prompt = (
        "You are validating extracted data from a passport and a G-28 form. "
        "Check whether the G-28 client matches the passport holder. "
        "Also check for inconsistent or suspicious data (e.g., missing required fields, "
        "invalid date order, mismatched names, invalid email/phone formats). "
        "Return ONLY valid JSON, no markdown or commentary.\n\n"
        "{\n"
        '  "passport_matches_client": null,\n'
        '  "issues": []\n'
        "}\n\n"
        f"Extracted:\n{payload_text}"
    )

    raw = generate_text(client, model, prompt, stage="Gemini validation")
    if not raw:
        return None
    result = extract_json_from_text(raw)
    if isinstance(result, dict):
        issues = result.get("issues")
        if issues is None:
            result["issues"] = []
        LOGGER.info("Stage: Gemini validation | result=ok")
        return result
    LOGGER.warning("Stage: Gemini validation | result=invalid")
    return None


def generate_fill_plan(
    extraction: Dict[str, Any], form_fields: List[Dict[str, Any]]
) -> Optional[List[Dict[str, str]]]:
    config = load_config().get("gemini", {})
    if not config.get("enabled"):
        LOGGER.info("Stage: Gemini fill | skipped (disabled)")
        return None
    if not config.get("fill_enabled", True):
        LOGGER.info("Stage: Gemini fill | skipped (fill disabled)")
        return None

    client = build_client(config)
    if client is None:
        return None

    model = config.get("model", "gemini-2.5-flash")
    fields_text = json.dumps(form_fields, ensure_ascii=True)
    extraction_text = json.dumps(extraction, ensure_ascii=True)
    prompt = (
        "You are filling a web form. Given the extracted data and the list of form fields, "
        "return ONLY valid JSON that maps values to existing labels. "
        "Use this format exactly:\n"
        "{\n"
        '  "fills": [\n'
        '    {"label": "...", "value": "..."}\n'
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- Use only labels from the provided fields list.\n"
        "- Omit fields you cannot confidently map.\n"
        "- Keep values as plain strings.\n\n"
        f"Form fields:\n{fields_text}\n\n"
        f"Extracted data:\n{extraction_text}"
    )

    raw = generate_text(client, model, prompt, stage="Gemini fill")
    if not raw:
        return None
    payload = extract_json_from_text(raw)
    if not isinstance(payload, dict):
        LOGGER.warning("Stage: Gemini fill | result=invalid")
        return None
    fills = payload.get("fills")
    if not isinstance(fills, list):
        LOGGER.warning("Stage: Gemini fill | result=missing fills")
        return None
    normalized: List[Dict[str, str]] = []
    for item in fills:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        value = str(item.get("value", "")).strip()
        if not label or not value:
            continue
        normalized.append({"label": label, "value": value})
    LOGGER.info("Stage: Gemini fill | result=%d items", len(normalized))
    return normalized

def build_client(config: Dict[str, Any]) -> Optional[genai.Client]:
    api_key = (config.get("api_key") or os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        LOGGER.warning("Stage: Gemini | skipped (missing api_key)")
        return None
    return genai.Client(api_key=api_key)


def trim_text(text: str, config: Dict[str, Any]) -> str:
    max_chars = int(config.get("max_chars", 12000))
    return text[:max_chars]


def generate_text(client: genai.Client, model: str, prompt: str, *, stage: str) -> Optional[str]:
    try:
        LOGGER.info("Stage: %s | request=sent | model=%s", stage, model)
        response = client.models.generate_content(model=model, contents=prompt)
        output = (response.text or "").strip()
        if output:
            LOGGER.info("Stage: %s | result=ok", stage)
        else:
            LOGGER.warning("Stage: %s | result=empty", stage)
        return output or None
    except Exception as exc:
        LOGGER.warning("Stage: %s | error=%s", stage, exc)
        return None


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = text[start : end + 1].strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None
