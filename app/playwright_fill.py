from __future__ import annotations

import os
import re
import time
from typing import Dict, List, Optional, Tuple

import logging

from playwright.sync_api import sync_playwright

from .config import load_config
from .schema import ExtractionResult
from .summarizer import generate_fill_plan

FORM_URL = "https://mendrika-alma.github.io/form-submission/"
LOGGER = logging.getLogger(__name__)


def fill_form(data: ExtractionResult) -> Dict[str, object]:
    url = os.getenv("FORM_URL", FORM_URL)
    headless = os.getenv("PLAYWRIGHT_HEADLESS", "false").lower() in ("1", "true", "yes")
    hold_seconds = int(os.getenv("PLAYWRIGHT_HOLD_SECONDS", "100"))
    provider = resolve_fill_provider()

    if provider == "browser_use":
        try:
            from .browser_use_fill import fill_form_with_browser_use
        except Exception as exc:  # pragma: no cover - optional dependency
            LOGGER.warning("Stage: browser_use import failed | error=%s", exc)
        else:
            LOGGER.info("Stage: fill provider | provider=browser_use")
            return fill_form_with_browser_use(data, url)

    filled_fields: List[str] = []
    LOGGER.info("Stage: Playwright launch | headless=%s", headless)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless)
        page = browser.new_page()
        page.set_default_timeout(3000)
        page.goto(url, wait_until="domcontentloaded")
        LOGGER.info("Stage: Playwright navigate | url=%s", url)

        if try_fill_with_gemini(page, data, filled_fields):
            LOGGER.info("Stage: Playwright fill | source=gemini")
        else:
            LOGGER.info("Stage: Playwright fill | source=heuristics")

        if not filled_fields:
            first, middle, last = split_name(data)
            if try_fill_text(page, [r"first name", r"given name"], first):
                filled_fields.append("passport.given_name")
            if try_fill_text(page, [r"middle name"], middle):
                filled_fields.append("passport.middle_name")
            if try_fill_text(page, [r"last name", r"surname", r"family name"], last):
                filled_fields.append("passport.surname")

            if try_fill_text(page, [r"date of birth", r"\bDOB\b", r"birth date"], data.passport.dob):
                filled_fields.append("passport.dob")

            if data.passport.sex:
                if try_select(page, [r"sex", r"gender"], data.passport.sex):
                    filled_fields.append("passport.sex")
                elif try_check_label(page, [data.passport.sex]):
                    filled_fields.append("passport.sex")

            if try_select(page, [r"country", r"nationality", r"citizenship"], data.passport.country):
                filled_fields.append("passport.country")
            else:
                if try_fill_text(page, [r"country", r"nationality", r"citizenship"], data.passport.country):
                    filled_fields.append("passport.country")

            if try_fill_text(
                page,
                [r"passport number", r"document number", r"passport no"],
                data.passport.number,
            ):
                filled_fields.append("passport.number")

            if try_fill_text(
                page,
                [r"passport.*expiry", r"expiration", r"expiry date", r"date of expiry"],
                data.passport.expiry_date,
            ):
                filled_fields.append("passport.expiry_date")

            client_first, client_middle, client_last = split_client_name(data)
            if try_fill_text(page, [r"client first name", r"applicant first name"], client_first):
                filled_fields.append("g28.client.given_name")
            if try_fill_text(page, [r"client middle name", r"applicant middle name"], client_middle):
                filled_fields.append("g28.client.middle_name")
            if try_fill_text(page, [r"client last name", r"applicant last name"], client_last):
                filled_fields.append("g28.client.surname")
            if try_fill_text(page, [r"client name", r"applicant name", r"person represented"], data.g28.client.full_name):
                filled_fields.append("g28.client.full_name")
            if try_fill_text(page, [r"client date of birth", r"applicant date of birth"], data.g28.client.dob):
                filled_fields.append("g28.client.dob")
            if try_fill_text(page, [r"client country", r"applicant country"], data.g28.client.country):
                filled_fields.append("g28.client.country")
            if try_fill_text(page, [r"client address", r"applicant address"], data.g28.client.address):
                filled_fields.append("g28.client.address")
            if try_fill_text(page, [r"client city", r"applicant city"], data.g28.client.city):
                filled_fields.append("g28.client.city")
            if try_select(page, [r"client state", r"applicant state"], data.g28.client.state):
                filled_fields.append("g28.client.state")
            elif try_fill_text(page, [r"client state", r"applicant state"], data.g28.client.state):
                filled_fields.append("g28.client.state")
            if try_fill_text(page, [r"client zip", r"applicant zip", r"client postal"], data.g28.client.zip):
                filled_fields.append("g28.client.zip")
            if try_fill_text(page, [r"client email", r"applicant email"], data.g28.client.email):
                filled_fields.append("g28.client.email")
            if try_fill_text(page, [r"client phone", r"applicant phone"], data.g28.client.phone):
                filled_fields.append("g28.client.phone")

            if try_fill_text(
                page, [r"attorney name", r"representative name", r"name of attorney"], data.g28.attorney.name
            ):
                filled_fields.append("g28.attorney.name")
            if try_fill_text(page, [r"firm", r"law firm", r"company"], data.g28.attorney.firm):
                filled_fields.append("g28.attorney.firm")
            if try_fill_text(
                page,
                [r"attorney address", r"representative address", r"address", r"street"],
                data.g28.attorney.address,
            ):
                filled_fields.append("g28.attorney.address")
            if try_fill_text(page, [r"attorney city", r"city"], data.g28.attorney.city):
                filled_fields.append("g28.attorney.city")
            if try_select(page, [r"attorney state", r"state"], data.g28.attorney.state):
                filled_fields.append("g28.attorney.state")
            elif try_fill_text(page, [r"attorney state", r"state"], data.g28.attorney.state):
                filled_fields.append("g28.attorney.state")
            if try_fill_text(page, [r"attorney zip", r"postal", r"zip"], data.g28.attorney.zip):
                filled_fields.append("g28.attorney.zip")
            if try_fill_text(page, [r"attorney email", r"email"], data.g28.attorney.email):
                filled_fields.append("g28.attorney.email")
            if try_fill_text(page, [r"attorney phone", r"telephone", r"tel"], data.g28.attorney.phone):
                filled_fields.append("g28.attorney.phone")

        if hold_seconds > 0 and not headless:
            LOGGER.info("Stage: Playwright hold | seconds=%d", hold_seconds)
            time.sleep(hold_seconds)

        browser.close()
        LOGGER.info("Stage: Playwright done | filled=%d", len(filled_fields))

    return {"status": "ok", "url": url, "filled_fields": filled_fields}


def split_name(data: ExtractionResult) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    first = data.passport.given_name
    last = data.passport.surname
    middle = None

    if not first and not last and data.passport.full_name:
        parts = data.passport.full_name.split()
        if len(parts) == 1:
            first = parts[0]
        elif len(parts) == 2:
            first, last = parts
        else:
            first = parts[0]
            last = parts[-1]
            middle = " ".join(parts[1:-1])

    return first, middle, last


def split_client_name(data: ExtractionResult) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    first = data.g28.client.given_name
    last = data.g28.client.surname
    middle = None

    if not first and not last and data.g28.client.full_name:
        parts = data.g28.client.full_name.split()
        if len(parts) == 1:
            first = parts[0]
        elif len(parts) == 2:
            first, last = parts
        else:
            first = parts[0]
            last = parts[-1]
            middle = " ".join(parts[1:-1])

    return first, middle, last


def try_fill_text(page, labels: List[str], value: Optional[str]) -> bool:
    if not value:
        LOGGER.info("Stage: fill text | result=skipped (empty value)")
        return False
    for label in labels:
        locator = page.get_by_label(re.compile(label, re.IGNORECASE))
        if locator.count():
            locator.first.fill(value)
            LOGGER.info("Stage: fill text | result=filled | label=%s | value=%s", label, value)
            return True
    LOGGER.info("Stage: fill text | result=no match | value=%s", value)
    return False


def try_select(page, labels: List[str], value: Optional[str]) -> bool:
    if not value:
        LOGGER.info("Stage: select option | result=skipped (empty value)")
        return False
    for label in labels:
        locator = page.get_by_label(re.compile(label, re.IGNORECASE))
        if locator.count():
            try:
                locator.first.select_option(label=value)
            except Exception:
                try:
                    locator.first.select_option(value=value)
                except Exception:
                    LOGGER.info("Stage: select option | result=failed | label=%s | value=%s", label, value)
                    return False
            LOGGER.info("Stage: select option | result=selected | label=%s | value=%s", label, value)
            return True
    LOGGER.info("Stage: select option | result=no match | value=%s", value)
    return False


def try_check_label(page, labels: List[str]) -> bool:
    for label in labels:
        locator = page.get_by_label(re.compile(label, re.IGNORECASE))
        if locator.count():
            try:
                locator.first.check()
            except Exception:
                locator.first.click()
            LOGGER.info("Stage: check label | result=checked | label=%s", label)
            return True
    LOGGER.info("Stage: check label | result=no match")
    return False


def resolve_fill_provider() -> str:
    env_provider = os.getenv("FILL_PROVIDER")
    if env_provider:
        return env_provider.strip().lower()
    config = load_config().get("fill", {})
    provider = config.get("provider", "playwright")
    return str(provider).strip().lower()


def try_fill_with_gemini(page, data: ExtractionResult, filled_fields: List[str]) -> bool:
    form_fields = collect_form_fields(page)
    if not form_fields:
        LOGGER.info("Stage: Gemini fill | skipped (no form fields)")
        return False
    extraction_payload = {
        "passport": data.passport.model_dump(),
        "g28": data.g28.model_dump(),
    }
    plan = generate_fill_plan(extraction_payload, form_fields)
    if not plan:
        return False
    apply_fill_plan(page, plan, filled_fields)
    return len(filled_fields) > 0


def collect_form_fields(page) -> List[Dict[str, str]]:
    script = """
    () => {
      const fields = [];
      const seen = new Set();

      function addField(labelText, control) {
        if (!labelText || !control) return;
        const label = labelText.trim();
        if (!label) return;
        const key = `${label}::${control.tagName}::${control.name || ''}::${control.id || ''}`;
        if (seen.has(key)) return;
        seen.add(key);
        fields.push({
          label,
          tag: control.tagName,
          type: control.type || '',
          name: control.name || '',
          id: control.id || '',
          placeholder: control.placeholder || ''
        });
      }

      document.querySelectorAll('label').forEach(label => {
        const text = label.innerText || label.textContent || '';
        let control = null;
        if (label.htmlFor) {
          control = document.getElementById(label.htmlFor);
        }
        if (!control) {
          control = label.querySelector('input, select, textarea');
        }
        addField(text, control);
      });

      document.querySelectorAll('input[aria-label], select[aria-label], textarea[aria-label]').forEach(control => {
        const text = control.getAttribute('aria-label') || '';
        addField(text, control);
      });

      return fields;
    }
    """
    try:
        return page.evaluate(script)
    except Exception as exc:
        LOGGER.warning("Stage: Gemini fill | field collection failed: %s", exc)
        return []


def apply_fill_plan(page, plan: List[Dict[str, str]], filled_fields: List[str]) -> None:
    for item in plan:
        label = item.get("label", "").strip()
        value = item.get("value", "").strip()
        if not label or not value:
            continue
        locator = page.get_by_label(re.compile(re.escape(label), re.IGNORECASE))
        if not locator.count():
            continue
        target = locator.first
        try:
            tag = target.evaluate("el => el.tagName")
            input_type = target.evaluate("el => el.type || ''")
        except Exception:
            tag = ""
            input_type = ""

        if tag == "SELECT":
            try:
                target.select_option(label=value)
            except Exception:
                try:
                    target.select_option(value=value)
                except Exception:
                    continue
        elif input_type in ("checkbox", "radio"):
            if value.lower() in ("yes", "true", "1", "checked", "x"):
                try:
                    target.check()
                except Exception:
                    target.click()
            else:
                continue
        else:
            target.fill(value)

        filled_fields.append(f"gemini:{label}")
