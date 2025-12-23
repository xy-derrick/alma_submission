from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .browser_use_fill import fill_form_with_browser_use
from .extractors import run_extraction, run_validation
from .playwright_fill import FORM_URL, fill_form, resolve_fill_provider
from .schema import ExtractionResult, ValidationResult

logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Document Extraction + Form Population")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/extract", response_model=ExtractionResult)
async def extract(
    passport: Optional[UploadFile] = File(None),
    g28: Optional[UploadFile] = File(None),
) -> ExtractionResult:
    if passport is None and g28 is None:
        raise HTTPException(status_code=400, detail="Provide a passport and/or G-28 file.")

    logging.info(
        "Stage: /extract request | passport=%s | g28=%s",
        passport.filename if passport else "none",
        g28.filename if g28 else "none",
    )
    passport_bytes = await passport.read() if passport else None
    g28_bytes = await g28.read() if g28 else None

    result = run_extraction(
        passport_bytes=passport_bytes,
        passport_name=passport.filename if passport else None,
        g28_bytes=g28_bytes,
        g28_name=g28.filename if g28 else None,
    )
    logging.info("Stage: /extract response | missing=%d", len(result.missing_fields))
    return result


@app.post("/populate")
async def populate(payload: ExtractionResult) -> dict:
    payload.compute_missing()
    logging.info("Stage: /populate request | missing=%d", len(payload.missing_fields))
    provider = resolve_fill_provider()
    if provider == "browser_use":
        url = os.getenv("FORM_URL", FORM_URL)
        result = await fill_form_with_browser_use(payload, url)
    else:
        result = await run_in_threadpool(fill_form, payload)
    logging.info("Stage: /populate response | filled=%d", len(result.get("filled_fields", [])))
    return result


@app.post("/validate", response_model=ValidationResult)
async def validate(payload: ExtractionResult) -> ValidationResult:
    logging.info("Stage: /validate request")
    return await run_in_threadpool(run_validation, payload)
