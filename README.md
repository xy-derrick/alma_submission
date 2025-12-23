# Passport + G-28 Extractor

Local FastAPI app for uploading a passport and G-28, extracting structured fields, and populating the test form with Playwright.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
playwright install
```

OCR and PDF rendering dependencies:
- Tesseract OCR must be installed and available on your PATH for OCR (`pytesseract`).
- PyMuPDF (`pymupdf`) is used for PDF rendering and text extraction (no Poppler required).
- PassportEye (`passporteye`) is used for MRZ extraction when available.

## Run

```bash
uvicorn app.main:app --reload
```

Open `http://localhost:8000`.

## Workflow

1) Upload a passport and G-28 (PDF/JPG/PNG).
2) Click **Extract** to run MRZ/OCR/PDF parsing and normalize into the schema.
3) Review fields in the UI and edit if needed.
4) Click **Validate** to let Gemini check for mismatches or issues.
5) Click **Populate Form** to fill the test form without submitting.

## Gemini summary

Configure Gemini in `config.json`:

- Set `"enabled": true`
- Add your `"api_key"` (or set `GEMINI_API_KEY` in the environment)
- Adjust `"model"` or `"max_chars"` if needed
- Toggle `"summary_enabled"`, `"structured_enabled"`, or `"fill_enabled"` as desired

The extracted text (PDF or OCR) is sent to Gemini for a summary and optional structured JSON that fills the fields in the UI.

## Endpoints

- `GET /health`
- `POST /extract` (multipart form with `passport` and/or `g28`)
- `POST /validate` (JSON payload with extracted fields)
- `POST /populate` (JSON payload with extracted fields)

## Playwright settings

- `FORM_URL` overrides the default test form URL.
- `PLAYWRIGHT_HEADLESS=true` to run headless.
- `PLAYWRIGHT_HOLD_SECONDS=8` to keep the browser open briefly for demo.

## Browser-use fill

To use the browser-use agent instead of the Playwright filler:

- Set `"fill": { "provider": "browser_use" }` in `config.json` or set `FILL_PROVIDER=browser_use`.
- Ensure `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) is set in your `.env` or environment.
