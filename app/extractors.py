from __future__ import annotations

import io
import json
import logging
import os
import re
import tempfile
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageOps

from .schema import ExtractionResult, ValidationResult
from .summarizer import extract_structured_fields, summarize_text, validate_extraction

try:
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pytesseract = None

try:
    from passporteye import read_mrz  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    read_mrz = None

try:
    import fitz  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    fitz = None

try:
    from pdf2image import convert_from_bytes  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    convert_from_bytes = None

try:
    from pypdf import PdfReader  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    PdfReader = None


LOGGER = logging.getLogger(__name__)

COUNTRY_MAP = {
    "USA": "United States",
    "GBR": "United Kingdom",
    "CAN": "Canada",
    "MEX": "Mexico",
    "IND": "India",
    "CHN": "China",
    "DEU": "Germany",
    "FRA": "France",
    "ESP": "Spain",
    "ITA": "Italy",
    "BRA": "Brazil",
    "AUS": "Australia",
    "JPN": "Japan",
    "KOR": "South Korea",
}

US_STATE_CODES = {
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
    "DC",
}


def run_extraction(
    passport_bytes: Optional[bytes],
    passport_name: Optional[str],
    g28_bytes: Optional[bytes],
    g28_name: Optional[str],
) -> ExtractionResult:
    result = ExtractionResult()

    if passport_bytes and passport_name:
        LOGGER.info("Stage: passport extraction | filename=%s", passport_name)
        extract_passport(passport_bytes, passport_name, result)
    else:
        LOGGER.info("Stage: passport extraction | skipped (no file)")

    if g28_bytes and g28_name:
        LOGGER.info("Stage: G-28 extraction | filename=%s", g28_name)
        extract_g28(g28_bytes, g28_name, result)
    else:
        LOGGER.info("Stage: G-28 extraction | skipped (no file)")

    result.compute_missing()
    LOGGER.info(
        "Stage: normalization | missing=%d fields", len(result.missing_fields)
    )
    return result


def run_validation(result: ExtractionResult) -> ValidationResult:
    validation_result = ValidationResult()
    validation = validate_extraction(result.model_dump())
    if validation:
        validation_result.passport_matches_client = validation.get("passport_matches_client")
        issues = validation.get("issues") or []
        if isinstance(issues, list):
            validation_result.issues = [str(issue) for issue in issues]
        log_extracted_content("Gemini validation", validation)
    return validation_result


def extract_passport(file_bytes: bytes, filename: str, result: ExtractionResult) -> None:
    LOGGER.info("Stage: passport load images | tool=%s", "pdf2image" if filename.lower().endswith(".pdf") else "PIL")
    images = load_images_from_bytes(file_bytes, filename)
    if not images:
        LOGGER.warning("No images extracted from passport upload.")
        return
    LOGGER.info("Stage: passport load images | result=%d images", len(images))

    primary = images[0]
    if read_mrz is not None:
        LOGGER.info("Stage: passport MRZ extraction | tool=passporteye")
        mrz_data = extract_mrz_with_passporteye(primary)
        if mrz_data:
            LOGGER.info("Stage: passport MRZ parse | result=valid")
            log_extracted_content("passporteye MRZ parsed", mrz_data)
            apply_passport_from_passporteye(result, mrz_data)
            passport_text = build_passport_text_from_passporteye(mrz_data)
            apply_gemini_for_passport(result, passport_text)
            return
        LOGGER.info("Stage: passport MRZ extraction | result=not found")

    LOGGER.info("Stage: passport MRZ extraction | tool=pytesseract")
    mrz_lines = extract_mrz_lines(primary)
    if mrz_lines:
        LOGGER.info("Stage: passport MRZ extraction | result=lines found | lines=%d", len(mrz_lines))
        log_extracted_content("passport MRZ lines", list(mrz_lines))
        LOGGER.info("Stage: passport MRZ parse | result=attempt")
        mrz_data = parse_mrz_td3(mrz_lines[0], mrz_lines[1])
        if mrz_data:
            LOGGER.info("Stage: passport MRZ parse | result=valid")
            log_extracted_content("passport MRZ parsed", mrz_data)
            apply_passport_from_mrz(result, mrz_data)
            passport_text = build_passport_text_from_mrz(mrz_data)
            apply_gemini_for_passport(result, passport_text)
            return
        LOGGER.info("Stage: passport MRZ parse | result=invalid check digits")
    else:
        LOGGER.info("Stage: passport MRZ extraction | result=not found")

    LOGGER.info("Stage: passport OCR fallback | tool=pytesseract")
    text = ocr_full_text(images)
    LOGGER.info("Stage: passport OCR fallback | result=chars=%d", len(text))
    log_extracted_content("passport OCR text", text)
    apply_passport_from_ocr(result, text)
    apply_gemini_for_passport(result, text)


def extract_g28(file_bytes: bytes, filename: str, result: ExtractionResult) -> None:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        pdf_text = extract_pdf_text(file_bytes)
        if pdf_text:
            LOGGER.info("Stage: G-28 PDF text extraction | result=chars=%d", len(pdf_text))
            log_extracted_content("G-28 PDF text", pdf_text)
            structured = extract_structured_fields(pdf_text)
            if structured:
                log_extracted_content("G-28 Gemini JSON", structured)
                apply_structured_fields(result, structured, overwrite=True)
            else:
                apply_pdf_text_to_attorney_fields(result, pdf_text)
                LOGGER.info("Stage: G-28 PDF text apply | result=filled attorney fields")
            summary = summarize_text(pdf_text)
            if summary:
                result.summary = merge_summary(result.summary, "G-28", summary)
                log_extracted_content("G-28 Gemini summary", summary)
        else:
            LOGGER.info("Stage: G-28 PDF text extraction | result=empty")
        return

    LOGGER.info("Stage: G-28 OCR fallback | tool=pytesseract")
    images = load_images_from_bytes(file_bytes, filename)
    if not images:
        LOGGER.warning("No images extracted from G-28 upload.")
        return
    LOGGER.info("Stage: G-28 OCR fallback | result=%d images", len(images))

    text = ocr_full_text(images)
    LOGGER.info("Stage: G-28 OCR fallback | result=chars=%d", len(text))
    log_extracted_content("G-28 OCR text", text)
    ocr_fields = extract_attorney_from_text(text)
    LOGGER.info("Stage: G-28 OCR mapping | result=%d fields", len(ocr_fields))
    if ocr_fields:
        log_extracted_content("G-28 OCR fields", ocr_fields)
    for path, value in ocr_fields.items():
        set_field(result, path, value, source="OCR")
    if text:
        summary = summarize_text(text)
        if summary:
            result.summary = merge_summary(result.summary, "G-28", summary)
            log_extracted_content("G-28 Gemini summary", summary)
        structured = extract_structured_fields(text)
        if structured:
            log_extracted_content("G-28 Gemini JSON", structured)
            apply_structured_fields(result, structured, overwrite=True)


def load_images_from_bytes(file_bytes: bytes, filename: str) -> List[Image.Image]:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        if fitz is not None:
            try:
                LOGGER.info("Stage: PDF render | tool=pymupdf")
                return render_pdf_with_pymupdf(file_bytes)
            except Exception as exc:  # pragma: no cover - runtime dependency
                LOGGER.warning("PyMuPDF rendering failed: %s", exc)
        if convert_from_bytes is None:
            LOGGER.warning("pdf2image is not available; skipping PDF rendering.")
            return []
        try:
            LOGGER.info("Stage: PDF render | tool=pdf2image")
            images = convert_from_bytes(file_bytes)
            LOGGER.info("Stage: PDF render | result=%d pages", len(images))
            return images
        except Exception as exc:  # pragma: no cover - runtime dependency
            LOGGER.warning("PDF rendering failed: %s", exc)
            return []

    try:
        image = Image.open(io.BytesIO(file_bytes))
        rgb = image.convert("RGB")
        LOGGER.info("Stage: image load | result=1 image | size=%dx%d", rgb.size[0], rgb.size[1])
        return [rgb]
    except Exception as exc:  # pragma: no cover - runtime dependency
        LOGGER.warning("Image open failed: %s", exc)
        return []


def preprocess_for_ocr(image: Image.Image) -> Image.Image:
    img = image.convert("L")
    img = ImageOps.autocontrast(img)
    width, height = img.size
    scale = 1400 / max(width, 1)
    if scale > 1:
        img = img.resize((int(width * scale), int(height * scale)))
    return img


def ocr_image(image: Image.Image, *, whitelist: Optional[str] = None, psm: int = 6) -> str:
    if pytesseract is None:
        LOGGER.warning("OCR skipped: pytesseract is not available.")
        return ""
    config = f"--oem 1 --psm {psm}"
    if whitelist:
        config += f" -c tessedit_char_whitelist={whitelist}"
    return pytesseract.image_to_string(image, config=config)


def ocr_full_text(images: Iterable[Image.Image]) -> str:
    text_blocks = []
    for image in images:
        prepped = preprocess_for_ocr(image)
        text_blocks.append(ocr_image(prepped))
    return "\n".join(text_blocks)


def extract_mrz_lines(image: Image.Image) -> Optional[Tuple[str, str]]:
    width, height = image.size
    crop_top = int(height * 0.62)
    mrz_crop = image.crop((0, crop_top, width, height))
    prepped = preprocess_for_ocr(mrz_crop)
    raw = ocr_image(prepped, whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<", psm=6)
    lines = []
    for line in raw.splitlines():
        cleaned = re.sub(r"[^A-Z0-9<]", "", line.upper())
        if len(cleaned) >= 30:
            lines.append(cleaned)
    if len(lines) < 2:
        return None

    candidates = sorted(lines, key=len, reverse=True)
    line1 = next((l for l in candidates if l.startswith("P<") or l.startswith("P")), candidates[0])
    line2 = next((l for l in candidates if l is not line1), candidates[1])
    line1 = line1[:44].ljust(44, "<")
    line2 = line2[:44].ljust(44, "<")
    return line1, line2


def parse_mrz_td3(line1: str, line2: str) -> Optional[Dict[str, str]]:
    if len(line1) != 44 or len(line2) != 44:
        return None

    passport_number = line2[0:9]
    passport_check = line2[9]
    nationality = line2[10:13]
    birth_date = line2[13:19]
    birth_check = line2[19]
    sex = line2[20]
    expiry_date = line2[21:27]
    expiry_check = line2[27]
    personal_number = line2[28:42]
    personal_check = line2[42]
    final_check = line2[43]

    if (
        check_digit(passport_number) != passport_check
        or check_digit(birth_date) != birth_check
        or check_digit(expiry_date) != expiry_check
    ):
        return None

    composite = (
        passport_number
        + passport_check
        + birth_date
        + birth_check
        + expiry_date
        + expiry_check
        + personal_number
        + personal_check
    )
    if check_digit(composite) != final_check:
        return None

    issuing_country = line1[2:5]
    name_block = line1[5:]
    surname, given = split_mrz_name(name_block)

    return {
        "passport_number": passport_number.replace("<", "").strip(),
        "nationality": nationality.replace("<", "").strip(),
        "issuing_country": issuing_country.replace("<", "").strip(),
        "surname": surname,
        "given_name": given,
        "dob": normalize_mrz_date(birth_date, is_expiry=False),
        "expiry_date": normalize_mrz_date(expiry_date, is_expiry=True),
        "sex": normalize_sex(sex),
    }


def apply_passport_from_mrz(result: ExtractionResult, data: Dict[str, str]) -> None:
    full_name = " ".join([data.get("given_name", ""), data.get("surname", "")]).strip()
    if full_name:
        set_field(result, "passport.full_name", full_name, source="MRZ")
    set_field(result, "passport.given_name", data.get("given_name"), source="MRZ")
    set_field(result, "passport.surname", data.get("surname"), source="MRZ")
    set_field(result, "passport.dob", data.get("dob"), source="MRZ")
    set_field(
        result,
        "passport.country",
        normalize_country(data.get("nationality")),
        source="MRZ",
    )
    set_field(result, "passport.sex", data.get("sex"), source="MRZ")
    set_field(result, "passport.number", data.get("passport_number"), source="MRZ")
    set_field(
        result,
        "passport.issue_country",
        normalize_country(data.get("issuing_country")),
        source="MRZ",
    )
    set_field(result, "passport.expiry_date", data.get("expiry_date"), source="MRZ")


def apply_passport_from_passporteye(result: ExtractionResult, data: Dict[str, str]) -> None:
    given_name = (data.get("names") or "").replace("<<", " ").replace("<", " ").strip()
    surname = (data.get("surname") or "").replace("<<", " ").replace("<", " ").strip()
    full_name = " ".join([given_name, surname]).strip()

    set_field(result, "passport.full_name", full_name, source="PASSPORTEYE")
    set_field(result, "passport.given_name", given_name, source="PASSPORTEYE")
    set_field(result, "passport.surname", surname, source="PASSPORTEYE")
    set_field(
        result,
        "passport.dob",
        normalize_mrz_date(data.get("date_of_birth", ""), is_expiry=False),
        source="PASSPORTEYE",
    )
    set_field(
        result,
        "passport.country",
        normalize_country(data.get("nationality")),
        source="PASSPORTEYE",
    )
    set_field(result, "passport.sex", normalize_sex(data.get("sex", "")), source="PASSPORTEYE")
    set_field(result, "passport.number", data.get("number"), source="PASSPORTEYE")
    set_field(
        result,
        "passport.issue_country",
        normalize_country(data.get("country")),
        source="PASSPORTEYE",
    )
    set_field(
        result,
        "passport.expiry_date",
        normalize_mrz_date(data.get("expiration_date", ""), is_expiry=True),
        source="PASSPORTEYE",
    )


def apply_passport_from_ocr(result: ExtractionResult, text: str) -> None:
    full_name = extract_value_after_label(text, ["surname", "name"])
    given_name = extract_value_after_label(text, ["given name", "given names", "first name"])
    surname = extract_value_after_label(text, ["surname", "last name"])
    passport_number = extract_value_after_label(text, ["passport no", "passport number", "document no"])
    nationality = extract_value_after_label(text, ["nationality", "country"])
    dob = find_date_near(text, ["date of birth", "birth", "dob"])
    expiry = find_date_near(text, ["expiry", "expiration", "date of expiry"])

    if not full_name and (given_name or surname):
        full_name = " ".join([given_name or "", surname or ""]).strip()

    log_extracted_content(
        "passport OCR fields",
        {
            "passport.full_name": full_name,
            "passport.given_name": given_name,
            "passport.surname": surname,
            "passport.dob": dob,
            "passport.country": nationality,
            "passport.number": passport_number,
            "passport.expiry_date": expiry,
        },
    )
    set_field(result, "passport.full_name", full_name, source="OCR")
    set_field(result, "passport.given_name", given_name, source="OCR")
    set_field(result, "passport.surname", surname, source="OCR")
    set_field(result, "passport.dob", dob, source="OCR")
    set_field(result, "passport.country", normalize_country(nationality), source="OCR")
    set_field(result, "passport.number", passport_number, source="OCR")
    set_field(result, "passport.expiry_date", expiry, source="OCR")


def extract_g28_from_pdf_form(file_bytes: bytes) -> Dict[str, str]:
    if fitz is not None:
        LOGGER.info("Stage: G-28 PDF form extraction | tool=pymupdf")
        try:
            mapped = extract_g28_from_pdf_form_pymupdf(file_bytes)
        except Exception as exc:  # pragma: no cover - runtime dependency
            LOGGER.warning("PyMuPDF form extraction failed: %s", exc)
            mapped = {}
        LOGGER.info("Stage: G-28 PDF form extraction | result=%d mapped fields", len(mapped))
        if mapped:
            return mapped

    if PdfReader is None:
        LOGGER.warning("PDF form extraction skipped: pypdf is not available.")
        return {}
    try:
        LOGGER.info("Stage: G-28 PDF form extraction | tool=pypdf")
        reader = PdfReader(io.BytesIO(file_bytes))
        fields = reader.get_fields() or {}
        LOGGER.info("Stage: G-28 PDF form extraction | result=%d raw fields", len(fields))
    except Exception as exc:  # pragma: no cover - runtime dependency
        LOGGER.warning("PDF form read failed: %s", exc)
        return {}

    mapped: Dict[str, str] = {}
    for key, field in fields.items():
        value = field.get("/V") if isinstance(field, dict) else None
        if value is None:
            continue
        value_text = str(value).strip()
        if not value_text:
            continue
        key_lower = str(key).lower()
        if "attorney" in key_lower and "name" in key_lower:
            mapped["g28.attorney.name"] = value_text
        elif "firm" in key_lower:
            mapped["g28.attorney.firm"] = value_text
        elif "email" in key_lower:
            mapped["g28.attorney.email"] = value_text
        elif "phone" in key_lower or "tel" in key_lower:
            mapped["g28.attorney.phone"] = value_text
        elif "address" in key_lower or "street" in key_lower:
            mapped["g28.attorney.address"] = value_text
        elif "city" in key_lower:
            mapped["g28.attorney.city"] = value_text
        elif "state" in key_lower:
            mapped["g28.attorney.state"] = value_text
        elif "zip" in key_lower or "postal" in key_lower:
            mapped["g28.attorney.zip"] = value_text
    return mapped


def extract_g28_from_pdf_form_pymupdf(file_bytes: bytes) -> Dict[str, str]:
    if fitz is None:
        return {}
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    mapped: Dict[str, str] = {}
    widget_count = 0
    for page in doc:
        widgets = page.widgets() or []
        for widget in widgets:
            widget_count += 1
            name = (widget.field_name or "").strip()
            value = (widget.field_value or "").strip()
            if not name or not value:
                continue
            key_lower = name.lower()
            if "attorney" in key_lower and "name" in key_lower:
                mapped["g28.attorney.name"] = value
            elif "firm" in key_lower:
                mapped["g28.attorney.firm"] = value
            elif "email" in key_lower:
                mapped["g28.attorney.email"] = value
            elif "phone" in key_lower or "tel" in key_lower:
                mapped["g28.attorney.phone"] = value
            elif "address" in key_lower or "street" in key_lower:
                mapped["g28.attorney.address"] = value
            elif "city" in key_lower:
                mapped["g28.attorney.city"] = value
            elif "state" in key_lower:
                mapped["g28.attorney.state"] = value
            elif "zip" in key_lower or "postal" in key_lower:
                mapped["g28.attorney.zip"] = value
    LOGGER.info("Stage: G-28 PDF form extraction | PyMuPDF widgets=%d", widget_count)
    return mapped


def extract_pdf_text(file_bytes: bytes) -> str:
    if fitz is None:
        LOGGER.warning("PDF text extraction skipped: PyMuPDF is not available.")
        return ""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:  # pragma: no cover - runtime dependency
        LOGGER.warning("PDF text extraction failed: %s", exc)
        return ""
    pages_text: List[str] = []
    for page in doc:
        items: List[Tuple[float, float, str]] = []
        text_dict = page.get_text("dict")
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                line_text = "".join(span.get("text", "") for span in line.get("spans", []))
                line_text = line_text.strip()
                if not line_text:
                    continue
                bbox = line.get("bbox", [0, 0, 0, 0])
                items.append((bbox[1], bbox[0], line_text))

        xs = [item[1] for item in items]
        use_columns = False
        if len(xs) >= 10:
            min_x = min(xs)
            max_x = max(xs)
            if (max_x - min_x) > 200:
                use_columns = True

        if use_columns:
            split_x = (min_x + max_x) / 2
            left_items = [item for item in items if item[1] <= split_x]
            right_items = [item for item in items if item[1] > split_x]
            left_items.sort(key=lambda item: (item[0], item[1]))
            right_items.sort(key=lambda item: (item[0], item[1]))
            ordered_items = left_items + right_items
        else:
            items.sort(key=lambda item: (item[0], item[1]))
            ordered_items = items
        page_text = "\n".join(text for _, __, text in ordered_items).strip()
        if not page_text:
            page_text = page.get_text().strip()
        if page_text:
            pages_text.append(page_text)

    combined = "\n\n".join(pages_text).strip()
    if combined:
        save_text_to_temp(combined, prefix="pdf_text_")
    return combined


def save_text_to_temp(text: str, *, prefix: str) -> None:
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".txt",
            prefix=prefix,
            delete=False,
        ) as handle:
            handle.write(text)
            temp_path = handle.name
        LOGGER.info("Stage: extract output saved | path=%s", temp_path)
    except Exception as exc:  # pragma: no cover - runtime dependency
        LOGGER.warning("Temp file write failed: %s", exc)


def render_pdf_with_pymupdf(file_bytes: bytes) -> List[Image.Image]:
    if fitz is None:
        return []
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    images: List[Image.Image] = []
    matrix = fitz.Matrix(2, 2)
    for page in doc:
        pix = page.get_pixmap(matrix=matrix)
        image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        images.append(image)
    LOGGER.info("Stage: PDF render | result=%d pages", len(images))
    return images


def extract_mrz_with_passporteye(image: Image.Image) -> Optional[Dict[str, str]]:
    if read_mrz is None:
        return None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            temp_path = handle.name
        image.save(temp_path, format="PNG")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            mrz = read_mrz(temp_path)
    except Exception as exc:  # pragma: no cover - runtime dependency
        LOGGER.warning("PassportEye MRZ extraction failed: %s", exc)
        return None
    finally:
        if "temp_path" in locals():
            try:
                os.remove(temp_path)
            except OSError:
                pass
    if mrz is None or not getattr(mrz, "valid", False):
        return None
    try:
        return mrz.to_dict()
    except Exception:
        return None


def apply_pdf_text_to_attorney_fields(result: ExtractionResult, pdf_text: str) -> None:
    text = pdf_text.strip()
    for path in (
        "g28.attorney.name",
        "g28.attorney.firm",
        "g28.attorney.address",
        "g28.attorney.city",
        "g28.attorney.state",
        "g28.attorney.zip",
        "g28.attorney.email",
        "g28.attorney.phone",
    ):
        set_field(result, path, text, source="PDF_TEXT")


def apply_gemini_for_passport(result: ExtractionResult, text: str) -> None:
    if not text:
        return
    structured = extract_structured_fields(text)
    if structured:
        log_extracted_content("passport Gemini JSON", structured)
        apply_structured_fields(result, structured, overwrite=True)
    summary = summarize_text(text)
    if summary:
        result.summary = merge_summary(result.summary, "Passport", summary)
        log_extracted_content("passport Gemini summary", summary)


def merge_summary(existing: Optional[str], label: str, summary: str) -> str:
    prefix = f"{label} summary:"
    if not existing:
        return f"{prefix}\n{summary}"
    return f"{existing}\n\n{prefix}\n{summary}"


def build_passport_text_from_passporteye(data: Dict[str, str]) -> str:
    return "\n".join(
        [
            f"Passport number: {data.get('number', '')}",
            f"Surname: {data.get('surname', '')}",
            f"Given names: {data.get('names', '')}",
            f"Nationality: {data.get('nationality', '')}",
            f"Issuing country: {data.get('country', '')}",
            f"Date of birth: {data.get('date_of_birth', '')}",
            f"Sex: {data.get('sex', '')}",
            f"Expiry date: {data.get('expiration_date', '')}",
        ]
    ).strip()


def build_passport_text_from_mrz(data: Dict[str, str]) -> str:
    return "\n".join(
        [
            f"Passport number: {data.get('passport_number', '')}",
            f"Surname: {data.get('surname', '')}",
            f"Given names: {data.get('given_name', '')}",
            f"Nationality: {data.get('nationality', '')}",
            f"Issuing country: {data.get('issuing_country', '')}",
            f"Date of birth: {data.get('dob', '')}",
            f"Sex: {data.get('sex', '')}",
            f"Expiry date: {data.get('expiry_date', '')}",
        ]
    ).strip()


def apply_structured_fields(
    result: ExtractionResult, payload: Dict[str, object], *, overwrite: bool
) -> None:
    passport_data = payload.get("passport")
    if isinstance(passport_data, dict):
        for key, value in passport_data.items():
            if not isinstance(key, str) or not hasattr(result.passport, key):
                continue
            if value in (None, "", []):
                continue
            if not overwrite and getattr(result.passport, key):
                continue
            set_field(result, f"passport.{key}", str(value), source="GEMINI_JSON")

    g28_data = payload.get("g28")
    if not isinstance(g28_data, dict):
        return

    for group_key, target in (("client", result.g28.client), ("attorney", result.g28.attorney)):
        group_data = g28_data.get(group_key)
        if not isinstance(group_data, dict):
            continue
        for key, value in group_data.items():
            if not isinstance(key, str) or not hasattr(target, key):
                continue
            if value in (None, "", []):
                continue
            if not overwrite and getattr(target, key):
                continue
            set_field(result, f"g28.{group_key}.{key}", str(value), source="GEMINI_JSON")


def log_extracted_content(stage: str, content: object, *, max_len: int = 800) -> None:
    if isinstance(content, str):
        snippet = re.sub(r"\s+", " ", content).strip()
        if len(snippet) > max_len:
            snippet = f"{snippet[:max_len]}... (truncated)"
        LOGGER.info("Stage: %s | content=%s", stage, snippet)
        return

    try:
        payload = json.dumps(content, ensure_ascii=True, sort_keys=True)
    except TypeError:
        payload = repr(content)

    if len(payload) > max_len:
        payload = f"{payload[:max_len]}... (truncated)"
    LOGGER.info("Stage: %s | content=%s", stage, payload)


def extract_attorney_from_text(text: str) -> Dict[str, str]:
    data: Dict[str, str] = {}

    email_match = re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", text, re.IGNORECASE)
    if email_match:
        data["g28.attorney.email"] = email_match.group(0)

    phone_match = re.search(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", text)
    if phone_match:
        data["g28.attorney.phone"] = phone_match.group(0)

    name = extract_value_after_label(text, ["attorney name", "name of attorney", "attorney"])
    if name:
        data["g28.attorney.name"] = name

    firm = extract_value_after_label(text, ["firm name", "law firm", "firm"])
    if firm:
        data["g28.attorney.firm"] = firm

    address = extract_value_after_label(text, ["address", "street"])
    if address:
        data["g28.attorney.address"] = address
        city, state, zip_code = parse_city_state_zip(address)
        if city:
            data["g28.attorney.city"] = city
        if state:
            data["g28.attorney.state"] = state
        if zip_code:
            data["g28.attorney.zip"] = zip_code

    if "g28.attorney.city" not in data:
        city_state_line = find_line_with_state(text)
        if city_state_line:
            city, state, zip_code = parse_city_state_zip(city_state_line)
            if city:
                data["g28.attorney.city"] = city
            if state:
                data["g28.attorney.state"] = state
            if zip_code:
                data["g28.attorney.zip"] = zip_code

    return data


def find_line_with_state(text: str) -> Optional[str]:
    for line in text.splitlines():
        tokens = re.split(r"\s+", line.strip())
        for token in tokens:
            if token.upper().strip(",") in US_STATE_CODES:
                return line.strip()
    return None


def parse_city_state_zip(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    match = re.search(r"([A-Za-z .'-]+),?\s+([A-Z]{2})\s+(\d{5}(?:-\d{4})?)", text)
    if match:
        return match.group(1).strip(), match.group(2).strip(), match.group(3).strip()
    return None, None, None


def extract_value_after_label(text: str, labels: Iterable[str]) -> Optional[str]:
    for label in labels:
        pattern = rf"{re.escape(label)}\s*[:\-]?\s*(.+)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            value = value.split("\n", 1)[0].strip()
            if value:
                return value
    return None


def find_date_near(text: str, keywords: Iterable[str]) -> Optional[str]:
    for line in text.splitlines():
        lower = line.lower()
        if any(keyword in lower for keyword in keywords):
            date_text = extract_date_from_text(line)
            if date_text:
                return normalize_date(date_text)

    date_text = extract_date_from_text(text)
    return normalize_date(date_text) if date_text else None


def extract_date_from_text(text: str) -> Optional[str]:
    match = re.search(
        r"(\d{4}[\/\-.]\d{1,2}[\/\-.]\d{1,2}|\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4})",
        text,
    )
    if match:
        return match.group(1)
    return None


def normalize_date(date_text: Optional[str]) -> Optional[str]:
    if not date_text:
        return None
    date_text = date_text.strip()
    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y.%m.%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%d.%m.%Y",
        "%m-%d-%Y",
        "%m/%d/%Y",
        "%m.%d.%Y",
        "%d-%m-%y",
        "%d/%m/%y",
        "%m-%d-%y",
        "%m/%d/%y",
    ]
    for fmt in formats:
        try:
            parsed = datetime.strptime(date_text, fmt)
            return parsed.date().isoformat()
        except ValueError:
            continue
    return None


def normalize_mrz_date(date_text: str, *, is_expiry: bool) -> Optional[str]:
    if not date_text or len(date_text) != 6 or not date_text.isdigit():
        return None
    year = int(date_text[0:2])
    month = int(date_text[2:4])
    day = int(date_text[4:6])
    current_year = datetime.utcnow().year % 100
    if is_expiry:
        century = 2000 if year < 70 else 1900
    else:
        century = 1900 if year > current_year else 2000
    try:
        return datetime(century + year, month, day).date().isoformat()
    except ValueError:
        return None


def normalize_country(country: Optional[str]) -> Optional[str]:
    if not country:
        return None
    country = country.strip().upper()
    return COUNTRY_MAP.get(country, country.title())


def normalize_sex(sex: str) -> Optional[str]:
    value = sex.strip().upper()
    if value in ("M", "MALE"):
        return "Male"
    if value in ("F", "FEMALE"):
        return "Female"
    return None


def split_mrz_name(name_block: str) -> Tuple[str, str]:
    parts = name_block.split("<<", 1)
    surname = parts[0].replace("<", " ").strip()
    given = parts[1].replace("<", " ").strip() if len(parts) > 1 else ""
    return surname, given


MRZ_VALUES: Dict[str, int] = {str(i): i for i in range(10)}
MRZ_VALUES.update({chr(ord("A") + i): 10 + i for i in range(26)})
MRZ_VALUES["<"] = 0


def check_digit(data: str) -> str:
    weights = [7, 3, 1]
    total = 0
    for idx, char in enumerate(data):
        total += MRZ_VALUES.get(char, 0) * weights[idx % 3]
    return str(total % 10)


def set_field(result: ExtractionResult, path: str, value: Optional[str], *, source: str) -> None:
    if value is None:
        return
    value = value.strip() if isinstance(value, str) else value
    if not value:
        return
    target = result
    parts = path.split(".")
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)
    result.source[path] = source
