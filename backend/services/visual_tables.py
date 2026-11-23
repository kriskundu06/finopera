"""Vision-assisted extraction for financial statement tables embedded as images.

The text RAG pipeline remains useful for narrative pages. This module handles
statement pages whose tables are images or whose OCR text loses row/column
relationships, then emits page-grounded text for the existing evidence pipeline.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pymupdf
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

_API_KEY = os.getenv("GEMINI_API_KEY")
_MODEL = os.getenv("GEMINI_VISION_MODEL") or os.getenv(
    "GEMINI_MODEL", "gemini-2.5-flash"
)
_client = genai.Client(api_key=_API_KEY) if _API_KEY else None

_STATEMENT_PATTERNS = {
    "income_statement": re.compile(
        r"profit\s+and\s+loss|income\s+statement|statement\s+of\s+"
        r"(?:consolidated\s+)?profit|statement\s+of\s+operations",
        re.IGNORECASE,
    ),
    "balance_sheet": re.compile(
        r"balance\s+sheet|statement\s+of\s+financial\s+position",
        re.IGNORECASE,
    ),
    "cash_flow": re.compile(
        r"cash\s+flows?|statement\s+of\s+cash",
        re.IGNORECASE,
    ),
}


def _classify_statement(text: str) -> str | None:
    for statement, pattern in _STATEMENT_PATTERNS.items():
        if pattern.search(text or ""):
            return statement
    return None


def _is_image_heavy(page: pymupdf.Page) -> bool:
    """Check whether an embedded image covers a meaningful part of the page."""
    page_area = max(float(page.rect.width * page.rect.height), 1.0)
    covered_area = 0.0
    for image in page.get_images(full=True):
        xref = image[0]
        try:
            for rect in page.get_image_rects(xref):
                covered_area += max(0.0, rect.width * rect.height)
        except Exception:
            continue
    return covered_area / page_area >= 0.20


def _table_text_from_response(payload: dict[str, Any]) -> str:
    heading = str(payload.get("heading") or "").strip()
    units = str(payload.get("units") or "").strip()
    columns = payload.get("columns")
    rows = payload.get("rows")

    lines: list[str] = []
    if heading:
        lines.append(f"Statement heading: {heading}")
    if units:
        lines.append(f"Reported units and currency: {units}")

    if isinstance(columns, list) and columns:
        column_names = [str(item).strip() for item in columns]
        lines.append("Column headings: " + " | ".join(column_names))
    else:
        column_names = []

    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            label = str(row.get("label") or "").strip()
            values = row.get("values")
            if not label:
                continue
            if not isinstance(values, list):
                values = []
            rendered_values = [str(value).strip() for value in values]
            if column_names and len(rendered_values) == len(column_names):
                cells = [
                    f"{column_names[index]}: {value}"
                    for index, value in enumerate(rendered_values)
                ]
            else:
                cells = rendered_values
            lines.append(label + (" | " + " | ".join(cells) if cells else ""))

    return "\\n".join(lines).strip()


def _extract_page_table(
    page: pymupdf.Page,
    page_number: int,
    page_hint: str,
) -> dict[str, Any] | None:
    if _client is None:
        raise RuntimeError(
            "GEMINI_API_KEY is required for image-based financial table "
            "extraction. Add it to the project-root .env file."
        )

    pixmap = page.get_pixmap(dpi=240, colorspace=pymupdf.csRGB, alpha=False)
    image_bytes = pixmap.tobytes("png")

    prompt = f"""
You are reading a page image from a company's financial report.

PDF page number: {page_number}
Selectable text found on this page (may be incomplete): {page_hint[:1200]}

First determine whether this page contains a financial statement table:
income statement / profit and loss, balance sheet / statement of financial
position, or cash flow statement.

Return JSON with exactly these keys:
{{
  "is_financial_statement": true,
  "statement_type": "income_statement | balance_sheet | cash_flow | other",
  "heading": "exact printed statement heading",
  "units": "exact printed currency and scale, e.g. INR crore",
  "columns": ["each column heading in left-to-right order, including dates and period labels"],
  "rows": [
    {{"label": "printed row label", "values": ["cell 1", "cell 2"]}}
  ]
}}

Important rules:
- Transcribe every visible row, including subtotal and total rows.
- Preserve the exact left-to-right order of columns and map every cell to its
  corresponding column. Include period labels such as quarter ended, half year
  ended, and as-at dates.
- Preserve commas, decimal points, minus signs and parentheses for negatives.
- Keep a printed dash as "-"; do not turn it into zero.
- Do not calculate, infer, correct, or invent any number.
- If a cell is unreadable, write "[unclear]" for that cell. Do not guess.
- Combine wrapped row labels into one label.
- Do not omit a row because it appears to be a heading or subtotal.
- If this is not a financial statement table, set is_financial_statement to
  false, statement_type to "other", and rows to [].
- Return valid JSON only, without Markdown.
"""

    try:
        response = _client.models.generate_content(
            model=_MODEL,
            contents=[
                prompt,
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            ],
            config={
                "temperature": 0,
                "response_mime_type": "application/json",
                "max_output_tokens": 7000,
            },
        )
    except Exception as exc:
        raise RuntimeError(
            f"Gemini vision extraction failed on PDF page {page_number} "
            f"using model {_MODEL}: {type(exc).__name__}: {exc}"
        ) from exc

    if not isinstance(response.text, str) or not response.text.strip():
        raise RuntimeError(
            f"Gemini returned an empty visual-table response for PDF page {page_number}."
        )

    try:
        payload = json.loads(response.text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Gemini returned invalid visual-table JSON for PDF page {page_number}: {exc}"
        ) from exc

    if not isinstance(payload, dict) or not payload.get("is_financial_statement"):
        return None

    statement = payload.get("statement_type")
    if statement not in _STATEMENT_PATTERNS:
        statement = _classify_statement(
            str(payload.get("heading") or "") + " " + page_hint
        )
    if statement not in _STATEMENT_PATTERNS:
        return None

    table_text = _table_text_from_response(payload)
    if not table_text:
        return None

    return {
        "statement": statement,
        "text": (
            f"Visual table transcription from PDF page {page_number}. "
            "The following rows and column headings were read from the page image.\\n"
            + table_text
        ),
        "source": "uploaded_pdf",
        "document_id": None,
        "page": page_number,
        "chunk_number": 1,
        "source_type": "visual_table_transcription",
    }


def extract_visual_statement_pages(
    pdf_path: str | Path,
    document_id: str,
) -> list[dict[str, Any]]:
    """Find and transcribe image-based statement pages from the original PDF."""
    path = Path(pdf_path)
    if not path.is_file():
        return []

    passages: list[dict[str, Any]] = []
    with pymupdf.open(path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            native_text = page.get_text("text").strip()
            candidate_text = native_text
            has_images = bool(page.get_images(full=True))

            # OCR is used only to locate the relevant statement pages. The
            # image itself is sent to Gemini so table structure is not flattened.
            if has_images and len(candidate_text) < 2500:
                try:
                    text_page = page.get_textpage_ocr(
                        language="eng",
                        dpi=180,
                        full=True,
                    )
                    ocr_text = page.get_text("text", textpage=text_page).strip()
                    if len(ocr_text) > len(candidate_text):
                        candidate_text = ocr_text
                except Exception:
                    # If Tesseract is unavailable, native statement headings
                    # still work; full-page image-only PDFs are handled below.
                    pass

            statement_hint = _classify_statement(candidate_text)
            image_heavy = has_images and _is_image_heavy(page)
            image_only = image_heavy and len(native_text) < 80

            if not image_heavy:
                continue
            if statement_hint is None and not image_only:
                continue

            passage = _extract_page_table(
                page=page,
                page_number=page_number,
                page_hint=candidate_text,
            )
            if passage:
                passage["document_id"] = document_id
                passages.append(passage)
                print(
                    f"Vision table extracted: {passage['statement']} "
                    f"from PDF page {page_number}"
                )

    return passages
