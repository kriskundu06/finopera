"""Diagnose financial-statement retrieval and field extraction for one indexed PDF.

Usage:
    python debug_extract.py <64-character-document-id>

Run from the repository root with the backend virtual environment active.
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from collections import defaultdict

from backend.services.financial_extraction import (
    NUMERIC_FIELDS,
    STATEMENT_FIELDS,
    _retrieve_statement_passages,
    extract_financials,
)
from backend.services.vector_store import get_document_chunks


STATEMENT_PATTERNS = {
    "income_statement": re.compile(
        r"statement of profit and loss|income statement|statement of operations|"
        r"revenue from operations", re.I
    ),
    "balance_sheet": re.compile(
        r"balance sheet|total current assets|total current liabilities|total assets", re.I
    ),
    "cash_flow": re.compile(
        r"cash flow statement|statement of cash flows|cash flows from operating activities", re.I
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("document_id", help="SHA-256 document ID returned by upload")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    chunks = get_document_chunks(args.document_id)
    pages: dict[int, list[str]] = defaultdict(list)
    for chunk in chunks:
        page = chunk.get("page")
        if isinstance(page, int):
            pages[page].append(str(chunk.get("text", "")))

    print(f"Indexed chunks: {len(chunks)}")
    print(f"Indexed pages: {len(pages)}")
    image_dir = Path(__file__).resolve().parent / "data" / "document_pages" / args.document_id
    print(f"Page-image directory: {image_dir}")
    image_pages = sorted(
        int(match.group(1))
        for image in image_dir.glob("page_*.png")
        if (match := re.fullmatch(r"page_(\\d+)\\.png", image.name))
    ) if image_dir.is_dir() else []
    print(f"Persisted page images: {len(image_pages)}; pages={image_pages}")
    all_text = "\\n".join(text for values in pages.values() for text in values)

    print("\\nDOCUMENT COVERAGE")
    for name, pattern in STATEMENT_PATTERNS.items():
        matching_pages = sorted(
            page for page, texts in pages.items()
            if pattern.search("\\n".join(texts))
        )
        print(f"{name}: {'detected' if matching_pages else 'not detected'}; pages={matching_pages}")

    selected = _retrieve_statement_passages(args.document_id)
    print("\\nSELECTED PAGES")
    for statement, passages in selected.items():
        selected_pages = sorted({p.get("page") for p in passages if p.get("page") is not None})
        missing_images = sorted(set(selected_pages) - set(image_pages))
        print(
            f"{statement}: {len(passages)} chunks; pages={selected_pages}; "
            f"selected_pages_without_images={missing_images}"
        )

    print("\\nEXTRACTION (structured rejection details are logged above)")
    result = extract_financials(args.document_id)
    financials = result.get("financials", {})
    evidence = result.get("evidence", {})
    print(f"extraction_status={result.get('extraction_status')}")
    print(f"statement_errors={result.get('statement_errors')}")
    print(f"verified_fields={result.get('verified_fields')}")
    print(f"missing_fields={result.get('missing_fields')}")

    print("\\nFIELD STATUS")
    for field in sorted(NUMERIC_FIELDS):
        value = financials.get(field)
        reference = evidence.get(field)
        field_status = result.get("field_status", {}).get(field, {})
        status = field_status.get("status") or (
            "verified" if value is not None and reference else "not_verified"
        )
        print(
            f"{field}: status={status}; reason={field_status.get('reason')}; "
            f"value={value}; page={reference.get('page') if reference else None}; "
            f"quote={reference.get('quote') if reference else None}"
        )


if __name__ == "__main__":
    main()
