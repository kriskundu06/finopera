import hashlib
import re
from io import BytesIO
from pathlib import Path

import pymupdf
import requests
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.services.chunking import chunk_pages
from backend.services.financial_analysis import calculate_financial_metrics
from backend.services.financial_extraction import extract_financials
from backend.services.rag import answer_question
from backend.services.vector_store import index_chunks, search_chunks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCUMENT_PAGE_IMAGES = PROJECT_ROOT / "data" / "document_pages"


app = FastAPI(
    title="Due Diligence Copilot RAG Platform",
    description="Local-first company document analysis API",
    version="0.1.0",
)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=10)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    document_id: str = Field(min_length=1, max_length=64)
    top_k: int = Field(default=5, ge=1, le=10)


class FinancialAnalysisRequest(BaseModel):
    revenue: float | None = None
    gross_profit: float | None = None
    operating_income: float | None = None
    net_income: float | None = None

    current_assets: float | None = None
    current_liabilities: float | None = None

    total_debt: float | None = None
    total_equity: float | None = None
    total_assets: float | None = None

    operating_cash_flow: float | None = None
    capital_expenditure: float | None = None

    currency: str = "USD"
    units: str = "millions"


def validate_document_id(document_id: str) -> str:
    """Validate that document_id is a 64-character SHA-256 hex digest."""
    if not re.fullmatch(r"[a-fA-F0-9]{64}", document_id):
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid document_id. Expected a 64-character "
                "SHA-256 hexadecimal hash."
            ),
        )
    return document_id.lower()


@app.post("/financial-analysis")
def financial_analysis(request: FinancialAnalysisRequest):
    financials = request.model_dump()

    try:
        return calculate_financial_metrics(financials)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not calculate financial metrics: {exc}",
        ) from exc


# Allow the local React development server to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "AI Due Diligence Copilot API is running",
        "version": "0.1.0",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/documents/extract")
async def extract_pdf(file: UploadFile = File(...)):
    # Validate the uploaded file.
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file.",
        )

    # Read the uploaded PDF and calculate its content-based identity.
    pdf_bytes = await file.read()

    if not pdf_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(pdf_bytes) > 20 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="PDF must be 20 MB or smaller.",
        )

    document_id = hashlib.sha256(pdf_bytes).hexdigest()

    try:
        with pymupdf.open(
            stream=BytesIO(pdf_bytes),
            filetype="pdf",
        ) as pdf:
            pages = []

            ocr_pages = []
            ocr_skipped_pages = []

            for page_number, page in enumerate(pdf, start=1):
                page_text = page.get_text("text").strip()

                # Financial tables are often embedded as page images. PyMuPDF's
                # normal text extraction silently misses those figures, so OCR
                # image-heavy pages when their native text layer is sparse.
                has_images = bool(page.get_images(full=True))
                # Preserve a high-resolution page image whenever the native
                # text layer is absent or an embedded image may contain a table.
                needs_visual = len(page_text) < 250 or (
                    has_images and len(page_text) < 2500
                )
                page_image_path = None
                if needs_visual:
                    image_dir = DOCUMENT_PAGE_IMAGES / document_id
                    image_dir.mkdir(parents=True, exist_ok=True)
                    page_image_path = image_dir / f"page_{page_number}.png"
                    page.get_pixmap(dpi=250, alpha=False).save(str(page_image_path))

                # OCR improves searchable text, but it is no longer a hard
                # dependency: Gemini Vision can read the preserved page image.
                needs_ocr = needs_visual

                if needs_ocr:
                    try:
                        text_page = page.get_textpage_ocr(
                            language="eng",
                            dpi=300,
                            full=True,
                        )
                        ocr_text = page.get_text(
                            "text",
                            textpage=text_page,
                        ).strip()

                        # full=True OCR includes the native text layer as well
                        # as text recognised from embedded page images. Prefer
                        # it whenever it returns usable content, even if its
                        # character count is shorter than the native layer.
                        if len(ocr_text) >= 40:
                            page_text = ocr_text
                            ocr_pages.append(page_number)
                        else:
                            # Keep native text if OCR was inconclusive; record
                            # the page so the upload diagnostics expose the gap.
                            ocr_skipped_pages.append({
                                "page": page_number,
                                "reason": (
                                    "OCR returned too little usable text "
                                    f"({len(ocr_text)} characters)"
                                ),
                            })
                    except Exception as ocr_exc:
                        ocr_skipped_pages.append({
                            "page": page_number,
                            "reason": (
                                "OCR unavailable; preserved page image for "
                                f"Gemini Vision ({type(ocr_exc).__name__})"
                            ),
                        })
                        print(
                            f"OCR skipped on PDF page {page_number}; "
                            "vision image retained for extraction: "
                            f"{type(ocr_exc).__name__}: {ocr_exc}"
                        )

                if page_image_path is not None:
                    image_note = (
                        "[PAGE IMAGE AVAILABLE FOR VISUAL EXTRACTION. "
                        "This note is indexing metadata, not source text. "
                        "The original image may contain an income statement, "
                        "balance sheet, cash flow statement, revenue, net income, "
                        "total assets, total liabilities, total equity, operating "
                        "activities, reporting-period headers and numeric amounts.]"
                    )
                    page_text = f"{page_text}\n{image_note}".strip()

                pages.append({
                    "page": page_number,
                    "text": page_text,
                    "ocr_used": page_number in ocr_pages,
                })

            extracted_text = "\n\n".join(
                f"[Page {page['page']}]\n{page['text']}"
                for page in pages
                if page["text"]
            )

            chunks = chunk_pages(
                pages=pages,
                filename=file.filename,
                document_id=document_id,
            )
            indexed_count = index_chunks(chunks)

    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Could not read or index this PDF. Check that it is valid.",
        ) from exc

    return {
        "filename": file.filename,
        "document_id": document_id,
        "total_pages": len(pages),
        "pages_with_text": sum(bool(page["text"]) for page in pages),
        "character_count": len(extracted_text),
        "text": extracted_text,
        "pages": pages,
        "ocr_required": bool(ocr_skipped_pages),
        "ocr_pages": ocr_pages,
        "ocr_skipped_pages": ocr_skipped_pages,
        "visual_pages": [
            page_number
            for page_number in range(1, len(pages) + 1)
            if (DOCUMENT_PAGE_IMAGES / document_id / f"page_{page_number}.png").exists()
        ],
        "chunk_count": len(chunks),
        "chunks": chunks,
        "indexed_count": indexed_count,
    }


@app.post("/search")
def search_documents(request: SearchRequest):
    matches = search_chunks(
        query=request.query,
        top_k=request.top_k,
    )

    return {
        "query": request.query,
        "result_count": len(matches),
        "results": matches,
    }


@app.post("/financial-extract")
def financial_extract_endpoint(
    document_id: str = Query(..., min_length=64, max_length=64),
    filename: str | None = Query(default=None, min_length=1),
):
    document_id = validate_document_id(document_id)

    try:
        return extract_financials(
            document_id=document_id,
            filename=filename,
        )
    except requests.RequestException as exc:
        print(f"Financial extraction HTTP error: {exc!r}")
        raise HTTPException(
            status_code=502,
            detail=f"Financial extraction request failed: {exc}",
        ) from exc
    except (ValueError, KeyError) as exc:
        print(f"Financial extraction parsing error: {exc!r}")
        raise HTTPException(
            status_code=502,
            detail=f"Could not parse extraction response: {exc}",
        ) from exc


@app.post("/financial-analyze-document")
def analyze_financial_document(
    document_id: str = Query(..., min_length=64, max_length=64),
    filename: str | None = Query(default=None, min_length=1),
):
    document_id = validate_document_id(document_id)

    extraction = extract_financials(
        document_id=document_id,
        filename=filename,
    )

    if not extraction.get("sources"):
        raise HTTPException(
            status_code=404,
            detail=(
                "No financial passages found for this document. "
                "Upload and index the PDF again if necessary."
            ),
        )

    financials = extraction.get("financials", {})
    analysis = calculate_financial_metrics(financials)

    return {
        "document_id": document_id,
        "message": (
            "Financial analysis completed. "
            "Verify figures against the source."
        ),
        "financials": financials,
        "analysis": analysis,
        "evidence": extraction.get("evidence", {}),
        "field_status": extraction.get("field_status", {}),
        "verified_fields": extraction.get("verified_fields", []),
        "sources": extraction.get("sources", []),
        "currency": extraction.get(
            "currency",
            financials.get("currency", "unspecified"),
        ),
        "units": extraction.get(
            "units",
            financials.get("units", "unspecified"),
        ),
        "reporting_period": extraction.get(
            "reporting_period",
            financials.get("reporting_period"),
        ),
    }


@app.post("/ask")
def ask_question_endpoint(request: AskRequest):
    # Keep document-scoped RAG queries tied to a valid SHA-256 document ID.
    document_id = validate_document_id(request.document_id)

    try:
        return answer_question(
            question=request.question,
            document_id=document_id,
            top_k=request.top_k,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
