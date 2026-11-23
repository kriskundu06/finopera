import hashlib
import logging
import re
from io import BytesIO
from pathlib import Path

import pymupdf
import requests
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, Body
from fastapi.responses import FileResponse, StreamingResponse
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.services.chunking import chunk_pages
from backend.services.financial_analysis import calculate_financial_metrics
from backend.services.financial_extraction import extract_financials
from backend.services.rag import answer_question, generate_due_diligence_report
from backend.services.vector_store import index_chunks, search_chunks


logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "data" / "uploads"


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
        logger.warning("Invalid financial-analysis input: %s", exc)
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
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    # Accept Vite's changing local dev ports without opening credentialed CORS.
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
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
                needs_ocr = has_images and len(page_text) < 1500

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
                        if len(ocr_text) >= 80:
                            page_text = ocr_text
                            ocr_pages.append(page_number)
                        else:
                            ocr_skipped_pages.append({
                                "page": page_number,
                                "reason": "OCR returned too little usable text",
                            })
                    except Exception as ocr_exc:
                        raise RuntimeError(
                            "OCR is required to read image-based tables on "
                            f"PDF page {page_number}, but OCR failed. Install "
                            "Tesseract OCR and ensure it is available on PATH. "
                            f"Original error: {type(ocr_exc).__name__}: {ocr_exc}"
                        ) from ocr_exc

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

            # Retain the original PDF so users can download the exact source
            # report later. The SHA-256 filename avoids trusting upload names.
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            stored_pdf = UPLOAD_DIR / f"{document_id}.pdf"
            if not stored_pdf.exists():
                stored_pdf.write_bytes(pdf_bytes)

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
        "chunk_count": len(chunks),
        "chunks": chunks,
        "indexed_count": indexed_count,
    }


@app.get("/documents/{document_id}/download")
def download_original_document(document_id: str, filename: str | None = Query(default=None, max_length=255)):
    document_id = validate_document_id(document_id)
    pdf_path = UPLOAD_DIR / f"{document_id}.pdf"

    if not pdf_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Original PDF is not available on this server. Upload the report again to enable downloading.",
        )

    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=Path(filename).name if filename and Path(filename).name.lower().endswith(".pdf") else f"{document_id}.pdf",
    )


@app.post("/financial-analysis/report")
def download_financial_analysis_report(payload: dict = Body(...)):
    """Generate a portable PDF summary of extracted financials and their evidence."""
    from io import BytesIO
    from xml.sax.saxutils import escape
    from reportlab.graphics.shapes import Drawing, String, Rect, Line

    financials = payload.get("financials") or {}
    evidence = payload.get("evidence") or {}
    analysis = (payload.get("analysis") or {}).get("metrics") or {}
    if not isinstance(financials, dict) or not financials:
        raise HTTPException(status_code=400, detail="No financial figures are available to export.")

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=42, leftMargin=42, topMargin=42, bottomMargin=42)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("FinOpera — Financial Analysis Report", styles["Title"]),
        Spacer(1, 8),
        Paragraph(f"Source: {escape(str(payload.get('filename') or 'Uploaded report'))}", styles["Normal"]),
        Paragraph(f"Reporting period: {escape(str(payload.get('reporting_period') or 'Not identified'))}", styles["Normal"]),
        Paragraph(f"Currency: {escape(str(payload.get('currency') or 'Unspecified'))} · Units: {escape(str(payload.get('units') or 'Unspecified'))}", styles["Normal"]),
        Paragraph(f"Extraction status: {escape(str(payload.get('extraction_status') or 'unknown'))}", styles["Normal"]),
        Spacer(1, 14),
        Paragraph("Visual analysis", styles["Heading2"]),
    ]

    def add_bar_chart(title, items, percent=False):
        available = [
            (label, float(financials[key]))
            for label, key in items
            if isinstance(financials.get(key), (int, float))
            and not isinstance(financials.get(key), bool)
        ]
        if len(available) < 2:
            return
        width = 500
        row_height = 24
        height = 42 + row_height * len(available)
        drawing = Drawing(width, height)
        drawing.add(String(0, height - 12, title, fontName="Helvetica-Bold", fontSize=9, fillColor=colors.HexColor("#243047")))
        label_width = 132
        plot_left = label_width
        plot_width = width - label_width - 12
        values = [value for _, value in available]
        low = min(0.0, min(values))
        high = max(0.0, max(values))
        span = high - low
        if span == 0:
            span = 1.0
        scale = plot_width / span
        zero_x = plot_left + (0 - low) * scale
        drawing.add(Line(zero_x, 10, zero_x, height - 24, strokeColor=colors.HexColor("#94A3B8"), strokeWidth=0.7))
        for index, (label, value) in enumerate(available):
            y = height - 34 - index * row_height
            drawing.add(String(0, y + 3, label[:23], fontName="Helvetica", fontSize=7, fillColor=colors.HexColor("#334155")))
            value_x = plot_left + (value - low) * scale
            bar_x = min(zero_x, value_x)
            bar_width = max(1, abs(value_x - zero_x))
            drawing.add(Rect(bar_x, y, bar_width, 11, fillColor=colors.HexColor("#38A3A5"), strokeColor=None))
            shown = f"{value:,.2f}%" if percent else f"{value:,.2f}"
            text_x = min(width - 62, value_x + 4) if value >= 0 else max(plot_left, value_x - 54)
            drawing.add(String(text_x, y + 2, shown, fontName="Helvetica", fontSize=6.5, fillColor=colors.HexColor("#243047")))
        story.append(drawing)
        story.append(Spacer(1, 8))

    add_bar_chart("Core income statement figures", [
        ("Revenue", "revenue"),
        ("Operating income", "operating_income"),
        ("Net income", "net_income"),
    ])
    add_bar_chart("Core balance sheet figures", [
        ("Current assets", "current_assets"),
        ("Current liabilities", "current_liabilities"),
        ("Total debt", "total_debt"),
        ("Total equity", "total_equity"),
        ("Total assets", "total_assets"),
    ])
    ratio_values = [
        (key.replace("_", " ").title(), float(value))
        for key, value in analysis.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if len(ratio_values) >= 2:
        width, row_height = 500, 24
        height = 42 + row_height * len(ratio_values)
        drawing = Drawing(width, height)
        drawing.add(String(0, height - 12, "Calculated ratios", fontName="Helvetica-Bold", fontSize=9, fillColor=colors.HexColor("#243047")))
        label_width, plot_width = 190, 298
        values = [value for _, value in ratio_values]
        low, high = min(0.0, min(values)), max(0.0, max(values))
        span = max(1.0, high - low)
        scale = plot_width / span
        zero_x = label_width + (0 - low) * scale
        drawing.add(Line(zero_x, 10, zero_x, height - 24, strokeColor=colors.HexColor("#94A3B8"), strokeWidth=0.7))
        for index, (label, value) in enumerate(ratio_values):
            y = height - 34 - index * row_height
            drawing.add(String(0, y + 3, label[:30], fontName="Helvetica", fontSize=7, fillColor=colors.HexColor("#334155")))
            value_x = label_width + (value - low) * scale
            drawing.add(Rect(min(zero_x, value_x), y, max(1, abs(value_x - zero_x)), 11, fillColor=colors.HexColor("#6477C8"), strokeColor=None))
            drawing.add(String(min(width - 48, value_x + 4) if value >= 0 else max(label_width, value_x - 42), y + 2, f"{value:,.2f}", fontName="Helvetica", fontSize=6.5, fillColor=colors.HexColor("#243047")))
        story.append(drawing)
        story.append(Spacer(1, 10))

    story.append(Paragraph(
        "Charts include only extracted numeric values; unavailable fields are omitted, not treated as zero.",
        styles["Italic"],
    ))
    story.append(Spacer(1, 10))

    rows = [["Statement field", "Value", "Evidence"]]
    for field, value in financials.items():
        if field in {"currency", "units", "reporting_period"}:
            continue
        ref = evidence.get(field) if isinstance(evidence.get(field), dict) else {}
        source = "Calculated" if str(ref.get("type", "")).startswith("calculated") else (
            f"{ref.get('filename') or payload.get('filename') or 'Source'} · p. {ref.get('page')}"
            if ref.get("page") else "Not available in extracted evidence"
        )
        rows.append([field.replace("_", " ").title(), "—" if value is None else str(value), source])
    table = Table(rows, colWidths=[155, 115, 205], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243047")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F9")]),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([table, Spacer(1, 16), Paragraph("Calculated ratios", styles["Heading2"])])
    ratio_rows = [["Metric", "Value"]] + [[key.replace("_", " ").title(), "—" if value is None else str(value)] for key, value in analysis.items()]
    if len(ratio_rows) > 1:
        ratios = Table(ratio_rows, colWidths=[235, 235], repeatRows=1)
        ratios.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243047")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(ratios)
    story.extend([Spacer(1, 18), Paragraph("Important: Model-assisted extraction. Verify material figures against the original PDF before relying on them.", styles["Italic"])])
    doc.build(story)
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="application/pdf", headers={"Content-Disposition": 'attachment; filename="finopera-financial-analysis.pdf"'})


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
        logger.exception("Financial extraction dependency request failed")
        raise HTTPException(
            status_code=502,
            detail="The financial extraction provider could not be reached. Please try again.",
        ) from exc
    except RuntimeError as exc:
        logger.exception("Financial extraction could not complete")
        message = str(exc)
        status_code = 503 if "GEMINI_API_KEY is missing" in message else 502
        detail = (
            "Financial extraction is not configured. Add GEMINI_API_KEY to the backend environment."
            if status_code == 503
            else "The financial extraction provider could not complete the request. Please retry."
        )
        raise HTTPException(status_code=status_code, detail=detail) from exc
    except (ValueError, KeyError) as exc:
        logger.exception("Financial extraction response validation failed")
        raise HTTPException(
            status_code=502,
            detail="The financial extraction provider returned an invalid response.",
        ) from exc


@app.post("/financial-analyze-document")
def analyze_financial_document(
    document_id: str = Query(..., min_length=64, max_length=64),
    filename: str | None = Query(default=None, min_length=1),
):
    document_id = validate_document_id(document_id)

    try:
        extraction = extract_financials(
            document_id=document_id,
            filename=filename,
        )
    except requests.RequestException as exc:
        logger.exception("Financial analysis dependency request failed")
        raise HTTPException(
            status_code=502,
            detail="The financial extraction provider could not be reached. Please try again.",
        ) from exc
    except RuntimeError as exc:
        logger.exception("Financial analysis extraction failed")
        message = str(exc)
        status_code = 503 if "GEMINI_API_KEY is missing" in message else 502
        detail = (
            "Financial extraction is not configured. Add GEMINI_API_KEY to the backend environment."
            if status_code == 503
            else "The financial extraction provider could not complete the request. Please retry."
        )
        raise HTTPException(status_code=status_code, detail=detail) from exc
    except (ValueError, KeyError) as exc:
        logger.exception("Financial analysis extraction response was invalid")
        raise HTTPException(
            status_code=502,
            detail="The financial extraction provider returned an invalid response.",
        ) from exc

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
            extraction.get("message")
            or "Financial analysis completed. Verify figures against the source."
        ),
        "extraction_status": extraction.get("extraction_status", "completed"),
        "statement_errors": extraction.get("statement_errors", {}),
        "verified_fields": extraction.get("verified_fields", []),
        "financials": financials,
        "analysis": analysis,
        "evidence": extraction.get("evidence", {}),
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
    except RuntimeError as exc:
        logger.exception("Document question answering failed")
        message = str(exc)
        status_code = 503 if "GEMINI_API_KEY is missing" in message else 502
        detail = (
            "AI Analyst is not configured. Add GEMINI_API_KEY to the backend environment."
            if status_code == 503
            else "AI Analyst could not complete the request. Please try again."
        )
        raise HTTPException(status_code=status_code, detail=detail) from exc
    except requests.RequestException as exc:
        logger.exception("AI Analyst dependency request failed")
        raise HTTPException(
            status_code=502,
            detail="AI Analyst could not reach its provider. Please try again.",
        ) from exc


@app.post("/due-diligence/report")
def create_due_diligence_report(payload: dict = Body(...)):
    """Generate a comprehensive, evidence-cited due-diligence PDF."""
    document_id = validate_document_id(payload.get("document_id", ""))
    financials = payload.get("financials") or {}
    analysis = payload.get("analysis") or {}
    metrics = analysis.get("metrics", {}) if isinstance(analysis, dict) else {}
    try:
        result = generate_due_diligence_report(
            document_id=document_id,
            financials=financials if isinstance(financials, dict) else {},
            metrics=metrics if isinstance(metrics, dict) else {},
            reporting_period=payload.get("reporting_period"),
            filename=payload.get("filename"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("Due-diligence report generation failed")
        message = str(exc)
        status_code = 503 if "GEMINI_API_KEY is missing" in message else 502
        detail = (
            "AI Due Diligence is not configured. Add GEMINI_API_KEY to the backend environment."
            if status_code == 503
            else "The AI due-diligence report could not be generated. Please retry."
        )
        raise HTTPException(status_code=status_code, detail=detail) from exc

    from xml.sax.saxutils import escape
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import KeepTogether, PageBreak

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=48, leftMargin=48,
        topMargin=52, bottomMargin=48,
        title="FinOpera Comprehensive Due Diligence Report",
        author="FinOpera AI Analyst",
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportSubtitle", parent=styles["Normal"], alignment=TA_CENTER,
        textColor=colors.HexColor("#64748B"), fontSize=9, leading=14,
    ))
    styles.add(ParagraphStyle(
        name="ReportBody", parent=styles["BodyText"], fontSize=9,
        leading=14, spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="ReportSection", parent=styles["Heading2"], fontSize=13,
        leading=17, textColor=colors.HexColor("#243047"),
        spaceBefore=14, spaceAfter=7, keepWithNext=True,
    ))
    story = [
        Spacer(1, 36),
        Paragraph("FINOPERA", ParagraphStyle(
            name="Brand", parent=styles["Title"], alignment=TA_CENTER,
            textColor=colors.HexColor("#65A30D"), fontSize=14, leading=18,
        )),
        Spacer(1, 14),
        Paragraph("Comprehensive Due Diligence Report", ParagraphStyle(
            name="CoverTitle", parent=styles["Title"], alignment=TA_CENTER,
            textColor=colors.HexColor("#172033"), fontSize=23, leading=29,
        )),
        Spacer(1, 12),
        Paragraph(escape(str(payload.get("filename") or "Uploaded report")), styles["ReportSubtitle"]),
        Paragraph("Reporting period: " + escape(str(payload.get("reporting_period") or "Not identified")), styles["ReportSubtitle"]),
        Spacer(1, 12),
        Paragraph("AI-assisted preliminary assessment · Evidence-linked · Confidential working document", styles["ReportSubtitle"]),
        Spacer(1, 28),
        Paragraph("Scope and important limitations", styles["Heading2"]),
        Paragraph(
            "This report is a preliminary, document-based diligence aid generated from the uploaded material. "
            "It is not an audit, legal opinion, valuation, credit rating, or investment recommendation. "
            "Statements should be checked against the original filing and independent sources before decisions are made.",
            styles["ReportBody"],
        ),
        PageBreak(),
        Paragraph("Due diligence assessment", styles["Title"]),
        Spacer(1, 10),
    ]
    import re
    for raw_line in result["report"].splitlines():
        line = raw_line.strip()
        if not line:
            story.append(Spacer(1, 3))
            continue
        safe = escape(line)
        safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
        safe = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", safe)
        heading = re.match(r"^(?:#{1,3}\s+)?(\d{1,2}[.)]\s+.+|#{1,3}\s+.+)$", line)
        if heading:
            title = re.sub(r"^#{1,3}\s+", "", line)
            story.append(Paragraph(escape(title), styles["ReportSection"]))
        elif line.startswith(("- ", "• ", "* ")):
            story.append(Paragraph("• " + safe[2:], styles["ReportBody"]))
        else:
            story.append(Paragraph(safe, styles["ReportBody"]))

    story.extend([
        Spacer(1, 16),
        Paragraph("Evidence index", styles["Heading2"]),
        Paragraph("The source IDs below correspond to citations used in the report.", styles["ReportBody"]),
    ])
    source_rows = [["Source", "Document", "Page"]]
    for source in result["sources"]:
        source_rows.append([
            source["source_id"], str(source["source"])[:70],
            str(source["page"] if source["page"] is not None else "Not available"),
        ])
    source_table = Table(source_rows, colWidths=[55, 350, 65], repeatRows=1)
    source_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243047")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F9")]),
        ("FONTSIZE", (0, 0), (-1, -1), 8), ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(source_table)

    def add_page_number(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#64748B"))
        canvas.drawString(48, 28, "FinOpera · Preliminary due diligence · Verify against original sources")
        canvas.drawRightString(A4[0] - 48, 28, f"Page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    buffer.seek(0)
    return StreamingResponse(
        buffer, media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="finopera-due-diligence-report.pdf"'},
    )
