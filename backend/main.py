import requests
from fastapi import Query
from io import BytesIO
from backend.services.chunking import chunk_pages
import pymupdf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from backend.services.vector_store import (
    index_chunks,
    search_chunks,
)
from backend.services.rag import answer_question
from backend.services.financial_analysis import calculate_financial_metrics
from backend.services.financial_extraction import extract_financials
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

    # Read the uploaded PDF into memory.
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

    try:
        with pymupdf.open(
            stream=BytesIO(pdf_bytes),
            filetype="pdf",
        ) as pdf:
            pages = []

            for page_number, page in enumerate(pdf, start=1):
                page_text = page.get_text("text").strip()

                pages.append({
                    "page": page_number,
                    "text": page_text,
                })

            extracted_text = "\n\n".join(
                f"[Page {page['page']}]\n{page['text']}"
                for page in pages
                if page["text"]
            )
            chunks = chunk_pages(
                pages=pages,
                filename=file.filename,
            )
            indexed_count = index_chunks(chunks)

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Could not read this PDF. Check that it is valid.",
        ) from exc

    return {
        "filename": file.filename,
        "total_pages": len(pages),
        "pages_with_text": sum(bool(p["text"]) for p in pages),
        "character_count": len(extracted_text),
        "text": extracted_text,
        "pages": pages,
        "ocr_required": not bool(extracted_text),
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
@app.post("/ask")
def ask_documents(request: AskRequest):
    try:
        result = answer_question(
            question=request.question,
            top_k=request.top_k,
        )

        return {
            "question": request.question,
            **result,
        }

    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

@app.get("/financial-evidence")
def get_financial_evidence():
    financial_queries = {
        "revenue": "total revenue and revenue by year",
        "profit": "gross profit operating income and net income",
        "assets_and_debt": "total assets total liabilities total debt and shareholders equity",
        "cash_flow": "cash flow from operating activities and capital expenditures",
        "liquidity": "current assets and current liabilities",
    }

    evidence = {}

    for category, query in financial_queries.items():
        results = search_chunks(query=query, top_k=3)

        evidence[category] = [
            {
                "source": result["source"],
                "page": result["page"],
                "text": result["text"],
            }
            for result in results
        ]

    return {
        "message": "Retrieved financial evidence from indexed documents.",
        "evidence": evidence,
    }

@app.post("/financial-extract")
def financial_extract_endpoint(filename: str = Query(..., min_length=1)):
    try:
        return extract_financials(filename=filename)
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
def analyze_financial_document(filename: str = Query(..., min_length=1)):
    extraction = extract_financials(filename=filename)

    if not extraction.get("sources"):
        raise HTTPException(
            status_code=404,
            detail=f"No financial passages found for '{filename}'. Upload the PDF again.",
        )

    financials = extraction.get("financials", {})
    analysis = calculate_financial_metrics(financials)

    return {
        "message": (
            "Financial analysis completed. "
            "Verify figures against the source."
        ),
        "financials": financials,
        "analysis": analysis,
        "evidence": extraction.get("evidence", {}),
        "sources": extraction.get("sources", []),
        "currency": extraction.get("currency", financials.get("currency", "unspecified")),
        "units": extraction.get("units", financials.get("units", "unspecified")),
        "reporting_period": extraction.get(
            "reporting_period",
            financials.get("reporting_period"),
        ),
    }