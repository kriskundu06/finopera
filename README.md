# FinOpera — Financial Due Diligence RAG Platform

FinOpera is a financial document analysis application that uses retrieval-augmented generation (RAG) to help users extract financial information from company reports, calculate financial ratios, ask questions about source documents, and review potential financial risk indicators.

## Features

- **PDF processing:** Extract text from financial documents and organize it into page-aware chunks.
- **Financial extraction:** Retrieve financial statement information using Gemini, with supporting evidence where available.
- **Financial analysis:** Calculate financial ratios from extracted financial data.
- **RAG-powered Q&A:** Ask questions about uploaded documents and inspect retrieved source passages.
- **Risk screening:** Review preliminary risk indicators alongside supporting evidence.
- **Interactive dashboard:** Explore extracted financial information through a web interface.

## Technology Stack

**Backend**
- Python
- FastAPI
- PyMuPDF
- ChromaDB
- Google Gemini API

**Frontend**
- React
- TypeScript
- Vite
- Tailwind CSS
- Recharts
- Lucide React

## Project Structure

```text
due-diligence-rag/
├── backend/
│   ├── main.py
│   └── services/
│       ├── chunking.py
│       ├── financial_analysis.py
│       ├── financial_extraction.py
│       ├── rag.py
│       └── vector_store.py
├── frontend/
│   ├── src/
│   ├── package.json
│   └── vite.config.ts
├── tests/
├── requirements.txt
├── .env.example
└── .gitignore
```

## Prerequisites

- Python 3.10 or a compatible version for the dependencies
- Node.js and npm
- A Google Gemini API key

## Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd due-diligence-rag
```

### 2. Set up the backend

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install Python dependencies:

```powershell
pip install -r requirements.txt
```

Create your local `.env` file by copying `.env.example`, then set your own Gemini API key and the model name supported by your project.

```powershell
Copy-Item .env.example .env
```

Start the backend from the project root:

```powershell
python -m uvicorn backend.main:app --reload
```

### 3. Set up the frontend

Open another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite in your terminal.

## Configuration

The backend uses these environment variables:

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Your private Google Gemini API key |
| `GEMINI_MODEL` | The Gemini model name used by the application |

Never commit `.env` or publish API keys. The `.env.example` file should contain placeholders only.

## Limitations

- Financial extraction and risk indicators depend on the quality and coverage of the source documents.
- AI-generated results may be incomplete or incorrect and should be checked against the original filings.
- Risk indicators are preliminary screening aids, not investment advice or a substitute for professional financial due diligence.
- Deployment and end-to-end validation should be confirmed before treating the application as production-ready.

## Roadmap

- Improve extraction reliability and handling of API errors.
- Expand automated tests and financial-data validation.
- Improve document-level source isolation and evidence traceability.

## Disclaimer

FinOpera is an educational portfolio project. Its outputs are for informational purposes only and should not be relied on as the sole basis for investment or financial decisions.
