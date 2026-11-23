

import { useRef, useState } from 'react'
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from 'recharts'

import {

  Activity,

  AlertTriangle,

  Info,


  Landmark,


  ArrowDownRight,

  ArrowUpRight,

  BriefcaseBusiness,

  ChartNoAxesCombined,

  CircleHelp,

  FileSearch,

  FileText,

  Gauge,

  LayoutDashboard,

  Menu,

  Send,
  Settings,

  Bot,

  UserRound,

  LoaderCircle,

  ExternalLink,

  ShieldAlert,


  Upload,

  X,

} from 'lucide-react'

const navigation = [
  { label: 'Dashboard', page: 'Overview', icon: LayoutDashboard },
  { label: 'Documents', page: 'Overview', icon: FileText },
  { label: 'AI Analyst', page: 'Ask AI', icon: Bot },
  { label: 'Financial Statements', page: 'Financial analysis', icon: ChartNoAxesCombined },
  { label: 'Settings', page: 'Settings', icon: Settings },
]

const metrics = [

  {

    label: 'Total revenue',

    value: '$109.9M',

    change: '+12.8%',

    positive: true,

    icon: Activity,

    note: 'vs. previous period',

  },

  {

    label: 'Net profit margin',

    value: '56.94%',

    change: '+4.2%',

    positive: true,

    icon: ChartNoAxesCombined,

    note: 'vs. previous period',

  },

  {

    label: 'Current ratio',

    value: '1.92×',

    change: 'Healthy',

    positive: true,

    icon: Gauge,

    note: 'Short-term liquidity',

  },

  {

    label: 'Total debt',

    value: '$79.5M',

    change: '0.17×',

    positive: true,

    icon: BriefcaseBusiness,

    note: 'Debt-to-equity ratio',

  },

]

type FinancialAnalysisResponse = {

  message?: string

  financials?: Record<string, number | string | null | undefined>

  analysis?: {

    metrics?: Record<string, number | null | undefined>

    interpretation_notes?: string[]

    currency?: string

    units?: string

  }

  evidence?: Record<string, any>

  sources?: Array<{ source?: string; page?: number }>

  currency?: string

  units?: string

  reporting_period?: string | null

}

type AskSource = {

  source_id?: string

  source?: string

  page?: number

  chunk_number?: number

  text?: string

}

type AskResponse = {

  question?: string

  answer?: string

  sources?: AskSource[]

  detail?: string

}

type ChatMessage = {

  id: string

  role: 'user' | 'assistant'

  content: string

  sources?: AskSource[]

}

const SUGGESTED_QUESTIONS = [

  'What are the company’s main financial risks?',

  'What is the company’s total debt, and which page supports it?',

  'Summarize revenue and profitability for this reporting period.',

]

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '')

function formatAmount(value: unknown, currency = 'USD', units = 'millions') {

  if (typeof value !== 'number' || !Number.isFinite(value)) return '—'

  const symbol = currency === 'USD' ? '$' : currency === 'INR' ? '₹' : `${currency} `

  const normalizedUnits = units.toLowerCase()

  let amount = value

  let suffix = ''

  if (normalizedUnits.includes('million')) {

    if (Math.abs(value) >= 1000) { amount = value / 1000; suffix = 'B' }

    else suffix = 'M'

  } else if (normalizedUnits.includes('billion')) suffix = 'B'

  else if (normalizedUnits.includes('thousand')) suffix = 'K'

  return `${symbol}${amount.toLocaleString(undefined, { maximumFractionDigits: 2 })}${suffix}`

}

function formatPercent(value: unknown) {

  return typeof value === 'number' && Number.isFinite(value) ? `${value.toFixed(2)}%` : '—'

}

function formatRatio(value: unknown) {

  return typeof value === 'number' && Number.isFinite(value) ? `${value.toFixed(2)}×` : '—'

}

function App() {

  const [activePage, setActivePage] = useState('Overview')

  const [selectedFile, setSelectedFile] = useState<File | null>(null)

  const [dragging, setDragging] = useState(false)

  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)

  const [isAnalyzing, setIsAnalyzing] = useState(false)

  const [uploadError, setUploadError] = useState('')

  const [uploadMessage, setUploadMessage] = useState('')

  const [documentInfo, setDocumentInfo] = useState<{
    filename: string
    document_id: string
    total_pages: number
    indexed_count: number
  } | null>(null)

  const [financialResult, setFinancialResult] = useState<FinancialAnalysisResponse | null>(null)

  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([])

  const [chatInput, setChatInput] = useState('')

  const [isAsking, setIsAsking] = useState(false)

  const [chatError, setChatError] = useState('')

  const chatEndRef = useRef<HTMLDivElement>(null)

  const fileInputRef = useRef<HTMLInputElement>(null)

  function selectFile(file?: File) {

    if (!file) return

    if (file.type !== 'application/pdf' && !file.name.toLowerCase().endsWith('.pdf')) {

      setUploadError('Please select a PDF file.')

      return

    }

    if (file.size > 20 * 1024 * 1024) {

      setUploadError('PDF must be 20 MB or smaller.')

      return

    }

    setUploadError('')

    setUploadMessage('')

    setSelectedFile(file)

  }

  async function handleUpload() {

    if (!selectedFile || isAnalyzing) return

    setIsAnalyzing(true)

    setUploadError('')

    setUploadMessage('Uploading and indexing PDF…')

    setFinancialResult(null)

    setDocumentInfo(null)

    try {

      const formData = new FormData()

      formData.append('file', selectedFile)

      const uploadResponse = await fetch(`${API_BASE_URL}/documents/extract`, { method: 'POST', body: formData })

      const uploadPayload = await uploadResponse.json().catch(() => ({}))

      if (!uploadResponse.ok) throw new Error(uploadPayload.detail || `PDF upload failed (${uploadResponse.status}).`)

      if (uploadPayload.ocr_required) throw new Error('This PDF contains no selectable text. OCR is not enabled for this document yet.')

      if (typeof uploadPayload.document_id !== 'string' || !uploadPayload.document_id) {
        throw new Error('The backend did not return a document ID. Please update the backend and try uploading again.')
      }
      setDocumentInfo({
        filename: uploadPayload.filename || selectedFile.name,
        document_id: uploadPayload.document_id,
        total_pages: uploadPayload.total_pages ?? 0,
        indexed_count: uploadPayload.indexed_count ?? 0,
      })

      setUploadMessage('PDF indexed. Extracting financial figures and calculating ratios…')

      const analysisResponse = await fetch(`${API_BASE_URL}/financial-analyze-document?document_id=${encodeURIComponent(uploadPayload.document_id)}&filename=${encodeURIComponent(uploadPayload.filename || selectedFile.name)}`,{ method: 'POST' })

      const analysisPayload = await analysisResponse.json().catch(() => ({}))

      if (!analysisResponse.ok) throw new Error(analysisPayload.detail || `Financial analysis failed (${analysisResponse.status}).`)

      setFinancialResult(analysisPayload)

      setUploadMessage('Analysis complete. Dashboard metrics now use the uploaded report.')

      setActivePage('Financial analysis')

    } catch (error) {

      setUploadError(error instanceof Error ? error.message : 'Something went wrong while analyzing the PDF.')

      setUploadMessage('')

    } finally { setIsAnalyzing(false) }

  }

  async function askQuestion(questionOverride?: string) {

    const question = (questionOverride ?? chatInput).trim()

    if (!question || isAsking) return

    if (!documentInfo?.document_id) {
      setChatError('Please upload a PDF before asking questions.')
      return
    }

    const userMessage: ChatMessage = {

      id: `${Date.now()}-user`,

      role: 'user',

      content: question,

    }

    setChatMessages((current) => [...current, userMessage])

    setChatInput('')

    setChatError('')

    setIsAsking(true)

    try {

      const response = await fetch(`${API_BASE_URL}/ask`, {

        method: 'POST',

        headers: { 'Content-Type': 'application/json' },

        body: JSON.stringify({
          question,
          document_id: documentInfo.document_id,
          top_k: 5,
        }),

      })

      const payload = await response.json().catch(() => ({} as AskResponse)) as AskResponse

      if (!response.ok) {

        const detail = typeof payload.detail === 'string' ? payload.detail : `Question failed (${response.status}).`

        throw new Error(detail)

      }

      if (!payload.answer || typeof payload.answer !== 'string') {

        throw new Error('The assistant returned an empty answer. Please try rephrasing your question.')

      }

      setChatMessages((current) => [

        ...current,

        {

          id: `${Date.now()}-assistant`,

          role: 'assistant',

          content: payload.answer as string,

          sources: Array.isArray(payload.sources) ? payload.sources : [],

        },

      ])

    } catch (error) {

      setChatError(error instanceof Error ? error.message : 'Could not reach the AI assistant. Check that the backend is running.')

    } finally {

      setIsAsking(false)

      window.setTimeout(() => chatEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }), 50)

    }

  }

  const financials = financialResult?.financials ?? {}

  const analysis = financialResult?.analysis?.metrics ?? {}

  const currency = financialResult?.currency || String(financials.currency || 'USD')

  const units = financialResult?.units || String(financials.units || 'millions')

  const metricCards = [

    { ...metrics[0], value: financialResult ? formatAmount(financials.revenue, currency, units) : '—', change: financialResult ? 'From report' : 'Awaiting report', note: financialResult ? 'Reported revenue' : 'Upload a report to populate' },

    { ...metrics[1], value: financialResult ? formatPercent(analysis.net_profit_margin) : '—', change: financialResult ? 'Calculated' : 'Awaiting report', note: financialResult ? 'Net income / revenue' : 'Upload a report to populate' },

    { ...metrics[2], value: financialResult ? formatRatio(analysis.current_ratio) : '—', change: financialResult ? 'Calculated' : 'Awaiting report', note: financialResult ? 'Current assets / liabilities' : 'Upload a report to populate' },

    { ...metrics[3], value: financialResult ? formatAmount(financials.total_debt, currency, units) : '—', change: financialResult ? `D/E ${formatRatio(analysis.debt_to_equity)}` : 'Awaiting report', note: financialResult ? 'Reported total debt' : 'Upload a report to populate' },

  ]

  const getEvidenceReference = (field: string) => {

    const item = financialResult?.evidence?.[field]

    if (!item || typeof item !== 'object') return null

    if (typeof item.page === 'number' || typeof item.filename === 'string' || typeof item.quote === 'string') {

      return {

        filename: typeof item.filename === 'string' ? item.filename : documentInfo?.filename,

        page: typeof item.page === 'number' ? item.page : null,

        quote: typeof item.quote === 'string' ? item.quote : null,

      }

    }

    return null

  }

  return (

    <div className="min-h-screen bg-[#090d18] text-slate-200">

      {mobileMenuOpen && (

        <button

          aria-label="Close navigation"

          className="fixed inset-0 z-30 bg-black/60 lg:hidden"

          onClick={() => setMobileMenuOpen(false)}

        />

      )}

      <aside

        className={`fixed inset-y-0 left-0 z-40 flex ${sidebarCollapsed ? 'w-[76px]' : 'w-[252px]'} flex-col border-r border-white/[0.07] bg-[#0d1220] transition-transform duration-200 ${

          mobileMenuOpen ? 'translate-x-0' : '-translate-x-full'

        } lg:translate-x-0`}

      >

        <div className={`flex h-[76px] items-center gap-3 border-b border-white/[0.07] ${sidebarCollapsed ? 'justify-center px-2' : 'px-6'}`}>

          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-lime-500/15 text-lime-400">

            <ChartNoAxesCombined size={23} />

          </div>

          {!sidebarCollapsed && <div>

            <p className="text-[15px] font-semibold tracking-tight text-white">FinOpera</p>

            <p className="mt-0.5 text-xs text-slate-500">RAG-Powered Due Diligence</p>

          </div>}

          <button

            className="ml-auto rounded-lg p-2 text-slate-500 hover:bg-white/5 lg:hidden"

            onClick={() => setMobileMenuOpen(false)}

            aria-label="Close menu"

          >

            <X size={18} />

          </button>

        </div>

        <div className="px-4 pt-7">

          {!sidebarCollapsed && <p className="mb-3 px-3 text-[10px] font-semibold uppercase tracking-[0.19em] text-slate-600">

            Workspace

          </p>}

          <nav className="space-y-1">

            {navigation.map(({ label, page: pageKey, icon: Icon }) => {

              const active = activePage === pageKey

              return (

                <button

                  key={label}

                  onClick={() => {

                    setActivePage(pageKey)

                    setMobileMenuOpen(false)

                  }}

                  className={`flex w-full items-center gap-3 rounded-xl px-3 py-3 text-left text-[13px] transition ${

                    active

                      ? 'bg-lime-500/15 font-medium text-lime-300'

                      : 'text-slate-400 hover:bg-white/[0.04] hover:text-slate-200'

                  }`}

                >

                  <Icon size={18} strokeWidth={1.8} />

                  {!sidebarCollapsed && <span>{label}</span>}

                  {label === 'AI Analyst' && (

                    <span className="ml-auto rounded-md bg-lime-500/20 px-2 py-0.5 text-[10px] text-lime-300">

                      AI

                    </span>

                  )}

                </button>

              )

            })}

          </nav>

        </div>

        <div className="mt-auto p-4">

          <button

            onClick={() => {

              setActivePage('Settings')

              setMobileMenuOpen(false)

            }}

            className={`flex w-full items-center gap-3 rounded-xl px-3 py-3 text-sm transition ${

              activePage === 'Settings'

                ? 'bg-lime-500/15 font-medium text-lime-300'

                : 'text-slate-400 hover:bg-white/5 hover:text-white'

            }`}

          >

            <CircleHelp size={18} />

            Settings

          </button>

          <div className="mt-2 border-t border-white/[0.07] pt-4">

            <p className="text-xs text-slate-600">FinOpera</p>

            <p className="mt-1 text-[10px] text-slate-700">RAG-Powered Due Diligence Platform · v1.0</p>

          </div>

        </div>

      </aside>

      <div className={`min-h-screen ${sidebarCollapsed ? 'lg:pl-[76px]' : 'lg:pl-[252px]'}`}>

        <header className="sticky top-0 z-20 flex h-[76px] items-center justify-between border-b border-white/[0.07] bg-[#090d18]/90 px-5 backdrop-blur-xl sm:px-8">

          <div className="flex min-w-0 items-center gap-3">

            <button

              className="rounded-lg p-2 text-slate-400 hover:bg-white/5"

              onClick={() => { if (window.innerWidth < 1024) setMobileMenuOpen(true); else setSidebarCollapsed(value => !value) }}

              aria-label="Open menu"

            >

              <Menu size={21} />

            </button>

            <div className="min-w-0">

              <div className="flex items-center gap-2 text-xs text-slate-500">

                Workspace <span>/</span>

                <span className="text-slate-300">{activePage}</span>

              </div>

              <h1 className="mt-1 truncate text-base font-semibold text-white">

                {activePage}

              </h1>

            </div>

          </div>

          <div className="min-w-[1px] flex-1" aria-hidden="true" />

        </header>

        <main className="mx-auto max-w-[1600px] space-y-7 px-5 py-7 sm:px-8 sm:py-9">

          {activePage === 'Settings' ? (

            <section className="mx-auto w-full max-w-5xl space-y-7">

              <div>

                <div className="mb-3 flex items-center gap-2">

                  <CircleHelp size={17} className="text-lime-300" />

                  <span className="text-[11px] font-medium uppercase tracking-[0.16em] text-lime-300">Product guide</span>

                </div>

                <h2 className="text-2xl font-semibold tracking-tight text-white sm:text-3xl">Settings</h2>

                <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-400">

                  Learn how FinOpera turns financial reports into searchable evidence, extracted financial metrics, and a first-pass risk review.

                </p>

              </div>

              <div className="grid gap-4 sm:grid-cols-2">

                <article className="rounded-2xl border border-white/[0.07] bg-[#0e1422] p-5 sm:p-6">

                  <div className="flex items-center gap-3">

                    <div className="rounded-xl bg-lime-400/10 p-2.5 text-lime-300"><Upload size={19} /></div>

                    <h3 className="text-sm font-semibold text-white">1. Upload a report</h3>

                  </div>

                  <p className="mt-3 text-sm leading-6 text-slate-400">From Overview, choose <strong className="font-medium text-slate-200">Analyze a report</strong> and upload a PDF financial report. FinOpera accepts PDFs up to 20 MB. Use a report with selectable text for the best results; scanned PDFs may not extract reliably.</p>

                </article>

                <article className="rounded-2xl border border-white/[0.07] bg-[#0e1422] p-5 sm:p-6">

                  <div className="flex items-center gap-3">

                    <div className="rounded-xl bg-cyan-400/10 p-2.5 text-cyan-300"><FileText size={19} /></div>

                    <h3 className="text-sm font-semibold text-white">2. Read and index the PDF</h3>

                  </div>

                  <p className="mt-3 text-sm leading-6 text-slate-400">The backend extracts text from the PDF, splits it into smaller passages, and indexes those passages for retrieval. Page and source metadata are retained so relevant passages can be shown with answers.</p>

                </article>

                <article className="rounded-2xl border border-white/[0.07] bg-[#0e1422] p-5 sm:p-6">

                  <div className="flex items-center gap-3">

                    <div className="rounded-xl bg-emerald-400/10 p-2.5 text-emerald-300"><Landmark size={19} /></div>

                    <h3 className="text-sm font-semibold text-white">3. Extract financials</h3>

                  </div>

                  <p className="mt-3 text-sm leading-6 text-slate-400">The financial extraction service uses the report to identify available figures such as revenue, profit, assets, liabilities, debt, equity, and cash flow. FinOpera then calculates supported metrics, including margins, liquidity and leverage ratios, and free cash flow where the required inputs are available.</p>

                </article>

                <article className="rounded-2xl border border-white/[0.07] bg-[#0e1422] p-5 sm:p-6">

                  <div className="flex items-center gap-3">

                    <div className="rounded-xl bg-lime-400/10 p-2.5 text-lime-300"><Bot size={19} /></div>

                    <h3 className="text-sm font-semibold text-white">4. Ask questions with evidence</h3>

                  </div>

                  <p className="mt-3 text-sm leading-6 text-slate-400">Open Ask AI and ask a question about the report. FinOpera retrieves relevant indexed passages and uses them to prepare an answer. Check the source references and page numbers rather than relying on an answer alone.</p>

                </article>

                <article className="rounded-2xl border border-white/[0.07] bg-[#0e1422] p-5 sm:p-6">

                  <div className="flex items-center gap-3">

                    <div className="rounded-xl bg-amber-400/10 p-2.5 text-amber-300"><ShieldAlert size={19} /></div>

                    <h3 className="text-sm font-semibold text-white">5. Review risk indicators</h3>

                  </div>

                  <p className="mt-3 text-sm leading-6 text-slate-400">Risk assessment presents a preliminary screen of available liquidity, leverage, and cash-generation measures. Indicators are heuristic signals to help direct review—not a validated credit rating or a prediction of default.</p>

                </article>

                <article className="rounded-2xl border border-white/[0.07] bg-[#0e1422] p-5 sm:p-6">

                  <div className="flex items-center gap-3">

                    <div className="rounded-xl bg-rose-400/10 p-2.5 text-rose-300"><Info size={19} /></div>

                    <h3 className="text-sm font-semibold text-white">6. Verify before deciding</h3>

                  </div>

                  <p className="mt-3 text-sm leading-6 text-slate-400">AI extraction can misread tables, units, periods, or context. Confirm material figures in the original report, check whether amounts are in thousands or millions, and ensure ratios compare compatible reporting periods before using the analysis.</p>

                </article>

              </div>

              <article className="rounded-2xl border border-amber-400/20 bg-amber-400/[0.045] p-5 sm:p-6">

                <div className="flex items-center gap-3">

                  <AlertTriangle size={19} className="shrink-0 text-amber-300" />

                  <h3 className="text-sm font-semibold text-white">Important limitations</h3>

                </div>

                <ul className="mt-3 list-disc space-y-2 pl-5 text-sm leading-6 text-slate-400">

                  <li>FinOpera supports analysis; it does not replace professional accounting, legal, credit, or investment advice.</li>

                  <li>Results depend on the quality and completeness of the uploaded report and the extraction model.</li>

                  <li>Risk thresholds are screening rules, not a validated risk-prediction model.</li>

                  <li>Do not treat missing values as zero or assume a single reporting period shows a trend.</li>

                </ul>

              </article>

              <div className="flex flex-wrap gap-3">

                <button onClick={() => setActivePage('Overview')} className="inline-flex items-center gap-2 rounded-xl bg-lime-500 px-4 py-3 text-sm font-medium text-white transition hover:bg-lime-400"><Upload size={16} /> Analyze a report</button>

                <button onClick={() => setActivePage('Ask AI')} className="inline-flex items-center gap-2 rounded-xl border border-white/[0.1] px-4 py-3 text-sm font-medium text-slate-200 transition hover:bg-white/[0.04]"><Bot size={16} /> Open Ask AI</button>

              </div>

            </section>

          ) : activePage === 'Ask AI' ? (

            <section className="mx-auto flex w-full max-w-5xl flex-col gap-5">

              <div>

                <div className="mb-3 flex items-center gap-2">

                  <span className="h-2 w-2 rounded-full bg-emerald-400" />

                  <span className="text-[11px] font-medium uppercase tracking-[0.16em] text-emerald-400">Report-grounded assistant</span>

                </div>

                <h2 className="text-2xl font-semibold tracking-tight text-white sm:text-3xl">Ask your financial report</h2>

                <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-400">Ask questions about the indexed documents. Answers include retrieved source passages so you can verify the details in context.</p>

              </div>

              <div className="overflow-hidden rounded-2xl border border-white/[0.07] bg-[#0e1422]">

                <div className="flex items-center gap-3 border-b border-white/[0.07] px-5 py-4">

                  <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-lime-400/10 text-lime-300"><Bot size={20} /></div>

                  <div className="min-w-0 flex-1">

                    <p className="text-sm font-semibold text-white">FinOpera AI</p>

                    <p className="mt-1 text-xs text-slate-500">Retrieval-augmented answers · Gemini</p>

                  </div>

                  <span className="rounded-lg border border-emerald-400/15 bg-emerald-400/[0.06] px-2.5 py-1 text-[10px] text-emerald-300">Source-aware</span>

                </div>

                <div className="min-h-[340px] space-y-5 p-4 sm:p-6">

                  {chatMessages.length === 0 ? (

                    <div className="mx-auto flex max-w-xl flex-col items-center py-8 text-center">

                      <div className="flex h-14 w-14 items-center justify-center rounded-2xl border border-lime-400/20 bg-lime-400/10 text-lime-300"><FileSearch size={25} /></div>

                      <h3 className="mt-4 text-base font-semibold text-white">What would you like to know?</h3>

                      <p className="mt-2 text-sm leading-6 text-slate-400">Ask about financial performance, debt, liquidity, or details from a particular page.</p>

                      <div className="mt-6 grid w-full gap-2 text-left sm:grid-cols-1">

                        {SUGGESTED_QUESTIONS.map((question) => (

                          <button key={question} type="button" disabled={isAsking} onClick={() => askQuestion(question)} className="flex items-center justify-between gap-3 rounded-xl border border-white/[0.07] bg-white/[0.015] px-4 py-3 text-left text-xs text-slate-300 transition hover:border-lime-400/30 hover:bg-lime-400/[0.04] disabled:opacity-50">

                            <span>{question}</span><ExternalLink size={14} className="shrink-0 text-slate-600" />

                          </button>

                        ))}

                      </div>

                    </div>

                  ) : (

                    chatMessages.map((message) => (

                      <div key={message.id} className={`flex items-start gap-3 ${message.role === 'user' ? 'flex-row-reverse' : ''}`}>

                        <div className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-xl ${message.role === 'assistant' ? 'bg-lime-400/10 text-lime-300' : 'bg-lime-500/15 text-lime-300'}`}>

                          {message.role === 'assistant' ? <Bot size={18} /> : <UserRound size={18} />}

                        </div>

                        <div className={`min-w-0 max-w-[88%] rounded-2xl border px-4 py-3 ${message.role === 'assistant' ? 'border-white/[0.07] bg-white/[0.025]' : 'border-lime-400/20 bg-lime-400/[0.07]'}`}>

                          <p className="whitespace-pre-wrap break-words text-sm leading-6 text-slate-200">{message.content}</p>

                          {message.role === 'assistant' && message.sources && message.sources.length > 0 && (

                            <div className="mt-4 border-t border-white/[0.07] pt-3">

                              <p className="mb-2 flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500"><FileText size={13} /> Retrieved sources</p>

                              <div className="space-y-2">

                                {message.sources.map((source, index) => (

                                  <div key={`${source.source_id ?? 'source'}-${source.source ?? 'file'}-${source.page ?? index}-${index}`} className="rounded-lg border border-white/[0.06] bg-black/10 px-3 py-2.5">

                                    <div className="flex flex-wrap items-center gap-2 text-[11px]">

                                      <span className="rounded bg-lime-400/10 px-1.5 py-0.5 font-medium text-lime-300">{source.source_id || `S${index + 1}`}</span>

                                      <span className="break-all text-slate-300">{source.source || 'Source document'}</span>

                                      {typeof source.page === 'number' && <span className="text-slate-500">· Page {source.page}</span>}

                                    </div>

                                    {source.text && <p className="mt-2 line-clamp-4 whitespace-pre-wrap text-[11px] leading-5 text-slate-500">{source.text}</p>}

                                  </div>

                                ))}

                              </div>

                            </div>

                          )}

                        </div>

                      </div>

                    ))

                  )}

                  {isAsking && (

                    <div className="flex items-start gap-3">

                      <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-lime-400/10 text-lime-300"><Bot size={18} /></div>

                      <div className="flex items-center gap-2 rounded-2xl border border-white/[0.07] bg-white/[0.025] px-4 py-3 text-sm text-slate-400"><LoaderCircle size={16} className="animate-spin text-lime-300" /> Searching sources and preparing an answer…</div>

                    </div>

                  )}

                  <div ref={chatEndRef} />

                </div>

                {chatError && <div role="alert" className="mx-4 mb-3 rounded-xl border border-rose-400/20 bg-rose-400/[0.06] px-4 py-3 text-xs leading-5 text-rose-300 sm:mx-6">{chatError}</div>}

                <form onSubmit={(event) => { event.preventDefault(); void askQuestion() }} className="border-t border-white/[0.07] bg-black/10 p-4 sm:p-5">

                  <label htmlFor="ask-ai-input" className="mb-2 block text-[11px] font-medium text-slate-500">Your question</label>

                  <div className="flex items-end gap-2 rounded-xl border border-white/[0.09] bg-[#090d18] p-2 focus-within:border-lime-400/50">

                    <textarea id="ask-ai-input" rows={2} value={chatInput} onChange={(event) => setChatInput(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); void askQuestion() } }} placeholder="Ask about revenue, debt, liquidity, or a source page…" className="max-h-36 min-h-[44px] flex-1 resize-y bg-transparent px-2 py-2 text-sm text-slate-200 outline-none placeholder:text-slate-600" disabled={isAsking} />

                    <button type="submit" disabled={!chatInput.trim() || isAsking} className="inline-flex h-10 shrink-0 items-center gap-2 rounded-lg bg-lime-500 px-3 text-xs font-medium text-white transition hover:bg-lime-400 disabled:cursor-not-allowed disabled:opacity-40"><Send size={15} /> Ask</button>

                  </div>

                  <div className="mt-3 flex flex-wrap items-center justify-between gap-2">

                    <p className="text-[10px] leading-4 text-slate-600">Answers may be incomplete. Verify important figures against the cited report passages.</p>

                    {chatMessages.length > 0 && <button type="button" onClick={() => { setChatMessages([]); setChatError('') }} className="text-[11px] text-slate-500 transition hover:text-slate-300">Clear conversation</button>}

                  </div>

                </form>

              </div>

            </section>

          ) : (

            <>

          <section className="flex flex-col justify-between gap-5 md:flex-row md:items-end">

            <div>

              <h2 className="text-2xl font-semibold tracking-tight text-white sm:text-3xl">

                Financial overview

              </h2>

              <p className="mt-2 max-w-xl text-sm leading-6 text-slate-400">

                FinOpera transforms financial reports into evidence-backed insights. Analyze financial performance, investigate potential risks, and ask questions grounded in source documents.

              </p>

            </div>

            <button

              onClick={() => fileInputRef.current?.click()}

              className="inline-flex items-center justify-center gap-2 self-start rounded-xl bg-lime-500 px-4 py-3 text-sm font-medium text-white shadow-lg shadow-lime-950/30 transition hover:bg-lime-400 md:self-auto"

            >

              <Upload size={16} />

              Analyze a report

            </button>

          </section>

          <section className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">

            {metricCards.map(({ label, value, change, positive, icon: Icon, note }) => (

              <article

                key={label}

                className="rounded-2xl border border-white/[0.07] bg-[#0e1422] p-5 transition hover:border-white/[0.13]"

              >

                <div className="flex items-center justify-between gap-3">

                  <p className="text-[13px] text-slate-400">{label}</p>

                  <div className="rounded-xl border border-white/[0.06] bg-white/[0.035] p-2 text-slate-400">

                    <Icon size={17} />

                  </div>

                </div>

                <p className="mt-5 text-[28px] font-semibold tracking-tight text-white">

                  {value}

                </p>

                <div className="mt-3 flex flex-wrap items-center gap-2">

                  <span

                    className={`inline-flex items-center gap-1 rounded-md px-2 py-1 text-[11px] font-medium ${

                      positive

                        ? 'bg-emerald-400/10 text-emerald-400'

                        : 'bg-rose-400/10 text-rose-400'

                    }`}

                  >

                    {positive ? (

                      <ArrowUpRight size={12} />

                    ) : (

                      <ArrowDownRight size={12} />

                    )}

                    {change}

                  </span>

                  <span className="text-[11px] text-slate-600">{note}</span>

                </div>

              </article>

            ))}

          </section>

          <section className="grid grid-cols-1 gap-5 xl:grid-cols-[minmax(0,1.25fr)_minmax(300px,0.75fr)]">

            <article className="rounded-2xl border border-white/[0.07] bg-[#0e1422] p-5 sm:p-6">

              <div className="flex items-center justify-between gap-3">

                <div>

                  <h3 className="text-sm font-semibold text-white">

                    Report workspace

                  </h3>

                  <p className="mt-1.5 text-xs text-slate-500">

                    Upload a financial report to get started

                  </p>

                </div>

                <FileText size={19} className="text-slate-500" />

              </div>

              <input

                ref={fileInputRef}

                type="file"

                accept=".pdf,application/pdf"

                className="hidden"

                onChange={(event) => selectFile(event.target.files?.[0])}

              />

              <button

                type="button"

                onClick={() => fileInputRef.current?.click()}

                onDragOver={(event) => {

                  event.preventDefault()

                  setDragging(true)

                }}

                onDragLeave={() => setDragging(false)}

                onDrop={(event) => {

                  event.preventDefault()

                  setDragging(false)

                  selectFile(event.dataTransfer.files?.[0])

                }}

                className={`mt-5 flex min-h-[180px] w-full flex-col items-center justify-center rounded-xl border border-dashed px-4 py-6 text-center transition ${

                  dragging

                    ? 'border-lime-400 bg-lime-400/10'

                    : 'border-slate-700 bg-white/[0.015] hover:border-lime-400/60 hover:bg-lime-400/[0.03]'

                }`}

              >

                <div className="mb-3 flex h-11 w-11 items-center justify-center rounded-xl border border-lime-400/20 bg-lime-400/10 text-lime-300">

                  <Upload size={20} />

                </div>

                <p className="text-sm font-medium text-slate-200">

                  {selectedFile ? selectedFile.name : 'Drop your PDF here'}

                </p>

                <p className="mt-2 text-xs text-slate-500">

                  {selectedFile

                    ? `${(selectedFile.size / 1024 / 1024).toFixed(2)} MB · Ready to select`

                    : 'or click to browse your files'}

                </p>

                <span className="mt-3 rounded-lg border border-white/[0.08] px-3 py-1.5 text-[11px] text-slate-400">

                  PDF files only

                </span>

              </button>

              <div className="mt-4 flex flex-wrap items-center justify-between gap-3">

                <p className="text-[11px] text-slate-600">

                  Financial statements · Annual reports · 10-K / 10-Q

                </p>

                <button

                  onClick={handleUpload}

                  disabled={!selectedFile || isAnalyzing}

                  className="rounded-xl bg-lime-500 px-4 py-2.5 text-xs font-medium text-white transition hover:bg-lime-400 disabled:cursor-not-allowed disabled:opacity-40"

                >

                  {isAnalyzing ? 'Analyzing…' : 'Analyze report'}

                </button>

              </div>

              {(isAnalyzing || uploadMessage || uploadError) && (

                <div role={uploadError ? 'alert' : 'status'} className={`mt-4 rounded-xl border px-4 py-3 text-xs leading-5 ${uploadError ? 'border-rose-400/20 bg-rose-400/[0.06] text-rose-300' : 'border-lime-400/20 bg-lime-400/[0.06] text-lime-200'}`}>

                  {isAnalyzing && <span className="mr-2 inline-block h-2 w-2 animate-pulse rounded-full bg-lime-300" />}

                  {uploadError || uploadMessage}

                </div>

              )}

              {documentInfo && financialResult && (

                <div className="mt-4 rounded-xl border border-emerald-400/15 bg-emerald-400/[0.04] p-4">

                  <p className="text-xs font-medium text-emerald-300">Analyzed report</p>

                  <p className="mt-1 break-all text-sm text-slate-200">{documentInfo.filename}</p>

                  <p className="mt-1 text-[11px] text-slate-500">{documentInfo.total_pages} pages · {documentInfo.indexed_count} indexed chunks{financialResult.reporting_period ? ` · ${financialResult.reporting_period}` : ''}</p>

                  <p className="mt-2 text-[11px] text-slate-500">Values shown in {units}; verify extracted figures against the source evidence.</p>

                </div>

              )}

            </article>

            <article className="rounded-2xl border border-white/[0.07] bg-[#0e1422] p-5 sm:p-6">

              <div className="flex items-center justify-between">

                <div>

                  <h3 className="text-sm font-semibold text-white">

                    AI assistant

                  </h3>

                  <p className="mt-1.5 text-xs text-slate-500">

                    Ask questions grounded in your reports

                  </p>

                </div>

                <div className="rounded-xl bg-lime-400/10 p-2 text-lime-300">

                  <FileSearch size={18} />

                </div>

              </div>

              <div className="mt-5 rounded-xl border border-white/[0.06] bg-white/[0.02] p-4">

                <p className="text-xs leading-6 text-slate-400">

                  Upload a report to explore financial figures, understand key

                  ratios, and ask questions with traceable source references.

                </p>

              </div>

              <p className="mb-3 mt-5 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-600">

                Example questions

              </p>

              <div className="space-y-2">

                {[

                  'How has revenue changed?',

                  'What are the main financial risks?',

                  'Show the evidence for total debt',

                ].map((question) => (

                  <button

                    key={question}

                    onClick={() => setActivePage('Ask AI')}

                    className="flex w-full items-center justify-between gap-3 rounded-xl border border-white/[0.06] px-3 py-3 text-left text-xs text-slate-400 transition hover:border-lime-400/30 hover:bg-lime-400/[0.04] hover:text-slate-200"

                  >

                    <span>{question}</span>

                    <ArrowUpRight size={14} className="shrink-0 text-slate-600" />

                  </button>

                ))}

              </div>

            </article>

          </section>

          <footer className="flex flex-col justify-between gap-2 border-t border-white/[0.06] pt-5 text-[10px] text-slate-600 sm:flex-row">

            <p>FinOpera · RAG-Powered Due Diligence Platform</p>

            <p>{financialResult ? 'Live analysis · Verify figures against source evidence' : 'Preview interface · Financial figures are illustrative until a report is analyzed'}</p>

          </footer>

            </>

          )}

        </main>

      </div>

    </div>

  )

}

export default App
