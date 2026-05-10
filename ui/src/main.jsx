import React, { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  Activity,
  BarChart3,
  Brain,
  CheckCircle2,
  ChevronRight,
  Cpu,
  Download,
  FileText,
  GaugeCircle,
  Lightbulb,
  Loader2,
  Play,
  RefreshCcw,
  Send,
  Sparkles,
  Trophy,
  Upload,
  XCircle,
  Zap,
} from 'lucide-react';
import './index.css';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const emptyOptions = { A: '', B: '', C: '', D: '' };
const OPTION_LABELS = ['A', 'B', 'C', 'D'];

async function api(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch {
    }
    throw new Error(detail);
  }
  return response.json();
}

function StatusPill({ status }) {
  if (!status) {
    return (
      <span className="pill border-white/20 bg-white/10 text-white/80">
        <Loader2 className="h-3.5 w-3.5 animate-spin" /> Checking backend
      </span>
    );
  }
  if (status.status === 'offline') {
    return (
      <span className="pill border-rose-300/40 bg-rose-500/10 text-rose-100">
        <XCircle className="h-3.5 w-3.5" /> Backend offline
      </span>
    );
  }
  const ok = status.model_a_loaded && status.model_b_loaded;
  return (
    <span
      className={`pill border-white/20 ${
        ok ? 'bg-emerald-500/15 text-emerald-100' : 'bg-amber-500/15 text-amber-100'
      }`}
      title="Model A: option verifier · Model B: question + distractor generator · W2V: semantic blend"
    >
      <Cpu className="h-3.5 w-3.5" />
      <span className="font-semibold">A</span>
      <span className={status.model_a_loaded ? 'text-emerald-200' : 'text-rose-200'}>
        {status.model_a_loaded ? 'on' : 'off'}
      </span>
      <span className="opacity-60">·</span>
      <span className="font-semibold">B</span>
      <span className={status.model_b_loaded ? 'text-emerald-200' : 'text-rose-200'}>
        {status.model_b_loaded ? 'on' : 'off'}
      </span>
      <span className="opacity-60">·</span>
      <span className="font-semibold">W2V</span>
      <span className={status.model_b_word2vec_loaded ? 'text-emerald-200' : 'text-ink-300'}>
        {status.model_b_word2vec_loaded ? 'on' : 'off'}
      </span>
    </span>
  );
}

function Header({ tab, setTab, status }) {
  const tabs = [
    { id: 'quiz', label: 'Quiz', icon: Sparkles },
    { id: 'analytics', label: 'Analytics', icon: BarChart3 },
  ];
  return (
    <header className="sticky top-0 z-20 border-b border-white/5 bg-gradient-to-br from-ink-900 via-brand-900 to-ink-900">
      <div className="mx-auto flex max-w-7xl flex-col gap-4 px-4 py-4 md:flex-row md:items-center md:justify-between md:px-8">
        <div className="flex items-center gap-3">
          <div className="grid h-11 w-11 place-items-center rounded-2xl bg-gradient-to-br from-brand-400 to-brand-700 text-white shadow-lg shadow-brand-900/40">
            <Brain className="h-6 w-6" />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight text-white md:text-xl">RACE Quiz Generator</h1>
            <p className="text-xs text-ink-300">TF-IDF + Word2Vec reading-comprehension workflow</p>
          </div>
        </div>
        <nav className="flex items-center gap-1 rounded-2xl border border-white/10 bg-white/5 p-1 backdrop-blur">
          {tabs.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setTab(id)}
              className={`tab-btn ${tab === id ? 'tab-btn-active' : ''}`}
            >
              <Icon className="h-4 w-4" />
              {label}
            </button>
          ))}
        </nav>
        <StatusPill status={status} />
      </div>
    </header>
  );
}

const QUESTION_COUNT_OPTIONS = [3, 5, 7, 10];

function ArticlePanel({
  article,
  setArticle,
  question,
  setQuestion,
  options,
  setOptions,
  questionCount,
  setQuestionCount,
  loading,
  onLoadSample,
  onUpload,
  onGenerate,
}) {
  const charCount = article.length;
  const filledOptions = Object.values(options).filter(Boolean).length;
  return (
    <section className="surface flex h-full flex-col p-6">
      <div className="mb-4 flex items-center justify-between gap-3">
        <div>
          <h2 className="flex items-center gap-2 text-base font-semibold text-ink-900">
            <FileText className="h-4 w-4 text-brand-600" /> Article & setup
          </h2>
          <p className="text-xs text-ink-500">Paste a passage or load a RACE sample to begin.</p>
        </div>
        <button
          className="btn-ghost h-10 w-10 p-0"
          onClick={onLoadSample}
          disabled={loading === 'sample'}
          title="Load random RACE sample"
        >
          {loading === 'sample' ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <RefreshCcw className="h-4 w-4" />
          )}
        </button>
      </div>

      <div className="relative flex-1">
        <textarea
          className="field h-full min-h-[18rem] resize-none leading-6"
          value={article}
          onChange={(event) => setArticle(event.target.value)}
          placeholder="Paste a reading passage here…"
        />
        <div className="pointer-events-none absolute bottom-2 right-3 text-[11px] text-ink-400">
          {charCount.toLocaleString()} chars
        </div>
      </div>

      <label className="mt-3 flex cursor-pointer items-center justify-center gap-2 rounded-xl border border-dashed border-ink-300 bg-ink-50/60 px-4 py-2.5 text-sm font-medium text-ink-600 transition hover:border-brand-400 hover:text-brand-700">
        <Upload className="h-4 w-4" />
        Upload .txt article
        <input className="sr-only" type="file" accept=".txt,text/plain" onChange={onUpload} />
      </label>

      <div className="mt-4">
        <div className="mb-2 flex items-center justify-between">
          <label className="text-xs font-semibold uppercase tracking-wide text-ink-600">
            Number of questions
          </label>
        </div>
        <div className="grid grid-cols-4 gap-2">
          {QUESTION_COUNT_OPTIONS.map((count) => {
            const active = questionCount === count;
            return (
              <button
                key={count}
                onClick={() => setQuestionCount(count)}
                className={`rounded-xl border px-3 py-2 text-sm font-semibold transition ${
                  active
                    ? 'border-brand-500 bg-brand-50 text-brand-800 ring-2 ring-brand-200'
                    : 'border-ink-200 bg-white text-ink-700 hover:border-brand-300 hover:bg-brand-50/40'
                }`}
                type="button"
              >
                {count}
              </button>
            );
          })}
        </div>
        <div className="mt-2 flex items-center gap-2">
          <input
            type="range"
            min={1}
            max={10}
            step={1}
            value={questionCount}
            onChange={(event) => setQuestionCount(Number(event.target.value))}
            className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-ink-200 accent-brand-600"
          />
          <span className="min-w-[2.5rem] text-right text-sm font-semibold text-brand-700">
            {questionCount}
          </span>
        </div>
      </div>

      <details className="mt-4 rounded-xl border border-ink-200 bg-ink-50/60 p-4 text-sm">
        <summary className="cursor-pointer select-none font-semibold text-ink-700">
          Optional: anchor first question
          <span className="ml-2 text-xs font-normal text-ink-500">
            {filledOptions === 4 && question ? 'Will use your typed MCQ as Q1' : 'Skip to let AI generate all'}
          </span>
        </summary>
        <div className="mt-3 grid gap-3">
          <input
            className="field"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Question stem (optional)"
          />
          <div className="grid gap-2 sm:grid-cols-2">
            {OPTION_LABELS.map((label) => (
              <label key={label} className="grid grid-cols-[28px_minmax(0,1fr)] items-center gap-2">
                <span className="grid h-7 w-7 place-items-center rounded-lg bg-ink-900 text-xs font-bold text-white">
                  {label}
                </span>
                <input
                  className="field"
                  value={options[label]}
                  onChange={(event) => setOptions({ ...options, [label]: event.target.value })}
                  placeholder={`Option ${label}`}
                />
              </label>
            ))}
          </div>
        </div>
      </details>

      <button
        className="btn-primary mt-5 h-11 w-full"
        onClick={onGenerate}
        disabled={loading === 'generate' || article.length < 20}
      >
        {loading === 'generate' ? (
          <>
            <Loader2 className="h-4 w-4 animate-spin" /> Generating…
          </>
        ) : (
          <>
            <Send className="h-4 w-4" /> Generate {questionCount} Question{questionCount === 1 ? '' : 's'}
          </>
        )}
      </button>
    </section>
  );
}

function QuestionPanel({
  generatedQuestions,
  activeIndex,
  setActiveIndex,
  selectedByQuestion,
  verificationByQuestion,
  setSelectedByQuestion,
  onVerify,
  loading,
}) {
  const activeQuiz = generatedQuestions[activeIndex] || null;
  const selected = selectedByQuestion[activeIndex] || '';
  const verification = verificationByQuestion[activeIndex] || null;
  const correctLabel = verification?.predicted_option || activeQuiz?.predicted_correct_option;
  const showResult = !!verification;

  return (
    <section className="surface flex h-full flex-col p-6">
      <div className="mb-4 flex items-center justify-between gap-3">
        <div>
          <h2 className="flex items-center gap-2 text-base font-semibold text-ink-900">
            <Sparkles className="h-4 w-4 text-brand-600" /> Question &amp; answer
          </h2>
          <p className="text-xs text-ink-500">
            {generatedQuestions.length
              ? `Question ${activeIndex + 1} of ${generatedQuestions.length}`
              : 'Generated multiple-choice items will appear here.'}
          </p>
        </div>
        {generatedQuestions.length > 0 && (
          <span className="pill-info">
            <Sparkles className="h-3.5 w-3.5" /> AI-generated
          </span>
        )}
      </div>

      {generatedQuestions.length > 0 && (
        <div className="mb-5 flex flex-wrap gap-2">
          {generatedQuestions.map((item, index) => {
            const sel = selectedByQuestion[index];
            const ver = verificationByQuestion[index];
            const answered = !!ver;
            const correct = ver?.is_correct;
            const isActive = activeIndex === index;
            return (
              <button
                key={`${item.question}-${index}`}
                className={`progress-dot px-3 ${
                  isActive
                    ? 'border-brand-500 bg-brand-50 text-brand-800'
                    : answered
                    ? correct
                      ? 'border-emerald-200 bg-emerald-50 text-emerald-700'
                      : 'border-rose-200 bg-rose-50 text-rose-700'
                    : sel
                    ? 'border-ink-300 bg-white text-ink-700'
                    : 'border-ink-200 bg-white text-ink-500 hover:border-brand-300'
                }`}
                onClick={() => setActiveIndex(index)}
              >
                <span className="flex items-center gap-1.5">
                  Q{index + 1}
                  {answered &&
                    (correct ? (
                      <CheckCircle2 className="h-3.5 w-3.5" />
                    ) : (
                      <XCircle className="h-3.5 w-3.5" />
                    ))}
                </span>
              </button>
            );
          })}
        </div>
      )}

      <p className="mb-5 min-h-[3.5rem] text-lg font-medium leading-7 text-ink-900">
        {activeQuiz?.question || (
          <span className="text-ink-400">
            Generate questions to begin. The article on the left is all you need.
          </span>
        )}
      </p>

      <div className="grid gap-3">
        {OPTION_LABELS.map((label) => {
          const text = activeQuiz?.options?.[label] || '';
          const isSelected = selected === label;
          const isCorrect = showResult && correctLabel === label;
          const isWrong = showResult && isSelected && !verification.is_correct;
          let cls = 'option-card';
          let letterCls = 'letter';
          if (isCorrect) {
            cls += ' option-card-correct';
            letterCls += ' letter-correct';
          } else if (isWrong) {
            cls += ' option-card-incorrect';
            letterCls += ' letter-incorrect';
          } else if (isSelected) {
            cls += ' option-card-selected';
            letterCls += ' letter-selected';
          }
          return (
            <button
              key={label}
              className={cls}
              onClick={() =>
                !showResult && setSelectedByQuestion({ ...selectedByQuestion, [activeIndex]: label })
              }
              disabled={!text || showResult}
            >
              <span className={letterCls}>{label}</span>
              <span className="text-sm font-medium leading-6 text-ink-800">
                {text || <span className="text-ink-400">No option supplied</span>}
              </span>
              {isCorrect && (
                <CheckCircle2 className="absolute right-4 top-1/2 -translate-y-1/2 text-emerald-600" />
              )}
              {isWrong && (
                <XCircle className="absolute right-4 top-1/2 -translate-y-1/2 text-rose-600" />
              )}
            </button>
          );
        })}
      </div>

      <button
        className="btn-primary mt-5 h-11 w-full"
        onClick={onVerify}
        disabled={!selected || loading === 'verify' || !activeQuiz || showResult}
      >
        {loading === 'verify' ? (
          <>
            <Loader2 className="h-4 w-4 animate-spin" /> Checking…
          </>
        ) : (
          <>
            <Play className="h-4 w-4" /> Check Answer
          </>
        )}
      </button>

      {verification && (
        <div
          className={`mt-4 animate-fade-up rounded-2xl border p-4 ${
            verification.is_correct
              ? 'border-emerald-200 bg-emerald-50 text-emerald-900'
              : 'border-rose-200 bg-rose-50 text-rose-900'
          }`}
        >
          <div className="flex items-start gap-3">
            <div
              className={`grid h-9 w-9 shrink-0 place-items-center rounded-xl ${
                verification.is_correct ? 'bg-emerald-600 text-white' : 'bg-rose-600 text-white'
              }`}
            >
              {verification.is_correct ? (
                <CheckCircle2 className="h-5 w-5" />
              ) : (
                <XCircle className="h-5 w-5" />
              )}
            </div>
            <div className="flex-1">
              <strong className="block text-sm font-bold">
                {verification.is_correct ? 'Correct!' : 'Try again'}
              </strong>
              <p className="mt-1 text-sm leading-6">
                {verification.explanation} Confidence:{' '}
                <strong>{Math.round(verification.confidence * 100)}%</strong>.
              </p>
            </div>
            {generatedQuestions.length > activeIndex + 1 && (
              <button
                className="btn-ghost h-9"
                onClick={() => setActiveIndex(activeIndex + 1)}
                title="Next question"
              >
                Next <ChevronRight className="h-4 w-4" />
              </button>
            )}
          </div>
        </div>
      )}
    </section>
  );
}

function HintsPanel({ activeQuiz, revealed, setRevealed }) {
  if (!activeQuiz) {
    return (
      <section className="surface p-6">
        <div className="mb-1 flex items-center gap-2 text-base font-semibold text-ink-900">
          <Lightbulb className="h-4 w-4 text-amber-500" /> Hints
        </div>
        <p className="text-sm text-ink-500">Generate questions to reveal graduated hints.</p>
      </section>
    );
  }
  const hints = activeQuiz.hints || [];
  const allRevealed = revealed >= hints.length;
  return (
    <section className="surface p-6">
      <div className="mb-3 flex items-center justify-between gap-3">
        <h3 className="flex items-center gap-2 text-base font-semibold text-ink-900">
          <Lightbulb className="h-4 w-4 text-amber-500" /> Hints
        </h3>
        <span className="pill-neutral">
          {Math.min(revealed, hints.length)} / {hints.length}
        </span>
      </div>
      <div className="grid gap-2">
        {hints.map((hint, index) => {
          const open = revealed > index;
          return (
            <button
              key={`${hint}-${index}`}
              className={`grid grid-cols-[28px_minmax(0,1fr)] items-start gap-3 rounded-xl border p-3 text-left transition ${
                open
                  ? 'border-amber-200 bg-amber-50/60 text-ink-800'
                  : 'border-ink-200 bg-white hover:border-amber-300'
              }`}
              onClick={() => setRevealed(Math.max(revealed, index + 1))}
            >
              <span className="grid h-7 w-7 place-items-center rounded-lg bg-amber-500 text-xs font-bold text-white">
                {index + 1}
              </span>
              <span className="text-sm leading-6">
                {open ? hint : <span className="text-ink-500">Tap to reveal hint {index + 1}</span>}
              </span>
            </button>
          );
        })}
      </div>
    </section>
  );
}

function pct(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return `${Math.round(value * 100)}%`;
}

function num(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  if (typeof value !== 'number') return value;
  return value.toFixed(digits);
}

function AnalyticsView({ metrics, onRefresh, onExport, loading }) {
  const requests = metrics?.session_requests ?? 0;
  const avgLatency = metrics?.average_latency_ms ?? 0;
  const modelA = metrics?.model_a?.logistic_regression?.exact_match_answer_accuracy;
  const stacking = metrics?.model_a?.stacking_classifier?.exact_match_answer_accuracy;
  const distF1 = metrics?.model_b?.evaluation?.distractors?.f1;
  const hintR2 = metrics?.model_b?.hint_scorer?.validation_r2;
  const recent = (metrics?.last_requests || []).slice().reverse();

  return (
    <section className="grid gap-5">
      <div className="surface flex flex-col gap-4 p-6">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="flex items-center gap-2 text-base font-semibold text-ink-900">
              <BarChart3 className="h-4 w-4 text-brand-600" /> Session analytics
            </h2>
            <p className="text-xs text-ink-500">
              Live model accuracy, latency and recent request log.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button className="btn-ghost h-9" onClick={onRefresh} disabled={loading === 'metrics'}>
              {loading === 'metrics' ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <RefreshCcw className="h-4 w-4" />
              )}{' '}
              Refresh
            </button>
            <button className="btn-primary h-9" onClick={onExport}>
              <Download className="h-4 w-4" /> Export logs
            </button>
          </div>
        </div>

        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          <div className="kpi">
            <span className="kpi-label flex items-center gap-1.5">
              <Activity className="h-3.5 w-3.5 text-brand-500" /> Requests
            </span>
            <span className="kpi-value">{requests}</span>
            <span className="kpi-sub">since backend started</span>
          </div>
          <div className="kpi">
            <span className="kpi-label flex items-center gap-1.5">
              <Zap className="h-3.5 w-3.5 text-amber-500" /> Avg latency
            </span>
            <span className="kpi-value">{avgLatency} ms</span>
            <span className="kpi-sub">per generate / verify call</span>
          </div>
          <div className="kpi">
            <span className="kpi-label flex items-center gap-1.5">
              <GaugeCircle className="h-3.5 w-3.5 text-emerald-500" /> Model A accuracy
            </span>
            <span className="kpi-value">{pct(modelA)}</span>
            <span className="kpi-sub">logistic regression (RACE)</span>
          </div>
          <div className="kpi">
            <span className="kpi-label flex items-center gap-1.5">
              <Cpu className="h-3.5 w-3.5 text-brand-500" /> Stacking accuracy
            </span>
            <span className="kpi-value">{pct(stacking)}</span>
            <span className="kpi-sub">LR + SVM + NB + RF</span>
          </div>
          <div className="kpi">
            <span className="kpi-label flex items-center gap-1.5">
              <Sparkles className="h-3.5 w-3.5 text-brand-500" /> Distractor F1
            </span>
            <span className="kpi-value">{pct(distF1)}</span>
            <span className="kpi-sub">Model B candidate ranker</span>
          </div>
          <div className="kpi">
            <span className="kpi-label flex items-center gap-1.5">
              <Brain className="h-3.5 w-3.5 text-brand-500" /> Hint scorer R²
            </span>
            <span className="kpi-value">{num(hintR2)}</span>
            <span className="kpi-sub">Ridge regression</span>
          </div>
        </div>
      </div>

      <div className="surface p-6">
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-sm font-semibold text-ink-900">Recent activity</h3>
          <span className="pill-neutral">{recent.length} entries</span>
        </div>
        {recent.length === 0 ? (
          <p className="rounded-xl border border-dashed border-ink-200 bg-ink-50/40 p-6 text-center text-sm text-ink-500">
            No requests yet. Generate or verify a question to populate the log.
          </p>
        ) : (
          <div className="overflow-hidden rounded-xl border border-ink-200">
            <table className="w-full text-sm">
              <thead className="bg-ink-50 text-xs uppercase tracking-wide text-ink-500">
                <tr>
                  <th className="px-4 py-2.5 text-left">Endpoint</th>
                  <th className="px-4 py-2.5 text-right">Latency</th>
                  <th className="px-4 py-2.5 text-right">Result</th>
                </tr>
              </thead>
              <tbody>
                {recent.map((row, index) => {
                  const result =
                    row.predicted_option ||
                    (row.question_count !== undefined ? `${row.question_count} q` : '—');
                  const isVerify = row.endpoint === 'verify';
                  return (
                    <tr
                      key={`${row.endpoint}-${index}`}
                      className={`border-t border-ink-100 ${
                        index % 2 === 0 ? 'bg-white' : 'bg-ink-50/40'
                      }`}
                    >
                      <td className="px-4 py-2.5">
                        <span
                          className={
                            isVerify
                              ? 'pill border-emerald-200 bg-emerald-50 text-emerald-700'
                              : 'pill-info'
                          }
                        >
                          {row.endpoint}
                        </span>
                      </td>
                      <td className="px-4 py-2.5 text-right font-mono text-xs text-ink-600">
                        {row.latency_ms} ms
                      </td>
                      <td className="px-4 py-2.5 text-right font-semibold text-ink-700">
                        {result}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </section>
  );
}

function App() {
  const [tab, setTab] = useState('quiz');
  const [article, setArticle] = useState('');
  const [question, setQuestion] = useState('');
  const [options, setOptions] = useState(emptyOptions);
  const [questionCount, setQuestionCount] = useState(5);
  const [result, setResult] = useState(null);
  const [activeIndex, setActiveIndex] = useState(0);
  const [selectedByQuestion, setSelectedByQuestion] = useState({});
  const [verificationByQuestion, setVerificationByQuestion] = useState({});
  const [revealedHintsByQuestion, setRevealedHintsByQuestion] = useState({});
  const [metrics, setMetrics] = useState(null);
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState('');
  const [error, setError] = useState('');

  const generatedQuestions = result?.questions || [];
  const activeQuiz = generatedQuestions[activeIndex] || null;
  const revealedHints = revealedHintsByQuestion[activeIndex] || 0;

  useEffect(() => {
    refreshStatus();
    refreshMetrics();
  }, []);

  async function refreshStatus() {
    try {
      setStatus(await api('/health'));
    } catch (err) {
      setStatus({
        status: 'offline',
        model_a_loaded: false,
        model_b_loaded: false,
        error: err.message,
      });
    }
  }

  async function refreshMetrics() {
    setLoading((l) => (l === '' ? 'metrics' : l));
    try {
      setMetrics(await api('/metrics'));
    } catch {
      setMetrics(null);
    } finally {
      setLoading((l) => (l === 'metrics' ? '' : l));
    }
  }

  function resetQuizState() {
    setResult(null);
    setActiveIndex(0);
    setSelectedByQuestion({});
    setVerificationByQuestion({});
    setRevealedHintsByQuestion({});
  }

  async function loadSample() {
    setLoading('sample');
    setError('');
    try {
      const sample = await api('/sample');
      setArticle(sample.article);
      setQuestion(sample.question);
      setOptions(sample.options);
      resetQuizState();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading('');
    }
  }

  async function generateQuiz() {
    setLoading('generate');
    setError('');
    setSelectedByQuestion({});
    setVerificationByQuestion({});
    setRevealedHintsByQuestion({});
    setActiveIndex(0);
    try {
      const allFilled = OPTION_LABELS.every((label) => options[label]);
      const safeCount = Math.max(1, Math.min(10, Number(questionCount) || 5));
      const payload = {
        article,
        question: question || undefined,
        options: allFilled && question ? options : undefined,
        question_count: safeCount,
      };
      setResult(await api('/generate', { method: 'POST', body: JSON.stringify(payload) }));
      refreshMetrics();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading('');
    }
  }

  async function uploadArticle(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    setError('');
    try {
      const text = await file.text();
      setArticle(text);
      resetQuizState();
    } catch (err) {
      setError(`Could not read uploaded file: ${err.message}`);
    }
  }

  async function verifyAnswer() {
    const selected = selectedByQuestion[activeIndex];
    if (!selected || !activeQuiz) return;
    setLoading('verify');
    setError('');
    try {
      const response = await api('/verify', {
        method: 'POST',
        body: JSON.stringify({
          article,
          question: activeQuiz.question,
          options: activeQuiz.options,
          selected_option: selected,
          correct_option: activeQuiz?.predicted_correct_option,
        }),
      });
      setVerificationByQuestion({ ...verificationByQuestion, [activeIndex]: response });
      refreshMetrics();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading('');
    }
  }

  function exportLogs() {
    window.location.href = `${API_BASE}/logs/export`;
  }

  const stats = useMemo(() => {
    const total = generatedQuestions.length;
    const verifications = Object.values(verificationByQuestion);
    const correct = verifications.filter((v) => v?.is_correct).length;
    return { total, answered: verifications.length, correct };
  }, [generatedQuestions, verificationByQuestion]);

  return (
    <main className="min-h-screen pb-12">
      <Header tab={tab} setTab={setTab} status={status} />

      {error && (
        <div className="mx-auto mt-4 max-w-7xl px-4 md:px-8">
          <div className="flex items-start gap-2 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-900">
            <XCircle className="mt-0.5 h-4 w-4 shrink-0" />
            <span>{error}</span>
          </div>
        </div>
      )}

      <div className="mx-auto max-w-7xl px-4 py-6 md:px-8">
        {tab === 'quiz' ? (
          <div className="grid gap-5">
            <div className="grid items-stretch gap-5 lg:grid-cols-2">
              <ArticlePanel
                article={article}
                setArticle={setArticle}
                question={question}
                setQuestion={setQuestion}
                options={options}
                setOptions={setOptions}
                questionCount={questionCount}
                setQuestionCount={setQuestionCount}
                loading={loading}
                onLoadSample={loadSample}
                onUpload={uploadArticle}
                onGenerate={generateQuiz}
              />
              <QuestionPanel
                generatedQuestions={generatedQuestions}
                activeIndex={activeIndex}
                setActiveIndex={setActiveIndex}
                selectedByQuestion={selectedByQuestion}
                verificationByQuestion={verificationByQuestion}
                setSelectedByQuestion={setSelectedByQuestion}
                onVerify={verifyAnswer}
                loading={loading}
              />
            </div>

            {stats.total > 0 && (
              <div className="grid gap-3 sm:grid-cols-3">
                <div className="kpi">
                  <span className="kpi-label">Score</span>
                  <span className="kpi-value">
                    {stats.correct}/{stats.total}
                  </span>
                  <span className="kpi-sub">
                    {stats.total
                      ? `${Math.round((stats.correct / stats.total) * 100)}% correct`
                      : '—'}
                  </span>
                </div>
                <div className="kpi">
                  <span className="kpi-label">Answered</span>
                  <span className="kpi-value">{stats.answered}</span>
                  <span className="kpi-sub">of {stats.total}</span>
                </div>
                <div className="kpi">
                  <span className="kpi-label">Remaining</span>
                  <span className="kpi-value">{stats.total - stats.answered}</span>
                  <span className="kpi-sub">to verify</span>
                </div>
              </div>
            )}

            <HintsPanel
              activeQuiz={activeQuiz}
              revealed={revealedHints}
              setRevealed={(value) =>
                setRevealedHintsByQuestion({ ...revealedHintsByQuestion, [activeIndex]: value })
              }
            />
          </div>
        ) : (
          <AnalyticsView
            metrics={metrics}
            onRefresh={refreshMetrics}
            onExport={exportLogs}
            loading={loading}
          />
        )}
      </div>
    </main>
  );
}

createRoot(document.getElementById('root')).render(<App />);
