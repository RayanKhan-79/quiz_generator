import React, { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { BarChart3, CheckCircle2, Download, Lightbulb, Loader2, Play, RefreshCcw, Send, Upload, XCircle } from 'lucide-react';
import './index.css';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const emptyOptions = { A: '', B: '', C: '', D: '' };

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
      // keep status text
    }
    throw new Error(detail);
  }
  return response.json();
}

function App() {
  const [article, setArticle] = useState('');
  const [question, setQuestion] = useState('');
  const [options, setOptions] = useState(emptyOptions);
  const [result, setResult] = useState(null);
  const [activeIndex, setActiveIndex] = useState(0);
  const [selectedByQuestion, setSelectedByQuestion] = useState({});
  const [verificationByQuestion, setVerificationByQuestion] = useState({});
  const [metrics, setMetrics] = useState(null);
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState('');
  const [error, setError] = useState('');
  const [revealedHintsByQuestion, setRevealedHintsByQuestion] = useState({});

  const generatedQuestions = result?.questions || [];
  const activeQuiz = generatedQuestions[activeIndex] || null;
  const activeQuestion = activeQuiz?.question || question;
  const activeOptions = activeQuiz?.options || options;
  const selected = selectedByQuestion[activeIndex] || '';
  const verification = verificationByQuestion[activeIndex] || null;
  const revealedHints = revealedHintsByQuestion[activeIndex] || 0;
  const allHintsUsed = activeQuiz && revealedHints >= activeQuiz.hints.length;

  useEffect(() => {
    refreshStatus();
    refreshMetrics();
  }, []);

  async function refreshStatus() {
    try {
      setStatus(await api('/health'));
    } catch (err) {
      setStatus({ status: 'offline', model_a_loaded: false, model_b_loaded: false, error: err.message });
    }
  }

  async function refreshMetrics() {
    try {
      setMetrics(await api('/metrics'));
    } catch {
      setMetrics(null);
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
      const payload = {
        article,
        question: question || undefined,
        options: Object.values(options).some(Boolean) ? options : undefined,
        question_count: 5,
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
    if (!selected || !activeQuestion) return;
    setLoading('verify');
    setError('');
    try {
      const response = await api('/verify', {
        method: 'POST',
        body: JSON.stringify({
          article,
          question: activeQuestion,
          options: activeOptions,
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

  const modelState = useMemo(() => {
    if (!status) return 'Checking backend';
    if (status.status === 'offline') return 'Backend offline';
    return `Model A ${status.model_a_loaded ? 'loaded' : 'missing'} | Model B ${status.model_b_loaded ? 'loaded' : 'missing'}`;
  }, [status]);

  return (
    <main className="min-h-screen bg-slate-100 text-slate-900">
      <header className="flex flex-col gap-4 bg-slate-800 px-5 py-6 text-slate-50 md:flex-row md:items-center md:justify-between md:px-8">
        <div>
          <h1 className="text-3xl font-bold tracking-normal">RACE Quiz Generator</h1>
          <p className="mt-1 text-sm text-slate-300">AI-generated TF-IDF reading comprehension workflow</p>
        </div>
        <div className="w-fit rounded-full border border-slate-300 px-3 py-2 text-sm">{modelState}</div>
      </header>

      {error && <div className="mx-3 mt-4 rounded-md bg-orange-50 px-4 py-3 text-orange-900 md:mx-5">{error}</div>}

      <section className="grid gap-5 p-3 md:p-5 xl:grid-cols-[minmax(0,1fr)_360px]">
        <div className="grid gap-5 lg:grid-cols-[minmax(320px,0.9fr)_minmax(360px,1.1fr)]">
          <section className="rounded-lg border border-slate-300 bg-white p-5 shadow-sm">
            <div className="mb-4 flex items-center justify-between gap-3">
              <h2 className="text-lg font-semibold">Article Input</h2>
              <button
                className="inline-flex h-10 w-10 items-center justify-center rounded-md bg-slate-200 text-slate-800 disabled:cursor-not-allowed disabled:opacity-60"
                onClick={loadSample}
                disabled={loading === 'sample'}
                title="Load random RACE sample"
              >
                {loading === 'sample' ? <Loader2 className="h-5 w-5 animate-spin" /> : <RefreshCcw className="h-5 w-5" />}
              </button>
            </div>
            <textarea
              className="min-h-72 w-full resize-y rounded-md border border-slate-300 px-3 py-2 leading-6 outline-teal-700"
              value={article}
              onChange={(event) => setArticle(event.target.value)}
              placeholder="Paste a reading passage..."
            />
            <label className="mt-3 flex min-h-10 cursor-pointer items-center justify-center gap-2 rounded-md border border-dashed border-slate-400 bg-slate-50 px-4 py-2 text-sm font-semibold text-slate-700">
              <Upload className="h-5 w-5" />
              Upload .txt article
              <input className="sr-only" type="file" accept=".txt,text/plain" onChange={uploadArticle} />
            </label>
            <input
              className="mt-3 w-full rounded-md border border-slate-300 px-3 py-2 outline-teal-700"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Optional existing first question"
            />
            <div className="my-3 grid gap-3 sm:grid-cols-2">
              {Object.keys(emptyOptions).map((label) => (
                <label key={label} className="grid grid-cols-[28px_minmax(0,1fr)] items-center gap-2">
                  <span className="font-semibold">{label}</span>
                  <input
                    className="min-w-0 rounded-md border border-slate-300 px-3 py-2 outline-teal-700"
                    value={options[label]}
                    onChange={(event) => setOptions({ ...options, [label]: event.target.value })}
                  />
                </label>
              ))}
            </div>
            <button
              className="inline-flex min-h-10 w-full items-center justify-center gap-2 rounded-md bg-teal-700 px-4 py-2 font-semibold text-white disabled:cursor-not-allowed disabled:opacity-60"
              onClick={generateQuiz}
              disabled={loading === 'generate' || article.length < 20}
            >
              {loading === 'generate' ? <Loader2 className="h-5 w-5 animate-spin" /> : <Send className="h-5 w-5" />} Generate 5 Questions
            </button>
          </section>

          <section className="rounded-lg border border-slate-300 bg-white p-5 shadow-sm">
            <div className="mb-4 flex items-center justify-between gap-3">
              <h2 className="text-lg font-semibold">Question & Answer</h2>
              {result?.ai_generated && <span className="rounded-full border border-slate-300 px-3 py-1 text-xs">AI-generated</span>}
            </div>

            {generatedQuestions.length > 0 && (
              <div className="mb-4 grid grid-cols-5 gap-2">
                {generatedQuestions.map((item, index) => (
                  <button
                    key={`${item.question}-${index}`}
                    className={`h-10 rounded-md border text-sm font-semibold ${
                      activeIndex === index ? 'border-teal-700 bg-teal-50 text-teal-900' : 'border-slate-300 bg-white text-slate-700'
                    }`}
                    onClick={() => setActiveIndex(index)}
                  >
                    Q{index + 1}
                  </button>
                ))}
              </div>
            )}

            <p className="mb-4 min-h-16 text-lg leading-7">{activeQuestion || 'Generate or load questions to begin.'}</p>
            <div className="mb-4 grid gap-3">
              {Object.entries(activeOptions).map(([label, text]) => {
                const isSelected = selected === label;
                const isPredicted = verification?.predicted_option === label;
                return (
                  <button
                    key={label}
                    className={`grid min-h-14 grid-cols-[36px_minmax(0,1fr)] items-center gap-3 rounded-md border p-3 text-left disabled:cursor-not-allowed disabled:opacity-60 ${
                      isSelected ? 'border-teal-700 bg-teal-50' : 'border-slate-300 bg-white'
                    } ${isPredicted ? 'ring-2 ring-amber-500' : ''}`}
                    onClick={() => setSelectedByQuestion({ ...selectedByQuestion, [activeIndex]: label })}
                    disabled={!text}
                  >
                    <strong className="grid h-8 w-8 place-items-center rounded-full bg-slate-600 text-white">{label}</strong>
                    <span>{text || 'No option supplied'}</span>
                  </button>
                );
              })}
            </div>
            <button
              className="inline-flex min-h-10 w-full items-center justify-center gap-2 rounded-md bg-teal-700 px-4 py-2 font-semibold text-white disabled:cursor-not-allowed disabled:opacity-60"
              onClick={verifyAnswer}
              disabled={!selected || loading === 'verify' || !activeQuestion}
            >
              {loading === 'verify' ? <Loader2 className="h-5 w-5 animate-spin" /> : <Play className="h-5 w-5" />} Check Answer
            </button>
            {verification && (
              <div className={`mt-4 grid grid-cols-[24px_minmax(0,1fr)] gap-3 rounded-md p-3 ${verification.is_correct ? 'bg-green-50 text-green-900' : 'bg-orange-50 text-orange-900'}`}>
                {verification.is_correct ? <CheckCircle2 className="h-5 w-5" /> : <XCircle className="h-5 w-5" />}
                <div>
                  <strong>{verification.is_correct ? 'Correct' : 'Try again'}</strong>
                  <p className="mt-1">{verification.explanation} Confidence: {Math.round(verification.confidence * 100)}%.</p>
                </div>
              </div>
            )}
          </section>
        </div>

        <aside className="grid content-start gap-5">
          <section className="rounded-lg border border-slate-300 bg-white p-5 shadow-sm">
            <div className="mb-4 flex items-center justify-between gap-3">
              <h2 className="text-lg font-semibold">Hints</h2>
              <Lightbulb className="h-5 w-5" />
            </div>
            {(activeQuiz?.hints || []).map((hint, index) => (
              <div className="border-t border-slate-200 py-3" key={`${hint}-${index}`}>
                <button
                  className="font-semibold text-teal-700"
                  onClick={() => setRevealedHintsByQuestion({ ...revealedHintsByQuestion, [activeIndex]: Math.max(revealedHints, index + 1) })}
                >
                  Hint {index + 1}
                </button>
                {revealedHints > index && <p className="mt-2 leading-6 text-slate-600">{hint}</p>}
              </div>
            ))}
            {activeQuiz && allHintsUsed && (
              <div className="mt-3 rounded-md bg-stone-100 p-3">
                Answer: {activeQuiz.predicted_correct_option} | {activeQuiz.predicted_answer_text}
              </div>
            )}
            {!activeQuiz && <p className="leading-6 text-slate-600">Generate questions to reveal graduated hints.</p>}
          </section>

          <section className="rounded-lg border border-slate-300 bg-white p-5 shadow-sm">
            <div className="mb-4 flex items-center justify-between gap-3">
              <h2 className="text-lg font-semibold">Analytics</h2>
              <BarChart3 className="h-5 w-5" />
            </div>
            <dl className="mb-4 grid grid-cols-3 gap-2">
              <div className="rounded-md border border-slate-200 p-3">
                <dt className="text-xs text-slate-600">Requests</dt>
                <dd className="mt-1 font-bold">{metrics?.session_requests ?? 0}</dd>
              </div>
              <div className="rounded-md border border-slate-200 p-3">
                <dt className="text-xs text-slate-600">Avg latency</dt>
                <dd className="mt-1 font-bold">{metrics?.average_latency_ms ?? 0} ms</dd>
              </div>
              <div className="rounded-md border border-slate-200 p-3">
                <dt className="text-xs text-slate-600">Model A</dt>
                <dd className="mt-1 font-bold">{Math.round((metrics?.model_a?.logistic_regression?.exact_match_answer_accuracy || 0) * 100)}%</dd>
              </div>
            </dl>
            <button className="inline-flex min-h-10 w-full items-center justify-center gap-2 rounded-md bg-slate-200 px-4 py-2 text-slate-900" onClick={exportLogs}>
              <Download className="h-5 w-5" /> Export Logs
            </button>
            <div className="mt-3 grid gap-2 text-sm">
              {(metrics?.last_requests || []).slice().reverse().map((row, index) => (
                <div className="grid grid-cols-[1fr_80px_40px] gap-2 rounded-md bg-slate-100 p-2" key={`${row.endpoint}-${index}`}>
                  <span>{row.endpoint}</span>
                  <span>{row.latency_ms} ms</span>
                  <span>{row.predicted_option || row.question_count || '-'}</span>
                </div>
              ))}
            </div>
          </section>
        </aside>
      </section>
    </main>
  );
}

createRoot(document.getElementById('root')).render(<App />);
