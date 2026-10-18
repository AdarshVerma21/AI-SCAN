import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Clock3,
  FileImage,
  Info,
  LoaderCircle,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  Upload,
  X,
} from "lucide-react";
import { predictImage } from "./api";
import type { PredictionResponse } from "./types";

const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp"];
const MAX_FILE_BYTES = 10 * 1024 * 1024;

function validateFile(file: File): string | null {
  if (!ACCEPTED_TYPES.includes(file.type)) return "Choose a JPEG, PNG, or WebP image.";
  if (file.size > MAX_FILE_BYTES) return "Images must be smaller than 10 MB.";
  return null;
}

function scorePercent(value: number | null): string | null {
  if (value === null || !Number.isFinite(value)) return null;
  return `${Math.round(value * 100)}%`;
}

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => () => {
    abortRef.current?.abort();
    if (preview) URL.revokeObjectURL(preview);
  }, [preview]);

  const chooseFile = useCallback((nextFile: File | undefined) => {
    if (!nextFile) return;
    const validationError = validateFile(nextFile);
    setError(validationError);
    setResult(null);
    if (validationError) {
      setFile(null);
      setPreview(null);
      return;
    }
    setFile(nextFile);
    setPreview(URL.createObjectURL(nextFile));
  }, []);

  const handleAnalyze = async () => {
    if (!file || isLoading) return;
    setError(null);
    setResult(null);
    setIsLoading(true);
    const controller = new AbortController();
    abortRef.current = controller;
    const timeout = window.setTimeout(() => controller.abort(), 45_000);
    try {
      const response = await predictImage(file, controller.signal);
      setResult(response);
      if (response.status !== "success") {
        setError(response.error || response.warnings[0] || "The detector is not ready.");
      }
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Analysis failed. Please try again.");
    } finally {
      window.clearTimeout(timeout);
      abortRef.current = null;
      setIsLoading(false);
    }
  };

  const reset = () => {
    abortRef.current?.abort();
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setIsLoading(false);
    if (inputRef.current) inputRef.current.value = "";
  };

  const isSuccess = result?.status === "success";
  const isNotReady = result?.status === "model_not_ready";

  return (
    <main className="app-shell">
      <nav className="topbar">
        <div className="brand"><span className="brand-mark"><ShieldCheck size={18} /></span> AiScan</div>
        <div className="topbar-note"><span className="status-dot" /> private, local-first analysis</div>
      </nav>

      <section className="hero">
        <div className="eyebrow"><Sparkles size={14} /> AI image authenticity</div>
        <h1>Understand what<br /><em>you’re looking at.</em></h1>
        <p className="hero-copy">Upload an image to run it through the SAFE and PROBE-DINOv2 detectors. Clear signals, no exaggerated certainty.</p>
      </section>

      <section className="workspace" aria-label="Image analysis workspace">
        {!result && (
          <div
            className={`drop-zone ${isDragging ? "is-dragging" : ""} ${preview ? "has-preview" : ""}`}
            onDragEnter={(event) => { event.preventDefault(); setIsDragging(true); }}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={() => setIsDragging(false)}
            onDrop={(event) => { event.preventDefault(); setIsDragging(false); chooseFile(event.dataTransfer.files[0]); }}
            onClick={() => inputRef.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") inputRef.current?.click(); }}
          >
            <input ref={inputRef} type="file" accept={ACCEPTED_TYPES.join(",")} onChange={(event) => chooseFile(event.target.files?.[0])} hidden />
            {preview ? (
              <div className="preview-wrap">
                <img src={preview} alt="Selected image preview" />
                <div className="preview-overlay"><span>Choose another image</span></div>
              </div>
            ) : (
              <>
                <div className="upload-icon"><Upload size={23} /></div>
                <h2>Drop an image here</h2>
                <p>or <span>browse from your device</span></p>
                <small>JPEG, PNG, or WebP · up to 10 MB</small>
              </>
            )}
          </div>
        )}

        {error && (
          <div className={`message ${isNotReady ? "message-warning" : "message-error"}`} role="alert">
            <AlertTriangle size={17} /><span>{error}</span>
            {!isNotReady && <button className="icon-button" onClick={() => setError(null)} aria-label="Dismiss error"><X size={16} /></button>}
          </div>
        )}

        {file && !result && (
          <div className="selected-file">
            <div className="file-info"><FileImage size={18} /><span><strong>{file.name}</strong><small>{(file.size / 1024 / 1024).toFixed(2)} MB</small></span></div>
            <button className="text-button" onClick={reset}>Remove</button>
          </div>
        )}

        {!result && (
          <button className="primary-button analyze-button" disabled={!file || isLoading} onClick={handleAnalyze}>
            {isLoading ? <><LoaderCircle className="spin" size={18} /> Analyzing image…</> : <><Sparkles size={18} /> Analyze image</>}
          </button>
        )}

        {result && (
          <div className={`result-card ${isSuccess ? (result.prediction === "Real" ? "result-real" : "result-ai") : "result-muted"}`}>
            <div className="result-topline"><span className="result-label">{isSuccess ? "Analysis complete" : "Analysis unavailable"}</span><button className="icon-button" onClick={reset} aria-label="Upload another image"><X size={18} /></button></div>
            {isSuccess ? (
              <>
                <div className="verdict">
                  <div className="verdict-icon">{result.prediction === "Real" ? <CheckCircle2 size={27} /> : <Sparkles size={27} />}</div>
                  <div><span className="verdict-kicker">This image is likely</span><h2>{result.prediction}</h2></div>
                </div>
                {result.ai_score !== null && <div className="score-block"><div className="score-heading"><span>AI-generated score</span><strong>{scorePercent(result.ai_score)}</strong></div><div className="score-track"><div className="score-fill" style={{ width: `${Math.max(0, Math.min(100, result.ai_score * 100))}%` }} /></div>{result.uncertainty && <p className="uncertainty"><Info size={14} /> Close to the decision threshold — treat this result with caution.</p>}</div>}
                <div className="model-grid">{Object.entries(result.model_scores).map(([name, score]) => <div className="model-score" key={name}><span>{name}</span><strong>{score === null ? "Unavailable" : score.toFixed(3)}</strong><small>AI-vs-real logit</small></div>)}</div>
              </>
            ) : <div className="not-ready"><AlertTriangle size={25} /><h2>Models aren’t ready yet</h2><p>The backend received your image, but it cannot run inference until its checkpoints and fusion artifact are configured.</p></div>}
            <div className="result-meta"><span><Clock3 size={15} /> {result.processing_time_ms.toFixed(0)} ms</span>{result.image?.width && <span>{result.image.width} × {result.image.height} · {result.image.format}</span>}</div>
            {result.warnings.map((warning) => <p className="warning-note" key={warning}><Info size={14} /> {warning}</p>)}
            <div className="result-actions"><button className="secondary-button" onClick={handleAnalyze} disabled={isLoading}><RefreshCw size={16} /> Retry</button><button className="primary-button" onClick={reset}><Upload size={16} /> Upload another</button></div>
          </div>
        )}
      </section>

      <footer><span>SAFE + PROBE-DINOv2</span><span>Results are probabilistic, not proof.</span></footer>
    </main>
  );
}

export default App;

