import { useEffect, useMemo, useRef, useState } from "react";

import {
  deleteDocument,
  queryDocument,
  uploadDocument,
} from "./api";

const SESSION_KEY = "production-rag-document";

const GITHUB_URL = "https://github.com/YOUR_USERNAME/YOUR_REPOSITORY";
const LINKEDIN_URL = "https://www.linkedin.com/in/YOUR_PROFILE";

const METRICS = [
  ["Dense Hit@5", "68.89%"],
  ["Hybrid Hit@5", "88.89%"],
  ["Reranked Hit@5", "93.33%"],
  ["Reranked MRR", "78.89%"],
  ["Answer Success", "97.78%"],
  ["Faithfulness", "96.89%"],
  ["Citation Support", "95.11%"],
  ["Citation Validity", "100%"],
  ["Refusal Accuracy", "100%"],
];

const STACK = [
  "Python",
  "FastAPI",
  "Qdrant",
  "Sentence Transformers",
  "BM25",
  "Reciprocal Rank Fusion",
  "Cross-Encoder Reranking",
  "Gemini",
  "Langfuse",
  "React",
  "Docker",
];

function loadStoredDocument() {
  try {
    const raw = sessionStorage.getItem(SESSION_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function CitationCard({ citation }) {
  return (
    <div className="citation-card">
      <div className="citation-source">Source {citation.source_id}</div>
      <div className="citation-details">
        <span>{citation.filename}</span>
        <span>Page {citation.page_number}</span>
        <span>Chunk {citation.chunk_index}</span>
      </div>
    </div>
  );
}

function Message({ message }) {
  const isUser = message.role === "user";

  return (
    <div className={`message-row ${isUser ? "user-row" : "assistant-row"}`}>
      <div className={`message-block ${isUser ? "user-message" : "assistant-message"}`}>
        <div className="message-label">{isUser ? "You" : "Assistant"}</div>

        <div className="message-bubble">
          {message.content
            .split("\n")
            .filter(Boolean)
            .map((paragraph, index) => (
              <p key={index}>{paragraph}</p>
            ))}
        </div>

        {!isUser && message.citations?.length > 0 && (
          <div className="message-sources">
            <div className="sources-title">Sources</div>
            <div className="citations-grid">
              {message.citations.map((citation) => (
                <CitationCard
                  key={`${message.id}-${citation.source_id}-${citation.chunk_index}`}
                  citation={citation}
                />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function App() {
  const [documentInfo, setDocumentInfo] = useState(loadStoredDocument);
  const [selectedFile, setSelectedFile] = useState(null);
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);

  const [uploading, setUploading] = useState(false);
  const [asking, setAsking] = useState(false);
  const [resetting, setResetting] = useState(false);

  const [uploadError, setUploadError] = useState("");
  const [queryError, setQueryError] = useState("");
  const [resetError, setResetError] = useState("");

  const chatEndRef = useRef(null);

  useEffect(() => {
    if (documentInfo) {
      sessionStorage.setItem(SESSION_KEY, JSON.stringify(documentInfo));
    } else {
      sessionStorage.removeItem(SESSION_KEY);
    }
  }, [documentInfo]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, asking]);

  const busy = uploading || asking || resetting;

  const canAsk = useMemo(
    () =>
      Boolean(
        documentInfo?.document_id &&
          question.trim() &&
          !busy,
      ),
    [documentInfo, question, busy],
  );

  async function handleUpload(event) {
    event.preventDefault();

    if (!selectedFile) {
      setUploadError("Choose a PDF first.");
      return;
    }

    if (!selectedFile.name.toLowerCase().endsWith(".pdf")) {
      setUploadError("Only PDF files are supported.");
      return;
    }

    setUploading(true);
    setUploadError("");
    setQueryError("");
    setResetError("");
    setMessages([]);

    try {
      const result = await uploadDocument(selectedFile);
      setDocumentInfo(result);
      setSelectedFile(null);
      setQuestion("");
    } catch (error) {
      setUploadError(error.message);
    } finally {
      setUploading(false);
    }
  }

  async function handleAsk(event) {
    event.preventDefault();

    const cleanedQuestion = question.trim();

    if (!documentInfo?.document_id || !cleanedQuestion) {
      return;
    }

    const userMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: cleanedQuestion,
    };

    setMessages((current) => [...current, userMessage]);
    setQuestion("");
    setAsking(true);
    setQueryError("");

    try {
      const result = await queryDocument(
        documentInfo.document_id,
        cleanedQuestion,
      );

      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: result.answer || "",
          citations: result.citations || [],
        },
      ]);
    } catch (error) {
      setQueryError(error.message);
    } finally {
      setAsking(false);
    }
  }

  async function handleReset() {
    if (!documentInfo?.document_id) {
      return;
    }

    setResetting(true);
    setResetError("");

    try {
      await deleteDocument(documentInfo.document_id);

      setDocumentInfo(null);
      setSelectedFile(null);
      setQuestion("");
      setMessages([]);
    } catch (error) {
      setResetError(
        `Could not delete the current document: ${error.message}`,
      );
    } finally {
      setResetting(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="site-header">
        <a className="brand" href="#top">Production RAG</a>

        <nav>
          <a href="#demo">Demo</a>
          <a href="#architecture">Architecture</a>
          <a href="#results">Results</a>
          <a href="#about">About</a>
          <a href={GITHUB_URL} target="_blank" rel="noreferrer">GitHub</a>
          <a href={LINKEDIN_URL} target="_blank" rel="noreferrer">LinkedIn</a>
        </nav>
      </header>

      <main className="container" id="top">
        <section className="hero">
          <p className="eyebrow">Production-grade RAG portfolio project</p>
          <h1>Hybrid retrieval, reranking, grounded answers.</h1>
          <p className="hero-copy">
            A production-oriented Retrieval-Augmented Generation system with
            PDF ingestion, dense + BM25 hybrid retrieval, Reciprocal Rank
            Fusion, cross-encoder reranking, citations, evaluation,
            observability, FastAPI, Qdrant, Docker, and a live React demo.
          </p>

          <div className="hero-actions">
            <a className="primary-link" href="#demo">Try the live demo</a>
            <a className="secondary-link" href="#results">See evaluation results</a>
          </div>
        </section>

        <section id="demo" className="section-block">
          <div className="section-intro">
            <p className="eyebrow">Live demo</p>
            <h2>Chat with your PDF</h2>
            <p>
              Upload a document, let the system index it, and ask grounded
              questions with page-level citations.
            </p>
          </div>

          <section className="panel">
            <div className="section-heading">
              <div>
                <span className="step">1</span>
                <h3>Document</h3>
              </div>

              {documentInfo && (
                <button
                  className="ghost-button"
                  type="button"
                  onClick={handleReset}
                  disabled={resetting}
                >
                  {resetting ? "Deleting..." : "Upload another"}
                </button>
              )}
            </div>

            {!documentInfo ? (
              <form className="upload-form" onSubmit={handleUpload}>
                <label className="upload-box">
                  <input
                    type="file"
                    accept="application/pdf,.pdf"
                    onChange={(event) => {
                      setSelectedFile(event.target.files?.[0] || null);
                      setUploadError("");
                    }}
                    disabled={uploading}
                  />

                  <div className="upload-icon">PDF</div>

                  <div>
                    <strong>
                      {selectedFile ? selectedFile.name : "Choose a PDF file"}
                    </strong>
                    <p>
                      {selectedFile
                        ? `${(selectedFile.size / (1024 * 1024)).toFixed(2)} MB`
                        : "Maximum upload size: 20 MB"}
                    </p>
                  </div>
                </label>

                <button
                  className="primary-button"
                  type="submit"
                  disabled={!selectedFile || uploading}
                >
                  {uploading ? "Uploading and indexing..." : "Upload document"}
                </button>

                {uploadError && (
                  <div className="error-message">{uploadError}</div>
                )}
              </form>
            ) : (
              <>
                <div className="document-ready">
                  <div className="status-dot" />
                  <div className="document-meta">
                    <strong>{documentInfo.filename}</strong>
                    <p>
                      {documentInfo.page_count} pages
                      <span>•</span>
                      {documentInfo.chunk_count} chunks
                    </p>
                  </div>
                  <span className="ready-badge">Indexed</span>
                </div>

                {resetError && (
                  <div className="error-message reset-error">{resetError}</div>
                )}
              </>
            )}
          </section>

          <section
            className={`panel chat-panel ${
              !documentInfo ? "panel-disabled" : ""
            }`}
          >
            <div className="section-heading">
              <div>
                <span className="step">2</span>
                <h3>Conversation</h3>
              </div>
            </div>

            <div className="chat-window">
              {!documentInfo ? (
                <div className="empty-state">
                  Upload a PDF to start asking questions.
                </div>
              ) : messages.length === 0 && !asking ? (
                <div className="empty-state">
                  Ask your first question about {documentInfo.filename}.
                </div>
              ) : (
                <>
                  {messages.map((message) => (
                    <Message key={message.id} message={message} />
                  ))}

                  {asking && (
                    <div className="message-row assistant-row">
                      <div className="message-block assistant-message">
                        <div className="message-label">Assistant</div>
                        <div className="typing-bubble">
                          <div className="spinner" />
                          <span>Retrieving context and generating answer...</span>
                        </div>
                      </div>
                    </div>
                  )}

                  <div ref={chatEndRef} />
                </>
              )}
            </div>

            {queryError && (
              <div className="error-message">{queryError}</div>
            )}

            <form className="composer" onSubmit={handleAsk}>
              <textarea
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                placeholder={
                  documentInfo
                    ? "Ask something about this document..."
                    : "Upload a PDF first"
                }
                rows={3}
                disabled={!documentInfo || asking}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();

                    if (canAsk) {
                      event.currentTarget.form?.requestSubmit();
                    }
                  }
                }}
              />

              <div className="composer-footer">
                <span className="hint">
                  Enter to send · Shift + Enter for a new line
                </span>

                <button
                  className="primary-button ask-button"
                  type="submit"
                  disabled={!canAsk}
                >
                  {asking ? "Generating..." : "Send"}
                </button>
              </div>
            </form>
          </section>
        </section>

        <section id="architecture" className="section-block">
          <div className="section-intro">
            <p className="eyebrow">Architecture</p>
            <h2>How the system works</h2>
          </div>

          <div className="architecture-flow">
            {[
              "PDF Upload",
              "Token-aware Chunking",
              "Dense Embeddings",
              "Qdrant + BM25",
              "RRF Fusion",
              "Cross-Encoder Reranking",
              "Gemini Generation",
              "Validated Citations",
            ].map((item, index, items) => (
              <div className="flow-item" key={item}>
                <div className="flow-card">{item}</div>
                {index < items.length - 1 && (
                  <div className="flow-arrow">→</div>
                )}
              </div>
            ))}
          </div>

          <div className="stack-grid">
            {STACK.map((item) => (
              <span className="stack-chip" key={item}>{item}</span>
            ))}
          </div>
        </section>

        <section id="results" className="section-block">
          <div className="section-intro">
            <p className="eyebrow">Evaluation</p>
            <h2>Measured, not just demoed</h2>
            <p>
              Retrieval and generation were evaluated separately to quantify
              improvements from dense retrieval to hybrid search and reranking.
            </p>
          </div>

          <div className="metrics-grid">
            {METRICS.map(([label, value]) => (
              <div className="metric-card" key={label}>
                <div className="metric-value">{value}</div>
                <div className="metric-label">{label}</div>
              </div>
            ))}
          </div>
        </section>

        <section id="about" className="section-block">
          <div className="section-intro">
            <p className="eyebrow">About</p>
            <h2>Built as an AI/ML engineering portfolio project</h2>
            <p>
              This project focuses on retrieval quality, evaluation,
              observability, API design, document isolation, deployment, and
              production-oriented engineering rather than only wrapping an LLM.
            </p>
          </div>

          <div className="about-card">
            <div>
              <h3>Ali Haider</h3>
              <p>
                M2 AI/ML student focused on production ML systems, LLM
                applications, information retrieval, and MLOps.
              </p>
            </div>

            <div className="about-links">
              <a href={'https://github.com/haider-sattar/production-rag'} target="_blank" rel="noreferrer">GitHub</a>
              <a href={'https://www.linkedin.com/in/ali-haider-467948329/'} target="_blank" rel="noreferrer">LinkedIn</a>
            </div>
          </div>
        </section>

        <footer>
          Production RAG · Hybrid Retrieval · Reranking · Evaluation · Docker
        </footer>
      </main>
    </div>
  );
}

export default App;
