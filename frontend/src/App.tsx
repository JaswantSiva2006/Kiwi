import { Dispatch, FormEvent, KeyboardEvent, SetStateAction, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import ReactMarkdown from "react-markdown";
import {
  Check,
  Circle,
  FileText,
  MessageSquarePlus,
  Paperclip,
  PanelLeft,
  Search,
  Send,
  Settings,
  Sparkles,
} from "lucide-react";
import { ChatMessage, createThread, getThreadMessages, streamChatMessage, uploadDocument } from "./api";
import "./styles.css";

type StoredThread = {
  threadId: string;
  title: string;
  createdAt: string;
  updatedAt: string;
};

type UploadedDocument = {
  documentId?: string;
  filename: string;
  status: "indexing" | "ready" | "failed";
  error?: string;
};

type MemorySyncStatus = "synced" | "syncing" | "error";

const THREADS_KEY = "kivi.chat.threads";
const DOCUMENTS_KEY = "kivi.chat.documents";

export default function App() {
  const [threads, setThreads] = useState<StoredThread[]>(() => loadThreads());
  const [activeThreadId, setActiveThreadId] = useState<string | null>(() => loadThreads()[0]?.threadId ?? null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [loading, setLoading] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [toolStatus, setToolStatus] = useState<string | null>(null);
  const [showMemoryToast, setShowMemoryToast] = useState(false);
  const [memorySyncStatus, setMemorySyncStatus] = useState<MemorySyncStatus>("synced");
  const [uploadedDocument, setUploadedDocument] = useState<UploadedDocument | null>(null);
  const [documents, setDocuments] = useState<UploadedDocument[]>(() => loadDocuments());
  const [pendingDocumentSearchIds, setPendingDocumentSearchIds] = useState<string[]>([]);
  const [uploadingDocument, setUploadingDocument] = useState(false);
  const composerRef = useRef<HTMLTextAreaElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const toastTimerRef = useRef<number | null>(null);

  const activeThread = useMemo(
    () => threads.find((thread) => thread.threadId === activeThreadId) ?? null,
    [threads, activeThreadId],
  );

  useEffect(() => {
    saveThreads(threads);
  }, [threads]);

  useEffect(() => {
    saveDocuments(documents);
  }, [documents]);

  useEffect(() => {
    if (!activeThreadId) {
      setMessages([]);
      return;
    }
    let cancelled = false;
    setHistoryLoading(true);
    setError(null);
    getThreadMessages(activeThreadId)
      .then((loaded) => {
        if (!cancelled) setMessages(loaded);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message || "Could not load this chat.");
      })
      .finally(() => {
        if (!cancelled) setHistoryLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [activeThreadId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, loading, toolStatus]);

  useEffect(() => {
    return () => {
      if (toastTimerRef.current) window.clearTimeout(toastTimerRef.current);
    };
  }, []);

  async function startNewChat() {
    setError(null);
    setLoading(true);
    try {
      const threadId = await createThread();
      const thread = {
        threadId,
        title: "New chat",
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      };
      setThreads((current) => [thread, ...current]);
      setActiveThreadId(threadId);
      setMessages([]);
      requestAnimationFrame(() => composerRef.current?.focus());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not create a new chat.");
    } finally {
      setLoading(false);
    }
  }

  async function submitMessage(event?: FormEvent) {
    event?.preventDefault();
    const text = draft.trim();
    if (!text || loading) return;

    let threadId = activeThreadId;
    setError(null);
    setLoading(true);
    setDraft("");

    try {
      if (!threadId) {
        threadId = await createThread();
        const now = new Date().toISOString();
        setThreads((current) => [{ threadId, title: titleFrom(text), createdAt: now, updatedAt: now }, ...current]);
        setActiveThreadId(threadId);
      }

      setMessages((current) => [...current, { role: "user", content: text }]);
      const oneShotDocumentIds = pendingDocumentSearchIds;
      await streamChatMessage(threadId, text, (event) => {
        if (event.type === "tool_activity") {
          setToolStatus(event.label || "Accessing tools...");
        } else if (event.type === "tool_activity_done") {
          setToolStatus(null);
        } else if (event.type === "assistant_response") {
          setMessages((current) => [...current, { role: "assistant", content: event.text }]);
        } else if (event.type === "memory_syncing") {
          setMemorySyncStatus("syncing");
        } else if (event.type === "memory_updated") {
          setMemorySyncStatus("synced");
          showToast();
        }
      }, {
        forceDocumentSearch: oneShotDocumentIds.length > 0,
        documentIds: oneShotDocumentIds,
      });
      if (oneShotDocumentIds.length > 0) {
        setPendingDocumentSearchIds([]);
      }
      touchThread(threadId, text);
      requestAnimationFrame(() => composerRef.current?.focus());
    } catch (err) {
      setMemorySyncStatus("error");
      setDraft(text);
      setMessages((current) => current.filter((message, index) => index !== current.length - 1 || message.role !== "user"));
      setError(err instanceof Error ? err.message : "Kivi could not answer this message.");
    } finally {
      setLoading(false);
    }
  }

  function showToast() {
    setShowMemoryToast(true);
    if (toastTimerRef.current) window.clearTimeout(toastTimerRef.current);
    toastTimerRef.current = window.setTimeout(() => setShowMemoryToast(false), 2600);
  }

  function touchThread(threadId: string, firstMessage: string) {
    setThreads((current) =>
      current.map((thread) =>
        thread.threadId === threadId
          ? {
              ...thread,
              title: thread.title === "New chat" ? titleFrom(firstMessage) : thread.title,
              updatedAt: new Date().toISOString(),
            }
          : thread,
      ),
    );
  }

  function onComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void submitMessage();
    }
  }

  function updateDraft(value: string) {
    setDraft(value);
    requestAnimationFrame(() => {
      const input = composerRef.current;
      if (!input) return;
      input.style.height = "auto";
      input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
    });
  }

  async function onDocumentSelected(file: File | undefined) {
    if (!file || uploadingDocument) return;
    if (file.type !== "application/pdf" && !file.name.toLowerCase().endsWith(".pdf")) {
      setUploadedDocument({ filename: file.name, status: "failed", error: "Only PDF files are supported." });
      return;
    }
    setError(null);
    setUploadingDocument(true);
    setUploadedDocument({ filename: file.name, status: "indexing" });
    upsertDocument(setDocuments, { filename: file.name, status: "indexing" });
    try {
      const result = await uploadDocument(file);
      if (result.status === "READY") {
        const readyDocument = {
          documentId: result.document_id,
          filename: result.filename || file.name,
          status: "ready" as const,
        };
        setUploadedDocument(readyDocument);
        upsertDocument(setDocuments, readyDocument);
        setPendingDocumentSearchIds([result.document_id]);
        showToast();
      } else {
        const failedDocument = {
          documentId: result.document_id,
          filename: result.filename || file.name,
          status: "failed",
          error: result.error || "Indexing failed.",
        } as const;
        setUploadedDocument(failedDocument);
        upsertDocument(setDocuments, failedDocument);
      }
    } catch (err) {
      const failedDocument = {
        filename: file.name,
        status: "failed",
        error: err instanceof Error ? err.message : "Indexing failed.",
      } as const;
      setUploadedDocument(failedDocument);
      upsertDocument(setDocuments, failedDocument);
    } finally {
      setUploadingDocument(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  return (
    <main className="app-shell">
      {showMemoryToast && <div className="memory-toast">Memory updated</div>}
      <aside className={`sidebar ${sidebarOpen ? "open" : "closed"}`}>
        <div className="brand">
          <div className="brand-mark">K</div>
          <div>
            <strong>Kivi</strong>
            <span>Personal intelligence</span>
          </div>
        </div>
        <button className="new-chat" onClick={startNewChat} disabled={loading}>
          <MessageSquarePlus size={18} />
          New conversation
        </button>
        <div className="sidebar-section documents-section">
          <div className="section-heading">
            <span>Documents</span>
            <button
              className="section-action"
              type="button"
              disabled={uploadingDocument || loading}
              onClick={() => fileInputRef.current?.click()}
              aria-label="Add PDF"
            >
              +
            </button>
          </div>
          {documents.length === 0 ? (
            <div className="sidebar-empty">
              <span>No indexed documents</span>
              <button type="button" onClick={() => fileInputRef.current?.click()} disabled={uploadingDocument || loading}>
                Add PDF
              </button>
            </div>
          ) : (
            documents.slice(0, 5).map((document) => <SidebarDocument key={document.filename} document={document} />)
          )}
        </div>
        <nav className="thread-list" aria-label="Recent chats">
          {threads.length === 0 && <div className="sidebar-empty">No conversations yet.</div>}
          {groupThreads(threads).map((group) => (
            <div className="thread-group" key={group.label}>
              <span>{group.label}</span>
              {group.threads.map((thread) => (
                <button
                  key={thread.threadId}
                  className={thread.threadId === activeThreadId ? "thread active" : "thread"}
                  onClick={() => setActiveThreadId(thread.threadId)}
                >
                  <span>{thread.title}</span>
                </button>
              ))}
            </div>
          ))}
        </nav>
        <div className="sidebar-footer">
          <MemorySyncIndicator status={memorySyncStatus} />
          <button className="sidebar-footer-button" type="button">
            <Settings size={15} />
            Settings
          </button>
        </div>
      </aside>

      <section className="conversation">
        <header className="topbar">
          <button className="icon-button" onClick={() => setSidebarOpen((open) => !open)} aria-label="Toggle sidebar">
            <PanelLeft size={20} />
          </button>
          <span>{activeThread?.title ?? "Kivi"}</span>
        </header>

        <div className="messages">
          {historyLoading ? (
            <div className="empty">Loading chat...</div>
          ) : messages.length === 0 ? (
            <div className="empty-state">
              <div className="empty-kicker">
                <Sparkles size={16} />
                Kivi is ready
              </div>
              <h1>Ask across memory, time, and documents.</h1>
              <div className="prompt-list">
                <button type="button" onClick={() => updateDraft("What changed since yesterday?")}>
                  What changed since yesterday?
                </button>
                <button type="button" onClick={() => updateDraft("Find what I said about Atlas.")}>
                  Find what I said about Atlas.
                </button>
                <button type="button" onClick={() => updateDraft("What does this document say about boundary conditions?")}>
                  What does this document say about boundary conditions?
                </button>
                <button type="button" onClick={() => updateDraft("What am I forgetting this week?")}>
                  What am I forgetting this week?
                </button>
              </div>
            </div>
          ) : (
            messages.map((message, index) => <MessageBubble key={`${message.role}-${index}`} message={message} />)
          )}
          {toolStatus && <ToolActivity label={toolStatus} />}
          {loading && !toolStatus && <div className="thinking">Kivi is thinking<span className="dots" /></div>}
          <div ref={bottomRef} />
        </div>

        <form className="composer-wrap" onSubmit={submitMessage}>
          {error && <div className="error">{error}</div>}
          {uploadedDocument && <DocumentChip document={uploadedDocument} />}
          <div className="composer">
            <input
              ref={fileInputRef}
              className="file-input"
              type="file"
              accept="application/pdf,.pdf"
              onChange={(event) => void onDocumentSelected(event.target.files?.[0])}
            />
            <button
              className="attach-button"
              type="button"
              disabled={uploadingDocument || loading}
              onClick={() => fileInputRef.current?.click()}
              aria-label="Attach PDF"
            >
              <Paperclip size={18} />
            </button>
            <textarea
              ref={composerRef}
              value={draft}
              onChange={(event) => updateDraft(event.target.value)}
              onKeyDown={onComposerKeyDown}
              placeholder="Message Kivi"
              rows={1}
              disabled={loading}
            />
            <button className="send-button" type="submit" disabled={!draft.trim() || loading} aria-label="Send message">
              <Send size={18} />
            </button>
          </div>
        </form>
      </section>
    </main>
  );
}

function SidebarDocument({ document }: { document: UploadedDocument }) {
  return (
    <div className={`sidebar-document ${document.status}`}>
      <FileText size={14} />
      <span>{document.filename}</span>
      {document.status === "ready" ? (
        <small className="ready-state">
          <Check size={12} />
          Ready
        </small>
      ) : document.status === "indexing" ? (
        <small>
          <Circle size={9} />
          Indexing
        </small>
      ) : (
        <small>Failed</small>
      )}
    </div>
  );
}

function MemorySyncIndicator({ status }: { status: MemorySyncStatus }) {
  const label = status === "syncing" ? "Ingesting memory..." : status === "error" ? "Memory sync issue" : "Memory synced";
  return (
    <div className={`memory-sync ${status}`}>
      <span />
      {label}
    </div>
  );
}

function DocumentChip({ document }: { document: UploadedDocument }) {
  return (
    <div className={`document-chip ${document.status}`} title={document.error}>
      <FileText size={16} />
      <span className="document-name">{document.filename}</span>
      {document.status === "ready" ? (
        <span className="document-state">
          <Check size={14} />
          Ready
        </span>
      ) : document.status === "failed" ? (
        <span className="document-state">Indexing failed</span>
      ) : (
        <span className="document-state">Indexing<span className="dots" /></span>
      )}
    </div>
  );
}

function ToolActivity({ label }: { label: string }) {
  return (
    <div className="tool-activity">
      <Search size={15} />
      <span>{label}</span>
      <span className="dots" />
    </div>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  return (
    <article className={`message ${message.role}`}>
      <div className="message-inner">
        <ReactMarkdown>{message.content}</ReactMarkdown>
      </div>
    </article>
  );
}

function loadThreads(): StoredThread[] {
  try {
    const raw = localStorage.getItem(THREADS_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveThreads(threads: StoredThread[]) {
  localStorage.setItem(THREADS_KEY, JSON.stringify(threads.slice(0, 40)));
}

function loadDocuments(): UploadedDocument[] {
  try {
    const raw = localStorage.getItem(DOCUMENTS_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveDocuments(documents: UploadedDocument[]) {
  localStorage.setItem(DOCUMENTS_KEY, JSON.stringify(documents.slice(0, 12)));
}

function groupThreads(threads: StoredThread[]) {
  const groups = [
    { label: "Today", threads: [] as StoredThread[] },
    { label: "Yesterday", threads: [] as StoredThread[] },
    { label: "Previous 7 Days", threads: [] as StoredThread[] },
    { label: "Older", threads: [] as StoredThread[] },
  ];
  const now = new Date();
  const today = startOfDay(now).getTime();
  const yesterday = today - 86400000;
  const previousWeek = today - 7 * 86400000;
  for (const thread of threads) {
    const time = new Date(thread.updatedAt || thread.createdAt).getTime();
    if (time >= today) groups[0].threads.push(thread);
    else if (time >= yesterday) groups[1].threads.push(thread);
    else if (time >= previousWeek) groups[2].threads.push(thread);
    else groups[3].threads.push(thread);
  }
  return groups.filter((group) => group.threads.length > 0);
}

function startOfDay(date: Date) {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate());
}

function upsertDocument(
  setDocuments: Dispatch<SetStateAction<UploadedDocument[]>>,
  document: UploadedDocument,
) {
  setDocuments((current) =>
    [document, ...current.filter((item) => (document.documentId ? item.documentId !== document.documentId : item.filename !== document.filename))].slice(0, 12),
  );
}

function titleFrom(text: string): string {
  const compact = text.replace(/\s+/g, " ").trim();
  return compact.length > 42 ? `${compact.slice(0, 39)}...` : compact || "New chat";
}

createRoot(document.getElementById("root")!).render(<App />);
