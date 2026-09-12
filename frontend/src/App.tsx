import { FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import ReactMarkdown from "react-markdown";
import { Brain, CalendarClock, MessageSquarePlus, PanelLeft, Search, Send, Sparkles } from "lucide-react";
import { ChatMessage, createThread, getThreadMessages, streamChatMessage } from "./api";
import "./styles.css";

type StoredThread = {
  threadId: string;
  title: string;
  createdAt: string;
  updatedAt: string;
};

const THREADS_KEY = "kivi.chat.threads";

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
  const composerRef = useRef<HTMLTextAreaElement | null>(null);
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
      await streamChatMessage(threadId, text, (event) => {
        if (event.type === "tool_activity") {
          setToolStatus(event.label || "Accessing tools...");
        } else if (event.type === "tool_activity_done") {
          setToolStatus(null);
        } else if (event.type === "assistant_response") {
          setMessages((current) => [...current, { role: "assistant", content: event.text }]);
        } else if (event.type === "memory_updated") {
          showToast();
        }
      });
      touchThread(threadId, text);
      requestAnimationFrame(() => composerRef.current?.focus());
    } catch (err) {
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

  return (
    <main className="app-shell">
      {showMemoryToast && <div className="memory-toast">Memory updated</div>}
      <aside className={`sidebar ${sidebarOpen ? "open" : "closed"}`}>
        <div className="brand">
          <div className="brand-mark">K</div>
          <div>
            <strong>Kivi</strong>
            <span>personal memory</span>
          </div>
        </div>
        <button className="new-chat" onClick={startNewChat} disabled={loading}>
          <MessageSquarePlus size={18} />
          New chat
        </button>
        <div className="sidebar-section">
          <span>Workspace</span>
          <div className="sidebar-pill">
            <Brain size={15} />
            Memory active
          </div>
          <div className="sidebar-pill">
            <CalendarClock size={15} />
            Calendar ready
          </div>
        </div>
        <nav className="thread-list" aria-label="Recent chats">
          {threads.map((thread) => (
            <button
              key={thread.threadId}
              className={thread.threadId === activeThreadId ? "thread active" : "thread"}
              onClick={() => setActiveThreadId(thread.threadId)}
            >
              <span>{thread.title}</span>
            </button>
          ))}
        </nav>
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
              <div className="empty-icon">
                <Sparkles size={26} />
              </div>
              <h1>How can Kivi help?</h1>
              <p>Start a fresh conversation, or ask about your memories, projects, and schedule.</p>
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
          <div className="composer">
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

function titleFrom(text: string): string {
  const compact = text.replace(/\s+/g, " ").trim();
  return compact.length > 42 ? `${compact.slice(0, 39)}...` : compact || "New chat";
}

createRoot(document.getElementById("root")!).render(<App />);
