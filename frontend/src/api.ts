export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

const API_BASE_URL = import.meta.env.VITE_KIVI_API_BASE_URL ?? "http://127.0.0.1:8000";

export async function createThread(): Promise<string> {
  const response = await fetch(`${API_BASE_URL}/api/chat/threads`, { method: "POST" });
  const data = await parseJson(response);
  return data.thread_id;
}

export async function getThreadMessages(threadId: string): Promise<ChatMessage[]> {
  const response = await fetch(`${API_BASE_URL}/api/chat/threads/${encodeURIComponent(threadId)}/messages`);
  const data = await parseJson(response);
  return data.messages ?? [];
}

export async function sendChatMessage(threadId: string, message: string): Promise<string> {
  const response = await fetch(`${API_BASE_URL}/api/chat/threads/${encodeURIComponent(threadId)}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      current_datetime: timezoneAwareNowIso(),
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
      locale: navigator.language || "en-US",
    }),
  });
  const data = await parseJson(response);
  return data.answer;
}

export type ChatStreamEvent =
  | { type: "tool_activity"; label: string; tools?: string[] }
  | { type: "tool_activity_done" }
  | { type: "assistant_response"; text: string }
  | { type: "memory_updated"; ingestion_id?: string; cursor?: string }
  | { type: "done"; thread_id: string }
  | { type: "error"; message: string };

export async function streamChatMessage(
  threadId: string,
  message: string,
  onEvent: (event: ChatStreamEvent) => void,
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/chat/threads/${encodeURIComponent(threadId)}/messages/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      current_datetime: timezoneAwareNowIso(),
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
      locale: navigator.language || "en-US",
    }),
  });
  if (!response.ok) {
    await parseJson(response);
  }
  if (!response.body) {
    throw new Error("Kivi request did not return a stream");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() ?? "";
    for (const eventBlock of events) {
      const parsed = parseSseEvent(eventBlock);
      if (parsed) {
        onEvent(parsed);
        if (parsed.type === "error") {
          throw new Error(parsed.message || "Kivi request failed");
        }
      }
    }
  }
}

async function parseJson(response: Response): Promise<any> {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || "Kivi request failed");
  }
  return data;
}

function timezoneAwareNowIso(): string {
  const date = new Date();
  const offsetMinutes = -date.getTimezoneOffset();
  const sign = offsetMinutes >= 0 ? "+" : "-";
  const absolute = Math.abs(offsetMinutes);
  const hours = String(Math.floor(absolute / 60)).padStart(2, "0");
  const minutes = String(absolute % 60).padStart(2, "0");
  const local = new Date(date.getTime() - date.getTimezoneOffset() * 60000).toISOString().slice(0, -1);
  return `${local}${sign}${hours}:${minutes}`;
}

function parseSseEvent(block: string): ChatStreamEvent | null {
  const dataLine = block
    .split("\n")
    .find((line) => line.startsWith("data:"));
  if (!dataLine) return null;
  return JSON.parse(dataLine.slice(5).trim()) as ChatStreamEvent;
}
