/**
 * Local conversation list and active-thread selection for the demo.
 *
 * Two keys, both small:
 *
 *   jobbuddy_demo_conversations  [{ threadId, userId, label, createdAt, updatedAt }]
 *   jobbuddy_demo_active_thread  "<threadId>"
 *
 * **No transcript, no progress, no graph state.** Those live in the LangGraph
 * checkpoint and come back from `/api/history` and `/api/completion`. A second copy
 * here would be a cache nobody invalidates, and it would drift the moment the same
 * thread was touched from another browser.
 *
 * Titles are derived deterministically from the first user message. Naming a
 * conversation is not worth a model call, and a title that changed on every turn
 * would be worse than a plain one.
 */

const LIST_KEY = 'jobbuddy_demo_conversations';
const ACTIVE_KEY = 'jobbuddy_demo_active_thread';
const MAX_CONVERSATIONS = 20;
const TITLE_MAX = 34;

export const FALLBACK_TITLE = 'New Career Conversation';

export interface DemoConversation {
  threadId: string;
  userId: number;
  label: string;
  createdAt: string;
  updatedAt: string;
}

export interface DemoIdentity {
  threadId: string;
  userId: number;
}

// --------------------------------------------------------------- storage ----
// Every access is guarded: localStorage throws in private-mode browsers and is
// absent during server rendering. A demo that crashes because storage is
// unavailable would be worse than one that simply starts fresh.

function readList(): DemoConversation[] {
  if (typeof window === 'undefined') return [];
  try {
    const parsed = JSON.parse(window.localStorage.getItem(LIST_KEY) ?? '[]');
    return Array.isArray(parsed) ? (parsed as DemoConversation[]) : [];
  } catch {
    return [];
  }
}

function writeList(conversations: DemoConversation[]): void {
  try {
    window.localStorage.setItem(LIST_KEY, JSON.stringify(conversations.slice(0, MAX_CONVERSATIONS)));
  } catch {
    /* quota or private mode: the session still works, it just will not persist */
  }
}

export function getActiveThreadId(): string | null {
  if (typeof window === 'undefined') return null;
  try {
    return window.localStorage.getItem(ACTIVE_KEY);
  } catch {
    return null;
  }
}

export function setActiveThreadId(threadId: string): void {
  try {
    window.localStorage.setItem(ACTIVE_KEY, threadId);
  } catch {
    /* ignore */
  }
}

/** Newest first. */
export function listConversations(): DemoConversation[] {
  return readList().sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
}

// ---------------------------------------------------------------- titles ----

/** Openers that carry no information about the conversation. */
const FILLER_PREFIXES = [
  "i'm exploring", 'i am exploring', "i'm looking for", 'i am looking for',
  'i want to help', 'i want to', 'i would like to', "i'd like to", 'i need help with',
  'i need help', 'i need to', 'can you help me with', 'can you help me', 'can you help',
  'help me with', 'help me', 'please help me', 'please', "i'm trying to",
  'i am trying to', "i'm interested in", 'i am interested in', "i'm thinking about",
  'i am thinking about', 'hi', 'hello', 'hey',
];

const STOP_WORDS = new Set([
  'a', 'an', 'the', 'and', 'or', 'but', 'so', 'because', 'is', 'am', 'are', 'was',
  'were', 'be', 'been', 'to', 'of', 'in', 'on', 'for', 'with', 'about', 'into',
  'from', 'at', 'by', 'my', 'me', 'i', 'it', 'that', 'this', 'want', 'need', 'like',
  'get', 'help', 'really', 'very', 'just', 'right', 'now', 'more', 'some', 'any',
  'can', 'could', 'would', 'should', 'will', 'move', 'moving', 'make', 'making',
  'do', 'doing', 'find', 'finding', 'looking', 'exploring', 'planning', 'plan',
  'thinking', 'interested', 'popular', 'good', 'best', 'new',
]);

/** Words that read better in their conventional casing. */
const CASING: Record<string, string> = {
  ai: 'AI', ml: 'ML', ux: 'UX', ui: 'UI', qa: 'QA', hr: 'HR',
  sre: 'SRE', devops: 'DevOps', pm: 'PM', mle: 'MLE', llm: 'LLM',
};

function titleCase(word: string): string {
  const lower = word.toLowerCase();
  if (CASING[lower]) return CASING[lower];
  return lower.charAt(0).toUpperCase() + lower.slice(1);
}

/**
 * A concise, deterministic title from the first user message.
 *
 *   "I'm exploring AI Engineer roles and want help planning my job search."
 *     -> "AI Engineer Roles Job Search"
 *
 * Strip a leading filler phrase, drop stop-words, keep the first few meaningful
 * words, title-case them, and bound the length. Deliberately rule-based: anything
 * cleverer would either need a model call or become a small NLP project that fails
 * in ways nobody can predict. Where the rules find nothing useful, it falls back to
 * a clean truncation rather than inventing meaning.
 */
export function titleFromFirstMessage(text: string): string {
  const cleaned = text.replace(/\s+/g, ' ').trim();
  if (!cleaned) return FALLBACK_TITLE;

  let body = cleaned.toLowerCase();
  for (const prefix of FILLER_PREFIXES) {
    if (body.startsWith(`${prefix} `)) {
      body = body.slice(prefix.length + 1);
      break;
    }
  }

  const words = body
    .replace(/[^\p{L}\p{N}\s+#.-]/gu, ' ')
    .split(' ')
    .map((w) => w.replace(/^[.-]+|[.-]+$/g, ''))
    .filter(Boolean);

  const meaningful = words.filter((w) => !STOP_WORDS.has(w));
  if (meaningful.length === 0) {
    // Nothing survived the filter. Fall back to a clean truncation of what was said
    // rather than inventing a topic.
    const plain = words.slice(0, 4).map(titleCase).join(' ');
    return plain ? bound(plain) : FALLBACK_TITLE;
  }

  // One meaningful word is a real topic but a poor label on its own, so it gets a
  // neutral qualifier. Earlier this fell back to *every* word including the filler,
  // which produced things like "Move Into AI Because It".
  if (meaningful.length === 1) {
    return bound(titleCase(meaningful[0]) + ' Career Conversation');
  }

  const chosen = meaningful.slice(0, 5);
  let title = chosen.map(titleCase).join(' ');
  return bound(title);
}

/** Clip to TITLE_MAX on a word boundary where one is close enough. */
function bound(title: string): string {
  if (title.length <= TITLE_MAX) return title || FALLBACK_TITLE;
  const clipped = title.slice(0, TITLE_MAX);
  const lastSpace = clipped.lastIndexOf(' ');
  return (lastSpace > TITLE_MAX * 0.5 ? clipped.slice(0, lastSpace) : clipped) || FALLBACK_TITLE;
}

// ----------------------------------------------------------------- CRUD ----

/**
 * Create or update a record.
 *
 * A title assigned once is never rewritten. A conversation whose name changed on
 * every turn would be impossible to find again in the list.
 */
export function rememberConversation(threadId: string, userId: number, title?: string): void {
  const now = new Date().toISOString();
  const list = readList();
  const index = list.findIndex((c) => c.threadId === threadId);

  if (index === -1) {
    list.unshift({
      threadId,
      userId,
      label: title ?? FALLBACK_TITLE,
      createdAt: now,
      updatedAt: now,
    });
  } else {
    const record = list[index];
    list[index] = {
      ...record,
      label: record.label === FALLBACK_TITLE && title ? title : record.label,
      updatedAt: now,
    };
  }
  writeList(list);
}

/**
 * Remove one record from the local list.
 *
 * **Local only.** The backend checkpoint for this thread still exists in Postgres —
 * this demo has no deletion endpoint and does not invent one. Removing the record
 * makes the conversation unreachable from this browser, not deleted.
 *
 * Returns the remaining list so the caller can pick a fallback selection.
 */
export function deleteConversation(threadId: string): DemoConversation[] {
  const remaining = readList().filter((c) => c.threadId !== threadId);
  writeList(remaining);
  return remaining.sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
}

// ------------------------------------------------------------- identity ----

function randomUserId(): number {
  // A positive integer: the backend types `user_id` as int and rejects <= 0.
  return Math.floor(Math.random() * 900_000) + 100_000;
}

export function mintIdentity(): DemoIdentity {
  const threadId =
    typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
      ? crypto.randomUUID()
      : `demo-${Date.now()}-${Math.random().toString(16).slice(2, 10)}`;
  return { threadId, userId: randomUserId() };
}

/**
 * The conversation to show on load.
 *
 * Order matters, and it is the whole point of this function: the **persisted active
 * selection** wins. Previously nothing recorded which thread was being viewed, so a
 * refresh fell through to whatever identity was written last — always the newest
 * conversation, never the one on screen.
 *
 * Falls back to the most recent record only when the active id names nothing that
 * still exists, and mints a fresh thread when the list is empty.
 */
export function resolveActiveIdentity(): DemoIdentity {
  const list = listConversations();
  const activeId = getActiveThreadId();

  const active = activeId ? list.find((c) => c.threadId === activeId) : undefined;
  if (active) return { threadId: active.threadId, userId: active.userId };

  if (list.length > 0) {
    const newest = list[0];
    setActiveThreadId(newest.threadId);
    return { threadId: newest.threadId, userId: newest.userId };
  }

  const fresh = mintIdentity();
  rememberConversation(fresh.threadId, fresh.userId);
  setActiveThreadId(fresh.threadId);
  return fresh;
}
