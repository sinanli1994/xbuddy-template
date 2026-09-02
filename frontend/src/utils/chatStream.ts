/**
 * What a `message` SSE event should do to the live transcript.
 *
 * Extracted from ChatArea for one reason: it is the rule that broke, and inside the
 * component it could only be checked by reading the source. A text scan cannot see a
 * disabled branch — a mutation that wrapped the handler in `if (true) {} else` passed
 * every structural check — so the decision lives here, where a probe can drive it
 * with a real event sequence and compare the result against the persisted transcript.
 *
 * The rule it encodes:
 *
 *   A turn streams the assistant's reply twice. Once as `token` events, which the
 *   caller accumulates into the placeholder bubble, and once as a `message` event
 *   carrying the finished text. Rendering both puts the reply on screen twice, so
 *   the duplicate is dropped.
 *
 *   But not every assistant message has tokens behind it. `implementation_node`
 *   appends a readiness line with no model call, so that line arrives *only* as a
 *   `message` event. Dropping every `message` event — which is what the transcript
 *   used to do — meant the line was checkpointed, returned by /history, and appeared
 *   out of nowhere on refresh.
 *
 * So: a `message` event repeating text already on screen is noise; a `message` event
 * carrying anything else is the only chance the user has to see it live.
 */

/** Assistant lines already visible for the current turn. */
export interface TranscriptProgress {
  /** Text accumulated from `token` events into the placeholder bubble. */
  accumulated: string;
  /** Normalised text of extra bubbles already appended this turn. */
  shown: string[];
  /** How many extra bubbles have been appended this turn. */
  extraBubbles: number;
}

export type MessageEventOutcome =
  | { kind: 'ignore'; reason: string }
  /** Nothing streamed into the placeholder, so use it rather than leaving it blank. */
  | { kind: 'fill-placeholder'; text: string }
  /** A genuinely new assistant message: its own bubble, after the streamed reply. */
  | { kind: 'append-bubble'; text: string };

/** The `content` payload of a `message` SSE frame, as the service serialises it. */
export interface MessagePayload {
  type?: string;
  content?: unknown;
}

export function planMessageEvent(
  payload: MessagePayload | null | undefined,
  progress: TranscriptProgress
): MessageEventOutcome {
  const text = typeof payload?.content === 'string' ? payload.content : '';
  const normalized = text.trim();

  if (payload?.type !== 'ai') {
    // Human echoes and tool traffic are not transcript the user needs replayed.
    return { kind: 'ignore', reason: 'not an assistant message' };
  }
  if (!normalized) {
    return { kind: 'ignore', reason: 'empty content' };
  }
  if (normalized === progress.accumulated.trim()) {
    return { kind: 'ignore', reason: 'already streamed as tokens' };
  }
  if (progress.shown.includes(normalized)) {
    return { kind: 'ignore', reason: 'already shown this turn' };
  }
  if (progress.accumulated === '' && progress.extraBubbles === 0) {
    return { kind: 'fill-placeholder', text };
  }
  return { kind: 'append-bubble', text };
}
