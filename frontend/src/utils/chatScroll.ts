export const CHAT_BOTTOM_FOLLOW_THRESHOLD = 96;

export interface ChatScrollMetrics {
  scrollHeight: number;
  scrollTop: number;
  clientHeight: number;
}

export type ChatScrollTrigger = 'new-user-message' | 'restored-history' | 'stream-token';

/** Whether the reader is close enough to the bottom that streaming should follow. */
export function isNearChatBottom(metrics: ChatScrollMetrics): boolean {
  return metrics.scrollHeight - metrics.scrollTop - metrics.clientHeight <= CHAT_BOTTOM_FOLLOW_THRESHOLD;
}

/**
 * Pick the only allowed scroll behavior for each event.
 *
 * New messages get one smooth reveal. Token growth is direct (and therefore cannot
 * restart a smooth animation every few milliseconds), and stops entirely when the
 * reader has moved away from the bottom.
 */
export function chatScrollBehavior(
  trigger: ChatScrollTrigger,
  isNearBottom: boolean,
): ScrollBehavior | null {
  if (trigger === 'new-user-message') return 'smooth';
  if (trigger === 'restored-history') return 'auto';
  return isNearBottom ? 'auto' : null;
}
