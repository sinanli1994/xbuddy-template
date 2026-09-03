/**
 * Behavioural check: the live transcript must equal the restored one.
 *
 *   npx tsx scripts/stream-parity-probe.ts
 *
 * Replays the SSE frames the backend actually sends for a final turn through the
 * same rule ChatArea uses, builds the transcript that would be on screen, and
 * compares it against the assistant messages /history returns for that thread.
 *
 * This exists because the structural checks could not see the bug. A mutation that
 * disabled the whole `message` branch left every string those checks look for in
 * place and passed all of them; only running the rule catches it.
 *
 * Offline and free: no server, no backend, no model.
 */

import { planMessageEvent, type TranscriptProgress } from '../src/utils/chatStream';
import { readFileSync } from 'node:fs';

const REPLY = 'Locked in — that is your action plan agreed and the last section closed.';
// Read the canonical backend literal, so the probe cannot silently test stale copy.
const readySource = readFileSync(new URL('../../src/agents/xbuddy/nodes/implementation.py', import.meta.url), 'utf8')
  .split('FINAL_OUTPUT_READY_MESSAGE = (')[1]?.split('\n)')[0];
if (!readySource) throw new Error('Missing canonical readiness message');
const READY = [...readySource.matchAll(/"([^"]*)"|'([^']*)'/g)].map(m => m[1] ?? m[2]).join('');
const CONFIRMATION = 'yes, that plan looks right';

interface Frame {
  type: string;
  content?: unknown;
}

/**
 * What the service emits for the last turn. The reply arrives twice — as tokens
 * while the model writes it, then as a `message` event once the node returns — and
 * the readiness line arrives only as a `message`, because no model produced it.
 */
function finalTurnFrames(): Frame[] {
  const tokens = REPLY.match(/.{1,12}/g) ?? [];
  return [
    { type: 'metadata', content: { thread_id: 't', user_id: 1 } },
    ...tokens.map((chunk) => ({ type: 'token', content: chunk })),
    { type: 'message', content: { type: 'human', content: CONFIRMATION } },
    { type: 'message', content: { type: 'ai', content: REPLY } },
    { type: 'message', content: { type: 'ai', content: READY } },
    { type: 'completion', content: { collection_complete: true, artifact_available: true } },
  ];
}

/** Drive the frames exactly as the component does, and return the visible bubbles. */
function renderTranscript(frames: Frame[]): string[] {
  let accumulated = '';
  const shown: string[] = [];
  let extraBubbles = 0;
  const extras: string[] = [];

  for (const frame of frames) {
    if (frame.type === 'token') {
      accumulated += String(frame.content);
    } else if (frame.type === 'message') {
      const progress: TranscriptProgress = { accumulated, shown, extraBubbles };
      const outcome = planMessageEvent(frame.content as never, progress);
      if (outcome.kind === 'fill-placeholder') {
        accumulated = outcome.text;
        shown.push(outcome.text.trim());
      } else if (outcome.kind === 'append-bubble') {
        extraBubbles += 1;
        shown.push(outcome.text.trim());
        extras.push(outcome.text);
      }
    }
  }
  // Deliberately unfiltered. The placeholder bubble is already on screen by the
  // time these frames arrive, so a rule that never fills it leaves an empty
  // assistant bubble in the transcript — filtering that out here would hide
  // exactly the defect the probe exists to catch.
  return [accumulated, ...extras];
}

/** What /history returns for the same turn: both assistant messages, in order. */
const PERSISTED = [REPLY, READY];

let failures = 0;
const check = (label: string, ok: boolean, detail = '') => {
  if (!ok) failures++;
  console.log(`  [${ok ? 'PASS' : 'FAIL'}] ${label}${detail ? ` — ${detail}` : ''}`);
};

const live = renderTranscript(finalTurnFrames());

check(
  'the live transcript matches the one /history restores',
  JSON.stringify(live) === JSON.stringify(PERSISTED),
  JSON.stringify(live)
);
check('the readiness line is visible before any refresh', live.includes(READY));
check('the reply appears exactly once', live.filter((b) => b === REPLY).length === 1);
check('no bubble is empty', live.every((b) => b.trim().length > 0));
check('the user message is not echoed as a bubble', !live.includes(CONFIRMATION));

// A turn with no extra message must still render exactly one bubble.
const ordinary = renderTranscript([
  { type: 'token', content: 'Tell me about ' },
  { type: 'token', content: 'your background.' },
  { type: 'message', content: { type: 'ai', content: 'Tell me about your background.' } },
]);
check(
  'an ordinary turn is still one bubble',
  JSON.stringify(ordinary) === JSON.stringify(['Tell me about your background.']),
  JSON.stringify(ordinary)
);

// A message-only turn (tokens suppressed) must fill the placeholder, not append.
const noTokens = renderTranscript([
  { type: 'message', content: { type: 'ai', content: 'Reply without tokens.' } },
]);
check(
  'a turn with no tokens still renders its reply',
  JSON.stringify(noTokens) === JSON.stringify(['Reply without tokens.']),
  JSON.stringify(noTokens)
);

// The confirmation-first graph now finalizes without a preceding chat reply.
const readyOnly = renderTranscript([
  { type: 'message', content: { type: 'ai', content: READY } },
]);
check('confirmation-first final turn is exactly one ready bubble',
  JSON.stringify(readyOnly) === JSON.stringify([READY]));

const proposalText = '### First-Draft Action Plan\n\n**1. Improve Application Materials**\n\n- Tailor the CV.\n\n**2. Close Skill Gaps**\n\n- Build a relevant project.\n\n**3. Network Intentionally**\n\n- Contact practitioners.\n\nWhich steps would you adjust?';
const proposalOnly = renderTranscript([
  { type: 'message', content: { type: 'ai', content: proposalText } },
]);
check('validated proposal-only turn fills one bubble without raw JSON tokens',
  JSON.stringify(proposalOnly) === JSON.stringify([proposalText]));

// A message event repeating the streamed reply must not add a bubble.
const repeated = renderTranscript([
  { type: 'token', content: 'Same text.' },
  { type: 'message', content: { type: 'ai', content: 'Same text.' } },
  { type: 'message', content: { type: 'ai', content: 'Same text.' } },
]);
check(
  'a repeated message event does not duplicate a bubble',
  JSON.stringify(repeated) === JSON.stringify(['Same text.']),
  JSON.stringify(repeated)
);

// And a repeated *extra* message must not either. This is the case the previous
// one cannot reach: both copies above are caught by the token comparison, so the
// "already shown this turn" rule never runs.
const repeatedExtra = renderTranscript([
  { type: 'token', content: REPLY },
  { type: 'message', content: { type: 'ai', content: REPLY } },
  { type: 'message', content: { type: 'ai', content: READY } },
  { type: 'message', content: { type: 'ai', content: READY } },
]);
check(
  'a repeated extra message does not duplicate a bubble',
  JSON.stringify(repeatedExtra) === JSON.stringify([REPLY, READY]),
  JSON.stringify(repeatedExtra)
);

console.log(`\nlive/history transcript parity: ${failures} failed`);
if (failures) process.exit(1);
