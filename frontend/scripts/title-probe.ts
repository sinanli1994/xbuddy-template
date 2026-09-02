/**
 * Behavioural check for the deterministic conversation-title rules.
 *
 *   npx tsx scripts/title-probe.ts
 *
 * Offline and free: the whole point of the rules is that naming a conversation costs
 * nothing.
 *
 * Cases carrying `expect` are pinned to exact wording, because the wording is the
 * behaviour being fixed -- "Targeting AI Engineer Role Focused" was a passing title
 * under a shape-only check. Cases without `expect` keep the looser shape assertions,
 * so tuning the stop-word list does not turn the check red for no reason.
 */

import {
  FALLBACK_TITLE,
  listConversations,
  rememberConversation,
  titleFromFirstMessage,
} from '../src/utils/demoConversations';

const storage = new Map<string, string>();
Object.defineProperty(globalThis, 'window', {
  configurable: true,
  value: {
    localStorage: {
      getItem: (key: string) => storage.get(key) ?? null,
      setItem: (key: string, value: string) => storage.set(key, value),
      removeItem: (key: string) => storage.delete(key),
    },
  },
});

const CASES: Array<{
  input: string;
  expectDifferentFromInput?: boolean;
  expect?: string;
  note?: string;
}> = [
  // The reported regression. `focused` opens "focused on LLM applications" and the
  // five-token window used to stop right after it, leaving a dangling participle.
  {
    input: "I'm targeting an AI Engineer role focused on LLM applications, RAG systems, and AI agents.",
    expect: 'Targeting AI Engineer Role',
    note: 'must not end in "Focused"',
    expectDifferentFromInput: true,
  },
  {
    input: "I'm looking for a Data Analyst role in Toronto.",
    expect: 'Data Analyst Role Toronto',
    expectDifferentFromInput: true,
  },
  // Pre-existing behaviour that the qualifier rules must leave alone.
  {
    input: 'I want to switch from backend engineering to machine learning.',
    expect: 'Switch Backend Engineering',
    note: 'unchanged by the qualifier rules',
    expectDifferentFromInput: true,
  },
  // Only the dangling-tail guard fixes this one: no QUALIFIER_BOUNDARIES phrase
  // appears, so the five-token window ends on the participle itself.
  {
    input: 'Which companies should I be targeting?',
    expect: 'Which Companies',
    note: 'must not end in "Targeting"',
    expectDifferentFromInput: true,
  },
  { input: '', expect: FALLBACK_TITLE },
  { input: '   ', expect: FALLBACK_TITLE, expectDifferentFromInput: false },
  // A qualifier phrase mid-sentence is cut; the topic in front of it survives.
  {
    input: "I'm a backend engineer with experience in distributed systems.",
    expect: 'Backend Engineer',
    expectDifferentFromInput: true,
  },
  {
    input: 'I want a Product Manager role within fintech.',
    expect: 'Product Manager Role',
    expectDifferentFromInput: true,
  },
  { input: "I'm exploring AI Engineer roles and want help planning my job search.", expectDifferentFromInput: true },
  { input: "I'm targeting AI Engineer Role", expect: 'Targeting AI Engineer Role', expectDifferentFromInput: true },
  { input: 'I want to move into AI because it is really popular right now.', expectDifferentFromInput: true },
  { input: 'I need help preparing for technical interviews.', expectDifferentFromInput: true },
  { input: 'Can you help me switch from backend engineering to machine learning?', expectDifferentFromInput: true },
  { input: 'hello', expectDifferentFromInput: false },
];

const MAX = 34;
let failures = 0;

for (const { input, expectDifferentFromInput, expect, note } of CASES) {
  const title = titleFromFirstMessage(input);
  const problems: string[] = [];

  if (expect !== undefined && title !== expect) {
    problems.push(`expected ${JSON.stringify(expect)}${note ? ` (${note})` : ''}`);
  }
  // A trailing participle is the defect class, not just the one reported string.
  if (/\b(?:focused|focusing|targeting|related|seeking|using|based|working)$/i.test(title)) {
    problems.push('ends on a dangling participle');
  }

  if (title.length > MAX) problems.push(`longer than ${MAX} chars`);
  if (!title.trim()) problems.push('empty');
  if (expectDifferentFromInput && title === input.trim()) {
    problems.push('is the verbatim first message');
  }
  if (titleFromFirstMessage(input) !== title) problems.push('not deterministic');
  if (/^M(?:\s|$)/.test(title)) problems.push('contains a stray apostrophe fragment');

  const mark = problems.length === 0 ? 'PASS' : 'FAIL';
  if (problems.length) failures++;
  console.log(
    `  [${mark}] ${JSON.stringify(input.slice(0, 46))}\n         -> ${JSON.stringify(title)}` +
      (problems.length ? `  (${problems.join('; ')})` : '')
  );
}

if (titleFromFirstMessage('') !== FALLBACK_TITLE) {
  console.log(`  [FAIL] empty input should fall back to ${FALLBACK_TITLE}`);
  failures++;
}

const threadId = 'title-probe-thread';
const firstMessage = "I'm moving from backend development into AI Engineer roles.";
const secondMessage = 'Actually, I would also consider data engineering.';
rememberConversation(threadId, 7);
rememberConversation(threadId, 7, titleFromFirstMessage(firstMessage));
const titleAfterFirst = listConversations().find((item) => item.threadId === threadId)?.label;
if (!titleAfterFirst || titleAfterFirst === FALLBACK_TITLE) {
  console.log('  [FAIL] first user message should replace the fallback title immediately');
  failures++;
}

// Simulate a title persisted by the pre-fix apostrophe tokenizer. A restore
// supplies the title derived from the actual first message and should repair
// only the exact "M <correct title>" legacy form.
const legacyRecords = JSON.parse(storage.get('jobbuddy_demo_conversations') ?? '[]');
legacyRecords[0].label = `M ${titleAfterFirst}`;
storage.set('jobbuddy_demo_conversations', JSON.stringify(legacyRecords));
rememberConversation(threadId, 7, titleAfterFirst);
const titleAfterLegacyRepair = listConversations().find(
  (item) => item.threadId === threadId,
)?.label;
if (titleAfterLegacyRepair !== titleAfterFirst) {
  console.log('  [FAIL] restore should repair the exact legacy apostrophe title');
  failures++;
}

rememberConversation(threadId, 7, titleFromFirstMessage(secondMessage));
const titleAfterSecond = listConversations().find((item) => item.threadId === threadId)?.label;
if (titleAfterSecond !== titleAfterFirst) {
  console.log('  [FAIL] later user messages must not rename the conversation');
  failures++;
}

const storedMetadata = JSON.parse(storage.get('jobbuddy_demo_conversations') ?? '[]');
const forbiddenKeys = ['messages', 'transcript', 'progress', 'completion', 'finalPlan'];
if (
  storedMetadata.some((record: Record<string, unknown>) =>
    forbiddenKeys.some((key) => Object.hasOwn(record, key))
  )
) {
  console.log('  [FAIL] conversation metadata storage contains transcript/progress data');
  failures++;
}

console.log(`\n  deterministic title cases plus first-message timing checks: ${failures} failed`);
process.exit(failures ? 1 : 0);
