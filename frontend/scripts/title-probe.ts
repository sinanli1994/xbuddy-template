/**
 * Behavioural check for the deterministic conversation-title rules.
 *
 *   npx tsx scripts/title-probe.ts
 *
 * Offline and free: the whole point of the rules is that naming a conversation costs
 * nothing. This asserts the shape of the output rather than exact wording, so tuning
 * the stop-word list does not turn the check red for no reason.
 */

import { FALLBACK_TITLE, titleFromFirstMessage } from '../src/utils/demoConversations';

const CASES: Array<{ input: string; expectDifferentFromInput?: boolean }> = [
  { input: "I'm exploring AI Engineer roles and want help planning my job search.", expectDifferentFromInput: true },
  { input: 'I want to move into AI because it is really popular right now.', expectDifferentFromInput: true },
  { input: 'I need help preparing for technical interviews.', expectDifferentFromInput: true },
  { input: 'Can you help me switch from backend engineering to machine learning?', expectDifferentFromInput: true },
  { input: 'hello', expectDifferentFromInput: false },
  { input: '   ', expectDifferentFromInput: false },
];

const MAX = 34;
let failures = 0;

for (const { input, expectDifferentFromInput } of CASES) {
  const title = titleFromFirstMessage(input);
  const problems: string[] = [];

  if (title.length > MAX) problems.push(`longer than ${MAX} chars`);
  if (!title.trim()) problems.push('empty');
  if (expectDifferentFromInput && title === input.trim()) {
    problems.push('is the verbatim first message');
  }
  if (titleFromFirstMessage(input) !== title) problems.push('not deterministic');

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

console.log(`\n  ${CASES.length + 1 - failures} passed, ${failures} failed`);
process.exit(failures ? 1 : 0);
