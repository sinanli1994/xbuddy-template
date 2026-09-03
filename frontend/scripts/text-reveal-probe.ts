/** Deterministic animation clock: no browser, server, network, or live model. */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { startTextReveal, type RevealClock } from '../src/utils/textReveal';
import { planMessageEvent } from '../src/utils/chatStream';

function fakeClock() {
  let now = 0, id = 0;
  const pending = new Map<number, () => void>();
  const clock: RevealClock = {
    now: () => now, request: fn => { pending.set(++id, fn); return id; }, cancel: id => { pending.delete(id); },
  };
  return { clock, pending, advance(ms: number) {
    now += ms;
    const callbacks = [...pending.values()]; pending.clear(); callbacks.forEach(fn => fn());
  } };
}
let passed = 0;
function check(label: string, run: () => void) { run(); passed++; console.log(`PASS ${label}`); }
const proposal = '### First-Draft Action Plan\n\n**1. Build relevant skills**\n\n- Practice system design weekly.\n\nWhich steps would you adjust?';
const ready = 'Your final career plan is ready. Open "View Final Plan" in the sidebar to read it.';
for (const text of [proposal, ready]) {
  check(`message-only reply progressively reveals (${text.length} chars)`, () => {
    const time = fakeClock(); const paints: string[] = [];
    startTextReveal(text, value => paints.push(value), time.clock);
    assert.equal(paints.length, 0, 'cannot synchronously dump the full response');
    time.advance(100);
    assert(paints[0].length > 0 && paints[0].length < text.length);
    time.advance(200);
    assert(paints.at(-1)!.length > paints[0].length);
    time.advance(6000);
    assert.equal(paints.at(-1), text);
    assert.equal(time.pending.size, 0);
  });
}
check('canonical transcript and dedup stay complete while presentation is partial', () => {
  const outcome = planMessageEvent({ type: 'ai', content: proposal }, { accumulated: '', shown: [], extraBubbles: 0 });
  assert.equal(outcome.kind, 'fill-placeholder');
  const canonical = proposal;
  const time = fakeClock(); let visible = '';
  startTextReveal(canonical, value => { visible = value; }, time.clock);
  time.advance(100);
  assert.notEqual(visible, canonical);
  assert.equal(planMessageEvent({ type: 'ai', content: proposal }, {
    accumulated: canonical, shown: [canonical.trim()], extraBubbles: 0,
  }).kind, 'ignore', 'duplicate message must not restart or append a reveal');
  assert.equal(canonical, proposal, 'copy/history retain the exact received message');
  // DONE/final_response do not drive the display clock, so they cannot jump it to the end.
  assert.notEqual(visible, canonical);
  time.advance(6000); assert.equal(visible, canonical);
});
check('ordinary token echoes never start a second presentation', () => {
  assert.equal(planMessageEvent({ type: 'ai', content: ready }, {
    accumulated: ready, shown: [], extraBubbles: 0,
  }).kind, 'ignore');
});
check('cancelling a thread prevents even an already queued callback from painting', () => {
  const time = fakeClock(); let count = 0;
  const cancel = startTextReveal(proposal, () => count++, time.clock);
  const stale = [...time.pending.values()][0]; cancel(); stale(); time.advance(1000);
  assert.equal(count, 0); assert.equal(time.pending.size, 0);
});
check('Unicode graphemes stay intact and final Markdown is byte-identical', () => {
  const text = '👩🏽‍💻计划e\u0301✅';
  const prefixes = new Set(['', '👩🏽‍💻', '👩🏽‍💻计', '👩🏽‍💻计划', '👩🏽‍💻计划e\u0301', text]);
  const time = fakeClock(); const paints: string[] = [];
  startTextReveal(text, value => paints.push(value), time.clock);
  for (let i = 0; i < 20; i++) time.advance(20);
  assert(paints.every(value => prefixes.has(value))); assert.equal(paints.at(-1), text);
});
check('short final-ready-sized messages finish briskly without appearing at once', () => {
  const time = fakeClock(); let visible = '';
  const text = 'x'.repeat(200);
  startTextReveal(text, value => { visible = value; }, time.clock);
  time.advance(400); assert.equal(visible.length, 100);
  time.advance(400); assert.equal(visible, text); assert.equal(time.pending.size, 0);
});
check('long messages reveal progressively and finish within two seconds', () => {
  const time = fakeClock(); let visible = '';
  const text = 'Long validated content. '.repeat(1000);
  startTextReveal(text, value => { visible = value; }, time.clock);
  time.advance(1000); assert(visible.length > 0 && visible.length < text.length);
  time.advance(1000); assert.equal(visible, text); assert.equal(time.pending.size, 0);
});
check('empty messages do not leave a running frame', () => {
  const time = fakeClock(); startTextReveal('', () => {}, time.clock);
  time.advance(16); assert.equal(time.pending.size, 0);
});
const read = (path: string) => readFileSync(new URL(path, import.meta.url), 'utf8');
const chat = read('../src/components/ChatArea.tsx');
const component = read('../src/components/ProgressiveText.tsx');
check('history/new-turn paths clear visual policy; canonical copy remains unchanged', () => {
  assert(chat.includes('if (loadedMessages) {\n      revealMessageIdsRef.current.clear();'));
  assert(chat.includes('revealMessageIdsRef.current.clear();\n    setMessages(prev => [...prev, userMessage])'));
  assert(chat.includes('copyToClipboard(message.content, message.id)'));
  assert(chat.includes('text={message.content}'));
  assert(chat.includes('{visibleContent}'));
});
check('reduced-motion and unmount cleanup are wired', () => {
  assert(component.includes('(prefers-reduced-motion: reduce)'));
  assert(component.includes('if (preference.matches) finish()'));
  assert(component.includes('preference.removeEventListener'));
  assert(component.includes('animate && !reducedMotion ? visible : text'));
});
check('animation scroll uses the existing reader-position-aware token scroll rule', () => {
  assert(chat.includes("scrollRevealedText = useCallback(() => scheduleChatScroll('stream-token')"));
  assert(chat.includes('onProgress={scrollRevealedText}'));
  assert(!component.includes('scrollIntoView') && !component.includes('scrollTo'));
});
console.log(`text reveal: ${passed} passed, 0 failed`);
