/** Offline real-component render checks, including deliberately broken markup. */
import React from 'react';
import assert from 'node:assert/strict';
import { renderToStaticMarkup } from 'react-dom/server';
import JobBuddyProgress from '../src/components/JobBuddyProgress';

const noop = () => {};
const base = {
  threadId: 'local-fixture', userId: 7,
  conversations: Array.from({ length: 40 }, (_, i) => ({
    threadId: `fixture-${i}`, userId: 7, label: `Conversation ${i}`,
    createdAt: new Date(i).toISOString(), updatedAt: new Date(i).toISOString(),
  })),
  onNewConversation: noop, onSelectConversation: noop, onDeleteConversation: noop,
  finalPlanError: null, onViewFinalPlan: noop, onRetryFinalPlan: noop,
};
const sections = ['Career Goal', 'Background', 'Job Preferences', 'Skill Assessment', 'Action Plan']
  .map((name, i) => ({ id: `section-${i}`, name, status: 'done' as const }));
type Props = React.ComponentProps<typeof JobBuddyProgress>;
function render(available: boolean, loaded: boolean, error: string | null = null) {
  return renderToStaticMarkup(<JobBuddyProgress {...base} finalPlanReady={loaded} finalPlanError={error}
    completion={{ sections, collection_complete: true, artifact_available: available }} />);
}

function divBlock(html: string, marker: string): string {
  const index = html.indexOf(marker);
  assert(index >= 0, `missing ${marker}`);
  const start = html.lastIndexOf('<div', index);
  let depth = 0;
  for (const m of html.slice(start).matchAll(/<\/?div\b[^>]*>/g)) {
    depth += m[0].startsWith('</') ? -1 : 1;
    if (!depth) return html.slice(start, start + m.index! + m[0].length);
  }
  throw Error('unclosed div');
}

function assertHierarchy(html: string, available: boolean, loaded: boolean) {
  const progress = divBlock(html, 'data-testid="workflow-progress"');
  const summary = divBlock(html, 'data-testid="completion-summary"');
  assert(progress.includes(summary), 'completion must be inside Your Progress');
  const labels = ['Your Progress', ...sections.map(s => s.name), '>Collection<', '>Final Plan<'];
  let previous = -1;
  for (const label of labels) {
    const index = progress.indexOf(label);
    assert(index > previous, `wrong order: ${label}`);
    previous = index;
  }
  assert.equal(html.split('data-testid="completion-summary"').length - 1, 1, 'no duplicated summary');
  assert(html.indexOf(summary) < html.indexOf('+ Start a new conversation'));
  assert(html.indexOf('+ Start a new conversation') < html.indexOf('Recent Conversations'));
  assert(html.indexOf('Recent Conversations') < html.indexOf('Developer Details'));
  assert.equal(summary.includes('>View Final Plan</button>'), available && loaded,
    'view requires backend availability AND loaded document');
}

let passed = 0;
function check(name: string, fn: () => void) { fn(); passed++; console.log(`PASS ${name}`); }

for (const available of [false, true]) for (const loaded of [false, true]) {
  check(`hierarchy / availability=${available} loaded=${loaded}`, () => {
    const html = render(available, loaded);
    assertHierarchy(html, available, loaded);
    assert(html.includes(available ? '>Ready<' : '>Not Ready<'));
    assert(html.includes('>Complete<'));
  });
}
check('retrieval failure offers Retry but no empty viewer', () => {
  const html = render(true, false, 'offline');
  assertHierarchy(html, true, false);
  assert(html.includes('could not be loaded') && html.includes('>Retry</button>'));
});
check('no backend projection remains neutral', () => {
  const props: Props = { ...base, completion: null, finalPlanReady: false };
  const html = renderToStaticMarkup(<JobBuddyProgress {...props} />);
  assert(html.includes('Progress appears here once JobBuddy replies.'));
  assert(!html.includes('>Done<') && !html.includes('>View Final Plan</button>'));
});
check('forty conversations retain a bounded internal scroll region', () => {
  const html = render(true, true);
  assert(html.includes('overflow-y:auto') && html.includes('min-height:114px'));
  assert(html.includes('max-height:clamp(114px, 26vh, 190px)'));
  assert(!/position:(absolute|fixed)/.test(html));
});
check('teeth: moving completion below history is rejected', () => {
  const html = render(true, true);
  const summary = divBlock(html, 'data-testid="completion-summary"');
  const moved = html.replace(summary, '').replace('<details', summary + '<details');
  assert.throws(() => assertHierarchy(moved, true, true));
});
check('teeth: duplicate summary is rejected', () => {
  const html = render(true, true);
  const summary = divBlock(html, 'data-testid="completion-summary"');
  assert.throws(() => assertHierarchy(html.replace(summary, summary + summary), true, true));
});
check('teeth: premature View Final Plan is rejected', () => {
  assert.throws(() => assertHierarchy(render(true, true), true, false));
});
console.log(`sidebar layout: ${passed} passed, 0 failed`);
