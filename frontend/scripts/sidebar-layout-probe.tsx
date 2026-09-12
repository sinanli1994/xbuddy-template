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
  resume: null, onUploadResume: noop, onRetryResumeStatus: noop,
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

// ------------------------------------------------------------ resume card ----

const meta = { filename: 'Jordan Avery CV.pdf', pageCount: 2, chunkCount: 12, indexedAt: '2026-09-11T12:00:00Z' };
function renderResume(resume: Props['resume']) {
  return renderToStaticMarkup(<JobBuddyProgress {...base} finalPlanReady={false} completion={null} resume={resume} />);
}
function assertResumePlacement(html: string) {
  const card = html.indexOf('data-testid="resume-card"');
  assert(card >= 0, 'missing resume card');
  assert(html.indexOf('data-testid="completion-summary"') < card, 'resume card must follow progress');
  assert(card < html.indexOf('+ Start a new conversation'), 'resume card must precede the new-conversation control');
}

check('no conversation selected renders no resume card', () => {
  assert(!renderResume(null).includes('resume-card'));
});
check('no resume offers an upload', () => {
  const html = renderResume({ kind: 'none' });
  assertResumePlacement(html);
  assert(html.includes('data-state="none"') && html.includes('>Upload PDF</button>'));
  assert(html.includes('accept="application/pdf,.pdf"'));
  assert(!html.includes('✓'));
});
check('checking shows no upload and no claim', () => {
  const html = renderResume({ kind: 'checking' });
  assert(html.includes('Checking for a resume') && !html.includes('>Upload PDF</button>') && !html.includes('✓'));
});
check('uploading names the file and claims nothing yet', () => {
  const html = renderResume({ kind: 'uploading', filename: 'cv.pdf', previous: null });
  assert(html.includes('Reading cv.pdf') && !html.includes('✓') && !html.includes('passages'));
});
check('indexed shows the filename, passage count, and Replace', () => {
  const html = renderResume({ kind: 'indexed', meta });
  assertResumePlacement(html);
  assert(html.includes('✓ Jordan Avery CV.pdf') && html.includes('12 passages · 2 pages'));
  assert(html.includes('>Replace</button>'));
});
check('an upload error keeps the previous resume visible as still in use', () => {
  const html = renderResume({ kind: 'error', message: 'That PDF is password-protected.', retry: 'upload', previous: meta });
  assert(html.includes('role="alert"') && html.includes('That PDF is password-protected.'));
  assert(html.includes('Still using Jordan Avery CV.pdf.') && html.includes('>Choose another PDF</button>'));
  assert(!html.includes('✓'), 'an error never renders as indexed');
});
check('a status error offers Retry, not "no resume"', () => {
  const html = renderResume({ kind: 'error', message: "We couldn't check your resume right now.", retry: 'status', previous: null });
  assert(html.includes('>Retry</button>') && !html.includes('>Upload PDF</button>') && !html.includes('Still using'));
});
check('teeth: a card rendered below the conversation list is rejected', () => {
  const html = renderResume({ kind: 'indexed', meta });
  const card = divBlock(html, 'data-testid="resume-card"');
  const moved = html.replace(card, '').replace('<details', card + '<details');
  assert.throws(() => assertResumePlacement(moved));
});
console.log(`sidebar layout: ${passed} passed, 0 failed`);
