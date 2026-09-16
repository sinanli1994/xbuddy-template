/**
 * The welcome card's optional resume upload, offline and with no paid call.
 *
 *   npx tsx scripts/welcome-resume-probe.tsx
 *
 * Renders the real components and drives the real drag/drop/picker handlers with
 * fake events and real `File` objects. What this cannot see — the proxy, the bearer
 * token, multipart delivery — is measured by resume-proxy-probe.mjs; the page's
 * wiring by check-demo-contract.mjs.
 */
import React from 'react';
import assert from 'node:assert/strict';
import { renderToStaticMarkup } from 'react-dom/server';

import JobBuddyProgress from '../src/components/JobBuddyProgress';
import JobBuddyWelcome from '../src/components/JobBuddyWelcome';
import { resumeDropHandlers } from '../src/components/ResumeDropZone';
import { RESUME_MAX_BYTES, type ResumeMeta, type ResumeView } from '../src/utils/resume';

// Next compiles JSX with the automatic runtime, so a component like JobBuddyWelcome
// never imports React. tsx uses the classic runtime and looks React up at render time.
(globalThis as { React?: typeof React }).React = React;

let passed = 0;
function check(name: string, fn: () => void) {
  fn();
  passed++;
  console.log(`PASS ${name}`);
}

const noop = () => {};
const meta: ResumeMeta = { filename: 'Resume - Sinan(Andy) Li.pdf', pageCount: 1, chunkCount: 6, indexedAt: '2026-09-16T12:00:00Z' };
const newer: ResumeMeta = { filename: 'Resume 2026.pdf', pageCount: 2, chunkCount: 9, indexedAt: '2026-09-16T13:00:00Z' };

/** Visible text, entities decoded, so assertions read like the copy. */
function text(html: string): string {
  return html
    .replace(/<[^>]+>/g, '')
    .replace(/&#x27;/g, "'")
    .replace(/&quot;/g, '"')
    .replace(/&amp;/g, '&');
}

function welcome(resume: ResumeView | null) {
  return renderToStaticMarkup(<JobBuddyWelcome resume={resume} />);
}

function sidebar(resume: ResumeView | null) {
  return renderToStaticMarkup(
    <JobBuddyProgress
      completion={null} threadId="thread-a" userId={7} conversations={[]}
      onNewConversation={noop} onSelectConversation={noop} onDeleteConversation={noop}
      finalPlanReady={false} finalPlanError={null} onViewFinalPlan={noop} onRetryFinalPlan={noop}
      resume={resume} onUploadResume={noop} onRetryResumeStatus={noop}
    />
  );
}

const pdf = (name = 'cv.pdf', size = 1024) => new File([new Uint8Array(size)], name, { type: 'application/pdf' });

/** A recorder for one set of handlers. */
function harness(busy = false) {
  const uploads: File[] = [];
  const rejections: string[] = [];
  const dragging: boolean[] = [];
  const handlers = resumeDropHandlers({
    busy,
    onUpload: (file) => uploads.push(file),
    onReject: (message) => rejections.push(message),
    setDragging: (value) => dragging.push(value),
    contains: (container, node) => container === 'zone' && node === 'child',
  });
  return { handlers, uploads, rejections, dragging };
}

function dropEvent(files: File[]) {
  const event = { prevented: 0, dataTransfer: { files, dropEffect: '' }, preventDefault() { this.prevented++; } };
  return event;
}

// ---------------------------------------------------------------- 1, 2: copy ----

check('1. a new conversation shows the optional resume area', () => {
  const html = welcome({ kind: 'none' });
  assert(html.includes('data-testid="welcome-resume"') && html.includes('data-testid="resume-drop-zone"'));
  const t = text(html);
  assert(t.includes('Have a resume? (Optional)'));
  assert(t.includes('Drag & drop your PDF resume here') && t.includes('Upload Resume') && t.includes('PDF · up to 2 MB'));
  assert(html.includes('accept="application/pdf,.pdf"'));
});

check('2. the copy makes clear the user can continue without a resume', () => {
  const t = text(welcome({ kind: 'none' }));
  assert(t.includes("I'll help you build a personalized career plan across five areas"));
  for (const area of ['Career Goal', 'Background', 'Job Preferences', 'Skill Assessment', 'Action Plan']) {
    assert(t.includes(area), area);
  }
  assert(t.includes("You'll still confirm anything before I treat it as fact."));
  const optional = t.indexOf('Have a resume?');
  const skip = t.indexOf("No resume? That's completely fine — just answer the questions as we go.");
  const start = t.indexOf("To get started, tell me what kind of career move or role you're considering.");
  assert(optional >= 0 && optional < skip && skip < start, 'optional upload, then the skip path, then the Career Goal question');
});

// ------------------------------------------------- 3–6: picker and drag/drop ----

check('3. choosing a file hands it to the existing upload path', () => {
  const h = harness();
  const file = pdf();
  h.handlers.onChoose([file]);
  assert.deepEqual(h.uploads, [file]);
  assert.deepEqual(h.rejections, []);
});

check('4. dropping a PDF uploads it and never lets the browser open the file', () => {
  const h = harness();
  const file = pdf();
  const enter = dropEvent([]);
  h.handlers.onDragEnter(enter);
  const over = dropEvent([]);
  h.handlers.onDragOver(over);
  const drop = dropEvent([file]);
  h.handlers.onDrop(drop);
  assert.equal(enter.prevented + over.prevented + drop.prevented, 3, 'every drag event cancels the default');
  assert.equal(over.dataTransfer.dropEffect, 'copy');
  assert.deepEqual(h.dragging, [true, false], 'highlighted on enter, cleared on drop');
  assert.deepEqual(h.uploads, [file]);
});

check('5. an unsupported type is refused before any upload', () => {
  const h = harness();
  const docx = new File([new Uint8Array(10)], 'cv.docx', { type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' });
  const drop = dropEvent([docx]);
  h.handlers.onDrop(drop);
  assert.equal(drop.prevented, 1);
  assert.deepEqual(h.uploads, []);
  assert.match(h.rejections[0] ?? '', /PDF/);
});

check('6. an oversized file is refused before any upload', () => {
  const h = harness();
  h.handlers.onDrop(dropEvent([pdf('big.pdf', RESUME_MAX_BYTES + 1)]));
  assert.deepEqual(h.uploads, []);
  assert.match(h.rejections[0] ?? '', /2 MB/);
});

check('several files at once are refused rather than guessing which one was meant', () => {
  const h = harness();
  h.handlers.onDrop(dropEvent([pdf('a.pdf'), pdf('b.pdf')]));
  assert.deepEqual(h.uploads, []);
  assert.match(h.rejections[0] ?? '', /one PDF/);
});

check('drag-over highlight survives moving across the zone and clears on leaving it', () => {
  const h = harness();
  h.handlers.onDragEnter(dropEvent([]));
  h.handlers.onDragLeave({ preventDefault: noop, currentTarget: 'zone', relatedTarget: 'child' });
  assert.deepEqual(h.dragging, [true], 'entering a child is not leaving');
  h.handlers.onDragLeave({ preventDefault: noop, currentTarget: 'zone', relatedTarget: 'elsewhere' });
  assert.deepEqual(h.dragging, [true, false]);
});

check('the rendered zone reflects the drag-over state it is given', () => {
  // renderToStaticMarkup cannot drag; the starting state is what it can pin.
  const html = welcome({ kind: 'none' });
  assert(html.includes('data-dragging="false"'));
});

// ----------------------------------------------------- 7–10: upload states ----

check('7. uploading renders progress and blocks a duplicate submission', () => {
  const html = welcome({ kind: 'uploading', filename: 'cv.pdf', previous: null });
  const t = text(html);
  assert(t.includes('Reading cv.pdf…') && html.includes('aria-busy="true"') && html.includes('aria-live="polite"'));
  assert(!t.includes('Upload Resume'), 'no second submit while one is in flight');
  assert(t.includes('You can keep chatting'), 'the conversation is not blocked');
  const h = harness(true);
  const drop = dropEvent([pdf()]);
  h.handlers.onDrop(drop);
  h.handlers.onChoose([pdf()]);
  assert.equal(drop.prevented, 1, 'still never navigates');
  assert.deepEqual(h.uploads, [], 'a drop or pick while busy is ignored');
  const over = dropEvent([]);
  h.handlers.onDragOver(over);
  assert.equal(over.dataTransfer.dropEffect, 'none');
});

check('8. indexed shows ready, the filename, and its metadata', () => {
  const html = welcome({ kind: 'indexed', meta });
  const t = text(html);
  assert(t.includes('✓ Resume ready') && t.includes('Resume - Sinan(Andy) Li.pdf') && t.includes('6 passages · 1 page'));
  assert(t.includes("I'll use it when we reach Background and Skill Assessment."));
  assert(html.includes('>Replace</button>'));
  assert(!t.includes('Drag & drop') && !t.includes('No resume?'), 'no upload prompt once one is attached');
  assert(t.includes('To get started, tell me'), 'Career Goal is still where the conversation starts');
});

check('9. Replace offers another file through the same upload path', () => {
  const h = harness(false); // indexed is not busy
  const replacement = pdf('Resume 2026.pdf');
  h.handlers.onDrop(dropEvent([replacement]));
  h.handlers.onChoose([replacement]);
  assert.deepEqual(h.uploads, [replacement, replacement]);
  // The next backend-confirmed view replaces the old filename everywhere.
  for (const html of [welcome({ kind: 'indexed', meta: newer }), sidebar({ kind: 'indexed', meta: newer })]) {
    assert(text(html).includes('Resume 2026.pdf') && !text(html).includes('Sinan(Andy)'));
  }
});

check('10. an upload error is shown and stays recoverable', () => {
  const html = welcome({ kind: 'error', message: 'That PDF is password-protected.', retry: 'upload', previous: meta });
  const t = text(html);
  assert(html.includes('role="alert"') && t.includes('That PDF is password-protected.'));
  assert(t.includes('Still using Resume - Sinan(Andy) Li.pdf.'));
  assert(html.includes('data-testid="resume-drop-zone"') && t.includes('Upload Resume'), 'upload again is right there');
  assert(t.includes('To get started, tell me'), 'the conversation is not broken');
  const status = welcome({ kind: 'error', message: "We couldn't check your resume right now.", retry: 'status', previous: null });
  assert(status.includes('>Retry</button>') && !text(status).includes('Upload Resume'));
});

// ------------------------------------------------ 11, 12: one state, two views ----

const VIEWS: ResumeView[] = [
  { kind: 'checking' },
  { kind: 'none' },
  { kind: 'uploading', filename: 'cv.pdf', previous: meta },
  { kind: 'indexed', meta },
  { kind: 'error', message: 'That file is larger than 2 MB. Please upload a smaller PDF.', retry: 'upload', previous: meta },
  { kind: 'error', message: "We couldn't check your resume right now.", retry: 'status', previous: null },
];

function assertAgree(view: ResumeView, welcomeHtml: string, sidebarHtml: string) {
  const state = (html: string, testid: string) =>
    html.match(new RegExp(`data-testid="${testid}" data-state="([a-z]+)"`))?.[1];
  assert.equal(state(welcomeHtml, 'welcome-resume'), view.kind);
  assert.equal(state(sidebarHtml, 'resume-card'), view.kind);
  const shared =
    view.kind === 'indexed' ? [view.meta.filename, `${view.meta.chunkCount} passages`]
    : view.kind === 'uploading' ? [`Reading ${view.filename}`]
    : view.kind === 'error' ? [view.message] : [];
  for (const fact of shared) {
    assert(text(welcomeHtml).includes(fact) && text(sidebarHtml).includes(fact), fact);
  }
}

check('11. the welcome card and the sidebar render the same resume state', () => {
  for (const view of VIEWS) assertAgree(view, welcome(view), sidebar(view));
});

check('teeth: a sidebar showing a different state than the welcome card is caught', () => {
  assert.throws(() => assertAgree({ kind: 'indexed', meta }, welcome({ kind: 'indexed', meta }), sidebar({ kind: 'none' })));
  assert.throws(() =>
    assertAgree({ kind: 'indexed', meta: newer }, welcome({ kind: 'indexed', meta: newer }), sidebar({ kind: 'indexed', meta })),
  );
});

check('12. without the welcome card, the sidebar still shows status, Replace, and recovery', () => {
  // The sidebar takes no messages: it is independent of whether the welcome is shown.
  const indexed = sidebar({ kind: 'indexed', meta });
  assert(text(indexed).includes('✓ Resume - Sinan(Andy) Li.pdf') && indexed.includes('>Replace</button>'));
  const failed = sidebar({ kind: 'error', message: 'Scanned image.', retry: 'upload', previous: null });
  assert(failed.includes('>Choose another PDF</button>'));
  const none = sidebar({ kind: 'none' });
  assert(none.includes('>Add a PDF</button>'), 'a resume can still be added mid-conversation');
});

// ------------------------------------------- 13, 14: skip path and restoration ----

check('13. the no-resume path still starts with the Career Goal question', () => {
  for (const html of [welcome({ kind: 'none' }), welcome(null)]) {
    assert(text(html).trim().endsWith("To get started, tell me what kind of career move or role you're considering."));
  }
  assert(!welcome(null).includes('welcome-resume'), 'no resume state means no upload area, never a broken one');
});

check('14. a restored conversation checks first and claims nothing meanwhile', () => {
  const html = welcome({ kind: 'checking' });
  const t = text(html);
  assert(t.includes('Checking for a resume…'));
  assert(!html.includes('resume-drop-zone') && !t.includes('✓') && !t.includes('Upload Resume'));
});

console.log(`welcome resume: ${passed} passed, 0 failed`);
