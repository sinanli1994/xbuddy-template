/**
 * Offline checks for the resume card's rules (src/utils/resume.ts).
 *
 *   npx tsx scripts/resume-probe.ts
 *
 * The one that matters most: a resume view is shown only for the thread it was
 * loaded for, so switching conversations can never display another thread's resume.
 */
import assert from 'node:assert/strict';

import {
  RESUME_MAX_BYTES,
  checkResumeFile,
  chooseResumeFile,
  currentResume,
  errorMessageFrom,
  metaFromResponse,
  resumeForThread,
  type ThreadResume,
} from '../src/utils/resume';

let passed = 0;
function check(name: string, fn: () => void) {
  fn();
  passed++;
  console.log(`PASS ${name}`);
}

const meta = { filename: 'cv.pdf', pageCount: 2, chunkCount: 12, indexedAt: '2026-09-11T12:00:00Z' };
const onA: ThreadResume = { threadId: 'thread-a', view: { kind: 'indexed', meta } };

check('a resume is shown for its own thread', () => {
  assert.deepEqual(resumeForThread(onA, 'thread-a'), { kind: 'indexed', meta });
});
check("another thread never sees it", () => {
  assert.equal(resumeForThread(onA, 'thread-b'), null);
});
check('no selection shows nothing', () => {
  assert.equal(resumeForThread(onA, null), null);
  assert.equal(resumeForThread(null, 'thread-a'), null);
});
check('teeth: a selector that ignored the thread would be caught', () => {
  const leaky = (r: ThreadResume | null) => r?.view ?? null;
  assert.notEqual(leaky(onA), resumeForThread(onA, 'thread-b'));
});

check('the resume in effect survives an upload and a failed replace', () => {
  assert.equal(currentResume({ kind: 'indexed', meta }), meta);
  assert.equal(currentResume({ kind: 'uploading', filename: 'new.pdf', previous: meta }), meta);
  assert.equal(currentResume({ kind: 'error', message: 'x', retry: 'upload', previous: meta }), meta);
  assert.equal(currentResume({ kind: 'none' }), null);
  assert.equal(currentResume({ kind: 'checking' }), null);
  assert.equal(currentResume(null), null);
});

check('a PDF within the limit is accepted', () => {
  assert.equal(checkResumeFile({ name: 'CV.PDF', type: '', size: 1000 }), null);
  assert.equal(checkResumeFile({ name: 'resume', type: 'application/pdf', size: RESUME_MAX_BYTES }), null);
});
check('a non-PDF, an empty file, and an oversized file are refused before any request', () => {
  assert.match(checkResumeFile({ name: 'cv.docx', type: 'application/msword', size: 10 }) ?? '', /PDF/);
  assert.match(checkResumeFile({ name: 'cv.pdf', type: 'application/pdf', size: 0 }) ?? '', /empty/);
  assert.match(checkResumeFile({ name: 'cv.pdf', type: 'application/pdf', size: RESUME_MAX_BYTES + 1 }) ?? '', /2 MB/);
});

check('a drop or pick offers exactly one acceptable PDF', () => {
  const pdf = { name: 'cv.pdf', type: 'application/pdf', size: 1000 };
  assert.equal(chooseResumeFile([]), null);
  assert.equal(chooseResumeFile(null), null);
  assert.deepEqual(chooseResumeFile([pdf]), { file: pdf });
  assert.match((chooseResumeFile([pdf, pdf]) as { refusal: string }).refusal, /one PDF/);
  assert.match((chooseResumeFile([{ ...pdf, name: 'cv.png', type: 'image/png' }]) as { refusal: string }).refusal, /PDF/);
  assert.match((chooseResumeFile([{ ...pdf, size: RESUME_MAX_BYTES + 1 }]) as { refusal: string }).refusal, /2 MB/);
});
check('metadata is read only from a complete response', () => {
  assert.deepEqual(
    metaFromResponse({ filename: 'cv.pdf', page_count: 2, chunk_count: 12, indexed_at: '2026-09-11T12:00:00Z' }),
    meta
  );
  assert.equal(metaFromResponse({ filename: 'cv.pdf', page_count: 2 }), null);
  assert.equal(metaFromResponse(null), null);
  assert.equal(metaFromResponse('indexed'), null);
});
check('an error message comes from the proxy, or a plain fallback', () => {
  assert.equal(errorMessageFrom({ error: { code: 'no_text', message: 'Scanned image.' } }, 422), 'Scanned image.');
  assert.equal(errorMessageFrom({ error: 'Backend returned 500' }, 500), 'The resume could not be processed (500).');
  assert.equal(errorMessageFrom(null, 502), 'The resume could not be processed (502).');
});

console.log(`resume: ${passed} passed, 0 failed`);
