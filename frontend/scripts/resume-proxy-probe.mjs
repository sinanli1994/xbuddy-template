/**
 * The resume proxies, end to end through the real Next server, with no paid call.
 *
 *   node scripts/resume-proxy-probe.mjs      (requires `npm run build` first)
 *
 *   this client  ->  next start (:3101)  ->  /api/resume, /api/resume/status
 *                ->  fake backend (:8098), which records what it receives
 *
 * Proves what a source scan cannot: the bearer token is attached on the server and
 * never returned, the upload reaches the backend as multipart, only metadata comes
 * back to the browser, backend error text is relayed only for known user-facing
 * codes, and the resume's text never appears in the server's logs.
 */

import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import assert from 'node:assert/strict';

const BACKEND_PORT = 8098;
const NEXT_PORT = 3101;
const TOKEN = 'probe-token';
const MARKER = 'SYNTHETIC-RESUME-MARKER-7f3a';
const PDF = Buffer.from(`%PDF-1.4\n% ${MARKER}\n1 0 obj << >> endobj\n%%EOF\n`);

// ------------------------------------------------------------ fake backend ----

let reply = { status: 200, body: {} };
let received = [];

function startFakeBackend() {
  const server = createServer((req, res) => {
    const chunks = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => {
      received.push({ url: req.url, headers: req.headers, body: Buffer.concat(chunks) });
      res.writeHead(reply.status, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(reply.body));
    });
  });
  return new Promise((resolve) => server.listen(BACKEND_PORT, () => resolve(server)));
}

// ---------------------------------------------------------------- next ----

let logs = '';

function startNext() {
  const child = spawn('npx', ['next', 'start', '-p', String(NEXT_PORT)], {
    env: {
      ...process.env,
      JOBBUDDY_API_URL: `http://localhost:${BACKEND_PORT}`,
      JOBBUDDY_API_TOKEN: TOKEN,
      // See streaming-probe.mjs: NEXT_PUBLIC_API_ENV is inlined at build time, so
      // both URL branches point at the stub.
      JOBBUDDY_API_URL_LOCAL: `http://localhost:${BACKEND_PORT}`,
      NEXT_PUBLIC_API_ENV: 'production',
    },
    shell: process.platform === 'win32',
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  const record = (buf) => { logs += buf.toString(); };
  child.stdout.on('data', record);
  child.stderr.on('data', record);
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('next start did not become ready')), 45000);
    const onData = (buf) => {
      if (/Ready in|started server|Local:/i.test(buf.toString())) {
        clearTimeout(timer);
        resolve(child);
      }
    };
    child.stdout.on('data', onData);
    child.stderr.on('data', onData);
  });
}

// -------------------------------------------------------------- helpers ----

const base = `http://localhost:${NEXT_PORT}`;

async function upload({ file = PDF, filename = 'cv.pdf', userId = '7', threadId = 'thread-a' } = {}) {
  const form = new FormData();
  if (file) form.append('file', new Blob([file], { type: 'application/pdf' }), filename);
  if (threadId !== null) form.append('thread_id', threadId);
  if (userId !== null) form.append('user_id', userId);
  const response = await fetch(`${base}/api/resume`, { method: 'POST', body: form });
  return { status: response.status, text: await response.text() };
}

async function status(body = { thread_id: 'thread-a', user_id: 7 }) {
  const response = await fetch(`${base}/api/resume/status`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return { status: response.status, text: await response.text() };
}

let passed = 0;
let failed = 0;
async function check(name, fn) {
  received = [];
  try {
    await fn();
    passed++;
    console.log(`  PASS ${name}`);
  } catch (error) {
    failed++;
    console.log(`  FAIL ${name} — ${error.message}`);
  }
}

const INDEXED = {
  indexed: true, document_id: 'doc-internal-id', filename: 'cv.pdf',
  page_count: 2, chunk_count: 12, indexed_at: '2026-09-11T12:00:00Z',
};

// ---------------------------------------------------------------- run ----

let backend;
let next;
try {
  backend = await startFakeBackend();
  next = await startNext();

  await check('an upload reaches the backend as multipart with the bearer token', async () => {
    reply = { status: 200, body: INDEXED };
    const r = await upload();
    assert.equal(r.status, 200, r.text);
    const [call] = received;
    assert.equal(call.url, '/resume');
    assert.equal(call.headers.authorization, `Bearer ${TOKEN}`);
    assert.match(call.headers['content-type'], /^multipart\/form-data; boundary=/);
    const body = call.body.toString('latin1');
    assert(body.includes(MARKER), 'the file bytes reach the backend');
    assert(body.includes('name="thread_id"') && body.includes('thread-a'));
    assert(body.includes('name="user_id"') && /\r\n\r\n7\r\n/.test(body));
  });

  await check('the browser gets metadata only — no document id, no token', async () => {
    reply = { status: 200, body: { ...INDEXED, candidate_facts: { current_role: 'Secret Role' } } };
    const r = await upload();
    assert.deepEqual(JSON.parse(r.text), {
      indexed: true, filename: 'cv.pdf', page_count: 2, chunk_count: 12, indexed_at: '2026-09-11T12:00:00Z',
    });
    assert(!r.text.includes(TOKEN) && !r.text.includes('doc-internal-id') && !r.text.includes('Secret Role'));
  });

  await check('a user-facing backend error is relayed', async () => {
    reply = { status: 422, body: { detail: { code: 'no_text', message: 'It looks like a scanned image.' } } };
    const r = await upload();
    assert.equal(r.status, 422);
    assert.deepEqual(JSON.parse(r.text), { error: { code: 'no_text', message: 'It looks like a scanned image.' } });
  });

  await check('an internal backend error is not relayed', async () => {
    reply = { status: 500, body: { detail: 'Traceback (most recent call last): internal detail' } };
    const r = await upload();
    assert.equal(r.status, 500);
    assert(!r.text.includes('Traceback') && JSON.parse(r.text).error.code === 'upload_failed');
  });

  await check('an unknown error code is not relayed', async () => {
    reply = { status: 400, body: { detail: { code: 'db_exploded', message: 'relation resume_chunks ...' } } };
    const r = await upload();
    assert(!r.text.includes('resume_chunks') && JSON.parse(r.text).error.code === 'upload_failed');
  });

  await check('a 200 that is not indexed is not reported as indexed', async () => {
    reply = { status: 200, body: { indexed: false } };
    const r = await upload();
    assert.equal(r.status, 502);
    assert(!r.text.includes('"indexed":true'));
  });

  await check('404 and 429 get their own messages', async () => {
    reply = { status: 404, body: { detail: 'Thread not found' } };
    assert.equal(JSON.parse((await upload()).text).error.code, 'not_found');
    reply = { status: 429, body: { error: 'Rate limit exceeded' } };
    const limited = await upload();
    assert.equal(limited.status, 429);
    assert.equal(JSON.parse(limited.text).error.code, 'rate_limited');
  });

  await check('an oversized upload is refused without reaching the backend', async () => {
    // Just over the limit (caught on the parsed file) and well over it (caught by
    // the declared length, before the body is read).
    for (const extra of [2 * 1024 * 1024 + 1, 3 * 1024 * 1024]) {
      const r = await upload({ file: Buffer.concat([PDF, Buffer.alloc(extra)]) });
      assert.equal(r.status, 413);
      assert.equal(JSON.parse(r.text).error.code, 'too_large');
    }
    assert.equal(received.length, 0);
  });

  await check('missing identifiers are refused without reaching the backend', async () => {
    assert.equal((await upload({ userId: null })).status, 400);
    assert.equal((await upload({ threadId: null })).status, 400);
    assert.equal((await upload({ userId: 'abc' })).status, 400);
    assert.equal((await upload({ file: null })).status, 400);
    assert.equal(received.length, 0);
  });

  await check('status: an indexed resume returns metadata only', async () => {
    reply = {
      status: 200,
      body: { has_resume: true, filename: 'cv.pdf', page_count: 2, chunk_count: 12,
              indexed_at: '2026-09-11T12:00:00Z', document_id: 'doc-internal-id' },
    };
    const r = await status();
    assert.deepEqual(JSON.parse(r.text), {
      has_resume: true, filename: 'cv.pdf', page_count: 2, chunk_count: 12, indexed_at: '2026-09-11T12:00:00Z',
    });
    const [call] = received;
    assert.equal(call.url, '/resume/status');
    assert.equal(call.headers.authorization, `Bearer ${TOKEN}`);
    assert.deepEqual(JSON.parse(call.body.toString()), { thread_id: 'thread-a', user_id: 7 });
  });

  await check('status: another user\'s thread reads as no resume', async () => {
    reply = { status: 404, body: { detail: 'Thread not found' } };
    assert.deepEqual(JSON.parse((await status()).text), { has_resume: false });
  });

  await check('status: a backend failure is a failure, not "no resume"', async () => {
    reply = { status: 502, body: { detail: { code: 'status_unavailable', message: 'x' } } };
    const r = await status();
    assert.equal(r.status, 502);
    assert(!r.text.includes('has_resume'));
  });

  await check('status: identifiers are required', async () => {
    assert.equal((await status({ thread_id: 'thread-a' })).status, 400);
    assert.equal(received.length, 0);
  });

  await check("the resume's text never appears in the server logs", async () => {
    assert(!logs.includes(MARKER), 'the proxy logged file content');
  });
} catch (error) {
  console.error('  probe failed:', error.message);
  failed++;
} finally {
  // Kill the tree: next start spawns a child of its own (see streaming-probe.mjs).
  if (next) {
    try {
      if (process.platform === 'win32') {
        const { execSync } = await import('node:child_process');
        execSync(`taskkill /pid ${next.pid} /T /F`, { stdio: 'ignore' });
      } else {
        process.kill(-next.pid, 'SIGKILL');
      }
    } catch {
      next.kill();
    }
  }
  backend?.close();
  console.log(`\nresume proxy: ${passed} passed, ${failed} failed`);
  setTimeout(() => process.exit(failed ? 1 : 0), 300);
}
