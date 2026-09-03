/**
 * End-to-end streaming measurement, with no paid model call.
 *
 *   node scripts/streaming-probe.mjs
 *
 * Headers claiming "no-transform" prove nothing about whether bytes actually arrive
 * over time. This stands a fake backend in front of the real proxy and timestamps
 * every SSE frame the client receives, so "does it stream" becomes a measurement
 * rather than an inference.
 *
 *   fake backend (:8099, emits a token every 120ms)
 *     -> next start (:3100)  ->  /api/chat  ->  this client
 *
 * A streaming proxy produces inter-arrival gaps near 120ms. A buffering one delivers
 * everything in a single burst at the end, which is exactly what the browser showed.
 *
 * Requires `npm run build` first.
 */

import { spawn } from 'node:child_process';
import { createServer } from 'node:http';

const BACKEND_PORT = 8099;
const NEXT_PORT = 3100;
const TOKEN_COUNT = 10;
const TOKEN_DELAY_MS = 120;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ------------------------------------------------------------ fake backend ----

function startFakeBackend() {
  const server = createServer(async (req, res) => {
    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache, no-transform',
      Connection: 'keep-alive',
    });
    const send = (obj) => res.write(`data: ${JSON.stringify(obj)}\n\n`);

    send({ type: 'metadata', content: { thread_id: 'probe-thread', run_id: 'probe-run' } });
    const progress = (advanced) => ({
      collection_complete: false, artifact_available: false,
      sections: ['career_goal', 'background', 'job_preferences', 'skill_assessment', 'action_plan']
        .map((id, index) => ({ id, name: id, status: index === 0
          ? (advanced ? 'done' : 'in_progress') : (advanced && index === 1 ? 'in_progress' : 'pending') })),
    });
    send({ type: 'completion', content: progress(false) });
    for (let i = 0; i < TOKEN_COUNT; i++) {
      await sleep(TOKEN_DELAY_MS);
      send({ type: 'token', content: `w${i} ` });
      if (i === 3) send({ type: 'completion', content: progress(true) });
    }
    send({ type: 'message', content: { type: 'ai', content: 'w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 ' } });
    send({ type: 'section', content: { name: 'Career Goal', status: 'in_progress' } });
    send({
      type: 'completion',
      content: progress(true),
    });
    res.write('data: [DONE]\n\n');
    res.end();
  });
  return new Promise((resolve) => server.listen(BACKEND_PORT, () => resolve(server)));
}

// ---------------------------------------------------------------- next ----

function startNext() {
  const child = spawn('npx', ['next', 'start', '-p', String(NEXT_PORT)], {
    env: {
      ...process.env,
      JOBBUDDY_API_URL: `http://localhost:${BACKEND_PORT}`,
      JOBBUDDY_API_TOKEN: 'probe-token',
      // Both branches of jobbuddyApiUrl() point at the stub, because only one of
      // them can be steered from here. NEXT_PUBLIC_ values are inlined by
      // `next build`, so setting NEXT_PUBLIC_API_ENV at start time does nothing:
      // whatever .env.local said when the bundle was built is what the route sees.
      // With .env.local set to `local` for development, the probe was silently
      // pointing every request at localhost:8080 and failing on a connection error
      // that looked like a streaming bug.
      JOBBUDDY_API_URL_LOCAL: `http://localhost:${BACKEND_PORT}`,
      NEXT_PUBLIC_API_ENV: 'production',
    },
    shell: process.platform === 'win32',
    stdio: ['ignore', 'pipe', 'pipe'],
  });
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

// -------------------------------------------------------------- measure ----

async function measure() {
  const response = await fetch(`http://localhost:${NEXT_PORT}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      messages: [{ role: 'user', content: 'probe' }],
      userId: 4242,
      threadId: 'probe-thread',
      mode: 'stream',
    }),
  });

  if (!response.ok) throw new Error(`/api/chat returned ${response.status}`);

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const arrivals = [];
  const t0 = Date.now();
  let carry = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const text = carry + decoder.decode(value, { stream: true });
    const lines = text.split('\n');
    carry = lines.pop() ?? '';
    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const payload = line.slice(6).trim();
      const at = Date.now() - t0;
      if (payload === '[DONE]') {
        arrivals.push({ type: '[DONE]', at });
        continue;
      }
      try {
        arrivals.push({ ...JSON.parse(payload), at });
      } catch {
        arrivals.push({ type: 'UNPARSEABLE', at });
      }
    }
  }
  return arrivals;
}

// --------------------------------------------------------------- report ----

const backend = await startFakeBackend();
let next;
try {
  next = await startNext();
  await sleep(600);
  const arrivals = await measure();

  const tokens = arrivals.filter((a) => a.type === 'token');
  console.log(`\n  events received: ${arrivals.length}`);
  console.log(`  token events   : ${tokens.length} (backend emitted ${TOKEN_COUNT})`);
  console.log(`  sequence       : ${arrivals.map((a) => a.type).join(' -> ')}`);
  console.log(`  token arrivals : ${tokens.map((t) => `${t.at}ms`).join(', ')}`);

  const gaps = tokens.slice(1).map((t, i) => t.at - tokens[i].at);
  const spread = tokens.length > 1 ? tokens[tokens.length - 1].at - tokens[0].at : 0;
  console.log(`  gaps           : ${gaps.join(', ')}`);
  console.log(`  first->last    : ${spread}ms (expected ~${(TOKEN_COUNT - 1) * TOKEN_DELAY_MS}ms)`);

  const streaming = spread > ((TOKEN_COUNT - 1) * TOKEN_DELAY_MS) / 2;
  const advanceIndex = arrivals.findIndex((event) => event.type === 'completion' &&
    event.content.sections[0].status === 'done');
  const progressBeforeRemainingTokens = advanceIndex >= 0 &&
    arrivals.slice(advanceIndex + 1).some((event) => event.type === 'token');
  console.log(`  progress before remaining tokens: ${progressBeforeRemainingTokens ? 'PASS' : 'FAIL'}`);
  console.log(
    `\n  VERDICT: the proxy ${streaming ? 'STREAMS — tokens arrive spread over time' : 'BUFFERS — tokens arrive in one burst'}`
  );
  process.exitCode = streaming && progressBeforeRemainingTokens ? 0 : 1;
} catch (error) {
  console.error('  probe failed:', error.message);
  process.exitCode = 2;
} finally {
  // next start spawns a child of its own, so killing the wrapper leaves a live
  // server holding a lock on Next's SWC binary — which later broke npm ci with
  // EPERM. Kill the tree on Windows, the process group elsewhere.
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
  backend.close();
  setTimeout(() => process.exit(process.exitCode ?? 0), 300);
}
