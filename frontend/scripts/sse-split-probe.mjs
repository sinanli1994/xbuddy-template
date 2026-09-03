import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';

const routeSource = fs.readFileSync(new URL('../src/app/api/chat/route.ts', import.meta.url), 'utf8');
const routeCode = ts.transpileModule(routeSource, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
}).outputText;

const longMessage = 'A detailed production LLM/RAG plan with evaluation and observability. '.repeat(35);
const logicalEvents = [
  { type: 'token', content: 'Early incremental token. ' },
  { type: 'message', content: { type: 'ai', content: longMessage, run_id: 'probe-run', custom_data: { user_id: 7 } } },
  { type: 'section', content: { id: 'career_goal', status: 'in_progress' } },
  { type: 'completion', content: { collection_complete: false, artifact_available: false } },
];
const frames = logicalEvents.map((event) => `data: ${JSON.stringify(event)}\n\n`);
const doneFrame = 'data: [DONE]\n\n';
const wire = frames.join('') + doneFrame;

function splitAt(text, offsets) {
  const points = [...new Set(offsets)].filter((offset) => offset > 0 && offset < text.length).sort((a, b) => a - b);
  return points.map((point, index) => text.slice(index ? points[index - 1] : 0, point)).concat(text.slice(points.at(-1) ?? 0));
}

async function runRoute(chunks, code = routeCode) {
  const errors = [];
  const context = {
    exports: {}, Response, Request, ReadableStream, TextEncoder, TextDecoder, setTimeout, clearTimeout,
    process: { env: { NODE_ENV: 'production' } },
    console: { log() {}, warn() {}, error(...args) { errors.push(String(args[0])); } },
    require(id) {
      assert.equal(id, '@/lib/jobbuddyApi');
      return {
        DEFAULT_AGENT_ID: 'xbuddy',
        jobbuddyApiUrl: () => 'https://no-network.invalid',
        jobbuddyHeaders: () => ({ 'Content-Type': 'application/json' }),
      };
    },
    fetch: async () => new Response(new ReadableStream({
      start(controller) {
        for (const chunk of chunks) controller.enqueue(new TextEncoder().encode(chunk));
        controller.close();
      },
    })),
  };
  vm.runInNewContext(code, context);
  const response = await context.exports.POST(new Request('https://local-probe.invalid/api/chat', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ messages: [{ role: 'user', content: 'synthetic probe' }], userId: 7, threadId: 'synthetic-probe' }),
  }));
  const output = await response.text();
  const events = output.split('\n')
    .filter((line) => line.startsWith('data: '))
    .map((line) => line.slice(6))
    .map((data) => data === '[DONE]' ? data : JSON.parse(data));
  return { events, errors };
}

const messageStart = frames[0].length;
const messageFrame = frames[1];
const cases = {
  'all events in one chunk': [wire],
  'one complete event per chunk': [...frames, doneFrame],
  'message split inside JSON string': splitAt(wire, [messageStart + messageFrame.indexOf('production') + 4]),
  'message split immediately after data prefix': splitAt(wire, [messageStart + 6]),
  'message split before closing brace': splitAt(wire, [messageStart + messageFrame.lastIndexOf('}')]),
  'message across several chunks': splitAt(wire, [messageStart + 7, messageStart + 41, messageStart + 179, messageStart + 903]),
  'token split across chunks': splitAt(wire, [frames[0].indexOf('incremental') + 3]),
  'completion split across chunks': splitAt(wire, [frames.slice(0, 3).join('').length + 12]),
  'DONE split across chunks': splitAt(wire, [wire.indexOf('[DONE]') + 3]),
  'every 17 characters': splitAt(wire, Array.from({ length: Math.ceil(wire.length / 17) - 1 }, (_, i) => (i + 1) * 17)),
};

const baseline = await runRoute(cases['all events in one chunk']);
assert.equal(baseline.errors.length, 0);
for (const [name, chunks] of Object.entries(cases)) {
  const actual = await runRoute(chunks);
  assert.equal(actual.errors.length, 0, `${name}: proxy logged a parse error`);
  assert.deepEqual(actual.events, baseline.events, `${name}: forwarded logical events changed`);
}

const forwardedTypes = baseline.events.map((event) => typeof event === 'string' ? event : event.type);
assert.deepEqual(forwardedTypes, ['token', 'message', 'section', 'completion', 'final_response', '[DONE]']);

// Teeth: removing the carry must make the exact production failure observable.
const noCarrySource = routeSource.replace(
  'const chunk = carry + decoder.decode(value, { stream: true });',
  'const chunk = decoder.decode(value, { stream: true });',
);
assert.notEqual(noCarrySource, routeSource, 'carry mutation did not apply');
const noCarryCode = ts.transpileModule(noCarrySource, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
}).outputText;
const mutation = await runRoute(cases['message split inside JSON string'], noCarryCode);
assert.ok(
  mutation.errors.length > 0 || JSON.stringify(mutation.events) !== JSON.stringify(baseline.events),
  'teeth: removing carry unexpectedly preserved split-SSE behavior',
);

console.log(`SSE_SPLIT_OK ${Object.keys(cases).length} chunking cases + no-carry mutation rejected; events=${forwardedTypes.join(',')}`);
