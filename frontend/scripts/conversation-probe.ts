/** Offline conversation lifecycle regressions; never touches real browser storage. */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  deleteConversation, getActiveThreadId, listConversations, mintIdentity,
  rememberConversation, resolveActiveIdentity, setActiveThreadId,
} from '../src/utils/demoConversations';

const storage = new Map<string, string>();
Object.defineProperty(globalThis, 'window', { configurable: true, value: {
  localStorage: {
    getItem: (key: string) => storage.get(key) ?? null,
    setItem: (key: string, value: string) => storage.set(key, value),
    removeItem: (key: string) => storage.delete(key),
  },
} });
let passed = 0;
function check(label: string, run: () => void) {
  storage.clear(); run(); passed++; console.log(`PASS ${label}`);
}
check('first visit remains empty, including repeated initialization', () => {
  for (let i = 0; i < 3; i++) assert.equal(resolveActiveIdentity(), null);
  assert.deepEqual(listConversations(), []);
  assert.equal(getActiveThreadId(), null);
});
check('deleting the last selected conversation stays empty after reload', () => {
  rememberConversation('last', 7); setActiveThreadId('last');
  assert.deepEqual(deleteConversation('last'), []);
  assert.equal(getActiveThreadId(), null);
  for (let i = 0; i < 3; i++) assert.equal(resolveActiveIdentity(), null);
  assert.deepEqual(listConversations(), []);
});
check('one explicit new action after deletion creates exactly one row', () => {
  rememberConversation('last', 7); setActiveThreadId('last'); deleteConversation('last');
  assert.equal(resolveActiveIdentity(), null);
  const id = mintIdentity();
  rememberConversation(id.threadId, id.userId); setActiveThreadId(id.threadId);
  assert.equal(listConversations().length, 1);
  assert.deepEqual(resolveActiveIdentity(), id);
  assert.equal(listConversations().length, 1);
});
check('deleting an inactive conversation preserves selection', () => {
  rememberConversation('active', 7); rememberConversation('other', 8);
  setActiveThreadId('active'); deleteConversation('other');
  assert.deepEqual(resolveActiveIdentity(), { threadId: 'active', userId: 7 });
});
check('deleting the active conversation selects an existing row, never a new one', () => {
  rememberConversation('active', 7); rememberConversation('remaining', 8);
  setActiveThreadId('active'); deleteConversation('active');
  assert.deepEqual(resolveActiveIdentity(), { threadId: 'remaining', userId: 8 });
  assert.equal(listConversations().length, 1);
});
check('stale active selection with an empty list is cleared without minting', () => {
  setActiveThreadId('deleted'); assert.equal(resolveActiveIdentity(), null);
  assert.equal(getActiveThreadId(), null); assert.deepEqual(listConversations(), []);
});
check('corrupt stored list does not manufacture a conversation', () => {
  storage.set('jobbuddy_demo_conversations', '{broken');
  assert.equal(resolveActiveIdentity(), null); assert.deepEqual(listConversations(), []);
});

const page = readFileSync(new URL('../src/app/page.tsx', import.meta.url), 'utf8');
check('only the explicit new handler mints an identity', () => {
  const newHandler = page.slice(page.indexOf('const handleNewConversation'), page.indexOf('const handleFirstUserMessage'));
  assert.equal((page.match(/mintIdentity\(\)/g) ?? []).length, 1);
  assert(newHandler.includes('mintIdentity()'));
  const deletion = page.slice(page.indexOf('const handleDeleteConversation'), page.indexOf('\n  return ('));
  assert(deletion.includes('selectLocally(null)'));
  assert(!deletion.includes('rememberConversation('));
});
check('empty initialization skips restore and shows a start instruction', () => {
  assert(page.includes('if (id) void restore(id);'));
  assert(page.includes("else setRestoreState('ready');"));
  assert(page.includes('Start a new conversation” in the sidebar to begin.'));
  assert(page.includes("{identity && restoreState !== 'loading'"));
});
check('stale history and plan reads are guarded before committing state', () => {
  const restore = page.slice(page.indexOf('const restore ='), page.indexOf('const id = resolveActiveIdentity'));
  const guard = restore.indexOf('if (version !== selectionVersion.current) return;');
  assert(guard > 0 && guard < restore.indexOf('setLoadedMessages(restored)'));
  assert(guard < restore.indexOf('rememberConversation('));
  const plan = page.slice(page.indexOf('const loadFinalPlan'), page.indexOf('const restore ='));
  assert(plan.indexOf('if (version !== selectionVersion.current) return;') < plan.indexOf('setFinalPlan(data.'));
  const select = page.slice(page.indexOf('const selectLocally'), page.indexOf('const handleNewConversation'));
  assert(select.includes('selectionVersion.current += 1;'));
  assert(select.includes('setFinalPlan(null)') && select.includes('setCompletion(null)'));
});
console.log(`conversation lifecycle: ${passed} passed, 0 failed`);
