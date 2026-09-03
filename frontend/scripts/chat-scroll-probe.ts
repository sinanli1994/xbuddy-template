import {
  CHAT_BOTTOM_FOLLOW_THRESHOLD,
  chatScrollBehavior,
  isNearChatBottom,
} from '../src/utils/chatScroll';

let failures = 0;

function check(name: string, condition: boolean) {
  console.log(`  [${condition ? 'PASS' : 'FAIL'}] ${name}`);
  if (!condition) failures++;
}

check(
  'new user message gets one smooth reveal',
  chatScrollBehavior('new-user-message', false) === 'smooth',
);
check(
  'streaming while near bottom uses direct scrolling, never smooth animation',
  chatScrollBehavior('stream-token', true) === 'auto',
);
check(
  'manual upward scroll disables streaming auto-follow',
  chatScrollBehavior('stream-token', false) === null,
);
check(
  'a position inside the threshold counts as near bottom',
  isNearChatBottom({
    scrollHeight: 1000,
    scrollTop: 1000 - 500 - CHAT_BOTTOM_FOLLOW_THRESHOLD,
    clientHeight: 500,
  }),
);
check(
  'a position above the threshold counts as reading history',
  !isNearChatBottom({
    scrollHeight: 1000,
    scrollTop: 1000 - 500 - CHAT_BOTTOM_FOLLOW_THRESHOLD - 1,
    clientHeight: 500,
  }),
);
check(
  'restored history becomes visible without animation',
  chatScrollBehavior('restored-history', false) === 'auto',
);

console.log(`\n  ${6 - failures} passed, ${failures} failed`);
if (failures) process.exitCode = 1;
