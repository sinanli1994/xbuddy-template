/**
 * The responsive layout's structure and rules, offline.
 *
 *   npx tsx scripts/mobile-layout-probe.tsx
 *
 * Renders the real page and ChatArea, and reads the real globals.css. Layout itself —
 * widths, overflow, the drawer opening and closing, focus — needs a browser, and is
 * walked in headless Chrome at desktop and phone widths before a change ships.
 */
import React from 'react';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { renderToStaticMarkup } from 'react-dom/server';

import JobBuddyDemo from '../src/app/page';
import ChatArea from '../src/components/ChatArea';

// Next compiles JSX with the automatic runtime; tsx looks React up at render time.
(globalThis as { React?: typeof React }).React = React;

let passed = 0;
function check(name: string, fn: () => void) {
  fn();
  passed++;
  console.log(`PASS ${name}`);
}

const noop = () => {};
const css = readFileSync(new URL('../src/app/globals.css', import.meta.url), 'utf8').replace(/\/\*[^]*?\*\//g, ' ');

/** The declarations of one rule, outside or inside the mobile media query. */
function rule(source: string, selector: string): string {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const match = source.match(new RegExp(`(?:^|[}\\s,])${escaped}\\s*(?:,[^{]*)?\\{([^}]*)\\}`));
  assert(match, `missing rule: ${selector}`);
  return match[1].replace(/\s+/g, ' ');
}

function mobileBlock(source: string): string {
  const start = source.indexOf('@media (max-width: 767.98px)');
  assert(start >= 0, 'missing the 768px breakpoint');
  let depth = 0;
  for (let i = source.indexOf('{', start); i < source.length; i++) {
    if (source[i] === '{') depth++;
    if (source[i] === '}' && --depth === 0) return source.slice(source.indexOf('{', start) + 1, i);
  }
  throw new Error('unterminated media block');
}

function desktopSource(source: string): string {
  const start = source.indexOf('@media (max-width: 767.98px)');
  return start < 0 ? source : source.slice(0, start);
}

function assertResponsiveRules(source: string) {
  const desktop = desktopSource(source);
  const sidebar = rule(desktop, '.jb-sidebar');
  assert(sidebar.includes('width: 32%') && sidebar.includes('min-width: 280px') && sidebar.includes('max-width: 400px'),
    'desktop sidebar keeps its column width');
  assert(rule(desktop, '.jb-menu-button').includes('display: none'), 'mobile controls are hidden on desktop');

  const mobile = mobileBlock(source);
  const drawer = rule(mobile, '.jb-sidebar');
  assert(drawer.includes('position: fixed') && drawer.includes('min-width: 0'), 'the sidebar leaves the flow on a phone');
  assert(drawer.includes('transform: translateX(-100%)'), 'closed drawer is off-screen');
  assert(drawer.includes('visibility: hidden'), 'closed drawer is not focusable or announced');
  const open = rule(mobile, '.jb-sidebar[data-open="true"]');
  assert(open.includes('transform: none') && open.includes('visibility: visible'), 'open drawer is shown');
  const backdrop = rule(mobile, '.jb-backdrop[data-open="true"]');
  assert(backdrop.includes('display: block') && backdrop.includes('position: fixed') && backdrop.includes('inset: 0'),
    'open backdrop covers the chat');
  assert(rule(mobile, '.jb-menu-button').includes('display: inline-flex'), 'menu button appears on a phone');
  assert(rule(mobile, '.jb-drawer-close').includes('display: inline-flex'), 'close button appears on a phone');
}

// ------------------------------------------------------------------ CSS ----

check('the breakpoint rules: a column on desktop, a hidden drawer on a phone', () => {
  assertResponsiveRules(css);
});

check('teeth: a closed drawer that stays focusable is rejected', () => {
  assert.throws(() => assertResponsiveRules(css.replace('visibility: hidden;', '')));
});

check('teeth: mobile controls leaking onto desktop are rejected', () => {
  assert.throws(() => assertResponsiveRules(css.replace(/\.jb-menu-button,\s*\.jb-drawer-close,\s*\.jb-backdrop\s*\{\s*display: none;\s*\}/, '')));
});

check('nothing re-asserts a viewport unit that would fight the fixed-height app', () => {
  assert(!css.includes('100vh') && !css.includes('100dvh'));
});

// ------------------------------------------------------------------ page ----

// With no conversation selected (the server render), the page shows its placeholder.
const page = renderToStaticMarkup(<JobBuddyDemo />);
const asideTag = page.match(/<aside[^>]*>/)?.[0] ?? '';

check('one sidebar element, addressable by the menu buttons, closed by default', () => {
  assert.equal((page.match(/<aside/g) ?? []).length, 1, 'never a second copy of the sidebar');
  assert(asideTag.includes('id="jobbuddy-sidebar"') && asideTag.includes('class="jb-sidebar"'));
  assert(asideTag.includes('data-open="false"'));
  assert(asideTag.includes('aria-label="Progress, resume and conversations"'));
});

check('the sidebar width is not inline, so the phone breakpoint can override it', () => {
  const style = asideTag.match(/style="([^"]*)"/)?.[1] ?? '';
  assert(!/(^|;)\s*(min-|max-)?width:/.test(style), `inline width found: ${style}`);
  assert(style.includes('overflow-y:auto'), 'the sidebar still scrolls internally');
});

check('the existing sidebar content lives inside the drawer element', () => {
  const inside = page.slice(page.indexOf('<aside'), page.indexOf('</aside>'));
  for (const text of ['Your Progress', 'Career Goal', 'Action Plan', 'Start a new conversation', 'Recent Conversations']) {
    assert(inside.includes(text), text);
  }
});

check('the drawer has a labelled close button and a closed backdrop', () => {
  const inside = page.slice(page.indexOf('<aside'), page.indexOf('</aside>'));
  assert(/<button[^>]*class="jb-drawer-close"[^>]*aria-label="Close menu"/.test(inside));
  assert(/<div class="jb-backdrop" data-open="false" aria-hidden="true">/.test(page));
});

check('with no conversation, the placeholder still offers a way into the drawer', () => {
  const main = page.slice(page.indexOf('<main'), page.indexOf('</main>'));
  const button = main.match(/<button[^>]*class="jb-menu-button"[^>]*>/)?.[0] ?? '';
  assert(button.includes('aria-controls="jobbuddy-sidebar"') && button.includes('aria-expanded="false"'));
  assert(button.includes('aria-label="Open menu: progress, resume and conversations"'));
});

check('menu buttons carry no inline display, so CSS alone decides where they show', () => {
  for (const button of page.match(/<button[^>]*class="jb-(menu-button|drawer-close)"[^>]*>/g) ?? []) {
    assert(!/style="[^"]*display:/.test(button), button);
  }
});

// -------------------------------------------------------------- ChatArea ----

function chat(props: Partial<React.ComponentProps<typeof ChatArea>> = {}) {
  return renderToStaticMarkup(
    <ChatArea selectedAgent="xbuddy" userId={7} mode="stream" threadId="t-1" loadedMessages={[]}
      currentSection={null} onThreadIdChange={noop} onSectionUpdate={noop} {...props} />
  );
}

check('the chat header offers the menu button when the page provides one', () => {
  const html = chat({ onOpenMenu: noop, menuOpen: false });
  const header = html.slice(0, html.indexOf('</h1>'));
  const button = header.match(/<button[^>]*class="jb-menu-button"[^>]*>/)?.[0] ?? '';
  assert(button, 'menu button precedes the title');
  assert(button.includes('aria-label="Open menu: progress, resume and conversations"'));
  assert(button.includes('aria-controls="jobbuddy-sidebar"') && button.includes('aria-expanded="false"'));
  assert(chat({ onOpenMenu: noop, menuOpen: true }).includes('aria-expanded="true"'));
});

check('without a menu handler the chat header renders no control', () => {
  assert(!chat().includes('jb-menu-button'));
});

console.log(`mobile layout: ${passed} passed, 0 failed`);
