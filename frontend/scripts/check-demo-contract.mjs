/**
 * Structural checks for the JobBuddy demo path.
 *
 *   node scripts/check-demo-contract.mjs
 *
 * The repo's only test setup is Cypress, which needs a running server and a live
 * backend — too heavy for "is the token on the wrong side of the network boundary".
 * These are the checks worth having at zero cost: they read source text, add no
 * dependency, and run in milliseconds.
 *
 * They assert properties, not formatting, so they survive reindentation.
 */

import { readFileSync, existsSync, readdirSync, statSync } from 'node:fs';
import { join, dirname, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
// Line endings are normalized on read. Without this the comment stripper below
// silently gives up on a CRLF working tree: the `.` in its `//.*$` excludes
// carriage returns, so on CRLF input entire comments survive stripping
// and every "this word must not appear in the source" check reads prose as code.
// Git hands out CRLF on Windows checkouts, so which branch you are on decided
// whether these checks meant anything.
const read = (p) => readFileSync(join(ROOT, p), 'utf8').replace(/\r\n/g, '\n');

/**
 * Source with comments stripped.
 *
 * Checks about what the code *does* must not be satisfied — or broken — by prose
 * describing what it deliberately avoids. A file explaining "database_id is not read"
 * contains the string "database_id", and a naive scan reads that as a violation.
 */
const BLOCK_COMMENT = new RegExp('/\\*[\\s\\S]*?\\*/', 'g');
const LINE_COMMENT = new RegExp('(^|[^:])//.*$');

const code = (p) =>
  read(p)
    .replace(BLOCK_COMMENT, ' ')
    .split('\n')
    .map((line) => line.replace(LINE_COMMENT, '$1'))
    .join('\n');

// Every file reachable from the demo screen. Legacy files outside this list are
// Phase 2 deletions and deliberately not audited here.
const SERVER_ROUTES = [
  'src/app/api/chat/route.ts',
  'src/app/api/history/route.ts',
  'src/app/api/completion/route.ts',
  'src/app/api/final-output/route.ts',
  'src/app/api/resume/route.ts',
  'src/app/api/resume/status/route.ts',
];
const SERVER_LIB = ['src/lib/jobbuddyApi.ts'];
const CLIENT_FILES = [
  'src/app/page.tsx',
  'src/components/ChatArea.tsx',
  'src/components/JobBuddyProgress.tsx',
  'src/utils/demoConversations.ts',
  'src/components/JobBuddyWelcome.tsx',
  'src/components/FinalPlanPanel.tsx',
  'src/components/ResumeCard.tsx',
  'src/components/ResumeDropZone.tsx',
  'src/utils/resume.ts',
];
const DEMO_PATH = [...SERVER_ROUTES, ...SERVER_LIB, ...CLIENT_FILES];

const results = [];
const MISSING = DEMO_PATH.filter((f) => !existsSync(join(ROOT, f)));
const PRESENT = DEMO_PATH.filter((f) => existsSync(join(ROOT, f)));
const check = (label, ok, detail = '') => {
  results.push({ label, ok: Boolean(ok), detail });
  const mark = ok ? 'PASS' : 'FAIL';
  console.log(`  [${mark}] ${label}${detail ? ` — ${detail}` : ''}`);
};

check(
  'every file on the demo path exists',
  MISSING.length === 0,
  MISSING.length ? `missing: ${MISSING.join(', ')}` : `${PRESENT.length} files`
);

// --------------------------------------------------------------- secrets ----

const clientSources = CLIENT_FILES.filter((f) => existsSync(join(ROOT, f))).map((f) => ({
  f,
  text: read(f),
}));

check(
  'no client file references JOBBUDDY_API_TOKEN',
  clientSources.every(({ text }) => !text.includes('JOBBUDDY_API_TOKEN')),
  clientSources.filter(({ text }) => text.includes('JOBBUDDY_API_TOKEN')).map((c) => c.f).join(', ')
);

check(
  'no client file builds an Authorization header',
  clientSources.every(({ text }) => !/Authorization/i.test(text)),
  clientSources.filter(({ text }) => /Authorization/i.test(text)).map((c) => c.f).join(', ')
);

// A NEXT_PUBLIC_ prefix means Next inlines the value into the browser bundle, so a
// secret behind that prefix is a published secret.
const publicVars = PRESENT.flatMap((f) => read(f).match(/NEXT_PUBLIC_[A-Z0-9_]+/g) ?? []);
const publicSecrets = publicVars.filter((v) => /TOKEN|SECRET|PASSWORD|KEY/.test(v));
check(
  'no NEXT_PUBLIC_ token/secret anywhere in the demo path',
  publicSecrets.length === 0,
  publicSecrets.length ? [...new Set(publicSecrets)].join(', ') : `${publicVars.length} non-secret public vars`
);

for (const route of SERVER_ROUTES) {
  if (!existsSync(join(ROOT, route))) {
    check(`${route} exists`, false, 'the route file is missing');
    continue;
  }
  const text = read(route);
  check(
    `${route} owns the Authorization header`,
    // The call site, not the import. Matching `jobbuddyHeaders` anywhere in the file
    // stayed green when the outbound fetch was switched back to a bare
    // Content-Type header while the import sat unused at the top. The multipart
    // upload uses the token-only variant, so fetch can set the boundary itself.
    text.includes('headers: jobbuddyHeaders(),') || text.includes('headers: jobbuddyAuthHeaders(),'),
    'the outbound request attaches it'
  );
}

check(
  'the token helper is marked server-only',
  read('src/lib/jobbuddyApi.ts').includes("import \"server-only\"") ||
    read('src/lib/jobbuddyApi.ts').includes("import 'server-only'"),
  'importing it from a client component becomes a build error'
);

// ----------------------------------------------------------------- agent ----

const chat = read('src/app/api/chat/route.ts');
check('chat route default agent is xbuddy', read('src/lib/jobbuddyApi.ts').includes("DEFAULT_AGENT_ID = \"xbuddy\""));
check('chat route uses DEFAULT_AGENT_ID, not a literal', chat.includes('agentId = DEFAULT_AGENT_ID'));
check(
  'no FounderBuddy / value-canvas agent id in the demo path',
  PRESENT.every((f) => !/value-canvas|founder[-_ ]?buddy/i.test(code(f))),
  PRESENT.filter((f) => /value-canvas|founder[-_ ]?buddy/i.test(code(f))).join(', ')
);

// ------------------------------------------------------------- streaming ----

check('chat route pins the Node runtime', /export const runtime = 'nodejs'/.test(chat));
check('chat route is force-dynamic', /export const dynamic = 'force-dynamic'/.test(chat));
check('chat route sends text/event-stream', chat.includes("'Content-Type': 'text/event-stream'"));
check('chat route disables transformation', chat.includes('no-transform'));
check('chat route disables proxy buffering', chat.includes("'X-Accel-Buffering': 'no'"));
check(
  'chat route still uses POST fetch streaming, not EventSource',
  chat.includes('getReader()') && !chat.includes('new EventSource')
);

const chatArea = read('src/components/ChatArea.tsx');
for (const event of ['metadata', 'token', 'message', 'section', 'completion']) {
  check(`ChatArea handles the '${event}' SSE event`, chatArea.includes(`parsed.type === '${event}'`));
}
check("ChatArea terminates on [DONE]", chatArea.includes("=== '[DONE]'"));
// A literal regex, not the `re()` helper: that helper is declared further down the
// file, and calling it here hits the temporal dead zone.
check(
  'the completion event updates progress state',
  chatArea.includes('onCompletionUpdate?.(parsed.content as CompletionState)') &&
    /onCompletionUpdate=\{[^]{0,200}setCompletion\(next\)/.test(code('src/app/page.tsx'))
);
check(
  'SSE frames split across chunks are carried, not dropped',
  chatArea.includes('carry + decoder.decode') && chatArea.includes('carry = lines.pop()')
);

// -------------------------------------------------------------- sections ----

const progress = read('src/components/JobBuddyProgress.tsx');
check(
  'section rendering is data-driven',
  progress.includes('completion?.sections') && progress.includes('sections.map('),
  'renders the API array'
);
check(
  'no hardcoded section list in the demo path',
  PRESENT.every((f) => !/(mission|team_traction|invest_plan)/.test(code(f))) &&
    !/'career_goal'|"career_goal"/.test(code('src/components/JobBuddyProgress.tsx')),
  'neither FounderBuddy nor JobBuddy ids are duplicated client-side'
);
check(
  'progress panel does not read database_id',
  !code('src/components/JobBuddyProgress.tsx').includes('database_id')
);

// --------------------------------------------------------------- history ----

const history = read('src/app/api/history/route.ts');
const page = read('src/app/page.tsx');
check('history proxy requires user_id', /typeof user_id !== ['"]number['"]/.test(history));
check('history proxy forwards both identifiers', history.includes('JSON.stringify({ thread_id, user_id })'));
// A flat invariant, not a proximity window. The first version of this check looked
// for `.reverse(` within 200 characters of "messages", and a comment between the two
// was enough to push it out of range — it passed while the transcript was reversed.
// The page has no legitimate reason to reorder anything, so forbid it outright.
check(
  'history restoration preserves backend order',
  !/\.(reverse|sort)\(/.test(code('src/app/page.tsx')),
  'the page never reorders the transcript'
);
check(
  'history restoration maps human->user and ai->assistant',
  page.includes("m.type === 'human' ? ('user' as const) : ('assistant' as const)")
);
check('a failed restore surfaces an error state', page.includes("setRestoreState('error')"));

// -------------------------------------------------- forbidden dependencies ----

const DELETED_ENDPOINTS = ['/sync_section', '/refine_section', '/section_states', '/business_plan', '/generate_business_plan'];
for (const endpoint of DELETED_ENDPOINTS) {
  check(
    `demo path never calls ${endpoint}`,
    PRESENT.every((f) => !read(f).includes(endpoint))
  );
}

check(
  'demo path never touches Supabase',
  PRESENT.every((f) => {
    const t = read(f);
    return !t.includes('@supabase/supabase-js') && !t.includes("from('section_states')") &&
      !t.includes("from('final-outputs')") && !t.includes('@/lib/supabase');
  })
);

check(
  'the root page does not import the legacy Supabase-backed components',
  !/SectionDisplayPanel|BusinessPlanEditor|ProgressSidebar|ConversationHistory|ConfigPanel/.test(page)
);

check(
  'no fabricated final output is rendered',
  code('src/components/JobBuddyProgress.tsx').includes('onViewFinalPlan') &&
    !code('src/components/JobBuddyProgress.tsx').includes('final_output') &&
    code('src/components/FinalPlanPanel.tsx').includes('{plan}'),
  'the panel renders the backend document; the sidebar only opens it'
);

check('the history route exists', existsSync(join(ROOT, 'src/app/api/history/route.ts')));

// ------------------------------------------- Phase 1.1: one bubble, real streaming ----

const chatAreaCode = code('src/components/ChatArea.tsx');
const chatScrollCode = code('src/utils/chatScroll.ts');

check(
  'no standalone Thinking/Processing card',
  !chatAreaCode.includes("'Thinking...'") && !chatAreaCode.includes("'Processing...'"),
  'the loading state lives inside the single assistant bubble'
);

check(
  'the in-bubble loader is scoped to an empty assistant message',
  chatAreaCode.includes("message.role === 'assistant' && !message.content && isLoading"),
  'so the first token replaces it in place'
);

check(
  'token updates are not computed inside a state updater',
  !/setStreamingContent\(\s*prev\s*=>/.test(chatAreaCode),
  'React runs updaters at render time, which collapsed every token into one paint'
);

check(
  'token content accumulates synchronously',
  chatAreaCode.includes('accumulated += parsed.content') &&
    /let accumulated = ''/.test(chatAreaCode)
);

check(
  'each token issues its own setMessages',
  /accumulated \+= parsed\.content;[\s\S]{0,200}setMessages\(/.test(chatAreaCode),
  'no batching barrier between arrival and paint'
);

check(
  'rendering is not deferred until [DONE]',
  !/\[DONE\][\s\S]{0,300}setMessages\(prevMessages =>[\s\S]{0,200}content: accumulated/.test(chatAreaCode),
  'tokens paint as they arrive'
);

check(
  "the final 'message' event does not create a second bubble",
  /parsed\.type === 'message'\)\s*\{\s*\}/.test(chatAreaCode.replace(/\s+/g, ' ')) ||
    !/parsed\.type === 'message'[\s\S]{0,300}setMessages\(prev =>\s*\[\.\.\.prev/.test(chatAreaCode),
  'it never appends'
);

// ---------------------------------------------- Phase 1.1: JobBuddy wording ----

const FORBIDDEN_TEXT = [
  'XBuddy', 'FounderBuddy', 'value-canvas', 'Value Canvas', 'startup', 'Validate and refine',
  'Auto Reply', 'Start Auto', 'Stop Auto', 'AUTO MODE', 'business plan', 'Business Plan',
  'mission', 'traction', 'invest_plan', 'investment plan',
];
for (const phrase of FORBIDDEN_TEXT) {
  check(
    `root demo contains no "${phrase}"`,
    PRESENT.every((f) => !code(f).includes(phrase))
  );
}

check('assistant label is JobBuddy', chatAreaCode.includes("getAgentName = (_agentId: string) => 'JobBuddy'"));
check(
  'input placeholder addresses career goals',
  chatAreaCode.includes('Message JobBuddy about your career goals...')
);
check('subtitle is the JobBuddy one', chatAreaCode.includes('AI Career Planning Agent'));
check(
  'the composer has no auto-mode controls',
  !chatAreaCode.includes('handleAutoReply(false)') && !chatAreaCode.includes('handleAutoReply(true)'),
  'only [input] [Send] remains'
);


// ------------------------------------------------ Phase 1.2: layout + refresh ----

const progressCode = code('src/components/JobBuddyProgress.tsx');
const pageCode = code('src/app/page.tsx');

// Regexes are built with `new RegExp` from plain strings so no backslash has to
// survive a shell round trip — a mangled literal silently becomes a check that
// cannot fail, which is worse than no check at all.
const re = (pattern) => new RegExp(pattern);

check(
  'the new-conversation button is not pinned to the viewport bottom',
  !progressCode.includes("marginTop: 'auto'") && !pageCode.includes("marginTop: 'auto'"),
  'it sits under the title with the other controls'
);
check(
  'the new-conversation button lives in the sidebar panel',
  progressCode.includes('onNewConversation') && progressCode.includes('Start a new conversation')
);
check(
  'starting a new conversation makes no backend call',
  !re('handleReset[^]{0,400}fetch[(]').test(pageCode),
  'identity is minted locally; the backend hears nothing until the user sends'
);

// Anchored on `fontFamily`, which only the root container carries. A looser pattern
// matched <main>'s own block and stayed green while the root lost overflow:hidden —
// exactly the state where the outer page scrolls again.
check(
  'the app owns the viewport and the page itself does not scroll',
  re("height: '100%',[^]{0,60}overflow: 'hidden',[^]{0,60}fontFamily").test(pageCode),
  'checked on the root container, not the inner panel'
);
check(
  'the right panel is a flex column that constrains its child',
  re("flexDirection: 'column',[^]{0,140}height: '100%',[^]{0,40}overflow: 'hidden'").test(pageCode)
);
check(
  'the chat root can shrink below its content',
  chatAreaCode.includes('minHeight: 0'),
  'without this a flex item grows the page instead of scrolling'
);
check(
  'the message area is the scroll container',
  re("flex: 1,[^]{0,40}minHeight: 0,[^]{0,40}overflowY: 'auto'").test(chatAreaCode)
);
check(
  'the composer is outside the scroll container and cannot shrink',
  re("flexShrink: 0,[^]{0,40}padding: '16px 24px'").test(chatAreaCode) &&
    !re("overflowY: 'auto'[^]{0,600}<form onSubmit=[{]handleSubmit[}]").test(chatAreaCode),
  'the composer wrapper itself is shrink-0; the list scrolls above it'
);
check('the composer has a stable minimum height', chatAreaCode.includes("minHeight: '46px'"));

check(
  'the assistant label and copy button cannot overlap',
  !chatAreaCode.includes("position: 'absolute'") &&
    re("justifyContent: 'space-between',[^]{0,120}minHeight: '20px'").test(chatAreaCode),
  'one flex row with a reserved height, not a button floating over the label'
);
check(
  'the copy button is shrink-0 with a gap from the label',
  re("gap: '12px'[^]{0,1200}flexShrink: 0").test(chatAreaCode)
);
check(
  'the header reserves height even when the bubble is empty',
  chatAreaCode.includes("minHeight: '20px'")
);
check(
  'the copy button is hidden while the bubble is still empty',
  chatAreaCode.includes("message.role === 'assistant' && message.content && (")
);

check('the completion proxy route exists', existsSync(join(ROOT, 'src/app/api/completion/route.ts')));
const completionRoute = read('src/app/api/completion/route.ts');
check('completion proxy owns the Authorization header', completionRoute.includes('jobbuddyHeaders'));
check(
  'completion proxy requires both identifiers',
  re("typeof user_id !== ['\"]number['\"]").test(completionRoute) &&
    completionRoute.includes('JSON.stringify({ thread_id, user_id })')
);
check(
  'completion proxy never reads Supabase',
  !completionRoute.includes('supabase') && !completionRoute.includes('final-outputs')
);

check(
  'refresh calls both history and completion',
  pageCode.includes("fetch('/api/history'") && pageCode.includes("fetch('/api/completion'")
);
check(
  'the two restore calls are issued together',
  pageCode.includes('Promise.all('),
  'one loading state instead of two sequential flashes'
);
check(
  'progress is never reconstructed from messages',
  !re('completionData[^]{0,300}(backendMessages|historyData)').test(pageCode) &&
    !re('setCompletion[(][^]{0,200}(messages|backendMessages)').test(pageCode),
  'the projection is read from the response, never derived from the transcript'
);
check(
  'completion restoration preserves the backend section order',
  !re('sections[^]{0,160}[.](reverse|sort)[(]').test(pageCode)
);
check(
  'a restore failure is visible rather than silent',
  pageCode.includes("setRestoreState('error')")
);


// ------------------------------------- Phase 1.3: viewport, sidebar, conversations ----

const globalsCss = read('src/app/globals.css');
// Comment-stripped: the file's own comment explains the v3 directives by name, and a
// raw scan would match that prose rather than a live directive.
const globalsCode = read('src/app/globals.css').replace(BLOCK_COMMENT, ' ');
const conversationsCode = code('src/utils/demoConversations.ts');
const progressCode13 = code('src/components/JobBuddyProgress.tsx');
const pageCode13 = code('src/app/page.tsx');

// The header was permanently clipped because Tailwind v4 was installed while this
// file still used the v3 directives, so Preflight never loaded and the UA default
// `body { margin: 8px }` applied. The document became 100vh+16px tall, and
// scrollIntoView dragged it — hiding the header at every scroll position.
check(
  'globals.css uses the Tailwind v4 import, not the v3 directives',
  globalsCode.includes('@import "tailwindcss"') && !globalsCode.includes('@tailwind base'),
  'the v3 directives are inert under v4, so Preflight never loads'
);
check(
  'html and body are explicitly reset and own no scrolling',
  re('html,[^]{0,20}body[^]{0,200}margin: 0').test(globalsCss) &&
    re('html,[^]{0,20}body[^]{0,220}overflow: hidden').test(globalsCss),
  'the demo owns the viewport; the document never scrolls'
);
check(
  'viewport ownership is singular — nothing re-asserts 100vh',
  !pageCode13.includes('100vh') && !chatAreaCode.includes('100vh'),
  'heights derive from html/body instead of stacking viewport units'
);
check(
  'auto-scroll cannot move the document',
  !chatAreaCode.includes('scrollIntoView') &&
    chatAreaCode.includes('messagesPaneRef.current') &&
    chatAreaCode.includes('pane.scrollTo'),
  'only the message pane is addressed; document ancestors are never scrolled'
);
check(
  'streamed tokens never restart smooth scrolling',
  chatScrollCode.includes("return isNearBottom ? 'auto' : null") &&
    (chatScrollCode.match(/return 'smooth'/g) ?? []).length === 1,
  'token growth follows directly only while the reader remains near the bottom'
);
check(
  'manual upward scrolling disables auto-follow',
  chatAreaCode.includes('onScroll={(event)') &&
    chatAreaCode.includes('isNearBottomRef.current = isNearChatBottom(pane)') &&
    chatAreaCode.includes("trigger === 'stream-token' && !isNearBottomRef.current"),
  'the live ref is checked again inside the animation frame'
);
check(
  'new user messages still receive one smooth reveal',
  chatAreaCode.includes("scheduleChatScroll('new-user-message')") &&
    chatScrollCode.includes("if (trigger === 'new-user-message') return 'smooth'")
);

// -------------------------------------------------------- sidebar hierarchy ----

const orderOf = (needle) => progressCode13.indexOf(needle);
const idxProgress = orderOf('Your Progress');
const idxNew = orderOf('Start a new conversation');
const idxRecent = orderOf('Recent Conversations');
const idxStatus = orderOf('Final Plan');
const idxDev = orderOf('Developer Details');

check(
  'Progress appears before the New Conversation control',
  idxProgress > -1 && idxNew > -1 && idxProgress < idxNew,
  'not directly under the product title'
);
check(
  'New Conversation appears before Recent Conversations',
  idxNew > -1 && idxRecent > -1 && idxNew < idxRecent
);
check(
  'completion stays with progress and developer details stay last',
  idxProgress < idxStatus && idxStatus < idxNew && idxRecent < idxDev
);
check(
  'the new-conversation button is not bottom-pinned',
  !progressCode13.includes("marginTop: 'auto'") && !pageCode13.includes("marginTop: 'auto'")
);
check(
  'the current conversation is identifiable',
  progressCode13.includes('aria-current') && progressCode13.includes('conversation.threadId === threadId')
);
// Anchored on the opening tag and scoped to the aside's own style block. A looser
// window also matched the closing `</aside>` and ran on into <main>'s
// `overflow: 'hidden'`, so the check failed against a sidebar that was already correct.
const asideStyle = (pageCode13.split('<aside')[1] ?? '').split('>')[0];
check(
  'the sidebar scrolls rather than clipping its own overflow',
  asideStyle.includes("overflowY: 'auto'") && !asideStyle.includes("overflow: 'hidden'"),
  'a fixed-height sidebar could only hide the content below the fold'
);

// ------------------------------------------------------ conversation storage ----

check(
  'conversation records store no transcript or graph state',
  !conversationsCode.includes('messages') &&
    !conversationsCode.includes('section_states') &&
    !conversationsCode.includes('sections'),
  'identifiers, a label and timestamps only'
);
check(
  'conversation labels need no model call',
  !conversationsCode.includes('fetch(') &&
    !re('/api/|openai\\.|new OpenAI|anthropic').test(conversationsCode),
  'deterministic truncation of the first user message'
);
check(
  'switching a conversation restores through history + completion',
  re('handleSelectConversation[^]{0,600}restore[(]').test(pageCode13) &&
    pageCode13.includes("fetch('/api/history'") &&
    pageCode13.includes("fetch('/api/completion'")
);
check(
  'switching a conversation never calls the chat endpoint',
  !re('handleSelectConversation[^]{0,600}/api/chat').test(pageCode13)
);
check(
  'creating a conversation never calls the chat endpoint',
  !re('handleNewConversation[^]{0,500}fetch[(]').test(pageCode13),
  'no automatic greeting, so no model call'
);
check(
  'switching remounts the chat so no transcript bleeds across threads',
  pageCode13.includes('key={identity.threadId}')
);

// ------------------------------------------------------------------ chat ----

check(
  'Copy All copies only the visible transcript',
  chatAreaCode.includes('copyAllConversation') &&
    !re('copyAllConversation[^]{0,400}(thread_id|threadId|userId|collection_complete|sections)').test(
      chatAreaCode
    ),
  'no identifiers, progress or internal events'
);


// ------------------------- Phase 1.4: active thread, deletion, titles, onboarding ----

const convCode = code('src/utils/demoConversations.ts');
const page14 = code('src/app/page.tsx');
const prog14 = code('src/components/JobBuddyProgress.tsx');
const welcomeCode = code('src/components/JobBuddyWelcome.tsx');
const chat14 = code('src/components/ChatArea.tsx');

// ------------------------------------------------------------ active thread ----

check(
  'the active conversation is persisted separately from the list',
  convCode.includes("ACTIVE_KEY = 'jobbuddy_demo_active_thread'") &&
    convCode.includes('export function setActiveThreadId')
);
check(
  'load resolves the persisted active thread, not simply the newest',
  convCode.includes('export function resolveActiveIdentity') &&
    re('resolveActiveIdentity[^]{0,500}list.find[(][(]c[)] => c.threadId === activeId[)]').test(
      convCode
    ) &&
    re('if [(]active[)] return').test(convCode) &&
    page14.includes('resolveActiveIdentity()'),
  'the active id must actually select the record, not merely be read'
);
check(
  'an unknown active id falls back to the most recent record',
  re('resolveActiveIdentity[^]{0,700}list.length > 0[^]{0,200}setActiveThreadId').test(convCode)
);
check(
  'an empty list stays empty on reload',
  !convCode.slice(convCode.indexOf('export function resolveActiveIdentity')).includes('mintIdentity()') &&
    convCode.slice(convCode.indexOf('export function resolveActiveIdentity')).includes('return null;')
);
check(
  'selecting a conversation persists the new active thread',
  re('selectLocally[^]{0,200}setActiveThreadId').test(page14) &&
    re('handleSelectConversation[^]{0,300}selectLocally').test(page14)
);
check(
  'creating a conversation persists the new active thread',
  re('handleNewConversation[^]{0,400}selectLocally').test(page14)
);

// ---------------------------------------------------------------- deletion ----

check(
  'each conversation row has a delete control',
  prog14.includes('onDeleteConversation') && prog14.includes('aria-label={"Delete conversation'),
);
check(
  'the delete click cannot trigger row selection',
  re('onDeleteConversation[^]{0,600}stopPropagation').test(prog14) ||
    re('stopPropagation[^]{0,300}onDeleteConversation').test(prog14)
);
check(
  'deleting an inactive conversation does not change the selection',
  re('handleDeleteConversation[^]{0,900}conversation.threadId !== identity[?].threadId[^]{0,60}return').test(
    page14
  ),
  'it returns before touching the active identity'
);
check(
  'deleting the active conversation selects an existing fallback or nothing',
  re('handleDeleteConversation[^]{0,1200}remaining.length > 0[^]{0,300}restore[(]').test(page14) &&
    re('handleDeleteConversation[^]{0,1600}selectLocally[(]null[)]').test(page14),
  'another record if one remains, otherwise no conversation'
);
check(
  'deletion never calls the chat endpoint',
  !re('handleDeleteConversation[^]{0,1600}/api/chat').test(page14)
);
check(
  'deletion is local only and says so',
  read('src/utils/demoConversations.ts').includes('backend checkpoint'),
  'the Postgres checkpoint still exists; the record is only unreachable from here'
);

// ------------------------------------------------------------------ titles ----

check(
  'titles are generated deterministically with no network call',
  convCode.includes('export function titleFromFirstMessage') &&
    !re('titleFromFirstMessage[^]{0,1400}(fetch[(]|/api/|openai|llm)').test(convCode)
);
check(
  'titles are length-bounded',
  convCode.includes('TITLE_MAX') && re('function bound[^]{0,200}TITLE_MAX').test(convCode)
);
check(
  'the raw first message is not used verbatim as a title',
  re('for [(]const prefix of FILLER_PREFIXES[)]').test(convCode) &&
    re('words.filter[(][(]w[)] => !STOP_WORDS.has[(]w[)][)]').test(convCode),
  'both lists must be applied, not merely declared'
);
check(
  'a title is not rewritten once assigned',
  re('rememberConversation[^]{0,700}record.label === FALLBACK_TITLE').test(convCode),
  'so a conversation does not rename itself as it grows'
);
check(
  'there is a fallback title for an unusable first message',
  convCode.includes("FALLBACK_TITLE = 'New Career Conversation'")
);
check(
  'the first user message updates conversation metadata immediately',
  chatAreaCode.includes('onFirstUserMessage?.(userMessage.content)') &&
    pageCode13.includes('onFirstUserMessage={handleFirstUserMessage}') &&
    re('handleFirstUserMessage[^]{0,500}titleFromFirstMessage[(]content[)][^]{0,250}setConversations[(]listConversations[(][)][)]').test(pageCode13),
  'ChatArea reports the event; the parent that owns metadata updates the list'
);
check(
  'first-message title timing adds no model or network call',
  !re('handleFirstUserMessage[^]{0,700}(fetch[(]|/api/|openai|llm)').test(pageCode13)
);
check(
  'ChatArea does not own localStorage',
  !chatAreaCode.includes('localStorage'),
  'conversation metadata remains owned by the parent utility boundary'
);

// -------------------------------------------------------------- onboarding ----

check('a dedicated onboarding component exists', existsSync(join(ROOT, 'src/components/JobBuddyWelcome.tsx')));
check(
  'onboarding names all five JobBuddy areas',
  ['Career Goal', 'Background', 'Job Preferences', 'Skill Assessment', 'Action Plan'].every((a) =>
    welcomeCode.includes(a)
  )
);
check(
  'onboarding makes no network or model call',
  !re('fetch[(]|/api/|openai').test(welcomeCode)
);
check(
  'onboarding is a rendered component, never an entry in messages',
  !re('setMessages|messages.push|role: .assistant').test(welcomeCode) &&
    !re('JobBuddyWelcome[^]{0,200}setMessages').test(chat14),
  'so it cannot be mistaken for assistant output or copied as one'
);
check(
  'onboarding shows only while the thread has no messages',
  re('messages.length === 0 [?][^]{0,120}<JobBuddyWelcome').test(chat14),
  'a restored conversation shows its transcript instead'
);

// ------------------------------------------------------ progress honesty ----

const roadmapBlock = (prog14.split('ROADMAP.map(')[1] ?? '').slice(0, 700);
check(
  'the pre-first-turn roadmap is not presented as progress',
  prog14.includes('const ROADMAP') &&
    roadmapBlock.length > 0 &&
    !re('STATUS_LABEL|STATUS_COLOR|in_progress').test(roadmapBlock),
  'neutral names with no status label until the backend reports one'
);
check(
  'real statuses still come only from the backend projection',
  prog14.includes('completion?.sections')
);

// ----------------------------------------------------------- recent list ----

check(
  'the recent list has a multi-row minimum height',
  re("overflowY: 'auto',[^]{0,320}minHeight: 1[0-9][0-9]").test(prog14),
  'it previously collapsed to a single row'
);

// --------------------------------------------------------------- storage ----

check(
  'no transcript, progress or API response is persisted locally',
  !re('(messages|sections|collection_complete|artifact_available|section_states)').test(convCode),
  'the backend owns all of it'
);


// ------------------------------------- Phase 1.5: bounded list, scrollable sidebar ----

const prog15 = code('src/components/JobBuddyProgress.tsx');
const page15 = code('src/app/page.tsx');

check(
  'the sidebar can shrink so that it scrolls instead of overflowing',
  re("overflowY: 'auto',[^]{0,300}minHeight: 0").test(page15),
  'without minHeight: 0 a flex child pushes its overflow outside the box'
);
check(
  'the document still owns no scrolling',
  re("height: '100%',[^]{0,60}overflow: 'hidden',[^]{0,60}fontFamily").test(page15) &&
    read('src/app/globals.css').includes('overflow: hidden'),
  'root and html/body remain fixed to the viewport'
);
check(
  'the conversation list has a bounded height contract',
  prog15.includes('minHeight: 114') && re("maxHeight: 'clamp[(]").test(prog15),
  'a deliberate window, roughly three to five rows'
);
check(
  'the conversation list scrolls internally',
  re("overflowY: 'auto',[^]{0,400}maxHeight: 'clamp[(]").test(prog15)
);
check(
  'the conversation list cannot grow without limit',
  !re("overflowY: 'auto',[^]{0,400}flex: 1").test(prog15),
  'unrestricted growth is what let it consume the space the status blocks needed'
);
check(
  'the progress panel does not compete for height with the sidebar',
  !re("flexDirection: 'column', gap: 18, minHeight: 0, flex: 1").test(prog15),
  'normal flow, so each block lands after the one before it'
);

// Workflow completion belongs inside progress, before navigation/history.
const idxList15 = prog15.indexOf('Recent Conversations');
const idxStatus15 = prog15.indexOf('Final Plan');
const idxDev15 = prog15.indexOf('Developer Details');
const idxNew15 = prog15.indexOf('Start a new conversation');
const idxProg15 = prog15.indexOf('Your Progress');
const idxCollection15 = prog15.indexOf('label="Collection"');
const idxView15 = prog15.indexOf('View Final Plan');

check(
  'completion and final-plan action precede navigation; developer details stay last',
  idxCollection15 > idxProg15 && idxCollection15 < idxStatus15 && idxStatus15 < idxView15 &&
    idxView15 < idxNew15 && idxList15 < idxDev15
);
check(
  'progress including completion precedes new conversation and recent conversations',
  idxProg15 < idxNew15 && idxNew15 < idxList15,
  'progress, then new conversation, then recent conversations'
);
check(
  'the sidebar uses no absolute or fixed positioning',
  !re("position: '(absolute|fixed)'").test(prog15) && !re("position: '(absolute|fixed)'").test(page15)
);
check(
  'no negative margins are used to place sidebar blocks',
  !re('margin[A-Za-z]*: -').test(prog15)
);
check(
  'there are exactly three intentional scroll regions',
  [
    re("aside[^]{0,700}overflowY: 'auto'").test(page15),
    re("maxHeight: 'clamp[(]").test(prog15),
    re("flex: 1,[^]{0,40}minHeight: 0,[^]{0,40}overflowY: 'auto'").test(chatAreaCode),
  ].every(Boolean),
  'sidebar, conversation list, chat messages'
);


// ---------------------------------------------- Phase 2: legacy cleanup audit ----

/** Every source file under src/, so nothing escapes the audit by not being listed. */
function allSources(dir = join(ROOT, 'src')) {
  return readdirSync(dir).flatMap((entry) => {
    const full = join(dir, entry);
    return statSync(full).isDirectory() ? allSources(full) : full;
  });
}

const SOURCES = allSources().filter((f) => /\.(ts|tsx)$/.test(f));
const SOURCE_REL = SOURCES.map((f) => relative(ROOT, f).split('\\').join('/'));
const sourceCode = (f) => code(f);

// Product/domain terms only. Generic English words are excluded deliberately: a check
// that fails on the word "mission" in unrelated prose is noise, not a guard.
const LEGACY_TERMS = [
  'FounderBuddy', 'Founder Buddy', 'founder-buddy', 'value-canvas', 'Value Canvas',
  'team_traction', 'invest_plan', 'business_plan', 'BusinessPlan', 'startup idea',
  'sync_section', 'refine_section', 'section_states', 'final-outputs',
  'VALUE_CANVAS_API_TOKEN', 'VALUE_CANVAS_API_URL_PRODUCTION', 'VALUE_CANVAS_API_URL_LOCAL',
];

for (const term of LEGACY_TERMS) {
  const hits = SOURCE_REL.filter((f) => sourceCode(f).includes(term));
  check(`no legacy term "${term}" in frontend source`, hits.length === 0, hits.join(', '));
}

check(
  'no deleted backend endpoint is referenced anywhere in src',
  SOURCE_REL.every((f) => {
    const c = sourceCode(f);
    return !['/sync_section', '/refine_section', '/section_states', '/business_plan', '/generate_business_plan']
      .some((e) => c.includes(e));
  })
);

check(
  'no browser Supabase client remains',
  SOURCE_REL.every((f) => {
    const c = sourceCode(f);
    return !c.includes('@supabase/supabase-js') && !c.includes('@/lib/supabase');
  }),
  'all persistence goes through the backend'
);

check(
  'no stale link to a deleted page remains',
  SOURCE_REL.every((f) => {
    const c = sourceCode(f);
    return !c.includes('/business-plan') && !re("['\"]/section/").test(c);
  })
);

check(
  'every import in src resolves to a file that exists',
  (() => {
    const missing = [];
    for (const abs of SOURCES) {
      const text = readFileSync(abs, 'utf8');
      for (const m of text.matchAll(/from\s+['"](@\/[^'"]+)['"]/g)) {
        const target = join(ROOT, 'src', m[1].slice(2));
        const found = ['', '.ts', '.tsx', '/index.ts', '/index.tsx'].some((ext) =>
          existsSync(target + ext)
        );
        if (!found) missing.push(`${relative(ROOT, abs)} -> ${m[1]}`);
      }
    }
    if (missing.length) check.lastMissing = missing;
    return missing.length === 0;
  })(),
  'no import points at a deleted file'
);

check(
  'no client component imports the server-only token helper',
  SOURCE_REL.filter((f) => sourceCode(f).includes("'use client'")).every(
    (f) => !sourceCode(f).includes('@/lib/jobbuddyApi')
  )
);

// The user-facing surface is one page plus the three server routes.
const routeFiles = SOURCE_REL.filter((f) => /\/(page|route)\.tsx?$/.test(f)).sort();
check(
  'only the expected routes remain',
  JSON.stringify(routeFiles) ===
    JSON.stringify([
      'src/app/api/chat/route.ts',
      'src/app/api/completion/route.ts',
      'src/app/api/final-output/route.ts',
      'src/app/api/history/route.ts',
      'src/app/api/resume/route.ts',
      'src/app/api/resume/status/route.ts',
      'src/app/page.tsx',
    ]),
  routeFiles.join(', ')
);

check(
  'no transcript cache is written to localStorage anywhere',
  SOURCE_REL.every((f) => {
    const c = sourceCode(f);
    return !(c.includes('localStorage') && re('messages').test(c));
  }),
  'the checkpoint owns the transcript'
);

// package.json: the removed dependencies must stay removed.
const pkg = JSON.parse(read('package.json'));
const allDeps = { ...pkg.dependencies, ...pkg.devDependencies };
for (const dep of [
  '@supabase/supabase-js', '@tiptap/react', '@tiptap/starter-kit',
  '@tiptap/extension-placeholder', 'class-variance-authority', 'clsx',
  'tailwind-merge', 'lucide-react', 'openai', 'ai', '@ai-sdk/openai',
]) {
  check(`dependency "${dep}" stays removed`, !allDeps[dep]);
}
check('react-markdown is retained (the chat renders assistant markdown)', Boolean(allDeps['react-markdown']));

// .env.example must describe the current contract only.
const envExample = read('.env.example');
check(
  'env example names only the JobBuddy server vars',
  envExample.includes('JOBBUDDY_API_URL') &&
    envExample.includes('JOBBUDDY_API_TOKEN') &&
    !envExample.includes('VALUE_CANVAS') &&
    !envExample.includes('NEXT_PUBLIC_API_URL') &&
    !envExample.includes('SUPABASE')
);
check(
  'no NEXT_PUBLIC token or secret in the env example',
  !re('NEXT_PUBLIC_[A-Z_]*(TOKEN|SECRET|KEY)').test(envExample)
);


// ------------------------------------------ Phase 3: final-output retrieval ----

const page3 = code('src/app/page.tsx');
const prog3 = code('src/components/JobBuddyProgress.tsx');
const panel3 = code('src/components/FinalPlanPanel.tsx');

check('the final-output proxy route exists', existsSync(join(ROOT, 'src/app/api/final-output/route.ts')));

// Guarded: if the route is gone, the checks below must report that rather than
// crashing the whole run on a missing file.
const finalRouteExists = existsSync(join(ROOT, 'src/app/api/final-output/route.ts'));
const finalRoute = finalRouteExists ? read('src/app/api/final-output/route.ts') : '';
check(
  'the final-output proxy owns the Authorization header',
  finalRoute.includes('headers: jobbuddyHeaders(),')
);
check(
  'the final-output proxy requires both identifiers',
  re("typeof user_id !== ['\"]number['\"]").test(finalRoute) &&
    finalRoute.includes('JSON.stringify({ thread_id, user_id })')
);
check(
  'the final-output proxy never touches Supabase',
  finalRouteExists &&
    !code('src/app/api/final-output/route.ts').includes('supabase') &&
    !code('src/app/api/final-output/route.ts').includes('final-outputs')
);

check(
  'the plan is fetched only when the backend says one exists',
  re('completionData[?]\\.artifact_available[^]{0,80}loadFinalPlan').test(page3),
  'a thread without an artifact issues no request'
);
check(
  'the streamed completion event triggers a single fetch',
  re('next.artifact_available && !finalPlan[^]{0,80}loadFinalPlan').test(page3)
);
check(
  'the plan is never retrieved through the chat endpoint',
  !re('loadFinalPlan[^]{0,600}/api/chat').test(page3)
);
check(
  'retrieval never claims a plan the backend did not return',
  page3.includes('data.artifact_available ? data.final_output : null'),
  'availability from the response, not from the earlier completion'
);

check(
  'the final plan is never persisted locally',
  !re('localStorage[^]{0,200}(finalPlan|final_output|artifact)').test(page3) &&
    !code('src/utils/demoConversations.ts').includes('final_output'),
  'the backend stays the source of truth'
);

check(
  'switching a conversation clears the previous plan before restoring',
  re('selectLocally[^]{0,400}setFinalPlan\\(null\\)').test(page3),
  'so one thread\\u2019s plan cannot flash inside another'
);
check(
  'creating a conversation clears the plan and fetches nothing',
  re('handleNewConversation[^]{0,400}selectLocally').test(page3) &&
    !re('handleNewConversation[^]{0,500}fetch[(]').test(page3)
);

check(
  'the plan panel is read-only',
  panel3.includes('ReactMarkdown') &&
    !re('(contentEditable|<textarea|<input|onChange|useEditor)').test(panel3),
  'no editor, no regenerate, no export'
);
check(
  'the plan panel renders only what the backend returned',
  re('[{]plan[}]').test(panel3) && !re('(Career Direction|Positioning|Skill Priorities)').test(panel3),
  'section names come from the artifact, not from the component'
);
check(
  'a failed retrieval surfaces a retry rather than a fabricated plan',
  prog3.includes('onClick={onRetryFinalPlan}') && prog3.includes('could not be loaded'),
  'the button must be wired to the retry, not merely named in the props'
);
check(
  'the retry path calls only the final-output endpoint',
  re('onRetryFinalPlan[^]{0,200}loadFinalPlan').test(page3) &&
    !re('onRetryFinalPlan[^]{0,300}/api/chat').test(page3)
);
check(
  'View Final Plan appears only once the plan is actually loaded',
  prog3.includes('completion?.artifact_available && finalPlanReady'),
  'not merely when availability was announced'
);

check(
  'availability remains backend-driven',
  prog3.includes('completion?.artifact_available') &&
    !re('artifact_available = true').test(prog3)
);
check(
  'the internal `finished` flag stays out of the frontend',
  SOURCE_REL.every((f) => !re('\\bfinished\\b').test(sourceCode(f)))
);


// ------------------------------------------------- live / history parity ----
//
// A turn can persist more than one assistant message. `implementation` appends a
// readiness line with no model call behind it, so it arrives as a `message` event
// and never as tokens. ChatArea used to discard every `message` event outright,
// which meant that line showed up only after F5 — the transcript changed under the
// user on refresh.
//
// These read the streaming branch of ChatArea. They can prove the handler is wired
// and deduped; they cannot prove what renders. The manual checklist covers that.

const chatAreaParity = sourceCode('src/components/ChatArea.tsx');
const streamRule = sourceCode('src/utils/chatStream.ts');
const messageBranch = (() => {
  const start = chatAreaParity.indexOf("parsed.type === 'message'");
  if (start === -1) return '';
  const next = chatAreaParity.indexOf("parsed.type === 'section'", start);
  return chatAreaParity.slice(start, next === -1 ? chatAreaParity.length : next);
})();

check(
  'ChatArea handles the `message` SSE event at all',
  messageBranch.length > 0
);

check(
  'the decision lives in a rule a probe can run',
  re('planMessageEvent').test(messageBranch) && re('planMessageEvent').test(streamRule),
  'inline in the component it could only be checked by reading the source'
);

check(
  'both outcomes are wired up',
  re("'fill-placeholder'").test(messageBranch) && re("'append-bubble'").test(messageBranch),
  'an unhandled outcome silently drops the message again'
);

check(
  'an extra assistant message gets its own bubble id',
  re('extra-\\$\\{extraBubbleCount\\}').test(messageBranch),
  'reusing the streamed bubble id would overwrite the reply'
);

check(
  'the rule drops a message already streamed as tokens',
  re('accumulated').test(streamRule),
  'without this the reply appears twice: once from tokens, once from the event'
);

check(
  'only assistant messages become bubbles',
  re("!== 'ai'").test(streamRule)
);

// ------------------------------------------------------ title heuristics ----

const titleRules = sourceCode('src/utils/demoConversations.ts');

check(
  'the title rules cut at a qualifying clause',
  re('QUALIFIER_BOUNDARIES').test(titleRules) && re('focused on').test(titleRules),
  '"...role focused on X" used to title as "Targeting AI Engineer Role Focused"'
);

check(
  'a title cannot end on a dangling participle',
  re('DANGLING_TAIL').test(titleRules)
);

check(
  'titles still need no model call',
  !re('fetch\\(|/api/').test(titleRules),
  'naming a conversation is deterministic and local by design'
);

// ------------------------------------------------------ Phase 8: resume RAG ----
//
// Source checks for wiring a probe cannot see. The proxies' runtime behaviour —
// token attached, metadata only, errors filtered — is measured by
// scripts/resume-proxy-probe.mjs; the thread-scoping rule by scripts/resume-probe.ts.

const uploadRoute = code('src/app/api/resume/route.ts');
const statusRoute = code('src/app/api/resume/status/route.ts');
const page8 = code('src/app/page.tsx');
const lib8 = code('src/lib/jobbuddyApi.ts');
const resumeRules = code('src/utils/resume.ts');
const resumeCard = code('src/components/ResumeCard.tsx');

check(
  'the upload proxy forwards to /resume with the token-only header',
  uploadRoute.includes('/resume`') && uploadRoute.includes('headers: jobbuddyAuthHeaders(),') &&
    !uploadRoute.includes('Content-Type'),
  'a hand-set Content-Type would drop the multipart boundary'
);
check(
  'the token-only helper lives in the server-only module',
  re('export function jobbuddyAuthHeaders').test(lib8) &&
    re('jobbuddyHeaders[^]{0,200}\\.\\.\\.jobbuddyAuthHeaders\\(\\)').test(lib8),
  'one place builds the Authorization header'
);
check(
  'the status proxy forwards to /resume/status',
  statusRoute.includes('/resume/status`') && statusRoute.includes('headers: jobbuddyHeaders(),')
);
check(
  'neither resume proxy passes the backend body through wholesale',
  ![uploadRoute, statusRoute].some((t) => /NextResponse\.json\((await response\.json\(\)|data)\)/.test(t)) &&
    ![uploadRoute, statusRoute].some((t) => t.includes('document_id') || t.includes('candidate')),
  'metadata only: no document id, no candidate facts, no text'
);
check(
  'backend error text is relayed only for known user-facing codes',
  uploadRoute.includes('USER_FACING_CODES.has(detail.code)')
);
check(
  'a 404 status reads as no resume; other failures stay failures',
  re('status === 404[^]{0,120}has_resume: false').test(statusRoute) &&
    re('!response\\.ok[^]{0,400}status_unavailable').test(statusRoute)
);
check(
  'the resume status is restored with the conversation, outside its Promise.all',
  re('const restore = useCallback[^]{0,500}void loadResumeStatus\\(id\\)').test(page8) &&
    !re('Promise\\.all\\(\\[[^\\]]*resume').test(page8),
  'a resume status failure must not fail the restore'
);
check(
  'the resume view is scoped to the selected thread',
  page8.includes('resume={resumeForThread(resume, identity?.threadId ?? null)}') &&
    re('resume\\.threadId !== threadId').test(resumeRules) &&
    re('prev\\?\\.threadId === threadId').test(page8),
  'a view tagged with another thread is never shown or overwritten'
);
check(
  'switching conversations clears the resume view',
  re('const selectLocally[^]{0,900}setResume\\(null\\)').test(page8)
);
check(
  'a new conversation makes no resume request',
  !re('const handleNewConversation[^]{0,700}(fetch\\(|loadResumeStatus)').test(page8)
);
check(
  'the resume file goes only to /api/resume',
  re("fetch\\('/api/resume', \\{ method: 'POST', body: form \\}\\)").test(page8) &&
    !re('handleUploadResume[^]{0,2400}/api/chat').test(page8)
);
check(
  'late resume responses from a previous selection are dropped',
  ['const loadResumeStatus', 'const handleUploadResume'].every((fn) => {
    const start = page8.indexOf(fn);
    const body = start < 0 ? '' : page8.slice(start, start + 2600);
    // Both the success and the failure path of each request.
    return (body.match(/if \(version !== selectionVersion\.current\) return;/g) ?? []).length >= 2;
  })
);
check(
  'the card claims indexed only from backend metadata',
  resumeCard.includes("view.kind === 'indexed'") && !re("kind: 'indexed'").test(resumeCard)
);
check(
  'resume rules stay pure — no fetch, no storage',
  !re('fetch\\(|localStorage|sessionStorage').test(resumeRules)
);
check(
  'no resume data is stored in the browser',
  !re('localStorage[^\\n]*resume|resume[^\\n]*localStorage').test(page8)
);

// ------------------------------------------- welcome-card resume onboarding ----
//
// The welcome card is the primary upload surface and the sidebar its persistent
// status. Rendering and drag/drop rules are measured by welcome-resume-probe.tsx;
// these pin the wiring that keeps the two surfaces one state.

const dropZone = code('src/components/ResumeDropZone.tsx');
const chatResume = code('src/components/ChatArea.tsx');
const count = (text, pattern) => (text.match(pattern) ?? []).length;

check(
  'the welcome card offers the resume upload through handlers it is given',
  welcomeCode.includes('<ResumeDropZone') &&
    re('<JobBuddyWelcome[^]{0,300}onUploadResume=\\{onUploadResume\\}').test(chatResume)
);
check(
  'the welcome card and the sidebar receive the same resume view and upload handler',
  count(page8, /resume=\{resumeForThread\(resume, identity\?\.threadId \?\? null\)\}/g) === 2 &&
    count(page8, /onUploadResume=\{\(file\) => void handleUploadResume\(file\)\}/g) === 2,
  'one resume state, two surfaces — never a second copy that can drift'
);
check(
  'the drop zone makes no request and stores nothing',
  !re('fetch[(]|/api/|supabase|localStorage|sessionStorage|JOBBUDDY_API_TOKEN').test(dropZone)
);
check(
  'the drop zone claims indexed only from the view it is given',
  dropZone.includes("view.kind === 'indexed'") && !re("kind: 'indexed'").test(dropZone)
);
check(
  'every drag event on the drop zone cancels the browser default',
  ['onDragEnter', 'onDragOver', 'onDrop'].every((name) =>
    re(`${name}\\(event: DragEventLike\\) \\{\\s*event\\.preventDefault\\(\\);`).test(dropZone)
  ),
  'otherwise a dropped PDF opens in the tab instead of uploading'
);
check(
  'a file refused before upload lands in the shared resume state',
  re('const handleRejectResume[^]{0,600}updateResume\\(identity\\.threadId').test(page8)
);
check(
  'a resume upload never disables the chat input',
  !re('disabled=\\{[^}]*resume').test(chatResume)
);

// ----------------------------------------------------------------- report ----

const failed = results.filter((r) => !r.ok);
console.log(`\n${results.length - failed.length} passed, ${failed.length} failed`);
if (failed.length) {
  for (const f of failed) console.log(`  FAILED: ${f.label}`);
  process.exit(1);
}

