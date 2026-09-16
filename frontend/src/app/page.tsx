'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

import ChatArea, { type CompletionState, type Message } from '@/components/ChatArea';
import FinalPlanPanel from '@/components/FinalPlanPanel';
import JobBuddyProgress from '@/components/JobBuddyProgress';
import {
  deleteConversation,
  listConversations,
  mintIdentity,
  rememberConversation,
  resolveActiveIdentity,
  setActiveThreadId,
  titleFromFirstMessage,
  type DemoConversation,
  type DemoIdentity,
} from '@/utils/demoConversations';
import {
  checkResumeFile,
  currentResume,
  errorMessageFrom,
  metaFromResponse,
  resumeForThread,
  type ResumeView,
  type ThreadResume,
} from '@/utils/resume';

interface Section {
  database_id: number;
  name: string;
  status: string;
}

type RestoreState = 'loading' | 'ready' | 'error';

interface BackendMessage {
  type: string;
  content: string;
}

/**
 * The JobBuddy demo.
 *
 * Browser -> /api/chat, /api/history, /api/completion, /api/final-output and
 * /api/resume(/status) (Next server routes) -> Fly backend. The bearer token lives
 * only in those routes; nothing here knows it, and nothing here talks to Supabase.
 *
 * Viewport ownership is singular: `html, body { height: 100%; overflow: hidden }` in
 * globals.css, and every container below derives from it with `height: 100%`. Nothing
 * re-asserts `100vh` — that, combined with a missing body-margin reset, is what let an
 * offset accumulate and push the header out of the viewport entirely.
 */
export default function JobBuddyDemo() {
  const [identity, setIdentity] = useState<DemoIdentity | null>(null);
  const [conversations, setConversations] = useState<DemoConversation[]>([]);
  const [loadedMessages, setLoadedMessages] = useState<Message[]>([]);
  const [completion, setCompletion] = useState<CompletionState | null>(null);
  const [currentSection, setCurrentSection] = useState<Section | null>(null);
  const [restoreState, setRestoreState] = useState<RestoreState>('loading');
  const [restoreError, setRestoreError] = useState<string | null>(null);
  // The plan lives in React state only. It belongs to the backend; caching it in
  // localStorage would create a copy that goes stale the moment the plan changes.
  const [finalPlan, setFinalPlan] = useState<string | null>(null);
  const [planError, setPlanError] = useState<string | null>(null);
  const [planOpen, setPlanOpen] = useState(false);
  const planFetches = useRef(new Set<string>());
  // Tagged with its thread and shown only while that thread is selected, so one
  // conversation's resume can never appear in another. Memory only, like the plan.
  const [resume, setResume] = useState<ThreadResume | null>(null);
  // Invalidate reads/events from a selection that was deleted or switched away.
  const selectionVersion = useRef(0);
  const selectionAtRender = selectionVersion.current;

  useEffect(() => {
    if (process.env.NODE_ENV === 'development' && completion) {
      console.debug('[JobBuddy timing] progress-react-commit', {
        atMs: performance.now(),
        sections: completion.sections.map(({ id, status }) => ({ id, status })),
      });
    }
  }, [completion]);

  /**
   * Restore one thread from the backend.
   *
   * Transcript and progress are two independent reads of the same checkpoint, issued
   * together so a switch shows one loading state rather than two flashes. Progress is
   * never inferred from message text — that would be a guess wearing the costume of
   * state. Neither call reaches a model, so switching conversations is free.
   */
  /**
   * Fetch the finished plan for one thread.
   *
   * Only called when the completion projection says an artifact exists, so a thread
   * that has not finished never issues the request. It is a read: no model call, and
   * nothing here can regenerate the plan.
   */
  const loadFinalPlan = useCallback(async (id: DemoIdentity) => {
    const version = selectionVersion.current;
    // Intermediate and terminal completion can both announce the same artifact.
    // Keep this read single-flight; progress itself is never delayed by it.
    if (planFetches.current.has(id.threadId)) return;
    planFetches.current.add(id.threadId);
    setPlanError(null);
    try {
      const response = await fetch('/api/final-output', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ thread_id: id.threadId, user_id: id.userId }),
      });
      if (!response.ok) throw new Error(`Final plan request failed (${response.status})`);

      const data = (await response.json()) as {
        artifact_available: boolean;
        final_output: string | null;
      };
      // Never claim a plan the backend did not return. If it says unavailable, that
      // is the answer, even when completion said otherwise a moment ago.
      if (version !== selectionVersion.current) return;
      setFinalPlan(data.artifact_available ? data.final_output : null);
    } catch (error) {
      if (version !== selectionVersion.current) return;
      setPlanError(error instanceof Error ? error.message : 'Could not load the final plan');
      setFinalPlan(null);
    } finally {
      planFetches.current.delete(id.threadId);
    }
  }, []);

  /** Replace the resume view, but only if the stored one is still this thread's. */
  const updateResume = useCallback((threadId: string, view: ResumeView) => {
    setResume((prev) => (prev?.threadId === threadId ? { threadId, view } : prev));
  }, []);

  /**
   * Read whether this thread has a resume. One indexed read on the backend — no
   * model call. A failure is shown as a failure, never as "no resume", which would
   * invite a needless re-upload.
   */
  const loadResumeStatus = useCallback(async (id: DemoIdentity) => {
    const version = selectionVersion.current;
    const failed: ResumeView = {
      kind: 'error',
      message: "We couldn't check your resume right now.",
      retry: 'status',
      previous: null,
    };
    setResume({ threadId: id.threadId, view: { kind: 'checking' } });
    try {
      const response = await fetch('/api/resume/status', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ thread_id: id.threadId, user_id: id.userId }),
      });
      const data = await response.json().catch(() => null);
      if (version !== selectionVersion.current) return;
      if (!response.ok) {
        updateResume(id.threadId, { ...failed, message: errorMessageFrom(data, response.status) });
        return;
      }
      if (data?.has_resume !== true) {
        updateResume(id.threadId, { kind: 'none' });
        return;
      }
      const meta = metaFromResponse(data);
      updateResume(id.threadId, meta ? { kind: 'indexed', meta } : failed);
    } catch {
      if (version !== selectionVersion.current) return;
      updateResume(id.threadId, failed);
    }
  }, [updateResume]);

  const restore = useCallback(async (id: DemoIdentity) => {
    const version = selectionVersion.current;
    setRestoreState('loading');
    setRestoreError(null);
    // Beside history and completion, not inside their Promise.all: a resume status
    // failure must never fail the conversation's restore.
    void loadResumeStatus(id);

    const payload = {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ thread_id: id.threadId, user_id: id.userId }),
    };

    try {
      const [historyResponse, completionResponse] = await Promise.all([
        fetch('/api/history', payload),
        fetch('/api/completion', payload),
      ]);

      if (!historyResponse.ok) {
        throw new Error(`History request failed (${historyResponse.status})`);
      }
      if (!completionResponse.ok) {
        throw new Error(`Completion request failed (${completionResponse.status})`);
      }

      const historyData = await historyResponse.json();
      const backendMessages: BackendMessage[] = Array.isArray(historyData.messages)
        ? historyData.messages
        : [];

      // Order is the backend's, preserved exactly — the checkpoint already stores the
      // transcript chronologically and re-sorting here could only corrupt it.
      const restored = backendMessages
        .filter((m) => m.type === 'human' || m.type === 'ai')
        .map((m, index) => ({
          id: `restored-${index}`,
          role: m.type === 'human' ? ('user' as const) : ('assistant' as const),
          content: m.content,
        }));
      const completionData = (await completionResponse.json()) as CompletionState;
      if (version !== selectionVersion.current) return;
      setLoadedMessages(restored);
      setCompletion(
        Array.isArray(completionData.sections) && completionData.sections.length > 0
          ? completionData
          : null
      );

      // Title a restored thread from its own first user message. Deterministic and
      // local — naming a conversation is not worth a model call. Once assigned it is
      // never rewritten, so a thread does not rename itself as it grows.
      const firstUser = restored.find((m) => m.role === 'user');
      rememberConversation(
        id.threadId,
        id.userId,
        firstUser ? titleFromFirstMessage(firstUser.content) : undefined
      );
      setConversations(listConversations());

      // Conditional, and only after completion has answered: no artifact means no
      // request. The plan is fetched for the thread just restored, not the one that
      // happened to be selected when the request started.
      if (completionData?.artifact_available) {
        void loadFinalPlan(id);
      }

      setRestoreState('ready');
    } catch (error) {
      if (version !== selectionVersion.current) return;
      // A failed restore must not look like an empty conversation, or the user would
      // silently start talking over history they cannot see.
      setRestoreError(error instanceof Error ? error.message : 'Could not load your conversation');
      setRestoreState('error');
    }
  }, [loadFinalPlan, loadResumeStatus]);

  useEffect(() => {
    // The persisted active selection wins. Nothing used to record which thread was
    // being viewed, so a refresh fell back to whichever identity was written last —
    // always the newest conversation, never the one on screen.
    const id = resolveActiveIdentity();
    selectionVersion.current += 1;
    setIdentity(id);
    setConversations(listConversations());
    if (id) void restore(id);
    else setRestoreState('ready');
    return () => { selectionVersion.current += 1; };
  }, [restore]);

  /** Adopt a thread locally. No network call — the caller decides whether to restore. */
  const selectLocally = (id: DemoIdentity | null) => {
    selectionVersion.current += 1;
    setIdentity(id);
    setActiveThreadId(id?.threadId ?? null);
    setLoadedMessages([]);
    setCompletion(null);
    setCurrentSection(null);
    setRestoreError(null);
    // Cleared before any restore begins, so one conversation's plan can never flash
    // inside another while the next thread is still loading.
    setFinalPlan(null);
    setPlanError(null);
    setPlanOpen(false);
    setResume(null);
  };

  const handleNewConversation = () => {
    // Local only. A new conversation must not spend a model call just by existing —
    // no greeting is sent, and the backend hears nothing until the user types. The
    // onboarding card is a rendered component, not a fabricated assistant message.
    const id = mintIdentity();
    rememberConversation(id.threadId, id.userId);
    selectLocally(id);
    // A freshly minted thread cannot have a resume yet, so there is nothing to ask.
    setResume({ threadId: id.threadId, view: { kind: 'none' } });
    setRestoreState('ready');
    setConversations(listConversations());
  };

  /**
   * Upload a resume for the selected conversation, replacing any it had.
   *
   * The file goes to /api/resume only — never into the chat. The card says
   * "indexed" only when the backend confirmed it; a failed replace leaves the
   * previous resume in effect, because the backend swaps documents atomically.
   */
  const handleUploadResume = async (file: File) => {
    if (!identity) return;
    const id = identity;
    const version = selectionVersion.current;
    const previous = currentResume(resumeForThread(resume, id.threadId));

    const refusal = checkResumeFile(file);
    if (refusal) {
      updateResume(id.threadId, { kind: 'error', message: refusal, retry: 'upload', previous });
      return;
    }

    updateResume(id.threadId, { kind: 'uploading', filename: file.name, previous });
    const form = new FormData();
    form.append('file', file);
    form.append('thread_id', id.threadId);
    form.append('user_id', String(id.userId));

    try {
      // No Content-Type: the browser sets it, with the multipart boundary.
      const response = await fetch('/api/resume', { method: 'POST', body: form });
      const data = await response.json().catch(() => null);
      if (version !== selectionVersion.current) return;
      if (response.ok) {
        const meta = data?.indexed === true ? metaFromResponse(data) : null;
        // An unreadable success is not guessed at: ask the backend what is on file.
        if (meta) updateResume(id.threadId, { kind: 'indexed', meta });
        else void loadResumeStatus(id);
        return;
      }
      updateResume(id.threadId, {
        kind: 'error',
        message: errorMessageFrom(data, response.status),
        retry: 'upload',
        previous,
      });
    } catch {
      if (version !== selectionVersion.current) return;
      updateResume(id.threadId, {
        kind: 'error',
        message: 'Could not reach JobBuddy. Please try again.',
        retry: 'upload',
        previous,
      });
    }
  };

  /**
   * A file refused before any request — wrong type, too large, several at once. It
   * goes into the same resume state as a failed upload, so the welcome card and the
   * sidebar report it identically, and the resume already in effect stays in effect.
   */
  const handleRejectResume = (message: string) => {
    if (!identity) return;
    const previous = currentResume(resumeForThread(resume, identity.threadId));
    updateResume(identity.threadId, { kind: 'error', message, retry: 'upload', previous });
  };

  const handleFirstUserMessage = useCallback((content: string) => {
    if (!identity) return;
    // Metadata only: the transcript remains exclusively in the backend checkpoint.
    // rememberConversation changes a fallback label once and preserves it forever
    // after that, so later messages cannot rename the thread.
    rememberConversation(
      identity.threadId,
      identity.userId,
      titleFromFirstMessage(content),
    );
    setConversations(listConversations());
  }, [identity]);

  const handleSelectConversation = (conversation: DemoConversation) => {
    if (conversation.threadId === identity?.threadId) return;
    selectLocally({ threadId: conversation.threadId, userId: conversation.userId });
    // The same restoration path a refresh takes: history + completion, no model call.
    void restore({ threadId: conversation.threadId, userId: conversation.userId });
  };

  /**
   * Remove a conversation from the local list.
   *
   * Local only: this demo has no deletion endpoint and does not invent one, so the
   * backend checkpoint for the thread still exists in Postgres. The conversation
   * becomes unreachable from this browser, not deleted.
   */
  const handleDeleteConversation = (conversation: DemoConversation) => {
    const remaining = deleteConversation(conversation.threadId);
    setConversations(remaining);

    // Deleting a conversation the user is not looking at must not move them.
    if (conversation.threadId !== identity?.threadId) return;

    if (remaining.length > 0) {
      const next = remaining[0];
      selectLocally({ threadId: next.threadId, userId: next.userId });
      void restore({ threadId: next.threadId, userId: next.userId });
      return;
    }

    // Nothing left means no selection, not a new conversation.
    selectLocally(null);
    setRestoreState('ready');
  };

  return (
    <div
      style={{
        display: 'flex',
        height: '100%',
        overflow: 'hidden',
        fontFamily: 'system-ui, -apple-system, sans-serif',
        backgroundColor: '#ffffff',
      }}
    >
      <aside
        style={{
          width: '32%',
          minWidth: 280,
          maxWidth: 400,
          height: '100%',
          padding: 20,
          backgroundColor: '#f8fafc',
          borderRight: '1px solid #e2e8f0',
          display: 'flex',
          flexDirection: 'column',
          gap: 16,
          // The sidebar scrolls when the viewport is too short. minHeight: 0 is what
          // allows it to: without it a flex child refuses to shrink below its content
          // and pushes the overflow outside the box instead of scrolling inside it.
          overflowY: 'auto',
          minHeight: 0,
        }}
      >
        <JobBuddyProgress
          completion={completion}
          threadId={identity?.threadId ?? null}
          userId={identity?.userId ?? null}
          conversations={conversations}
          onNewConversation={handleNewConversation}
          onSelectConversation={handleSelectConversation}
          onDeleteConversation={handleDeleteConversation}
          finalPlanReady={Boolean(finalPlan)}
          finalPlanError={planError}
          onViewFinalPlan={() => setPlanOpen(true)}
          onRetryFinalPlan={() => identity && void loadFinalPlan(identity)}
          resume={resumeForThread(resume, identity?.threadId ?? null)}
          onUploadResume={(file) => void handleUploadResume(file)}
          onRetryResumeStatus={() => identity && void loadResumeStatus(identity)}
        />

        {restoreState === 'error' && (
          <div
            style={{
              flexShrink: 0,
              padding: '10px 12px',
              borderRadius: 8,
              backgroundColor: '#fef2f2',
              border: '1px solid #fecaca',
              fontSize: 13,
              color: '#991b1b',
            }}
          >
            <div style={{ marginBottom: 8 }}>Could not load this conversation. {restoreError}</div>
            <button
              onClick={() => identity && void restore(identity)}
              style={{
                padding: '5px 10px',
                fontSize: 12,
                fontWeight: 600,
                color: '#991b1b',
                backgroundColor: '#ffffff',
                border: '1px solid #fecaca',
                borderRadius: 6,
                cursor: 'pointer',
              }}
            >
              Retry
            </button>
          </div>
        )}
      </aside>

      <main
        style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          minWidth: 0,
          height: '100%',
          overflow: 'hidden',
        }}
      >
        {identity && restoreState !== 'loading' ? (
          <ChatArea
            key={identity.threadId}
            selectedAgent="xbuddy"
            userId={identity.userId}
            mode="stream"
            threadId={identity.threadId}
            loadedMessages={loadedMessages}
            currentSection={currentSection}
            onThreadIdChange={() => {
              /* the demo mints its own thread id, so the backend echo is a no-op */
            }}
            onSectionUpdate={(next) => {
              if (selectionAtRender === selectionVersion.current) setCurrentSection(next);
            }}
            onFirstUserMessage={handleFirstUserMessage}
            // The same view and handlers the sidebar receives: one resume state, two
            // surfaces, so the welcome card and the sidebar cannot disagree.
            resume={resumeForThread(resume, identity?.threadId ?? null)}
            onUploadResume={(file) => void handleUploadResume(file)}
            onRejectResume={handleRejectResume}
            onRetryResumeStatus={() => identity && void loadResumeStatus(identity)}
            onCompletionUpdate={(next) => {
              if (selectionAtRender !== selectionVersion.current) return;
              setCompletion(next);
              // Every event is the latest committed public projection. A plan
              // read is allowed only once an event reports artifact availability.
              if (next.artifact_available && !finalPlan) {
                void loadFinalPlan(identity);
              }
            }}
          />
        ) : (
          <div
            style={{
              flex: 1,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#94a3b8',
              fontSize: 14,
            }}
          >
            {restoreState === 'loading'
              ? 'Loading your conversation…'
              : 'Choose “Start a new conversation” in the sidebar to begin.'}
          </div>
        )}
      </main>

      {planOpen && finalPlan && (
        <FinalPlanPanel plan={finalPlan} onClose={() => setPlanOpen(false)} />
      )}
    </div>
  );
}
