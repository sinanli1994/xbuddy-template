'use client';

import { useCallback, useEffect, useState } from 'react';

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
 * Browser -> /api/chat, /api/history and /api/completion (Next server routes) -> Fly
 * backend. The bearer token lives only in those routes; nothing here knows it, and
 * nothing here talks to Supabase.
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
      setFinalPlan(data.artifact_available ? data.final_output : null);
    } catch (error) {
      setPlanError(error instanceof Error ? error.message : 'Could not load the final plan');
      setFinalPlan(null);
    }
  }, []);

  const restore = useCallback(async (id: DemoIdentity) => {
    setRestoreState('loading');
    setRestoreError(null);

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
      setLoadedMessages(restored);

      const completionData = (await completionResponse.json()) as CompletionState;
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
      // A failed restore must not look like an empty conversation, or the user would
      // silently start talking over history they cannot see.
      setRestoreError(error instanceof Error ? error.message : 'Could not load your conversation');
      setRestoreState('error');
    }
  }, [loadFinalPlan]);

  useEffect(() => {
    // The persisted active selection wins. Nothing used to record which thread was
    // being viewed, so a refresh fell back to whichever identity was written last —
    // always the newest conversation, never the one on screen.
    const id = resolveActiveIdentity();
    setIdentity(id);
    setConversations(listConversations());
    void restore(id);
  }, [restore]);

  /** Adopt a thread locally. No network call — the caller decides whether to restore. */
  const selectLocally = (id: DemoIdentity) => {
    setIdentity(id);
    setActiveThreadId(id.threadId);
    setLoadedMessages([]);
    setCompletion(null);
    setCurrentSection(null);
    setRestoreError(null);
    // Cleared before any restore begins, so one conversation's plan can never flash
    // inside another while the next thread is still loading.
    setFinalPlan(null);
    setPlanError(null);
    setPlanOpen(false);
  };

  const handleNewConversation = () => {
    // Local only. A new conversation must not spend a model call just by existing —
    // no greeting is sent, and the backend hears nothing until the user types. The
    // onboarding card is a rendered component, not a fabricated assistant message.
    const id = mintIdentity();
    rememberConversation(id.threadId, id.userId);
    selectLocally(id);
    setRestoreState('ready');
    setConversations(listConversations());
  };

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

    // Nothing left: a fresh empty thread, which shows onboarding. Still no model call.
    const fresh = mintIdentity();
    rememberConversation(fresh.threadId, fresh.userId);
    selectLocally(fresh);
    setRestoreState('ready');
    setConversations(listConversations());
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
            onSectionUpdate={setCurrentSection}
            onCompletionUpdate={(next) => {
              setCompletion(next);
              // The terminal completion event is the first moment a just-finished
              // plan exists. Fetch it once, and only then.
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
            Loading your conversation…
          </div>
        )}
      </main>

      {planOpen && finalPlan && (
        <FinalPlanPanel plan={finalPlan} onClose={() => setPlanOpen(false)} />
      )}
    </div>
  );
}
