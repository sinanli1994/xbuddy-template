'use client';

import React from 'react';
import type { CompletionState, PublicSection } from '@/components/ChatArea';
import type { DemoConversation } from '@/utils/demoConversations';
import type { ResumeView } from '@/utils/resume';
import ResumeCard from '@/components/ResumeCard';

/**
 * The five areas JobBuddy covers, shown as a neutral roadmap before the backend has
 * reported anything.
 *
 * This is workflow guidance, not progress. Every row renders grey with no status
 * label, so it cannot be mistaken for a real pending / in_progress / done status —
 * those only ever come from the backend projection. The moment /completion or a
 * stream reports real sections, this is replaced by them.
 */
const ROADMAP = ['Career Goal', 'Background', 'Job Preferences', 'Skill Assessment', 'Action Plan'];

/**
 * The JobBuddy sidebar.
 *
 * Order keeps collection, artifact status and its action inside progress — then new
 * conversation, recent conversations, developer details — with JobBuddy
 * content. The new-conversation control sits *between* progress and the list, where
 * it reads as "add another one of these", rather than under the title (where it looks
 * like a product action) or pinned to the bottom (where it drifts with content
 * height).
 *
 * Progress is rendered entirely from the backend's public completion projection. No
 * section list is hardcoded: the panel this replaced carried its section names as a
 * literal and was still doing so long after the backend's set had changed.
 * `database_id` is deliberately not read — it is not part of the public projection.
 */

const STATUS_COLOR: Record<PublicSection['status'], string> = {
  done: '#10b981',
  in_progress: '#6366f1',
  pending: '#94a3b8',
};

const STATUS_LABEL: Record<PublicSection['status'], string> = {
  done: 'Done',
  in_progress: 'In Progress',
  pending: 'Pending',
};

interface Props {
  completion: CompletionState | null;
  threadId: string | null;
  userId: number | null;
  conversations: DemoConversation[];
  onNewConversation: () => void;
  onSelectConversation: (conversation: DemoConversation) => void;
  onDeleteConversation: (conversation: DemoConversation) => void;
  finalPlanReady: boolean;
  finalPlanError: string | null;
  onViewFinalPlan: () => void;
  onRetryFinalPlan: () => void;
  /** The selected conversation's resume, or null when none is selected. */
  resume: ResumeView | null;
  onUploadResume: (file: File) => void;
  onRetryResumeStatus: () => void;
}

function SectionHeading({ children }: { children: React.ReactNode }) {
  return (
    <h2
      style={{
        fontSize: 11,
        fontWeight: 700,
        textTransform: 'uppercase',
        letterSpacing: '0.06em',
        color: '#64748b',
        margin: '0 0 8px',
      }}
    >
      {children}
    </h2>
  );
}

export default function JobBuddyProgress({
  completion,
  threadId,
  userId,
  conversations,
  onNewConversation,
  onSelectConversation,
  onDeleteConversation,
  finalPlanReady,
  finalPlanError,
  onViewFinalPlan,
  onRetryFinalPlan,
  resume,
  onUploadResume,
  onRetryResumeStatus,
}: Props) {
  const sections = completion?.sections ?? [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
      {/* 1. Product */}
      <div style={{ flexShrink: 0 }}>
        <h1 style={{ fontSize: 20, fontWeight: 700, color: '#0f172a', margin: 0 }}>JobBuddy</h1>
        <p style={{ fontSize: 13, color: '#64748b', margin: '4px 0 0' }}>
          AI Career Planning Agent
        </p>
      </div>

      {/* 2. Progress */}
      <div data-testid="workflow-progress" style={{ flexShrink: 0 }}>
        <SectionHeading>Your Progress</SectionHeading>
        {sections.length === 0 ? (
          <>
            <ul style={{ listStyle: 'none', margin: 0, padding: 0 }}>
              {ROADMAP.map((name) => (
                <li
                  key={name}
                  style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '6px 8px' }}
                >
                  <span
                    aria-hidden
                    style={{
                      width: 9,
                      height: 9,
                      borderRadius: '50%',
                      flexShrink: 0,
                      border: '1px solid #cbd5e1',
                      backgroundColor: 'transparent',
                    }}
                  />
                  <span style={{ flex: 1, fontSize: 14, color: '#94a3b8' }}>{name}</span>
                </li>
              ))}
            </ul>
            <p style={{ fontSize: 12, color: '#94a3b8', margin: '6px 0 0' }}>
              Progress appears here once JobBuddy replies.
            </p>
          </>
        ) : (
          <ul style={{ listStyle: 'none', margin: 0, padding: 0 }}>
            {sections.map((section) => {
              const active = section.status === 'in_progress';
              return (
                <li
                  key={section.id}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 10,
                    padding: '6px 8px',
                    borderRadius: 6,
                    // The active section is the one thing a viewer scans for, so it
                    // gets a background rather than only a colour difference.
                    backgroundColor: active ? '#eef2ff' : 'transparent',
                  }}
                >
                  <span
                    aria-hidden
                    style={{
                      width: 9,
                      height: 9,
                      borderRadius: '50%',
                      flexShrink: 0,
                      backgroundColor: STATUS_COLOR[section.status] ?? STATUS_COLOR.pending,
                    }}
                  />
                  <span
                    style={{
                      flex: 1,
                      fontSize: 14,
                      color: '#0f172a',
                      fontWeight: active ? 600 : 400,
                    }}
                  >
                    {section.name}
                  </span>
                  <span
                    style={{
                      fontSize: 11,
                      fontWeight: 600,
                      flexShrink: 0,
                      color: STATUS_COLOR[section.status] ?? STATUS_COLOR.pending,
                    }}
                  >
                    {STATUS_LABEL[section.status] ?? section.status}
                  </span>
                </li>
              );
            })}
          </ul>
        )}
        {/* Completion belongs to the same authoritative progress block. */}
        <div data-testid="completion-summary" style={{ flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 6, marginTop: 12, padding: '10px 8px 0', borderTop: '1px solid #e2e8f0' }}>
          <StatusLine
            label="Collection"
            value={completion?.collection_complete ? 'Complete' : 'In Progress'}
            done={Boolean(completion?.collection_complete)}
          />
          <StatusLine
            label="Final Plan"
            value={completion?.artifact_available ? 'Ready' : 'Not Ready'}
            done={Boolean(completion?.artifact_available)}
          />
          {completion?.artifact_available && finalPlanReady && (
            <button
              onClick={onViewFinalPlan}
              style={{
                padding: '9px 12px',
                borderRadius: 8,
                backgroundColor: '#ecfdf5',
                border: '1px solid #a7f3d0',
                fontSize: 13,
                fontWeight: 600,
                color: '#065f46',
                cursor: 'pointer',
                textAlign: 'left',
              }}
            >
              View Final Plan
            </button>
          )}

          {completion?.artifact_available && !finalPlanReady && (
            // The plan exists but could not be fetched. Saying so beats a button that
            // opens nothing, and beats pretending the plan is not ready.
            <div
              style={{
                padding: '8px 10px',
                borderRadius: 8,
                backgroundColor: finalPlanError ? '#fef2f2' : '#f8fafc',
                border: `1px solid ${finalPlanError ? '#fecaca' : '#e2e8f0'}`,
                fontSize: 12,
                color: finalPlanError ? '#991b1b' : '#64748b',
              }}
            >
              {finalPlanError ? (
                <>
                  <div style={{ marginBottom: 6 }}>
                    Your final plan is ready, but could not be loaded.
                  </div>
                  <button
                    onClick={onRetryFinalPlan}
                    style={{
                      padding: '4px 9px',
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
                </>
              ) : (
                'Loading your final plan…'
              )}
            </div>
          )}
        </div>
      </div>

      {/* 2b. Resume — belongs to this conversation, so it sits with its progress. */}
      {resume && (
        <div style={{ flexShrink: 0 }}>
          <SectionHeading>Your Resume</SectionHeading>
          <ResumeCard view={resume} onUpload={onUploadResume} onRetryStatus={onRetryResumeStatus} />
        </div>
      )}

      {/* 3. New conversation — between progress and the list, per the reference UI. */}
      <button
        onClick={onNewConversation}
        style={{
          flexShrink: 0,
          padding: '9px 12px',
          fontSize: 13,
          fontWeight: 600,
          color: '#4f46e5',
          backgroundColor: '#ffffff',
          border: '1px solid #c7d2fe',
          borderRadius: 8,
          cursor: 'pointer',
          textAlign: 'left',
        }}
        title="Start a new demo conversation"
      >
        + Start a new conversation
      </button>

      {/* 4. Bounded recent list scrolls internally; the sidebar also scrolls on short viewports. */}
      <div style={{ display: 'flex', flexDirection: 'column' }}>
        <SectionHeading>Recent Conversations</SectionHeading>
        {conversations.length === 0 ? (
          <p style={{ fontSize: 13, color: '#94a3b8', margin: 0 }}>
            Conversations you start will appear here.
          </p>
        ) : (
          <ul
            style={{
              listStyle: 'none',
              margin: 0,
              padding: 0,
              overflowY: 'auto',
              // A bounded window, not `flex: 1`. A row is ~38px, so this shows about
              // three rows at minimum and five at most, and the list scrolls inside
              // itself beyond that. clamp keeps the maximum from dominating a short
              // viewport, where the sidebar's own scrolling takes over.
              minHeight: 114,
              maxHeight: 'clamp(114px, 26vh, 190px)',
            }}
          >
            {conversations.map((conversation) => {
              const current = conversation.threadId === threadId;
              return (
                <li
                  key={conversation.threadId}
                  style={{ display: 'flex', alignItems: 'center', gap: 4, marginBottom: 4 }}
                >
                  <button
                    onClick={() => onSelectConversation(conversation)}
                    aria-current={current ? 'true' : undefined}
                    style={{
                      flex: 1,
                      minWidth: 0,
                      textAlign: 'left',
                      padding: '8px 10px',
                      fontSize: 13,
                      lineHeight: 1.35,
                      color: current ? '#3730a3' : '#334155',
                      backgroundColor: current ? '#eef2ff' : 'transparent',
                      border: current ? '1px solid #c7d2fe' : '1px solid transparent',
                      borderRadius: 6,
                      cursor: 'pointer',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                      fontWeight: current ? 600 : 400,
                    }}
                    title={conversation.label}
                  >
                    {conversation.label}
                  </button>
                  <button
                    onClick={(event) => {
                      // Without stopPropagation the click also reaches the row's
                      // select handler, so deleting would first switch to the
                      // conversation being deleted and restore a thread that is
                      // about to disappear.
                      event.stopPropagation();
                      onDeleteConversation(conversation);
                    }}
                    aria-label={"Delete conversation: " + conversation.label}
                    title="Remove from this list"
                    style={{
                      flexShrink: 0,
                      width: 24,
                      height: 24,
                      border: 'none',
                      borderRadius: 6,
                      backgroundColor: 'transparent',
                      color: '#94a3b8',
                      cursor: 'pointer',
                      fontSize: 13,
                      lineHeight: 1,
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.color = '#b91c1c';
                      e.currentTarget.style.backgroundColor = '#fef2f2';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.color = '#94a3b8';
                      e.currentTarget.style.backgroundColor = 'transparent';
                    }}
                  >
                    ✕
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </div>

      {/* 5. Developer details — always last. */}
      <details style={{ flexShrink: 0, fontSize: 12, color: '#64748b' }}>
        <summary style={{ cursor: 'pointer', userSelect: 'none' }}>Developer Details</summary>
        <dl
          style={{
            margin: '8px 0 0',
            display: 'grid',
            gridTemplateColumns: 'auto 1fr',
            gap: '4px 10px',
          }}
        >
          <dt style={{ color: '#94a3b8' }}>thread_id</dt>
          <dd style={{ margin: 0, wordBreak: 'break-all', fontFamily: 'ui-monospace, monospace' }}>
            {threadId ?? '—'}
          </dd>
          <dt style={{ color: '#94a3b8' }}>user_id</dt>
          <dd style={{ margin: 0, fontFamily: 'ui-monospace, monospace' }}>{userId ?? '—'}</dd>
          <dt style={{ color: '#94a3b8' }}>API mode</dt>
          <dd style={{ margin: 0 }}>SSE</dd>
          <dt style={{ color: '#94a3b8' }}>collection_complete</dt>
          <dd style={{ margin: 0 }}>{String(Boolean(completion?.collection_complete))}</dd>
          <dt style={{ color: '#94a3b8' }}>artifact_available</dt>
          <dd style={{ margin: 0 }}>{String(Boolean(completion?.artifact_available))}</dd>
        </dl>
      </details>
    </div>
  );
}

function StatusLine({ label, value, done }: { label: string; value: string; done: boolean }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
      <span style={{ color: '#64748b' }}>{label}</span>
      <span style={{ fontWeight: 600, color: done ? '#10b981' : '#64748b' }}>{value}</span>
    </div>
  );
}
