'use client';

import ResumeDropZone from '@/components/ResumeDropZone';
import type { ResumeView } from '@/utils/resume';

/**
 * The first-use card for a conversation with no history yet.
 *
 * This is **product guidance, not a model response.** It is rendered as its own
 * component, never pushed into `messages`, so it cannot be mistaken for assistant
 * output, cannot be copied by Copy All as if JobBuddy had said it, and cannot end up
 * in a transcript the backend never saw. Nothing here calls `/api/chat`.
 *
 * It is also the primary place to add a resume, because a resume belongs to this one
 * conversation and is most useful before it starts. Adding one stays optional: the
 * card says so, and the Career Goal question below it is where the conversation
 * actually begins. The upload itself is the page's — this card only offers it.
 *
 * It disappears as soon as the thread has real messages, so a restored conversation
 * shows its transcript instead of being re-introduced on every refresh.
 */

const AREAS = ['Career Goal', 'Background', 'Job Preferences', 'Skill Assessment', 'Action Plan'];

interface Props {
  /** The selected conversation's resume, or null when there is none to offer. */
  resume?: ResumeView | null;
  onUploadResume?: (file: File) => void;
  onRejectResume?: (message: string) => void;
  onRetryResumeStatus?: () => void;
}

const noop = () => {};

export default function JobBuddyWelcome({
  resume = null,
  onUploadResume = noop,
  onRejectResume = noop,
  onRetryResumeStatus = noop,
}: Props) {
  return (
    <div
      data-testid="jobbuddy-welcome"
      style={{
        maxWidth: 620,
        margin: '32px auto',
        padding: '24px 28px',
        backgroundColor: '#ffffff',
        border: '1px solid #e2e8f0',
        borderRadius: 12,
        boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
      }}
    >
      <h2 style={{ fontSize: 18, fontWeight: 700, color: '#0f172a', margin: 0 }}>
        Hi, I&apos;m JobBuddy 👋
      </h2>
      <p style={{ fontSize: 14, lineHeight: 1.6, color: '#475569', margin: '10px 0 0' }}>
        I&apos;ll help you build a personalized career plan across five areas:{' '}
        {AREAS.map((area, index) => (
          <span key={area}>
            <strong style={{ fontWeight: 600, color: '#0f172a' }}>{area}</strong>
            {index < AREAS.length - 2 ? ', ' : index === AREAS.length - 2 ? ', and ' : '.'}
          </span>
        ))}
      </p>

      {resume && (
        <ResumeDropZone
          view={resume}
          onUpload={onUploadResume}
          onReject={onRejectResume}
          onRetryStatus={onRetryResumeStatus}
        />
      )}

      {resume && resume.kind !== 'indexed' && (
        <p style={{ fontSize: 13, lineHeight: 1.55, color: '#64748b', margin: '10px 0 0' }}>
          No resume? That&apos;s completely fine — just answer the questions as we go.
        </p>
      )}

      <p
        style={{
          fontSize: 14,
          lineHeight: 1.6,
          color: '#0f172a',
          fontWeight: 600,
          margin: '18px 0 0',
          paddingTop: 14,
          borderTop: '1px solid #f1f5f9',
        }}
      >
        To get started, tell me what kind of career move or role you&apos;re considering.
      </p>
    </div>
  );
}
