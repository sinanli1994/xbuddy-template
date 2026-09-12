'use client';

import React, { useRef } from 'react';
import { currentResume, type ResumeView } from '@/utils/resume';

/**
 * The resume attached to the selected conversation.
 *
 * Presentational only: the page owns the state and the requests. What it renders
 * comes from the backend's status — this card never claims a resume the backend
 * did not report as indexed.
 */

interface Props {
  view: ResumeView;
  onUpload: (file: File) => void;
  onRetryStatus: () => void;
}

const buttonStyle: React.CSSProperties = {
  padding: '6px 10px',
  fontSize: 12,
  fontWeight: 600,
  color: '#4f46e5',
  backgroundColor: '#ffffff',
  border: '1px solid #c7d2fe',
  borderRadius: 6,
  cursor: 'pointer',
};

export default function ResumeCard({ view, onUpload, onRetryStatus }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const pick = () => input.current?.click();
  const inEffect = currentResume(view);

  return (
    <div
      data-testid="resume-card"
      data-state={view.kind}
      style={{
        flexShrink: 0,
        padding: '10px 12px',
        borderRadius: 8,
        backgroundColor: '#ffffff',
        border: '1px solid #e2e8f0',
        fontSize: 13,
        color: '#334155',
      }}
    >
      <input
        ref={input}
        type="file"
        accept="application/pdf,.pdf"
        hidden
        onChange={(event) => {
          const file = event.target.files?.[0];
          // Cleared so choosing the same file again still fires a change.
          event.target.value = '';
          if (file) onUpload(file);
        }}
      />

      {view.kind === 'checking' && (
        <div style={{ color: '#94a3b8' }}>Checking for a resume…</div>
      )}

      {view.kind === 'none' && (
        <>
          <div style={{ marginBottom: 8, color: '#64748b', lineHeight: 1.4 }}>
            Add your resume (PDF, up to 2 MB). JobBuddy uses it in Background and Skill
            Assessment, and records nothing from it until you confirm.
          </div>
          <button onClick={pick} style={buttonStyle}>
            Upload PDF
          </button>
        </>
      )}

      {view.kind === 'uploading' && (
        <div aria-live="polite" style={{ color: '#64748b' }}>
          Reading {view.filename}…
        </div>
      )}

      {view.kind === 'indexed' && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div
              style={{ fontWeight: 600, color: '#065f46', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
              title={view.meta.filename}
            >
              ✓ {view.meta.filename}
            </div>
            <div style={{ fontSize: 12, color: '#64748b' }}>
              {view.meta.chunkCount} passages · {view.meta.pageCount} {view.meta.pageCount === 1 ? 'page' : 'pages'}
            </div>
          </div>
          <button onClick={pick} style={buttonStyle}>
            Replace
          </button>
        </div>
      )}

      {view.kind === 'error' && (
        <div role="alert">
          <div style={{ color: '#991b1b', marginBottom: 6 }}>{view.message}</div>
          {inEffect && (
            <div style={{ fontSize: 12, color: '#64748b', marginBottom: 6 }}>
              Still using {inEffect.filename}.
            </div>
          )}
          <button onClick={view.retry === 'status' ? onRetryStatus : pick} style={buttonStyle}>
            {view.retry === 'status' ? 'Retry' : 'Choose another PDF'}
          </button>
        </div>
      )}
    </div>
  );
}
