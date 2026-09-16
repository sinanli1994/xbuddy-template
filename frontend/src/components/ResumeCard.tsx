'use client';

import React, { useRef } from 'react';
import { currentResume, type ResumeView } from '@/utils/resume';

/**
 * The resume attached to the selected conversation, as a compact status.
 *
 * The welcome card is where a resume is first added; this is where it stays visible
 * and manageable after the conversation starts — status, Replace, and recovery from
 * a failed upload or status check. Before anything is attached it is deliberately
 * quiet, so it never competes with the welcome card's upload.
 *
 * Presentational only: the page owns the state and the requests, and the welcome
 * card renders the same view. What it shows comes from the backend's status — it
 * never claims a resume the backend did not report as indexed.
 */

interface Props {
  view: ResumeView;
  onUpload: (file: File) => void;
  onRetryStatus: () => void;
}

const buttonStyle: React.CSSProperties = {
  padding: '5px 10px',
  fontSize: 12,
  fontWeight: 600,
  color: '#4f46e5',
  backgroundColor: '#ffffff',
  border: '1px solid #c7d2fe',
  borderRadius: 6,
  cursor: 'pointer',
  flexShrink: 0,
};

const linkStyle: React.CSSProperties = {
  padding: 0,
  fontSize: 12,
  fontWeight: 600,
  color: '#4f46e5',
  background: 'none',
  border: 'none',
  cursor: 'pointer',
};

export default function ResumeCard({ view, onUpload, onRetryStatus }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const pick = () => input.current?.click();
  const inEffect = currentResume(view);
  // Nothing attached yet: a plain line, not a card, so it reads as status.
  const quiet = view.kind === 'none' || view.kind === 'checking';

  return (
    <div
      data-testid="resume-card"
      data-state={view.kind}
      style={{
        flexShrink: 0,
        padding: quiet ? '0 8px' : '10px 12px',
        borderRadius: 8,
        backgroundColor: quiet ? 'transparent' : '#ffffff',
        border: quiet ? 'none' : '1px solid #e2e8f0',
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
        <div style={{ fontSize: 12, color: '#94a3b8' }}>Checking for a resume…</div>
      )}

      {view.kind === 'none' && (
        <div style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.5 }}>
          No resume attached to this conversation.{' '}
          <button type="button" onClick={pick} style={linkStyle}>
            Add a PDF
          </button>
        </div>
      )}

      {view.kind === 'uploading' && (
        <div aria-live="polite" style={{ color: '#64748b', overflowWrap: 'anywhere' }}>
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
            <div style={{ fontSize: 11, color: '#94a3b8' }}>Attached to this conversation</div>
          </div>
          <button type="button" onClick={pick} style={buttonStyle}>
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
          <button type="button" onClick={view.retry === 'status' ? onRetryStatus : pick} style={buttonStyle}>
            {view.retry === 'status' ? 'Retry' : 'Choose another PDF'}
          </button>
        </div>
      )}
    </div>
  );
}
