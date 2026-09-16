'use client';

import React, { useRef, useState } from 'react';
import { chooseResumeFile, currentResume, type ResumeView } from '@/utils/resume';

/**
 * The optional resume upload inside the welcome card — the primary place to add one.
 *
 * Presentational, like the sidebar's ResumeCard: the page owns the one resume state
 * and every request, and both surfaces render that same view. Nothing here fetches;
 * a chosen or dropped file is handed to `onUpload`, which is the page's existing
 * upload path to /api/resume.
 */

export interface ResumeControls {
  view: ResumeView;
  onUpload: (file: File) => void;
  onReject: (message: string) => void;
  onRetryStatus: () => void;
}

interface DragEventLike {
  preventDefault(): void;
  dataTransfer?: { files?: ArrayLike<File>; dropEffect?: string } | null;
  currentTarget?: unknown;
  relatedTarget?: unknown;
}

function containsNode(container: unknown, node: unknown): boolean {
  return (
    typeof Node !== 'undefined' &&
    container instanceof Node &&
    node instanceof Node &&
    container.contains(node)
  );
}

/**
 * Drag, drop and picker handling, as plain functions of their inputs.
 *
 * Kept free of hooks so the rules can be exercised without a browser. Every drag
 * event on the zone cancels the browser default — without that, a dropped PDF is
 * opened in the tab instead of uploaded.
 */
export function resumeDropHandlers({
  busy,
  onUpload,
  onReject,
  setDragging,
  contains = containsNode,
}: {
  busy: boolean;
  onUpload: (file: File) => void;
  onReject: (message: string) => void;
  setDragging: (dragging: boolean) => void;
  contains?: (container: unknown, node: unknown) => boolean;
}) {
  const offer = (files: ArrayLike<File> | null | undefined) => {
    const choice = chooseResumeFile(files);
    if (!choice) return;
    if ('refusal' in choice) onReject(choice.refusal);
    else onUpload(choice.file);
  };

  return {
    onDragEnter(event: DragEventLike) {
      event.preventDefault();
      if (!busy) setDragging(true);
    },
    onDragOver(event: DragEventLike) {
      event.preventDefault();
      if (event.dataTransfer) event.dataTransfer.dropEffect = busy ? 'none' : 'copy';
    },
    onDragLeave(event: DragEventLike) {
      // Moving between the zone's own children is not leaving it.
      if (contains(event.currentTarget, event.relatedTarget)) return;
      setDragging(false);
    },
    onDrop(event: DragEventLike) {
      event.preventDefault();
      setDragging(false);
      if (busy) return;
      offer(event.dataTransfer?.files);
    },
    onChoose(files: ArrayLike<File> | null | undefined) {
      if (!busy) offer(files);
    },
  };
}

const outlineButton: React.CSSProperties = {
  padding: '7px 14px',
  fontSize: 13,
  fontWeight: 600,
  color: '#4f46e5',
  backgroundColor: '#ffffff',
  border: '1px solid #c7d2fe',
  borderRadius: 8,
  cursor: 'pointer',
  flexShrink: 0,
};

export default function ResumeDropZone({ view, onUpload, onReject, onRetryStatus }: ResumeControls) {
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const busy = view.kind === 'uploading' || view.kind === 'checking';
  const handlers = resumeDropHandlers({ busy, onUpload, onReject, setDragging });
  const drag = {
    onDragEnter: handlers.onDragEnter,
    onDragOver: handlers.onDragOver,
    onDragLeave: handlers.onDragLeave,
    onDrop: handlers.onDrop,
  };
  const pick = () => {
    if (!busy) input.current?.click();
  };
  const inEffect = currentResume(view);

  return (
    <section
      data-testid="welcome-resume"
      data-state={view.kind}
      aria-label="Optional resume upload"
      style={{ margin: '18px 0 0' }}
    >
      <input
        ref={input}
        type="file"
        accept="application/pdf,.pdf"
        hidden
        onChange={(event) => {
          const files = event.target.files;
          handlers.onChoose(files ? Array.from(files) : null);
          // Cleared so choosing the same file again still fires a change.
          event.target.value = '';
        }}
      />

      {view.kind !== 'indexed' && (
        <>
          <div style={{ fontSize: 14, fontWeight: 600, color: '#0f172a' }}>
            Have a resume? <span style={{ fontWeight: 400, color: '#64748b' }}>(Optional)</span>
          </div>
          <p style={{ fontSize: 13, lineHeight: 1.55, color: '#475569', margin: '4px 0 10px' }}>
            Upload it here and I&apos;ll use it to help with your Background and Skill
            Assessment. You&apos;ll still confirm anything before I treat it as fact.
          </p>
        </>
      )}

      {view.kind === 'checking' && (
        <div style={{ fontSize: 13, color: '#94a3b8' }}>Checking for a resume…</div>
      )}

      {view.kind === 'error' && (
        <div role="alert" style={{ fontSize: 13, color: '#991b1b', margin: '0 0 8px' }}>
          {view.message}
          {inEffect && (
            <div style={{ color: '#64748b', marginTop: 2 }}>Still using {inEffect.filename}.</div>
          )}
        </div>
      )}

      {view.kind === 'error' && view.retry === 'status' && (
        <button type="button" onClick={onRetryStatus} style={outlineButton}>
          Retry
        </button>
      )}

      {(view.kind === 'none' || view.kind === 'uploading' ||
        (view.kind === 'error' && view.retry === 'upload')) && (
        <div
          data-testid="resume-drop-zone"
          data-dragging={dragging ? 'true' : 'false'}
          aria-busy={view.kind === 'uploading'}
          {...drag}
          style={{
            border: `1.5px dashed ${dragging ? '#6366f1' : '#cbd5e1'}`,
            backgroundColor: dragging ? '#eef2ff' : '#ffffff',
            borderRadius: 10,
            padding: '16px 12px',
            textAlign: 'center',
            transition: 'background-color 120ms ease, border-color 120ms ease',
          }}
        >
          {view.kind === 'uploading' ? (
            <div role="status" aria-live="polite" style={{ fontSize: 13, color: '#475569' }}>
              <div style={{ fontWeight: 600, overflowWrap: 'anywhere' }}>Reading {view.filename}…</div>
              <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 4 }}>
                You can keep chatting while this finishes.
              </div>
            </div>
          ) : (
            <>
              <div style={{ fontSize: 13, color: '#334155' }}>
                {dragging ? 'Drop your PDF to upload it' : 'Drag & drop your PDF resume here'}
              </div>
              <div style={{ fontSize: 12, color: '#94a3b8', margin: '4px 0 8px' }}>or</div>
              <button type="button" onClick={pick} disabled={busy} style={outlineButton}>
                Upload Resume
              </button>
              <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 8 }}>PDF · up to 2 MB</div>
            </>
          )}
        </div>
      )}

      {view.kind === 'indexed' && (
        <div
          data-testid="resume-ready"
          data-dragging={dragging ? 'true' : 'false'}
          {...drag}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 12,
            padding: '12px 14px',
            borderRadius: 10,
            border: `1px solid ${dragging ? '#6366f1' : '#a7f3d0'}`,
            backgroundColor: dragging ? '#eef2ff' : '#ecfdf5',
          }}
        >
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontSize: 13, fontWeight: 700, color: '#065f46' }}>✓ Resume ready</div>
            <div
              title={view.meta.filename}
              style={{
                fontSize: 13,
                color: '#0f172a',
                marginTop: 2,
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}
            >
              {view.meta.filename}
            </div>
            <div style={{ fontSize: 12, color: '#64748b' }}>
              {view.meta.chunkCount} passages · {view.meta.pageCount} {view.meta.pageCount === 1 ? 'page' : 'pages'}
            </div>
            <div style={{ fontSize: 12, color: '#475569', marginTop: 6 }}>
              I&apos;ll use it when we reach Background and Skill Assessment.
            </div>
          </div>
          <button type="button" onClick={pick} style={outlineButton}>
            Replace
          </button>
        </div>
      )}
    </section>
  );
}
