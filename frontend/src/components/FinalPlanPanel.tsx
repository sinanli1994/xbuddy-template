'use client';

import ReactMarkdown from 'react-markdown';

/**
 * The finished career plan, shown as a read-only overlay.
 *
 * An overlay rather than a route: the plan belongs to the conversation you are
 * already looking at, and pushing it to its own page would drop the chat context for
 * something you read once and dismiss.
 *
 * Read-only by construction — there is no editor, no regenerate, no export. The
 * backend owns the document; this renders the Markdown it returns and nothing else.
 * Nothing here is ever fabricated: when retrieval fails the caller shows an error
 * state instead of this panel.
 */

interface Props {
  plan: string;
  onClose: () => void;
}

export default function FinalPlanPanel({ plan, onClose }: Props) {
  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Final career plan"
      onClick={onClose}
      style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'rgba(15, 23, 42, 0.45)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 24,
        zIndex: 50,
      }}
    >
      <div
        // The backdrop closes the panel; a click inside it must not.
        onClick={(event) => event.stopPropagation()}
        style={{
          backgroundColor: '#ffffff',
          borderRadius: 12,
          border: '1px solid #e2e8f0',
          boxShadow: '0 10px 40px rgba(0,0,0,0.18)',
          width: 'min(760px, 100%)',
          maxHeight: '86vh',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            flexShrink: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 16,
            padding: '16px 24px',
            borderBottom: '1px solid #e2e8f0',
          }}
        >
          <h2 style={{ margin: 0, fontSize: 17, fontWeight: 700, color: '#0f172a' }}>
            Final Career Plan
          </h2>
          <button
            onClick={onClose}
            aria-label="Close the final plan"
            style={{
              flexShrink: 0,
              width: 28,
              height: 28,
              border: 'none',
              borderRadius: 6,
              backgroundColor: '#f1f5f9',
              color: '#475569',
              cursor: 'pointer',
              fontSize: 14,
              lineHeight: 1,
            }}
          >
            ✕
          </button>
        </div>

        {/* The plan is the only scroll region in the panel. */}
        <div
          style={{
            flex: 1,
            minHeight: 0,
            overflowY: 'auto',
            padding: '20px 24px 28px',
            fontSize: 14,
            lineHeight: 1.65,
            color: '#1e293b',
          }}
        >
          <ReactMarkdown
            components={{
              h1: ({ children }) => (
                <h1 style={{ fontSize: 20, fontWeight: 700, margin: '0 0 12px', color: '#0f172a' }}>
                  {children}
                </h1>
              ),
              h2: ({ children }) => (
                <h2
                  style={{
                    fontSize: 15,
                    fontWeight: 700,
                    margin: '22px 0 8px',
                    color: '#0f172a',
                  }}
                >
                  {children}
                </h2>
              ),
              h3: ({ children }) => (
                <h3 style={{ fontSize: 14, fontWeight: 600, margin: '16px 0 6px' }}>{children}</h3>
              ),
              p: ({ children }) => <p style={{ margin: '0 0 10px' }}>{children}</p>,
              ul: ({ children }) => (
                <ul style={{ margin: '0 0 12px', paddingLeft: 22 }}>{children}</ul>
              ),
              ol: ({ children }) => (
                <ol style={{ margin: '0 0 12px', paddingLeft: 22 }}>{children}</ol>
              ),
              li: ({ children }) => <li style={{ margin: '4px 0' }}>{children}</li>,
              strong: ({ children }) => <strong style={{ fontWeight: 600 }}>{children}</strong>,
            }}
          >
            {plan}
          </ReactMarkdown>
        </div>
      </div>
    </div>
  );
}
