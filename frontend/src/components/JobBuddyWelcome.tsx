'use client';

/**
 * The first-use card for a conversation with no history yet.
 *
 * This is **product guidance, not a model response.** It is rendered as its own
 * component, never pushed into `messages`, so it cannot be mistaken for assistant
 * output, cannot be copied by Copy All as if JobBuddy had said it, and cannot end up
 * in a transcript the backend never saw. Nothing here calls `/api/chat`.
 *
 * It disappears as soon as the thread has real messages, so a restored conversation
 * shows its transcript instead of being re-introduced on every refresh.
 */

const AREAS = [
  'Career Goal',
  'Background',
  'Job Preferences',
  'Skill Assessment',
  'Action Plan',
];

export default function JobBuddyWelcome() {
  return (
    <div
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
        I&apos;m your AI career planning assistant. We&apos;ll work through five areas
        together:
      </p>

      <ol
        style={{
          margin: '14px 0 0',
          paddingLeft: 20,
          fontSize: 14,
          lineHeight: 1.9,
          color: '#0f172a',
        }}
      >
        {AREAS.map((area) => (
          <li key={area}>{area}</li>
        ))}
      </ol>

      <p style={{ fontSize: 14, lineHeight: 1.6, color: '#475569', margin: '14px 0 0' }}>
        As we talk I&apos;ll organise your answers into structured progress — you can
        follow it in the panel on the left — and build a personalised job-search
        strategy from it.
      </p>

      <p
        style={{
          fontSize: 14,
          lineHeight: 1.6,
          color: '#0f172a',
          fontWeight: 600,
          margin: '14px 0 0',
        }}
      >
        To get started, tell me what kind of career move or role you&apos;re considering.
      </p>
    </div>
  );
}
