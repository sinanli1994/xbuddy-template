/**
 * Resume card state and the rules around it. Pure: no fetch, no storage, no DOM.
 *
 * A resume belongs to exactly one conversation — the backend scopes it by
 * (user_id, thread_id) — so the card's state carries the thread it describes and is
 * shown only while that thread is selected. Nothing about a resume is kept in the
 * browser beyond this in-memory view: the backend is the only record.
 */

/** The backend's limit (service.py RESUME_MAX_BYTES); checked here only to fail fast. */
export const RESUME_MAX_BYTES = 2 * 1024 * 1024;

export interface ResumeMeta {
  filename: string;
  pageCount: number;
  chunkCount: number;
  indexedAt: string;
}

export type ResumeView =
  | { kind: 'checking' }
  | { kind: 'none' }
  | { kind: 'uploading'; filename: string; previous: ResumeMeta | null }
  | { kind: 'indexed'; meta: ResumeMeta }
  | { kind: 'error'; message: string; retry: 'upload' | 'status'; previous: ResumeMeta | null };

/** A view, tagged with the one conversation it describes. */
export interface ThreadResume {
  threadId: string;
  view: ResumeView;
}

/** The view for the selected conversation, or null — never another thread's. */
export function resumeForThread(resume: ThreadResume | null, threadId: string | null): ResumeView | null {
  if (!resume || !threadId || resume.threadId !== threadId) return null;
  return resume.view;
}

/** The resume that is in effect behind a transient view, if any. */
export function currentResume(view: ResumeView | null): ResumeMeta | null {
  if (!view) return null;
  if (view.kind === 'indexed') return view.meta;
  if (view.kind === 'uploading' || view.kind === 'error') return view.previous;
  return null;
}

/** A client-side refusal, or null when the file may be sent. The backend re-checks all of it. */
export function checkResumeFile(file: { name: string; type: string; size: number }): string | null {
  const isPdf = file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf');
  if (!isPdf) return 'Please upload your resume as a PDF.';
  if (file.size === 0) return 'That file is empty.';
  if (file.size > RESUME_MAX_BYTES) return 'That file is larger than 2 MB. Please upload a smaller PDF.';
  return null;
}

/** Metadata from a proxy response; null when the response does not describe an indexed resume. */
export function metaFromResponse(data: unknown): ResumeMeta | null {
  if (!data || typeof data !== 'object') return null;
  const d = data as Record<string, unknown>;
  if (
    typeof d.filename !== 'string' ||
    typeof d.page_count !== 'number' ||
    typeof d.chunk_count !== 'number' ||
    typeof d.indexed_at !== 'string'
  ) {
    return null;
  }
  return { filename: d.filename, pageCount: d.page_count, chunkCount: d.chunk_count, indexedAt: d.indexed_at };
}

/** The message a failed proxy response carries, or a plain fallback. */
export function errorMessageFrom(data: unknown, status: number): string {
  const error = (data as { error?: { message?: unknown } } | null)?.error;
  if (error && typeof error.message === 'string' && error.message.trim()) return error.message;
  return `The resume could not be processed (${status}).`;
}
