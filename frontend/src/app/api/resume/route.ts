/**
 * Server-side proxy for a resume upload.
 *
 *   browser  ->  POST /api/resume          multipart (file, thread_id, user_id), no credentials
 *   server   ->  POST {JOBBUDDY_API_URL}/resume   multipart + Authorization: Bearer ...
 *
 * The file is streamed through untouched; nothing here reads, stores, or logs its
 * contents. The browser gets metadata back — filename, pages, passages, when — and
 * never the resume's text, the extracted candidate facts, or the document id.
 *
 * Supabase is never contacted. The backend extracts, embeds, and stores the resume;
 * it is the only component holding credentials for any of that.
 */

import { NextRequest, NextResponse } from "next/server";

import { jobbuddyApiUrl, jobbuddyAuthHeaders } from "@/lib/jobbuddyApi";
import { RESUME_MAX_BYTES } from "@/utils/resume";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// The file plus multipart framing and two short fields.
const MAX_REQUEST_BYTES = RESUME_MAX_BYTES + 64 * 1024;

// Codes whose backend message was written for the user (service.py `_resume_error`
// and the resume extraction/ingestion errors). Anything else — a 500, a validation
// body — is replaced with a generic message rather than forwarded.
const USER_FACING_CODES = new Set([
  "not_pdf",
  "too_large",
  "empty",
  "unreadable",
  "encrypted",
  "too_many_pages",
  "no_text",
  "no_chunks",
  "too_many_chunks",
  "embedding_failed",
  "persistence_failed",
  "invalid_thread",
]);

function refuse(status: number, code: string, message: string) {
  return NextResponse.json({ error: { code, message } }, { status });
}

async function backendError(response: Response) {
  if (response.status === 404) {
    return refuse(404, "not_found", "This conversation could not be found.");
  }
  if (response.status === 429) {
    return refuse(429, "rate_limited", "Too many uploads. Please wait a minute and try again.");
  }
  let detail: { code?: unknown; message?: unknown } | undefined;
  try {
    detail = (await response.json())?.detail;
  } catch {
    detail = undefined;
  }
  if (
    detail &&
    typeof detail.code === "string" &&
    USER_FACING_CODES.has(detail.code) &&
    typeof detail.message === "string"
  ) {
    return refuse(response.status, detail.code, detail.message);
  }
  return refuse(response.status, "upload_failed", "We couldn't process your resume right now. Please try again.");
}

export async function POST(req: NextRequest) {
  // Refuse an oversized body before reading it.
  const declared = Number(req.headers.get("content-length") ?? "0");
  if (declared > MAX_REQUEST_BYTES) {
    return refuse(413, "too_large", "That file is larger than 2 MB. Please upload a smaller PDF.");
  }

  let form: FormData;
  try {
    form = await req.formData();
  } catch {
    return refuse(400, "invalid_form", "Expected a multipart upload.");
  }

  const file = form.get("file");
  const threadId = form.get("thread_id");
  const userId = Number(form.get("user_id"));

  if (!(file instanceof File)) {
    return refuse(400, "invalid_form", "A resume file is required.");
  }
  // Both, always: the backend scopes the resume to exactly this user's thread.
  if (typeof threadId !== "string" || !threadId || !Number.isInteger(userId) || userId <= 0) {
    return refuse(400, "invalid_form", "thread_id (string) and user_id (number) are both required");
  }
  if (file.size > RESUME_MAX_BYTES) {
    return refuse(413, "too_large", "That file is larger than 2 MB. Please upload a smaller PDF.");
  }

  const outbound = new FormData();
  outbound.append("file", file, file.name || "resume.pdf");
  outbound.append("thread_id", threadId);
  outbound.append("user_id", String(userId));

  try {
    const response = await fetch(`${jobbuddyApiUrl()}/resume`, {
      method: "POST",
      headers: jobbuddyAuthHeaders(),
      body: outbound,
      cache: "no-store",
    });

    if (!response.ok) {
      return await backendError(response);
    }

    const data = await response.json();
    // Indexed only when the backend says so — it answers 200 only after the
    // resume was persisted, and this route does not second-guess it either way.
    if (data?.indexed !== true) {
      return refuse(502, "upload_failed", "We couldn't process your resume right now. Please try again.");
    }
    return NextResponse.json({
      indexed: true,
      filename: data.filename,
      page_count: data.page_count,
      chunk_count: data.chunk_count,
      indexed_at: data.indexed_at,
    });
  } catch (error) {
    console.error("[/api/resume] proxy failure:", error instanceof Error ? error.message : "unknown");
    return refuse(502, "unreachable", "Could not reach the backend. Please try again.");
  }
}
