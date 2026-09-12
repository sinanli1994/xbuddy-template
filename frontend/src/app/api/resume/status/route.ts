/**
 * Server-side proxy for a conversation's resume status.
 *
 *   browser  ->  POST /api/resume/status          (no credentials)
 *   server   ->  POST {JOBBUDDY_API_URL}/resume/status  + Authorization: Bearer ...
 *
 * Restored with the conversation, beside /api/history and /api/completion. One
 * indexed read on the backend: no model call, no embedding. Metadata only.
 *
 * Supabase is never contacted here.
 */

import { NextRequest, NextResponse } from "next/server";

import { jobbuddyApiUrl, jobbuddyHeaders } from "@/lib/jobbuddyApi";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(req: NextRequest) {
  let body: { thread_id?: string; user_id?: number };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  const { thread_id, user_id } = body;

  // Both, always: a resume belongs to exactly one user's thread.
  if (!thread_id || typeof user_id !== "number") {
    return NextResponse.json(
      { error: "thread_id (string) and user_id (number) are both required" },
      { status: 400 }
    );
  }

  try {
    const response = await fetch(`${jobbuddyApiUrl()}/resume/status`, {
      method: "POST",
      headers: jobbuddyHeaders(),
      body: JSON.stringify({ thread_id, user_id }),
      cache: "no-store",
    });

    // 404 is "not this user's thread" — answered as no resume, which is all the
    // caller may know. Same convention as /api/completion.
    if (response.status === 404) {
      return NextResponse.json({ has_resume: false });
    }

    if (!response.ok) {
      // Never forward the backend's body. A failure is not "no resume": saying so
      // would invite a needless re-upload.
      return NextResponse.json(
        { error: { code: "status_unavailable", message: "We couldn't check your resume right now." } },
        { status: response.status }
      );
    }

    const data = await response.json();
    if (data?.has_resume !== true) {
      return NextResponse.json({ has_resume: false });
    }
    return NextResponse.json({
      has_resume: true,
      filename: data.filename,
      page_count: data.page_count,
      chunk_count: data.chunk_count,
      indexed_at: data.indexed_at,
    });
  } catch (error) {
    console.error("[/api/resume/status] proxy failure:", error instanceof Error ? error.message : "unknown");
    return NextResponse.json(
      { error: { code: "unreachable", message: "We couldn't check your resume right now." } },
      { status: 502 }
    );
  }
}
