/**
 * Server-side proxy for conversation restoration.
 *
 *   browser  ->  POST /api/history        (no credentials)
 *   server   ->  POST {JOBBUDDY_API_URL}/history   + Authorization: Bearer ...
 *
 * The backend scopes the read by `user_id`, so both identifiers are required and
 * are forwarded unchanged. A thread belonging to a different user comes back as
 * 404 with no content — that is the backend's boundary, not something this route
 * re-implements.
 *
 * Supabase is never contacted here. History comes from the LangGraph checkpoint
 * through the backend, which is the only component holding credentials for it.
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

  // Both, always. Dropping user_id would ask the backend for an unscoped read,
  // which it rejects with 422 — better to fail here with a clear reason.
  if (!thread_id || typeof user_id !== "number") {
    return NextResponse.json(
      { error: "thread_id (string) and user_id (number) are both required" },
      { status: 400 }
    );
  }

  try {
    const response = await fetch(`${jobbuddyApiUrl()}/history`, {
      method: "POST",
      headers: jobbuddyHeaders(),
      body: JSON.stringify({ thread_id, user_id }),
      cache: "no-store",
    });

    // 404 means "no such thread for this user", which for a fresh demo identity is
    // the normal first-load answer rather than an error. Passed through as an empty
    // transcript so the client does not have to special-case a status code.
    if (response.status === 404) {
      return NextResponse.json({ thread_id, user_id, messages: [] });
    }

    if (!response.ok) {
      // Never forward the backend's body — it may carry internal detail.
      return NextResponse.json(
        { error: `Backend returned ${response.status}` },
        { status: response.status }
      );
    }

    return NextResponse.json(await response.json());
  } catch (error) {
    console.error("[/api/history] proxy failure:", error);
    return NextResponse.json({ error: "Could not reach the backend" }, { status: 502 });
  }
}
