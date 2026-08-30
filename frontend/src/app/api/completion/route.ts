/**
 * Server-side proxy for the public completion projection.
 *
 *   browser  ->  POST /api/completion         (no credentials)
 *   server   ->  POST {JOBBUDDY_API_URL}/completion  + Authorization: Bearer ...
 *
 * The refresh counterpart to `/api/history`. A reload restores the transcript from
 * one and the progress beside it from the other; neither is derived from the other,
 * because reconstructing progress from message text would be a guess.
 *
 * Supabase is never contacted. Progress comes from the LangGraph checkpoint through
 * the backend, which is the only component holding credentials for it.
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

  // Both, always. The backend scopes the read by user_id and rejects a request
  // without one; failing here says why.
  if (!thread_id || typeof user_id !== "number") {
    return NextResponse.json(
      { error: "thread_id (string) and user_id (number) are both required" },
      { status: 400 }
    );
  }

  try {
    const response = await fetch(`${jobbuddyApiUrl()}/completion`, {
      method: "POST",
      headers: jobbuddyHeaders(),
      body: JSON.stringify({ thread_id, user_id }),
      cache: "no-store",
    });

    // 404 is "not this user's thread". Answered as the empty projection so a client
    // renders five pending sections rather than a broken panel — it is the same
    // thing a brand-new thread looks like, which is all the caller may know.
    if (response.status === 404) {
      return NextResponse.json({
        collection_complete: false,
        artifact_available: false,
        sections: [],
      });
    }

    if (!response.ok) {
      // Never forward the backend's body; it may carry internal detail.
      return NextResponse.json(
        { error: `Backend returned ${response.status}` },
        { status: response.status }
      );
    }

    return NextResponse.json(await response.json());
  } catch (error) {
    console.error("[/api/completion] proxy failure:", error);
    return NextResponse.json({ error: "Could not reach the backend" }, { status: 502 });
  }
}
