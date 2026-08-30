/**
 * Server-side proxy for the finished career plan.
 *
 *   browser  ->  POST /api/final-output          (no credentials)
 *   server   ->  POST {JOBBUDDY_API_URL}/final_output  + Authorization: Bearer ...
 *
 * A read. It never reaches a model: the backend serves the plan the graph already
 * generated and stored, so opening the panel costs nothing.
 *
 * Supabase is never contacted. The `final-outputs` table exists, but the backend owns
 * it; the browser has no credentials for it and gets no path to it here.
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

  if (!thread_id || typeof user_id !== "number") {
    return NextResponse.json(
      { error: "thread_id (string) and user_id (number) are both required" },
      { status: 400 }
    );
  }

  try {
    const response = await fetch(`${jobbuddyApiUrl()}/final_output`, {
      method: "POST",
      headers: jobbuddyHeaders(),
      body: JSON.stringify({ thread_id, user_id }),
      cache: "no-store",
    });

    // 404 is "not this user's thread". Answered as "no plan" rather than an error:
    // it is the same thing a caller may know about a thread that is not theirs.
    if (response.status === 404) {
      return NextResponse.json({
        thread_id,
        user_id,
        artifact_available: false,
        final_output: null,
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
    console.error("[/api/final-output] proxy failure:", error);
    return NextResponse.json({ error: "Could not reach the backend" }, { status: 502 });
  }
}
