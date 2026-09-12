/**
 * Server-only configuration for talking to the JobBuddy backend on Fly.
 *
 * Importing this from a client component is a bug: it reads `JOBBUDDY_API_TOKEN`,
 * which has no `NEXT_PUBLIC_` prefix and is therefore `undefined` in the browser.
 * The token is attached here, in server routes, and never travels to the client.
 */

import "server-only";

export function jobbuddyApiUrl(): string {
  const isLocal = process.env.NEXT_PUBLIC_API_ENV === "local";
  const url = isLocal
    ? process.env.JOBBUDDY_API_URL_LOCAL || "http://localhost:8080"
    : process.env.JOBBUDDY_API_URL;

  if (!url) {
    throw new Error(
      "JOBBUDDY_API_URL is not configured. Set it in .env.local (see .env.example)."
    );
  }
  return url.replace(/\/$/, "");
}

/**
 * Headers for a backend call, including the bearer token when one is configured.
 *
 * The token is optional so a local backend running without `AUTH_SECRET` still
 * works — that is the same "local development convenience" branch `verify_bearer`
 * has. A deployed backend always has one, and refuses to start without it.
 */
export function jobbuddyHeaders(): Record<string, string> {
  return { "Content-Type": "application/json", ...jobbuddyAuthHeaders() };
}

/**
 * The bearer token alone, for a multipart upload. `fetch` must set that request's
 * Content-Type itself, because only it knows the multipart boundary.
 */
export function jobbuddyAuthHeaders(): Record<string, string> {
  const token = process.env.JOBBUDDY_API_TOKEN;
  return token ? { Authorization: `Bearer ${token}` } : {};
}

/** The only agent this demo serves. */
export const DEFAULT_AGENT_ID = "xbuddy";
