"""Live verification of PR 5 final-output persistence against real Supabase.

Explicitly opt-in. Nothing in `tests/` imports this, the filename does not match
`test_*.py`, and it sits outside `tests/` — so ordinary pytest never collects or
runs it. It performs real writes, which is exactly why it is not automatic.

    uv run python scripts/verify_final_output_persistence.py            # verify
    uv run python scripts/verify_final_output_persistence.py --keep     # skip cleanup

What it proves, in order:

    1. the `final-outputs` table exists and is reachable
    2. a first write inserts one row, status=current, hash recorded
    3. a second identical logical write leaves one row (idempotent)
    4. the stale transition preserves content
    5. safe regeneration overwrites content that is still the agent's own
    6. a simulated downstream edit is detected and NOT overwritten
    7. cleanup removes the rows it created

Secrets are never printed. The only environment detail reported is which variable
*name* supplied the key, never its value — the same discipline as
`supabase_client._resolve_backend_key`.

It writes under a dedicated thread_id prefixed `pr5-verify-` plus a random suffix,
so it cannot collide with real conversation data, and it deletes what it creates
unless `--keep` is given.
"""

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

load_dotenv(ROOT / ".env")

from agents.xbuddy.persistence import (
    STATUS_CURRENT,
    STATUS_STALE,
    content_fingerprint,
    is_downstream_edited,
    mark_final_output_stale,
    persist_final_output,
)

VERIFY_USER_ID = 999_001
FIRST = "# Verify run\n\n## Your Action Plan\n1. **First generation**\n"
REGENERATED = "# Verify run\n\n## Your Action Plan\n1. **Second generation**\n"
EDITED = FIRST + "\n\nA note the user typed themselves.\n"

passed: list[str] = []
failed: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    (passed if condition else failed).append(label)
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {label}{f' — {detail}' if detail else ''}")


def client():
    from integrations.supabase.supabase_client import SupabaseClient

    return SupabaseClient()


def read_row(thread_id: str) -> dict | None:
    return client().get_final_output(user_id=VERIFY_USER_ID, thread_id=thread_id)


def row_count(thread_id: str) -> int:
    result = (
        client()
        .client.table("final-outputs")
        .select("id")
        .eq("user_id", VERIFY_USER_ID)
        .eq("thread_id", thread_id)
        .execute()
    )
    return len(result.data or [])


async def main(keep: bool) -> int:
    thread_id = f"pr5-verify-{uuid.uuid4().hex[:8]}"
    print(f"Live verification against thread_id={thread_id}\n")

    # 0. Reachability, without revealing anything sensitive.
    try:
        from integrations.supabase.supabase_client import _resolve_backend_key

        _key, source = _resolve_backend_key()
        print(f"  backend key resolved from: {source}")
    except Exception as exc:  # noqa: BLE001 - report and stop
        print(f"  [FAIL] could not resolve Supabase credentials: {type(exc).__name__}")
        return 1

    try:
        read_row(thread_id)
        check("table `final-outputs` exists and is readable", True)
    except Exception as exc:  # noqa: BLE001
        check("table `final-outputs` exists and is readable", False, type(exc).__name__)
        print("\nApply supabase/migrations/002_final_outputs.sql, then re-run.")
        return 1

    try:
        # 1. First write.
        ok, reason = await persist_final_output(VERIFY_USER_ID, thread_id, FIRST)
        check("first write succeeds", ok, reason or "")
        row = read_row(thread_id)
        check("exactly one row exists", row_count(thread_id) == 1)
        check("status is current", (row or {}).get("status") == STATUS_CURRENT)
        check(
            "generated hash recorded",
            (row or {}).get("generated_content_hash") == content_fingerprint(FIRST),
        )

        # 2. Identical write is idempotent.
        ok, _ = await persist_final_output(VERIFY_USER_ID, thread_id, FIRST)
        check("second identical write succeeds", ok)
        check("still exactly one row", row_count(thread_id) == 1)

        # 3. Stale transition retains content.
        marked, reason = await mark_final_output_stale(VERIFY_USER_ID, thread_id)
        row = read_row(thread_id)
        check("stale marking succeeds", marked, reason or "")
        check("status is stale", (row or {}).get("status") == STATUS_STALE)
        check(
            "stale row retains its content",
            (row or {}).get("markdown_content") == FIRST,
        )
        check("stale row is still one row", row_count(thread_id) == 1)

        # 4. Safe regeneration overwrites untouched agent content.
        ok, reason = await persist_final_output(VERIFY_USER_ID, thread_id, REGENERATED)
        row = read_row(thread_id)
        check("regeneration over untouched content succeeds", ok, reason or "")
        check(
            "content replaced by the regenerated artifact",
            (row or {}).get("markdown_content") == REGENERATED,
        )
        check("status back to current", (row or {}).get("status") == STATUS_CURRENT)
        check(
            "hash updated to the new generation",
            (row or {}).get("generated_content_hash") == content_fingerprint(REGENERATED),
        )
        check("still exactly one row", row_count(thread_id) == 1)

        # 5. Simulated downstream edit must be detected and preserved. Written the
        #    way the frontend writes: content columns only, hash column untouched.
        client().client.table("final-outputs").update(
            {"content": EDITED, "markdown_content": EDITED}
        ).eq("user_id", VERIFY_USER_ID).eq("thread_id", thread_id).execute()
        row = read_row(thread_id)
        check("downstream edit detected by hash mismatch", is_downstream_edited(row))

        ok, reason = await persist_final_output(
            VERIFY_USER_ID, thread_id, "# Should never be written\n"
        )
        row = read_row(thread_id)
        check("regeneration over edited content is refused", ok is False)
        check(
            "refusal names the reason",
            bool(reason) and "has been edited" in (reason or ""),
        )
        check("edited content preserved intact", (row or {}).get("markdown_content") == EDITED)
        check("still exactly one row", row_count(thread_id) == 1)

    finally:
        if keep:
            print(f"\n--keep: leaving rows for thread_id={thread_id}")
        else:
            try:
                client().client.table("final-outputs").delete().eq(
                    "user_id", VERIFY_USER_ID
                ).eq("thread_id", thread_id).execute()
                remaining = row_count(thread_id)
                check("cleanup removed the test rows", remaining == 0, f"{remaining} left")
            except Exception as exc:  # noqa: BLE001
                check("cleanup removed the test rows", False, type(exc).__name__)

    print(f"\n{len(passed)} passed, {len(failed)} failed")
    if failed:
        for label in failed:
            print(f"  failed: {label}")
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep", action="store_true", help="Leave the verification rows in place."
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.keep)))
