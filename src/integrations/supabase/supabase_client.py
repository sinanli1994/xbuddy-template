"""Supabase client for database operations."""
import logging
from typing import Optional
from supabase import create_client, Client
from core.settings import settings

logger = logging.getLogger(__name__)

_supabase_client: Optional[Client] = None


# Backend key candidates, in resolution order. The newer sb_secret_... key wins;
# the legacy service-role JWT is the fallback so existing deployments keep working.
# Both bypass RLS, which is required: new tables get RLS enabled with no policies,
# so a publishable key would be denied every write.
_BACKEND_KEY_VARS = ("SUPABASE_SECRET_KEY", "SUPABASE_SERVICE_ROLE_KEY")


def _resolve_backend_key() -> tuple[str | None, str | None]:
    """Return (key_value, source_variable_name) for the server-side key.

    Only the *name* of the variable is ever surfaced to callers for logging —
    never the value.
    """
    for name in _BACKEND_KEY_VARS:
        raw = getattr(settings, name, None)
        if raw is None:
            continue
        value = raw.get_secret_value() if hasattr(raw, 'get_secret_value') else str(raw)
        if value and value.strip():
            return value, name
    return None, None


def get_supabase_client() -> Client:
    """Get or create Supabase client instance."""
    global _supabase_client

    if _supabase_client is None:
        supabase_url = getattr(settings, 'SUPABASE_URL', None)
        key_value, key_source = _resolve_backend_key()

        # Presence and source only — never the key itself.
        logger.info(
            "Supabase config check - URL: %s, key source: %s",
            supabase_url is not None,
            key_source or "none",
        )

        if not supabase_url or not key_value:
            error_msg = (
                "Supabase credentials not configured. Set SUPABASE_URL and "
                "SUPABASE_SECRET_KEY (or the legacy SUPABASE_SERVICE_ROLE_KEY)."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Creating Supabase client with URL: {supabase_url}")
        _supabase_client = create_client(supabase_url, key_value)
        logger.info("✅ Supabase client initialized successfully (key: %s)", key_source)

    return _supabase_client


class SupabaseClient:
    """Wrapper class for Supabase operations."""
    
    def __init__(self):
        self.client = get_supabase_client()
    
    def save_business_plan(
        self,
        user_id: int,
        thread_id: str,
        content: str,
        markdown_content: str,
        agent_id: str = "founder-buddy"
    ) -> dict:
        """Save business plan to Supabase (synchronous operation)."""
        try:
            logger.info(f"Saving business plan to Supabase - user_id: {user_id}, thread_id: {thread_id}, content_length: {len(content)}")
            
            result = self.client.table("business_plans").upsert({
                "user_id": user_id,
                "thread_id": thread_id,
                "agent_id": agent_id,
                "content": content,
                "markdown_content": markdown_content,
                "updated_at": "now()"
            }, on_conflict="user_id,thread_id").execute()
            
            logger.info(f"✅ Business plan saved successfully for user {user_id}, thread {thread_id}")
            logger.debug(f"Supabase response: {result.data if hasattr(result, 'data') else 'No data'}")
            return {"success": True, "data": result.data}
        except Exception as e:
            logger.error(f"❌ Error saving business plan to Supabase: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    def save_section_state(
        self,
        user_id: int,
        thread_id: str,
        section_id: str,
        content: dict,  # Tiptap JSON
        plain_text: str,
        status: str,
        satisfaction_status: Optional[str] = None,
        agent_id: str = "xbuddy"
    ) -> dict:
        """Save section state to Supabase (synchronous operation).

        `on_conflict` names the exact columns of the table's
        UNIQUE(user_id, thread_id, section_id) constraint. Without it the
        conflict target defaults to the UUID primary key; since no `id` is sent,
        a repeat write for the same section INSERTs and violates that unique
        constraint. The explicit target is what makes writes idempotent.

        `agent_id` defaults to "xbuddy" so it matches what the read paths filter
        on (service.py queries .eq("agent_id", agent_id) with the agent key).
        The old "founder-buddy" default wrote rows that those reads could never
        find. Note the column is not part of the unique constraint, so a second
        agent sharing a thread_id would update this row rather than add its own —
        tracked as a deferred schema improvement.
        """
        try:
            result = self.client.table("section_states").upsert(
                {
                    "user_id": user_id,
                    "thread_id": thread_id,
                    "agent_id": agent_id,
                    "section_id": section_id,
                    "content": content,
                    "plain_text": plain_text,
                    "status": status,
                    "satisfaction_status": satisfaction_status,
                    "updated_at": "now()",
                },
                on_conflict="user_id,thread_id,section_id",
            ).execute()

            logger.info(f"Section state saved: {section_id} for user {user_id}")
            return {"success": True, "data": result.data}
        except Exception as e:
            logger.error(f"Error saving section state: {e}")
            return {"success": False, "error": str(e)}
    
    def get_business_plan(
        self,
        user_id: int,
        thread_id: str
    ) -> Optional[dict]:
        """Get business plan from Supabase (synchronous operation)."""
        try:
            result = self.client.table("business_plans")\
                .select("*")\
                .eq("user_id", user_id)\
                .eq("thread_id", thread_id)\
                .maybe_single()\
                .execute()
            
            return result.data if result.data else None
        except Exception as e:
            logger.error(f"Error getting business plan: {e}")
            return None
    
    def get_section_states(
        self,
        user_id: int,
        thread_id: str
    ) -> list[dict]:
        """Get all section states for a thread (synchronous operation)."""
        try:
            result = self.client.table("section_states")\
                .select("*")\
                .eq("user_id", user_id)\
                .eq("thread_id", thread_id)\
                .execute()
            
            return result.data if result.data else []
        except Exception as e:
            logger.error(f"Error getting section states: {e}")
            return []

