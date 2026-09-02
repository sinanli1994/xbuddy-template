"""Grounded document titles, assembled before rendering or persistence.

Role identity uses conservative text normalization, not a model's guess about
occupations. Only explicit focus clauses supply a specialization; skills, proposed
actions and model-authored prose never supply facts for this title.
"""

import re
import unicodedata

from .models import XBuddyData

TITLE_MAX = 120
_FOCUS = re.compile(
    r"\s+(?:with (?:a )?focus on|focused on|focusing on|focus on|"
    r"speciali[sz](?:ing|ed) in)\s+", re.IGNORECASE,
)
_UNKNOWN = {"", "unknown", "not specified", "not provided", "not applicable", "n/a", "none",
            "unemployed", "not currently working", "between roles"}


def _text(value: str | None) -> str:
    return " ".join(unicodedata.normalize("NFKC", value or "").split()).strip(" .,:;")


def _parts(value: str | None) -> tuple[str, str]:
    parts = _FOCUS.split(_text(value), maxsplit=1)
    if len(parts) == 1:
        trailing_focus = re.fullmatch(r"(.+?)\s+with\s+(.+?)\s+focus", parts[0], re.IGNORECASE)
        if trailing_focus:
            parts = list(trailing_focus.groups())
    role = re.sub(r"\s+(?:roles?|positions?)$", "", parts[0], flags=re.IGNORECASE)
    focus = parts[1] if len(parts) == 2 else ""
    # Timeline belongs in the body, not a long headline. Never add one.
    focus = re.split(r"[.;]|\s+(?:within|over the next|in the next)\s+", focus, maxsplit=1,
                     flags=re.IGNORECASE)[0]
    return role, _text(focus)


def _bound(title: str) -> str:
    if len(title) <= TITLE_MAX:
        return title
    return title[:TITLE_MAX - 1].rsplit(" ", 1)[0].rstrip(" ,:;-") + "…"


def career_plan_title(data: XBuddyData) -> str:
    """Same/uncertain role => neutral plan; distinct known roles => transition.

    Ambiguous aliases are not resolved into invented equivalences. Multiple target
    roles use a neutral title, without asserting one definitive career transition.
    """
    targets = [_parts(value) for value in data.target_roles if _text(value).casefold() not in _UNKNOWN]
    if not targets:
        goal = _text(data.career_goal_summary)
        return _bound(f"Career Plan: {goal}" if goal else "Your Career Plan")
    if len(targets) > 1:
        return _bound("Career Plan: " + " / ".join(role for role, _ in targets))
    target, focus = targets[0]
    current, _ = _parts(data.current_role)
    known_current = current.casefold() not in _UNKNOWN
    related_labels = (current.casefold() == target.casefold()
                      or target.casefold().startswith(current.casefold() + " ")
                      or current.casefold().startswith(target.casefold() + " "))
    if known_current and not related_labels:
        return _bound(f"Transition from {current} to {target}")
    if not focus:
        # An explicitly stated focus in the goal is usable; skill gaps alone are not.
        goal_prefix, goal_focus = _parts(data.career_goal_summary)
        if not re.search(r"\b(?:not|never|avoid|without|no)\b", goal_prefix, re.IGNORECASE):
            focus = goal_focus
    return _bound(f"{target} Career Plan" + (f": {focus}" if focus else ""))
