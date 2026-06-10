"""Action classification and voice confirmation prompt generation."""
from __future__ import annotations

from enum import Enum


class ActionType(Enum):
    READ = "read"       # No confirmation needed
    CREATE = "create"   # Brief confirmation
    MODIFY = "modify"   # Brief confirmation
    SEND = "send"       # Full readback


_ACTION_MAP: dict[str, ActionType] = {
    "list_emails": ActionType.READ,
    "read_email": ActionType.READ,
    "get_email": ActionType.READ,
    "list_calendar_events": ActionType.READ,
    "list_tasks": ActionType.READ,
    "send_email": ActionType.SEND,
    "reply_email": ActionType.SEND,
    "forward_email": ActionType.SEND,
    "create_calendar_event": ActionType.CREATE,
    "add_task": ActionType.CREATE,
    "complete_task": ActionType.MODIFY,
    "update_calendar_event": ActionType.MODIFY,
    "reschedule_event": ActionType.MODIFY,
}


def classify_action(tool_name: str) -> ActionType:
    """Return the ActionType for a given tool function name."""
    return _ACTION_MAP.get(tool_name, ActionType.READ)


def format_confirmation(action_type: ActionType, **kwargs: str) -> str:
    """Build a natural spoken confirmation prompt for the given action."""
    if action_type == ActionType.SEND:
        to = kwargs.get("to", "")
        subject = kwargs.get("subject", "")
        body = kwargs.get("body", "")
        return (
            f"I'll send an email to {to}. "
            f"Subject: {subject}. "
            f"Message: {body}. "
            f"Send this?"
        )
    if action_type == ActionType.CREATE:
        title = kwargs.get("title", "")
        when = kwargs.get("when", "")
        where = kwargs.get("where", "")
        parts = [f"I'll create: {title}"]
        if when:
            parts.append(f"at {when}")
        if where:
            parts.append(f"at {where}")
        return " ".join(parts) + ". Go ahead?"
    if action_type == ActionType.MODIFY:
        description = kwargs.get("description", "make that change")
        return f"I'll {description}. Go ahead?"
    return ""  # READ — no confirmation needed
