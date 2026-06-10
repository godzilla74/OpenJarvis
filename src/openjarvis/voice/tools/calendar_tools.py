"""Google Calendar action tools for the voice assistant."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import httpx

_GCAL_BASE = "https://www.googleapis.com/calendar/v3"


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def format_event_summary(event: dict) -> str:
    title = event.get("summary", "Untitled event")
    start = event.get("start", {})
    dt_str = start.get("dateTime") or start.get("date", "")
    location = event.get("location", "")
    try:
        dt = datetime.fromisoformat(dt_str)
        time_str = dt.strftime("%A %b %d at %I:%M %p")
    except ValueError:
        time_str = dt_str
    parts = [f"{title} — {time_str}"]
    if location:
        parts.append(f"at {location}")
    return " ".join(parts)


def list_events(
    token: str,
    *,
    calendar_id: str = "primary",
    time_min: Optional[str] = None,
    time_max: Optional[str] = None,
    max_results: int = 10,
) -> list[str]:
    """Return event summary strings for the given calendar and time range."""
    params: dict = {
        "maxResults": max_results,
        "orderBy": "startTime",
        "singleEvents": "true",
    }
    if time_min:
        params["timeMin"] = time_min
    if time_max:
        params["timeMax"] = time_max
    if not time_min:
        params["timeMin"] = datetime.now(timezone.utc).isoformat()

    resp = httpx.get(
        f"{_GCAL_BASE}/calendars/{calendar_id}/events",
        headers=_headers(token),
        params=params,
    )
    resp.raise_for_status()
    return [format_event_summary(e) for e in resp.json().get("items", [])]


def create_event(
    token: str,
    *,
    title: str,
    start_datetime: str,
    end_datetime: str,
    description: str = "",
    location: str = "",
    calendar_id: str = "primary",
) -> str:
    """Create a calendar event. Returns the new event ID."""
    body = {
        "summary": title,
        "start": {"dateTime": start_datetime},
        "end": {"dateTime": end_datetime},
    }
    if description:
        body["description"] = description
    if location:
        body["location"] = location
    resp = httpx.post(
        f"{_GCAL_BASE}/calendars/{calendar_id}/events",
        headers=_headers(token),
        json=body,
    )
    resp.raise_for_status()
    return resp.json()["id"]


def update_event(
    token: str,
    *,
    event_id: str,
    calendar_id: str = "primary",
    **fields,
) -> None:
    """Patch an existing event with the provided fields."""
    resp = httpx.patch(
        f"{_GCAL_BASE}/calendars/{calendar_id}/events/{event_id}",
        headers=_headers(token),
        json=fields,
    )
    resp.raise_for_status()
