"""Google Tasks action tools for the voice assistant."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import httpx

_TASKS_BASE = "https://tasks.googleapis.com/tasks/v1"


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def format_task_summary(task: dict) -> str:
    title = task.get("title", "Untitled task")
    due = task.get("due", "")
    if due:
        try:
            dt = datetime.fromisoformat(due.replace("Z", "+00:00"))
            due_str = dt.strftime("due %b %d")
        except ValueError:
            due_str = due
        return f"{title} ({due_str})"
    return title


def _get_task_lists(token: str) -> list[dict]:
    resp = httpx.get(f"{_TASKS_BASE}/users/@me/lists", headers=_headers(token))
    resp.raise_for_status()
    return resp.json().get("items", [])


def list_tasks(
    token: str,
    *,
    task_list_id: Optional[str] = None,
    show_completed: bool = False,
) -> list[str]:
    """Return task summary strings from all task lists (or a specific one)."""
    if task_list_id:
        lists = [{"id": task_list_id}]
    else:
        lists = _get_task_lists(token)

    summaries: list[str] = []
    for tl in lists:
        params = {"showCompleted": str(show_completed).lower(), "showHidden": "false"}
        resp = httpx.get(
            f"{_TASKS_BASE}/lists/{tl['id']}/tasks",
            headers=_headers(token),
            params=params,
        )
        resp.raise_for_status()
        for task in resp.json().get("items", []):
            if task.get("status") != "completed":
                summaries.append(format_task_summary(task))
    return summaries


def add_task(
    token: str,
    *,
    title: str,
    notes: str = "",
    due: Optional[str] = None,
    task_list_id: Optional[str] = None,
) -> str:
    """Add a task. Returns the new task ID."""
    if not task_list_id:
        lists = _get_task_lists(token)
        task_list_id = lists[0]["id"] if lists else "@default"
    body: dict = {"title": title}
    if notes:
        body["notes"] = notes
    if due:
        body["due"] = due
    resp = httpx.post(
        f"{_TASKS_BASE}/lists/{task_list_id}/tasks",
        headers=_headers(token),
        json=body,
    )
    resp.raise_for_status()
    return resp.json()["id"]


def complete_task(token: str, *, task_id: str, task_list_id: Optional[str] = None) -> None:
    """Mark a task as completed."""
    if not task_list_id:
        lists = _get_task_lists(token)
        task_list_id = lists[0]["id"] if lists else "@default"
    resp = httpx.patch(
        f"{_TASKS_BASE}/lists/{task_list_id}/tasks/{task_id}",
        headers=_headers(token),
        json={"status": "completed"},
    )
    resp.raise_for_status()
