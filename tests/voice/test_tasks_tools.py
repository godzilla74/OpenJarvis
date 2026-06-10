"""Tests for Google Tasks tools with mocked httpx."""
import respx
import httpx

from openjarvis.voice.tools.tasks_tools import list_tasks, format_task_summary


def test_format_task_summary_with_due():
    task = {"title": "Buy groceries", "due": "2026-06-12T00:00:00.000Z", "status": "needsAction"}
    summary = format_task_summary(task)
    assert "Buy groceries" in summary
    assert "Jun 12" in summary


def test_format_task_summary_no_due():
    task = {"title": "Call dentist", "status": "needsAction"}
    summary = format_task_summary(task)
    assert "Call dentist" in summary


@respx.mock
def test_list_tasks_returns_summaries():
    respx.get("https://tasks.googleapis.com/tasks/v1/users/@me/lists").mock(
        return_value=httpx.Response(200, json={"items": [{"id": "list1", "title": "My Tasks"}]})
    )
    respx.get("https://tasks.googleapis.com/tasks/v1/lists/list1/tasks").mock(
        return_value=httpx.Response(200, json={
            "items": [{"title": "Walk the dog", "status": "needsAction"}]
        })
    )
    results = list_tasks(token="fake_token")
    assert len(results) == 1
    assert "Walk the dog" in results[0]
