"""Tests for calendar tools with mocked httpx."""
import respx
import httpx

from openjarvis.voice.tools.calendar_tools import list_events, format_event_summary


def test_format_event_summary():
    event = {
        "summary": "Team standup",
        "start": {"dateTime": "2026-06-11T09:00:00-04:00"},
        "end": {"dateTime": "2026-06-11T09:30:00-04:00"},
        "location": "Zoom",
    }
    summary = format_event_summary(event)
    assert "Team standup" in summary
    assert "09:00" in summary


@respx.mock
def test_list_events_returns_summaries():
    respx.get("https://www.googleapis.com/calendar/v3/calendars/primary/events").mock(
        return_value=httpx.Response(200, json={
            "items": [
                {
                    "summary": "Dentist",
                    "start": {"dateTime": "2026-06-12T14:00:00-04:00"},
                    "end": {"dateTime": "2026-06-12T15:00:00-04:00"},
                }
            ]
        })
    )
    results = list_events(token="fake_token", calendar_id="primary")
    assert len(results) == 1
    assert "Dentist" in results[0]
