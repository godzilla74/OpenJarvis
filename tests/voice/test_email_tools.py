"""Tests for email tools with mocked httpx."""
import pytest
import respx
import httpx

from openjarvis.voice.tools.email_tools import list_emails, format_email_summary


def test_format_email_summary():
    msg = {
        "id": "abc123",
        "snippet": "Let's meet Thursday at 2pm",
        "payload": {
            "headers": [
                {"name": "From", "value": "Sarah <sarah@example.com>"},
                {"name": "Subject", "value": "Meeting Thursday"},
                {"name": "Date", "value": "Tue, 10 Jun 2026 09:00:00 +0000"},
            ]
        },
    }
    summary = format_email_summary(msg)
    assert "Sarah" in summary
    assert "Meeting Thursday" in summary
    assert "Thursday at 2pm" in summary


@respx.mock
def test_list_emails_returns_summaries():
    respx.get("https://gmail.googleapis.com/gmail/v1/users/me/messages").mock(
        return_value=httpx.Response(200, json={
            "messages": [{"id": "msg1", "threadId": "t1"}]
        })
    )
    respx.get("https://gmail.googleapis.com/gmail/v1/users/me/messages/msg1").mock(
        return_value=httpx.Response(200, json={
            "id": "msg1",
            "snippet": "Project update attached",
            "payload": {
                "headers": [
                    {"name": "From", "value": "Bob <bob@example.com>"},
                    {"name": "Subject", "value": "Project update"},
                    {"name": "Date", "value": "Tue, 10 Jun 2026 10:00:00 +0000"},
                ]
            },
        })
    )
    results = list_emails(token="fake_token", max_results=5)
    assert len(results) == 1
    assert "Project update" in results[0]
