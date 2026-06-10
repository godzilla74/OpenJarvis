"""Tests for confirmation flow classifier."""
from openjarvis.voice.confirmation import ActionType, classify_action, format_confirmation


def test_read_needs_no_confirmation():
    assert classify_action("list_emails") == ActionType.READ


def test_send_email_needs_full_readback():
    assert classify_action("send_email") == ActionType.SEND


def test_reply_needs_full_readback():
    assert classify_action("reply_email") == ActionType.SEND


def test_create_event_needs_brief():
    assert classify_action("create_calendar_event") == ActionType.CREATE


def test_add_task_needs_brief():
    assert classify_action("add_task") == ActionType.CREATE


def test_update_event_needs_brief():
    assert classify_action("update_calendar_event") == ActionType.MODIFY


def test_send_confirmation_includes_all_fields():
    msg = format_confirmation(
        ActionType.SEND,
        subject="Re: Project update",
        to="sarah@example.com",
        body="Hi Sarah, sounds great. See you Thursday.",
    )
    assert "sarah@example.com" in msg
    assert "Re: Project update" in msg
    assert "sounds great" in msg
    assert "Send this?" in msg


def test_create_confirmation_is_brief():
    msg = format_confirmation(
        ActionType.CREATE,
        title="Dentist appointment",
        when="Thursday at 2pm",
    )
    assert "Dentist appointment" in msg
    assert "Thursday" in msg
    assert "Go ahead?" in msg
