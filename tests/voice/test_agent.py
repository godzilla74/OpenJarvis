"""Tests for VoiceAgent — routing, confirmation, memory persistence."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from openjarvis.voice.memory import VoiceMemory
from openjarvis.voice.agent import VoiceAgent


def _make_agent(tmp_path: Path, engine_response: str = "Here is your answer.") -> tuple[VoiceAgent, MagicMock, MagicMock]:
    """Build a VoiceAgent with fully-mocked engines and return (agent, local_engine, cloud_engine)."""
    local_engine = MagicMock()
    local_engine.generate.return_value = {"content": engine_response}

    cloud_engine = MagicMock()
    cloud_engine.generate.return_value = {"content": engine_response}

    memory = VoiceMemory(memory_path=tmp_path / "voice_memory.json")

    agent = VoiceAgent(
        memory=memory,
        local_engine=local_engine,
        local_model="mistral",
        cloud_engine=cloud_engine,
        cloud_model="claude-3-5-sonnet-20241022",
        gmail_tokens={"primary": "fake-gmail-token"},
        calendar_tokens={},
        tasks_tokens={},
    )
    return agent, local_engine, cloud_engine


def test_respond_read_no_confirmation(tmp_path):
    """Plain-text response with no tool call — confirmation should never be called."""
    agent, local_engine, _ = _make_agent(tmp_path, "You have 3 unread emails.")

    speak = MagicMock()
    listen = MagicMock(return_value="")

    result = agent.respond(
        "What emails do I have?",
        speak_confirmation=speak,
        listen_for_response=listen,
    )

    assert result == "You have 3 unread emails."
    speak.assert_not_called()
    listen.assert_not_called()
    local_engine.generate.assert_called_once()


def test_respond_affirmative_confirmation(tmp_path):
    """Tool call detected in LLM response → confirmation spoken → user says yes → tool executes."""
    tool_response_text = 'TOOL: send_email(to="alice@example.com", subject="Hello", body="Hi there")'
    agent, local_engine, _ = _make_agent(tmp_path, tool_response_text)

    speak = MagicMock()
    listen = MagicMock(return_value="yes")

    # Patch _execute_tool so we don't hit real network
    agent._execute_tool = MagicMock(return_value="Email sent.")

    result = agent.respond(
        "Send a hello email to Alice",
        speak_confirmation=speak,
        listen_for_response=listen,
    )

    speak.assert_called_once()
    listen.assert_called_once()
    agent._execute_tool.assert_called_once_with(
        "send_email",
        {"to": "alice@example.com", "subject": "Hello", "body": "Hi there"},
    )
    assert result == "Email sent."


def test_respond_cancel_on_no(tmp_path):
    """Tool call detected, user says 'no' → returns cancellation message without executing."""
    tool_response_text = 'TOOL: add_task(title="Buy milk")'
    agent, _, _ = _make_agent(tmp_path, tool_response_text)

    speak = MagicMock()
    listen = MagicMock(return_value="no")

    agent._execute_tool = MagicMock()

    result = agent.respond(
        "Add buy milk to my tasks",
        speak_confirmation=speak,
        listen_for_response=listen,
    )

    speak.assert_called_once()
    listen.assert_called_once()
    agent._execute_tool.assert_not_called()
    assert result == "Got it, cancelled."


def test_maybe_persist_fact(tmp_path):
    """'remember that...' in user input persists a fact to long-term memory."""
    agent, _, _ = _make_agent(tmp_path, "Got it, I'll remember that.")

    speak = MagicMock()
    listen = MagicMock(return_value="")

    agent.respond(
        "remember that my partner is John",
        speak_confirmation=speak,
        listen_for_response=listen,
    )

    # At least one fact containing "John" should be stored
    facts = agent._memory._store["facts"]
    assert any("John" in str(v) for v in facts.values()), (
        f"Expected a fact containing 'John' but got: {facts}"
    )


def test_cloud_engine_used_for_complex_input(tmp_path):
    """Inputs matching CLOUD routing signals use the cloud engine, not local."""
    agent, local_engine, cloud_engine = _make_agent(tmp_path, "Here is your proper reply.")

    speak = MagicMock()
    listen = MagicMock(return_value="")

    agent.respond(
        "Draft a proper reply to the email about the contract",
        speak_confirmation=speak,
        listen_for_response=listen,
    )

    cloud_engine.generate.assert_called_once()
    local_engine.generate.assert_not_called()


def test_memory_records_conversation_turns(tmp_path):
    """Both user and assistant turns are stored in session history after respond()."""
    agent, _, _ = _make_agent(tmp_path, "Sure, you have a meeting at 3pm.")

    speak = MagicMock()
    listen = MagicMock(return_value="")

    agent.respond(
        "What's on my calendar?",
        speak_confirmation=speak,
        listen_for_response=listen,
    )

    history = agent._memory.session_history
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "What's on my calendar?"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "Sure, you have a meeting at 3pm."


def test_parse_tool_call_extracts_name_and_args(tmp_path):
    """_parse_tool_call correctly parses TOOL: name(k=v, ...) format."""
    agent, _, _ = _make_agent(tmp_path)

    name, args = agent._parse_tool_call('TOOL: list_emails(query="is:unread", max_results=5)')
    assert name == "list_emails"
    assert args["query"] == "is:unread"
    assert args["max_results"] == "5"


def test_parse_tool_call_returns_empty_when_no_match(tmp_path):
    """_parse_tool_call returns ('', {}) when no TOOL pattern is present."""
    agent, _, _ = _make_agent(tmp_path)

    name, args = agent._parse_tool_call("You have 3 emails from your boss.")
    assert name == ""
    assert args == {}


def test_is_affirmative_recognises_variants(tmp_path):
    """_is_affirmative returns True for known affirmative words/phrases."""
    agent, _, _ = _make_agent(tmp_path)

    for phrase in ["yes", "yeah", "go ahead", "send it", "sure", "okay", "ok", "do it", "yep"]:
        assert agent._is_affirmative(phrase), f"Expected '{phrase}' to be affirmative"
    assert not agent._is_affirmative("no")
    assert not agent._is_affirmative("nope")
    assert not agent._is_affirmative("cancel that")
