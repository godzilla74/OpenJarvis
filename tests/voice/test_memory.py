"""Tests for voice memory (session + long-term)."""
from pathlib import Path

import pytest

from openjarvis.voice.memory import VoiceMemory


@pytest.fixture
def mem(tmp_path):
    return VoiceMemory(memory_path=tmp_path / "voice_memory.json")


def test_session_memory_starts_empty(mem):
    assert mem.session_history == []


def test_add_and_retrieve_session_turns(mem):
    mem.add_turn("user", "What's on my calendar?")
    mem.add_turn("assistant", "You have a meeting at 2pm.")
    assert len(mem.session_history) == 2
    assert mem.session_history[0]["role"] == "user"
    assert mem.session_history[1]["content"] == "You have a meeting at 2pm."


def test_remember_persists_to_disk(mem, tmp_path):
    mem.remember("user_name", "Justin")
    mem2 = VoiceMemory(memory_path=tmp_path / "voice_memory.json")
    assert mem2.recall("user_name") == "Justin"


def test_recall_missing_key_returns_none(mem):
    assert mem.recall("nonexistent_key") is None


def test_remember_standing_instruction(mem, tmp_path):
    mem.add_standing_instruction("Always BCC me on sent emails")
    mem2 = VoiceMemory(memory_path=tmp_path / "voice_memory.json")
    assert "Always BCC me on sent emails" in mem2.standing_instructions


def test_system_prompt_includes_memory(mem):
    mem.remember("user_name", "Justin")
    mem.add_standing_instruction("BCC on all emails")
    prompt = mem.build_system_prompt()
    assert "Justin" in prompt
    assert "BCC on all emails" in prompt
