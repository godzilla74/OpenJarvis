"""Tests for VoiceLoop orchestrator."""
from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helper: build a VoiceLoop with all heavy constructors mocked out
# ---------------------------------------------------------------------------

_PATCHES = [
    "openjarvis.voice.loop.VoiceMemory",
    "openjarvis.voice.loop.VoiceSTT",
    "openjarvis.voice.loop.VoiceTTS",
    "openjarvis.voice.loop.VoiceAgent",
    "openjarvis.voice.loop.WakeWordDetector",
]

# Common kwargs that satisfy VoiceLoop.__init__ signature
_INIT_KWARGS = dict(
    wake_word_model=Path("/fake/model.onnx"),
    stt_model="large-v3",
    tts_backend="kokoro",
    mic_device=None,
    local_engine=MagicMock(),
    local_model="local-model",
    cloud_engine=MagicMock(),
    cloud_model="cloud-model",
    gmail_tokens={},
    calendar_token="tok",
    tasks_token="tok",
    memory_path=None,
    silence_duration_s=1.5,
    confirmation_timeout_s=15.0,
)


def _make_loop():
    """Return a VoiceLoop with all heavyweight deps mocked.

    Because the imports are lazy (inside __init__), we patch the classes at
    their original module locations rather than on loop's module namespace.
    """
    from openjarvis.voice.loop import VoiceLoop

    with patch("openjarvis.voice.memory.VoiceMemory"), \
         patch("openjarvis.voice.stt.VoiceSTT"), \
         patch("openjarvis.voice.tts.VoiceTTS"), \
         patch("openjarvis.voice.agent.VoiceAgent"), \
         patch("openjarvis.voice.wake_word.WakeWordDetector"):
        loop = VoiceLoop(**_INIT_KWARGS)

    return loop


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestVoiceLoopStop:
    """test_stop_event_prevents_run — loop exits immediately when pre-stopped."""

    def test_stop_event_prevents_run(self):
        """Loop must exit within 1 second if _stop_event is already set."""
        loop = _make_loop()

        # Pre-set the stop event so the while-loop never iterates
        loop._stop_event.set()

        # Mock out the detector so start/stop are no-ops
        loop._detector = MagicMock()
        loop._tts = MagicMock()

        finished = threading.Event()

        def _run():
            # Patch signal.signal so it doesn't fail in a non-main thread
            with patch("openjarvis.voice.loop.signal"):
                loop.run()
            finished.set()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=1.0)

        assert finished.is_set(), "run() did not return within 1 second"


class TestWakeWord:
    """test_on_wake_sets_event — _on_wake() must set _wake_event."""

    def test_on_wake_sets_event(self):
        loop = _make_loop()

        assert not loop._wake_event.is_set(), "wake_event should start clear"
        loop._on_wake()
        assert loop._wake_event.is_set(), "_wake_event must be set after _on_wake()"


class TestStopMethod:
    """test_stop_sets_event — stop() must set _stop_event."""

    def test_stop_sets_event(self):
        loop = _make_loop()

        assert not loop._stop_event.is_set(), "_stop_event should start clear"
        loop.stop()
        assert loop._stop_event.is_set(), "_stop_event must be set after stop()"
