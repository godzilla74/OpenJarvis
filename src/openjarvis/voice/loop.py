"""Main voice assistant loop: wake word → STT → agent → TTS → repeat."""
from __future__ import annotations

import logging
import signal
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class VoiceLoop:
    """Orchestrates the full voice pipeline.

    Usage::
        loop = VoiceLoop(config=cfg)
        loop.run()   # blocks until SIGINT/SIGTERM
    """

    def __init__(
        self,
        *,
        wake_word_model: Path,
        stt_model: str = "large-v3",
        tts_backend: str = "kokoro",
        mic_device: Optional[int] = None,
        local_engine,
        local_model: str,
        cloud_engine,
        cloud_model: str,
        gmail_tokens: dict[str, str],
        calendar_token: str,
        tasks_token: str,
        memory_path: Optional[Path] = None,
        silence_duration_s: float = 1.5,
        confirmation_timeout_s: float = 15.0,
    ) -> None:
        from openjarvis.voice.memory import VoiceMemory
        from openjarvis.voice.stt import VoiceSTT
        from openjarvis.voice.tts import VoiceTTS
        from openjarvis.voice.agent import VoiceAgent
        from openjarvis.voice.wake_word import WakeWordDetector

        self._memory = VoiceMemory(memory_path=memory_path)
        self._stt = VoiceSTT(model_size=stt_model)
        self._tts = VoiceTTS(backend=tts_backend, device=mic_device)
        self._agent = VoiceAgent(
            memory=self._memory,
            local_engine=local_engine,
            local_model=local_model,
            cloud_engine=cloud_engine,
            cloud_model=cloud_model,
            gmail_tokens=gmail_tokens,
            calendar_token=calendar_token,
            tasks_token=tasks_token,
            confirmation_timeout_s=confirmation_timeout_s,
        )
        self._detector = WakeWordDetector(
            on_wake=self._on_wake,
            model_path=wake_word_model,
            device=mic_device,
        )
        self._mic_device = mic_device
        self._silence_duration_s = silence_duration_s
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()

    def _on_wake(self) -> None:
        """Called by WakeWordDetector when 'Hey Jarvis' is detected."""
        logger.info("Wake word detected")
        self._wake_event.set()

    def _record_and_transcribe(self) -> str:
        """Record the user's utterance and return transcribed text."""
        from openjarvis.voice.capture import record_utterance
        audio = record_utterance(
            device=self._mic_device,
            silence_duration_s=self._silence_duration_s,
        )
        return self._stt.transcribe(audio)

    def run(self) -> None:
        """Start the always-on voice loop. Blocks until SIGINT or stop() is called."""
        signal.signal(signal.SIGINT, lambda *_: self.stop())
        signal.signal(signal.SIGTERM, lambda *_: self.stop())

        logger.info("Starting voice loop — say 'Hey Jarvis' to begin")
        self._tts.speak("Jarvis is ready. Say Hey Jarvis to begin.")
        self._detector.start()

        try:
            while not self._stop_event.is_set():
                triggered = self._wake_event.wait(timeout=0.5)
                if not triggered:
                    continue
                self._wake_event.clear()

                self._tts.speak("Mm?")

                text = self._record_and_transcribe()
                if not text.strip():
                    continue

                logger.info("User said: %s", text)

                response = self._agent.respond(
                    text,
                    speak_confirmation=self._tts.speak,
                    listen_for_response=self._record_and_transcribe,
                )

                self._tts.speak(response)

        finally:
            self._detector.stop()
            logger.info("Voice loop stopped")

    def stop(self) -> None:
        """Signal the loop to stop after the current turn."""
        self._stop_event.set()
