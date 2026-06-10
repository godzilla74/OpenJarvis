"""TTS wrapper — text → spoken audio. Uses Kokoro if available, falls back to macOS say."""
from __future__ import annotations

import io
import subprocess


class VoiceTTS:
    """Synthesizes and plays text. Prefers KokoroTTSBackend; falls back to macOS say."""

    def __init__(self, backend: str = "kokoro", device: int | None = None) -> None:
        self._device = device
        self._backend = None
        if backend == "kokoro":
            try:
                from openjarvis.speech.kokoro_tts import KokoroTTSBackend
                self._backend = KokoroTTSBackend()
            except Exception:
                pass  # will use say fallback

    def speak(self, text: str) -> None:
        """Synthesize text and play through the speaker. Blocks until done."""
        if self._backend is not None:
            try:
                import io
                import sounddevice as sd
                import soundfile as sf
                result = self._backend.synthesize(text)
                buf = io.BytesIO(result.audio)
                data, samplerate = sf.read(buf, dtype="float32")
                sd.play(data, samplerate=samplerate, device=self._device)
                sd.wait()
                return
            except Exception:
                pass  # fall through to say
        # macOS say fallback — always available on Mac
        subprocess.run(["say", text], check=False)
