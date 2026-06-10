"""TTS wrapper — text → spoken audio via KokoroTTSBackend + sounddevice playback."""
from __future__ import annotations

import io

import numpy as np
import sounddevice as sd
import soundfile as sf


class VoiceTTS:
    """Wraps KokoroTTSBackend and plays audio via sounddevice."""

    def __init__(self, backend: str = "kokoro", device: int | None = None) -> None:
        try:
            from openjarvis.speech.kokoro_tts import KokoroTTSBackend
        except ImportError as exc:
            raise ImportError(
                "kokoro-onnx not installed. Run: uv pip install -e '.[speech]'"
            ) from exc
        self._backend = KokoroTTSBackend()
        self._device = device

    def speak(self, text: str) -> None:
        """Synthesize text and play it through the speaker. Blocks until done."""
        result = self._backend.synthesize(text)
        self._play_wav_bytes(result.audio)

    def _play_wav_bytes(self, wav_bytes: bytes) -> None:
        buf = io.BytesIO(wav_bytes)
        data, samplerate = sf.read(buf, dtype="float32")
        sd.play(data, samplerate=samplerate, device=self._device)
        sd.wait()
