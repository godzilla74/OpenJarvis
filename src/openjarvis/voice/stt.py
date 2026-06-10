"""STT wrapper — numpy array → transcribed text via FasterWhisperBackend."""
from __future__ import annotations

import io
import wave

import numpy as np

_SAMPLE_RATE = 16_000


def _array_to_wav_bytes(audio: np.ndarray) -> bytes:
    """Convert a 1-D float32 numpy array to 16-bit WAV bytes at 16 kHz."""
    pcm = (audio * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(_SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


class VoiceSTT:
    """Wraps FasterWhisperBackend for voice pipeline use."""

    def __init__(self, model_size: str = "large-v3", device: str = "auto") -> None:
        try:
            from openjarvis.speech.faster_whisper import FasterWhisperBackend
        except ImportError as exc:
            raise ImportError(
                "faster-whisper not installed. Run: uv pip install -e '.[speech]'"
            ) from exc
        self._backend = FasterWhisperBackend(model_size=model_size, device=device)

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a 1-D float32 numpy array. Returns text string."""
        wav_bytes = _array_to_wav_bytes(audio)
        result = self._backend.transcribe(wav_bytes, format="wav")
        return result.text
