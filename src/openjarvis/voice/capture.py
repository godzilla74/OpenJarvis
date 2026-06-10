"""Microphone capture with silence detection."""
from __future__ import annotations

import queue
import threading
from typing import Optional

import numpy as np
import sounddevice as sd


SAMPLE_RATE = 16_000
BLOCK_SIZE = 1_280  # 80 ms at 16 kHz — matches openwakeword chunk size


def list_input_devices() -> list[dict]:
    """Return all available audio input devices."""
    devices = sd.query_devices()
    return [
        {"index": i, "name": d["name"], "channels": d["max_input_channels"]}
        for i, d in enumerate(devices)
        if d["max_input_channels"] > 0
    ]


def find_device_index(name_fragment: str) -> Optional[int]:
    """Find a device index by partial name match (case-insensitive)."""
    for d in list_input_devices():
        if name_fragment.lower() in d["name"].lower():
            return d["index"]
    return None


def detect_silence(chunk: np.ndarray, *, threshold: float = 0.01) -> bool:
    """Return True if the audio chunk is below the silence threshold."""
    return float(np.max(np.abs(chunk))) < threshold


def merge_chunks(chunks: list[np.ndarray]) -> np.ndarray:
    """Concatenate a list of 1-D float32 arrays into one."""
    return np.concatenate(chunks).astype(np.float32)


def record_utterance(
    *,
    device: Optional[int] = None,
    silence_threshold: float = 0.01,
    silence_duration_s: float = 1.5,
    max_duration_s: float = 30.0,
) -> np.ndarray:
    """Record from microphone until silence, returning a 1-D float32 array at 16 kHz.

    Blocks until silence_duration_s of consecutive silence or max_duration_s.
    """
    silence_blocks = int(silence_duration_s * SAMPLE_RATE / BLOCK_SIZE)
    max_blocks = int(max_duration_s * SAMPLE_RATE / BLOCK_SIZE)

    audio_queue: queue.Queue[np.ndarray] = queue.Queue()
    stop_event = threading.Event()

    def _callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        audio_queue.put(indata[:, 0].copy())

    chunks: list[np.ndarray] = []
    consecutive_silence = 0

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=1,
        dtype="float32",
        device=device,
        callback=_callback,
    ):
        while not stop_event.is_set():
            chunk = audio_queue.get()
            chunks.append(chunk)
            if detect_silence(chunk, threshold=silence_threshold):
                consecutive_silence += 1
            else:
                consecutive_silence = 0
            if consecutive_silence >= silence_blocks:
                break
            if len(chunks) >= max_blocks:
                break

    return merge_chunks(chunks)
