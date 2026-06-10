"""Always-on wake word detector using openwakeword."""
from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

from openjarvis.voice.capture import BLOCK_SIZE, SAMPLE_RATE

_DEFAULT_MODEL = Path.home() / ".openjarvis" / "models" / "hey_jarvis.onnx"
_DETECTION_THRESHOLD = 0.5


class WakeWordDetector:
    """Listens continuously and fires a callback when 'Hey Jarvis' is detected.

    Usage::

        def on_wake():
            print("Wake word detected!")

        detector = WakeWordDetector(on_wake=on_wake)
        detector.start()   # non-blocking, runs in background thread
        # ... do other work ...
        detector.stop()
    """

    def __init__(
        self,
        *,
        on_wake: Callable[[], None],
        model_path: Optional[Path] = None,
        threshold: float = _DETECTION_THRESHOLD,
        device: Optional[int] = None,
    ) -> None:
        self._on_wake = on_wake
        self._model_path = model_path or _DEFAULT_MODEL
        self._threshold = threshold
        self._device = device
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._model = None

    def _load_model(self) -> None:
        from openwakeword.model import Model  # type: ignore[import]
        self._model = Model(
            wakeword_models=[str(self._model_path)],
            inference_framework="onnx",
        )

    def _run(self) -> None:
        self._load_model()
        audio_queue: queue.Queue[np.ndarray] = queue.Queue()

        def _callback(indata: np.ndarray, frames: int, time_info, status) -> None:
            audio_queue.put(indata[:, 0].copy())

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=1,
            dtype="float32",
            device=self._device,
            callback=_callback,
        ):
            while not self._stop_event.is_set():
                try:
                    chunk = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                prediction = self._model.predict(chunk)
                # Model key is the filename stem of the .onnx model
                score = prediction.get("hey_jarvis", 0.0)
                if score >= self._threshold:
                    self._model.reset()  # clear buffer to avoid double-trigger
                    self._on_wake()

    def start(self) -> None:
        """Start listening in a background thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background listener."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
