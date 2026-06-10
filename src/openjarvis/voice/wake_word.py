"""Always-on wake word detector using openwakeword."""
from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

from openjarvis.voice.capture import BLOCK_SIZE, SAMPLE_RATE

_DETECTION_THRESHOLD = 0.3


def _find_model(model_path: Optional[Path] = None) -> Path:
    """Return the wake word model path: custom trained → pre-built bundled."""
    if model_path is not None:
        return model_path
    custom = Path.home() / ".openjarvis" / "models" / "hey_jarvis.onnx"
    if custom.exists():
        return custom
    # Fall back to the pre-built model bundled with openwakeword — prefer .onnx
    try:
        import openwakeword as oww
        tflite_candidates = [Path(p) for p in oww.get_pretrained_model_paths() if "hey_jarvis" in p]
        # get_pretrained_model_paths() only returns .tflite; also check ONNX siblings
        onnx = next((p.with_suffix(".onnx") for p in tflite_candidates
                     if p.with_suffix(".onnx").exists()), None)
        tflite = next((p for p in tflite_candidates if p.exists()), None)
        if onnx:
            return onnx
        if tflite:
            return tflite
    except Exception:
        pass
    return custom  # let caller fail with a clear missing-file error


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
        self._model_path = _find_model(model_path)
        self._threshold = threshold
        self._device = device
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._model = None

    def _load_model(self) -> None:
        from openwakeword.model import Model  # type: ignore[import]
        framework = "tflite" if str(self._model_path).endswith(".tflite") else "onnx"
        self._model = Model(
            wakeword_models=[str(self._model_path)],
            inference_framework=framework,
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
                # Key is the model filename stem; check both custom and pre-built names
                score = max(
                    v for k, v in prediction.items()
                    if "hey_jarvis" in k
                ) if any("hey_jarvis" in k for k in prediction) else 0.0
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
