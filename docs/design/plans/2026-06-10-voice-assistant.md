# Hey Jarvis — Voice Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an always-on voice assistant — say "Hey Jarvis" → speak naturally → get spoken responses, with Gmail (multi-account), Calendar, and Tasks integration and persistent memory.

**Architecture:** A new `src/openjarvis/voice/` package contains the full pipeline (wake word → STT → agent → TTS → loop). `VoiceAgent` wraps `NativeReActAgent` with three tool sets (email, calendar, tasks), a complexity router switching between local MLX and Claude Sonnet, and smart confirmation. A `jarvis voice` CLI group provides `train`, `setup`, and the main `start` subcommand.

**Tech Stack:**
- Wake word: `openwakeword` — custom "Hey Jarvis" model trained on macOS `say`-generated audio
- STT: Faster-Whisper `large-v3` via existing `FasterWhisperBackend`
- LLM: Ollama (local, primary) + Anthropic Claude Sonnet (complex tasks fallback)
- TTS: Kokoro via existing `KokoroTTSBackend`, audio playback via `sounddevice`
- Audio I/O: `sounddevice` + `numpy`
- Tools: `httpx`-based wrappers over Gmail, Google Calendar, Google Tasks REST APIs

---

> **Implementation notes (verify before starting Task 12):**
> - `engine.chat()` API: verify the exact method name on `OllamaEngine` before writing `VoiceAgent`. Look at how `SimpleAgent._generate()` calls the engine internally and mirror that pattern.
> - `AnthropicEngine` path: confirm `from openjarvis.engine.anthropic import AnthropicEngine` exists, or find the correct import path via `grep -r "AnthropicEngine" src/`.
> - `DEFAULT_CONFIG_DIR`: confirm it is exported from `openjarvis.core.config` via `python -c "from openjarvis.core.config import DEFAULT_CONFIG_DIR; print(DEFAULT_CONFIG_DIR)"`.

---

## File Map

**New files:**
| File | Responsibility |
|---|---|
| `src/openjarvis/voice/__init__.py` | Package init |
| `src/openjarvis/voice/capture.py` | Mic recording with silence detection |
| `src/openjarvis/voice/wake_word.py` | openwakeword detector |
| `src/openjarvis/voice/stt.py` | Thin wrapper over FasterWhisperBackend |
| `src/openjarvis/voice/tts.py` | Thin wrapper over KokoroTTSBackend + audio playback |
| `src/openjarvis/voice/router.py` | Local vs cloud routing logic |
| `src/openjarvis/voice/confirmation.py` | Action classification + confirmation flow |
| `src/openjarvis/voice/memory.py` | Session + long-term memory |
| `src/openjarvis/voice/tools/__init__.py` | Tools package init |
| `src/openjarvis/voice/tools/email_tools.py` | List/read/send email via Gmail API |
| `src/openjarvis/voice/tools/calendar_tools.py` | List/create/update events via Calendar API |
| `src/openjarvis/voice/tools/tasks_tools.py` | List/add/complete tasks via Tasks API |
| `src/openjarvis/voice/agent.py` | VoiceAgent: NativeReActAgent + tools + routing + memory |
| `src/openjarvis/voice/loop.py` | Main orchestration loop |
| `src/openjarvis/cli/voice_cmd.py` | `jarvis voice` CLI group |
| `tests/voice/__init__.py` | Tests package |
| `tests/voice/test_router.py` | Router unit tests |
| `tests/voice/test_confirmation.py` | Confirmation classifier tests |
| `tests/voice/test_memory.py` | Memory read/write tests |
| `tests/voice/test_email_tools.py` | Email tools with mocked httpx |
| `tests/voice/test_calendar_tools.py` | Calendar tools with mocked httpx |
| `tests/voice/test_tasks_tools.py` | Tasks tools with mocked httpx |

**Modified files:**
| File | Change |
|---|---|
| `pyproject.toml` | Add `voice` extra with new deps |
| `src/openjarvis/core/config.py` | Add `VoiceAssistantConfig` dataclass + field on `JarvisConfig` |
| `src/openjarvis/cli/__init__.py` | Import and register `voice` command group |

---

## Task 1: Dependencies and Config

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/openjarvis/core/config.py`

- [ ] **Step 1: Add `voice` extra to pyproject.toml**

Open `pyproject.toml` and add after the `speech-deepgram` line:

```toml
speech-kokoro = ["kokoro-onnx>=0.4.0"]
voice = [
    "openwakeword>=0.6.0",
    "sounddevice>=0.4.6",
    "numpy>=1.24",
    "soundfile>=0.12",
    "google-auth-oauthlib>=1.0",
    "google-auth-httplib2>=0.2",
]
```

- [ ] **Step 2: Add VoiceAssistantConfig to config.py**

Open `src/openjarvis/core/config.py`. Find the existing `@dataclass(slots=True)` definitions. Add this new dataclass before the `JarvisConfig` class:

```python
@dataclass(slots=True)
class VoiceAccountConfig:
    credentials_path: str = ""
    label: str = ""  # e.g. "work email", "personal email"


@dataclass(slots=True)
class VoiceAssistantConfig:
    wake_word_model: str = ""  # path to .onnx model, empty = not trained yet
    stt_model: str = "large-v3"
    tts_backend: str = "kokoro"
    mic_device: str = ""  # empty = system default
    detection_threshold: float = 0.5
    silence_duration_s: float = 1.5
    confirmation_timeout_s: float = 15.0
    cloud_model: str = "claude-sonnet-4-6"
    local_model: str = "qwen2.5:14b"
    memory_path: str = ""  # empty = ~/.openjarvis/voice_memory.json
    gmail_accounts: list = field(default_factory=list)  # list of VoiceAccountConfig
```

- [ ] **Step 3: Register VoiceAssistantConfig on JarvisConfig**

In `src/openjarvis/core/config.py`, find the `JarvisConfig` dataclass and add this field after the `digest` field:

```python
voice_assistant: VoiceAssistantConfig = field(default_factory=VoiceAssistantConfig)
```

- [ ] **Step 4: Add "voice_assistant" to the section loader**

In `load_config()`, find the `top_sections` tuple (or equivalent section-mapping dict) and add `"voice_assistant"`. This ensures `[voice_assistant]` blocks in `~/.openjarvis/config.toml` are loaded.

- [ ] **Step 5: Install voice dependencies**

```bash
cd /Users/justinfarmer/Code/OpenJarvis
uv pip install -e ".[voice]"
```

Expected: packages install without errors. `python -c "import sounddevice; import openwakeword"` succeeds.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/openjarvis/core/config.py
git commit -m "feat(voice): add VoiceAssistantConfig and voice extra deps"
```

---

## Task 2: Audio Capture

**Files:**
- Create: `src/openjarvis/voice/capture.py`
- Create: `src/openjarvis/voice/__init__.py`
- Create: `tests/voice/__init__.py`

- [ ] **Step 1: Create package inits**

Create `src/openjarvis/voice/__init__.py` (empty):
```python
"""OpenJarvis voice assistant package."""
```

Create `tests/voice/__init__.py` (empty):
```python
```

- [ ] **Step 2: Write the failing test**

Create `tests/voice/test_capture.py`:

```python
"""Tests for audio capture (silence detection logic only — no real mic)."""
import numpy as np
import pytest

from openjarvis.voice.capture import detect_silence, merge_chunks


def test_detect_silence_quiet():
    chunk = np.zeros(1600, dtype=np.float32)
    assert detect_silence(chunk, threshold=0.01) is True


def test_detect_silence_loud():
    chunk = np.ones(1600, dtype=np.float32) * 0.5
    assert detect_silence(chunk, threshold=0.01) is False


def test_merge_chunks_produces_flat_array():
    a = np.array([0.1, 0.2], dtype=np.float32)
    b = np.array([0.3, 0.4], dtype=np.float32)
    result = merge_chunks([a, b])
    assert result.shape == (4,)
    np.testing.assert_allclose(result, [0.1, 0.2, 0.3, 0.4])
```

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/voice/test_capture.py -v
```
Expected: `ImportError: cannot import name 'detect_silence' from 'openjarvis.voice.capture'`

- [ ] **Step 4: Implement capture.py**

Create `src/openjarvis/voice/capture.py`:

```python
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

    Blocks until ``silence_duration_s`` of consecutive silence or ``max_duration_s``.
    """
    silence_blocks = int(silence_duration_s * SAMPLE_RATE / BLOCK_SIZE)
    max_blocks = int(max_duration_s * SAMPLE_RATE / BLOCK_SIZE)

    audio_queue: queue.Queue[np.ndarray] = queue.Queue()
    stop_event = threading.Event()

    def _callback(indata: np.ndarray, frames: int, time_info, status) -> None:  # noqa: ARG001
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
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/voice/test_capture.py -v
```
Expected: all 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/openjarvis/voice/ tests/voice/
git commit -m "feat(voice): add audio capture with silence detection"
```

---

## Task 3: STT and TTS Wrappers

**Files:**
- Create: `src/openjarvis/voice/stt.py`
- Create: `src/openjarvis/voice/tts.py`

These are thin wrappers that convert between numpy arrays and the bytes-based backends, and handle audio playback.

- [ ] **Step 1: Create stt.py**

```python
"""STT wrapper — numpy array → transcribed text via FasterWhisperBackend."""
from __future__ import annotations

import io
import wave

import numpy as np

from openjarvis.speech.faster_whisper import FasterWhisperBackend

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
        self._backend = FasterWhisperBackend(
            model_size=model_size, device=device, compute_type="float16"
        )

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a 1-D float32 numpy array. Returns text string."""
        wav_bytes = _array_to_wav_bytes(audio)
        result = self._backend.transcribe(wav_bytes, format="wav")
        return result.text.strip()
```

- [ ] **Step 2: Create tts.py**

```python
"""TTS wrapper — text → spoken audio via KokoroTTSBackend + sounddevice playback."""
from __future__ import annotations

import io

import numpy as np
import sounddevice as sd
import soundfile as sf

from openjarvis.speech.kokoro_tts import KokoroTTSBackend

_LONG_RESPONSE_WORDS = 150  # threshold to switch to OpenAI TTS (future)


class VoiceTTS:
    """Wraps KokoroTTSBackend and plays audio via sounddevice."""

    def __init__(self, backend: str = "kokoro", device: int | None = None) -> None:
        self._backend = KokoroTTSBackend()
        self._device = device

    def speak(self, text: str) -> None:
        """Synthesize ``text`` and play it through the speaker. Blocks until done."""
        result = self._backend.synthesize(text, output_format="wav")
        self._play_wav_bytes(result.audio)

    def _play_wav_bytes(self, wav_bytes: bytes) -> None:
        buf = io.BytesIO(wav_bytes)
        data, samplerate = sf.read(buf, dtype="float32")
        sd.play(data, samplerate=samplerate, device=self._device)
        sd.wait()
```

- [ ] **Step 3: Verify imports**

```bash
python -c "from openjarvis.voice.stt import VoiceSTT; from openjarvis.voice.tts import VoiceTTS; print('ok')"
```
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add src/openjarvis/voice/stt.py src/openjarvis/voice/tts.py
git commit -m "feat(voice): add STT and TTS wrappers"
```

---

## Task 4: Wake Word Training Script

**Files:**
- Create: `src/openjarvis/voice/train_wake_word.py`

This script generates synthetic "Hey Jarvis" audio samples using macOS `say`, then trains an openwakeword model. Run once before first use via `jarvis voice train`.

- [ ] **Step 1: Create train_wake_word.py**

```python
"""Generate synthetic training data and train an openwakeword 'hey jarvis' model.

Run via: jarvis voice train
Requires: macOS (uses `say` command), openwakeword[train] installed.
Output: ~/.openjarvis/models/hey_jarvis.onnx
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

MODEL_DIR = Path.home() / ".openjarvis" / "models"
TRAINING_DIR = Path.home() / ".openjarvis" / "training"
MODEL_OUT = MODEL_DIR / "hey_jarvis.onnx"

# macOS `say` voices — variety improves model robustness
VOICES = [
    "Alex", "Samantha", "Victoria", "Tom", "Fred",
    "Karen", "Moira", "Tessa", "Daniel", "Rishi",
]

# Slight variations on the phrase — improves recall at sentence boundaries
PHRASES = [
    "hey jarvis",
    "hey Jarvis",
    "Hey Jarvis",
    "hey, jarvis",
]


def generate_samples(output_dir: Path, n_per_voice: int = 20) -> None:
    """Generate TTS samples using macOS `say` and convert to 16 kHz WAV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for voice in VOICES:
        for phrase in PHRASES:
            for i in range(n_per_voice):
                aiff_path = output_dir / f"sample_{count:04d}.aiff"
                wav_path = output_dir / f"sample_{count:04d}.wav"
                # Generate AIFF via say
                subprocess.run(
                    ["say", "-v", voice, "-r", str(140 + (i % 40) - 20), phrase,
                     "-o", str(aiff_path)],
                    check=True, capture_output=True,
                )
                # Convert to 16 kHz mono WAV via afconvert (built into macOS)
                subprocess.run(
                    ["afconvert", "-f", "WAVE", "-d", "LEI16@16000",
                     str(aiff_path), str(wav_path)],
                    check=True, capture_output=True,
                )
                aiff_path.unlink()
                count += 1
    print(f"Generated {count} samples in {output_dir}")


def train(positive_dir: Path, output_dir: Path) -> Path:
    """Train openwakeword model on positive samples.

    openwakeword generates its own negative samples from ambient noise.
    See: https://github.com/dscripka/openWakeWord/blob/main/docs/training.md
    """
    try:
        from openwakeword.train import train_model  # type: ignore[import]
    except ImportError:
        print(
            "openwakeword training extras not installed.\n"
            "Run: uv pip install 'openwakeword[train]'",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_model(
        model_name="hey_jarvis",
        positive_dir=str(positive_dir),
        output_dir=str(output_dir),
        epochs=300,
    )
    return output_dir / "hey_jarvis.onnx"


def run_training(n_per_voice: int = 20) -> Path:
    """Full training pipeline: generate → train → return model path."""
    positive_dir = TRAINING_DIR / "positive"
    print("Step 1/2: Generating synthetic 'Hey Jarvis' training audio...")
    generate_samples(positive_dir, n_per_voice=n_per_voice)
    print("Step 2/2: Training wake word model (this takes ~5 minutes)...")
    model_path = train(positive_dir, MODEL_DIR)
    print(f"\nModel saved to: {model_path}")
    return model_path
```

- [ ] **Step 2: Verify the script imports cleanly**

```bash
python -c "from openjarvis.voice.train_wake_word import run_training; print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/openjarvis/voice/train_wake_word.py
git commit -m "feat(voice): add wake word training script (synthetic TTS data)"
```

---

## Task 5: Wake Word Detector

**Files:**
- Create: `src/openjarvis/voice/wake_word.py`

- [ ] **Step 1: Create wake_word.py**

```python
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

        def _callback(indata: np.ndarray, frames: int, time_info, status) -> None:  # noqa: ARG001
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
```

- [ ] **Step 2: Verify the module imports**

```bash
python -c "from openjarvis.voice.wake_word import WakeWordDetector; print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/openjarvis/voice/wake_word.py
git commit -m "feat(voice): add openwakeword wake word detector"
```

---

## Task 6: Complexity Router

**Files:**
- Create: `src/openjarvis/voice/router.py`
- Create: `tests/voice/test_router.py`

- [ ] **Step 1: Write the failing test**

Create `tests/voice/test_router.py`:

```python
"""Tests for LLM routing logic."""
from openjarvis.voice.router import Route, classify_route


def test_simple_query_routes_local():
    assert classify_route("What's on my calendar today?") == Route.LOCAL


def test_read_email_routes_local():
    assert classify_route("Read my emails") == Route.LOCAL


def test_add_task_routes_local():
    assert classify_route("Add buy milk to my tasks") == Route.LOCAL


def test_explicit_proper_reply_routes_cloud():
    assert classify_route("Write a proper reply to Sarah's email") == Route.CLOUD


def test_long_thread_signal_routes_cloud():
    assert classify_route("Draft a detailed response to the contract negotiation thread") == Route.CLOUD


def test_multi_step_routes_cloud():
    assert classify_route(
        "Check my emails, find the project proposal from last week, "
        "summarize it, and draft a reply accepting the terms"
    ) == Route.CLOUD
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/voice/test_router.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement router.py**

```python
"""Route voice requests between local Ollama model and cloud Claude Sonnet."""
from __future__ import annotations

import re
from enum import Enum


class Route(Enum):
    LOCAL = "local"
    CLOUD = "cloud"


# Phrases that signal a complex task requiring cloud reasoning
_CLOUD_SIGNALS = [
    r"\bproper (reply|response|email)\b",
    r"\bdetailed (reply|response|email|summary)\b",
    r"\bdraft\b.{0,40}\b(contract|proposal|negotiat|formal|important)\b",
    r"\b(then|and then|after that|finally)\b.{0,60}\b(reply|send|draft|write)\b",
    r"\bmulti.step\b",
    r"\bsummariz.{0,10} (and|then) (reply|respond|draft)\b",
]

_CLOUD_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _CLOUD_SIGNALS]

# If input is very long (lots of steps) it's a complex request
_COMPLEX_WORD_COUNT = 30


def classify_route(text: str) -> Route:
    """Return LOCAL for simple requests, CLOUD for complex ones."""
    for pattern in _CLOUD_PATTERNS:
        if pattern.search(text):
            return Route.CLOUD
    if len(text.split()) > _COMPLEX_WORD_COUNT:
        return Route.CLOUD
    return Route.LOCAL
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/voice/test_router.py -v
```
Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/openjarvis/voice/router.py tests/voice/test_router.py
git commit -m "feat(voice): add local/cloud complexity router"
```

---

## Task 7: Confirmation Classifier

**Files:**
- Create: `src/openjarvis/voice/confirmation.py`
- Create: `tests/voice/test_confirmation.py`

- [ ] **Step 1: Write the failing test**

Create `tests/voice/test_confirmation.py`:

```python
"""Tests for confirmation flow classifier."""
from openjarvis.voice.confirmation import ActionType, classify_action, format_confirmation


def test_read_needs_no_confirmation():
    assert classify_action("list_emails") == ActionType.READ


def test_send_email_needs_full_readback():
    assert classify_action("send_email") == ActionType.SEND


def test_reply_needs_full_readback():
    assert classify_action("reply_email") == ActionType.SEND


def test_create_event_needs_brief():
    assert classify_action("create_calendar_event") == ActionType.CREATE


def test_add_task_needs_brief():
    assert classify_action("add_task") == ActionType.CREATE


def test_update_event_needs_brief():
    assert classify_action("update_calendar_event") == ActionType.MODIFY


def test_send_confirmation_includes_all_fields():
    msg = format_confirmation(
        ActionType.SEND,
        subject="Re: Project update",
        to="sarah@example.com",
        body="Hi Sarah, sounds great. See you Thursday.",
    )
    assert "sarah@example.com" in msg
    assert "Re: Project update" in msg
    assert "sounds great" in msg
    assert "Send this?" in msg


def test_create_confirmation_is_brief():
    msg = format_confirmation(
        ActionType.CREATE,
        title="Dentist appointment",
        when="Thursday at 2pm",
    )
    assert "Dentist appointment" in msg
    assert "Thursday" in msg
    assert "Go ahead?" in msg
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/voice/test_confirmation.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement confirmation.py**

```python
"""Action classification and voice confirmation prompt generation."""
from __future__ import annotations

from enum import Enum


class ActionType(Enum):
    READ = "read"       # No confirmation needed
    CREATE = "create"   # Brief confirmation
    MODIFY = "modify"   # Brief confirmation
    SEND = "send"       # Full readback


_ACTION_MAP: dict[str, ActionType] = {
    "list_emails": ActionType.READ,
    "read_email": ActionType.READ,
    "get_email": ActionType.READ,
    "list_calendar_events": ActionType.READ,
    "list_tasks": ActionType.READ,
    "send_email": ActionType.SEND,
    "reply_email": ActionType.SEND,
    "forward_email": ActionType.SEND,
    "create_calendar_event": ActionType.CREATE,
    "add_task": ActionType.CREATE,
    "complete_task": ActionType.MODIFY,
    "update_calendar_event": ActionType.MODIFY,
    "reschedule_event": ActionType.MODIFY,
}


def classify_action(tool_name: str) -> ActionType:
    """Return the ActionType for a given tool function name."""
    return _ACTION_MAP.get(tool_name, ActionType.READ)


def format_confirmation(action_type: ActionType, **kwargs: str) -> str:
    """Build a natural spoken confirmation prompt for the given action."""
    if action_type == ActionType.SEND:
        to = kwargs.get("to", "")
        subject = kwargs.get("subject", "")
        body = kwargs.get("body", "")
        return (
            f"I'll send an email to {to}. "
            f"Subject: {subject}. "
            f"Message: {body}. "
            f"Send this?"
        )
    if action_type == ActionType.CREATE:
        title = kwargs.get("title", "")
        when = kwargs.get("when", "")
        where = kwargs.get("where", "")
        parts = [f"I'll create: {title}"]
        if when:
            parts.append(f"at {when}")
        if where:
            parts.append(f"at {where}")
        return " ".join(parts) + ". Go ahead?"
    if action_type == ActionType.MODIFY:
        description = kwargs.get("description", "make that change")
        return f"I'll {description}. Go ahead?"
    return ""  # READ — no confirmation needed
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/voice/test_confirmation.py -v
```
Expected: all 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/openjarvis/voice/confirmation.py tests/voice/test_confirmation.py
git commit -m "feat(voice): add action classifier and confirmation prompt builder"
```

---

## Task 8: Voice Memory

**Files:**
- Create: `src/openjarvis/voice/memory.py`
- Create: `tests/voice/test_memory.py`

- [ ] **Step 1: Write the failing test**

Create `tests/voice/test_memory.py`:

```python
"""Tests for voice memory (session + long-term)."""
import json
import tempfile
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/voice/test_memory.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement memory.py**

```python
"""Session and long-term memory for the voice assistant."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

_DEFAULT_MEMORY_PATH = Path.home() / ".openjarvis" / "voice_memory.json"

_EMPTY_STORE: dict[str, Any] = {
    "facts": {},
    "standing_instructions": [],
    "contacts": {},
}

_SYSTEM_PROMPT_TEMPLATE = """\
You are Jarvis, a personal AI assistant. You respond via voice — keep answers \
concise and natural-sounding, no markdown or bullet points.

{facts_section}\
{instructions_section}\
{contacts_section}\
"""


class VoiceMemory:
    """Two-layer memory: in-process session history + JSON long-term store."""

    def __init__(self, *, memory_path: Optional[Path] = None) -> None:
        self._path = memory_path or _DEFAULT_MEMORY_PATH
        self._session: list[dict[str, str]] = []
        self._store = self._load()

    # ---- session memory ----

    @property
    def session_history(self) -> list[dict[str, str]]:
        return list(self._session)

    def add_turn(self, role: str, content: str) -> None:
        """Append a conversation turn to the in-memory session."""
        self._session.append({"role": role, "content": content})

    def clear_session(self) -> None:
        self._session.clear()

    # ---- long-term memory ----

    def remember(self, key: str, value: str) -> None:
        """Store a fact persistently (e.g. remember("user_name", "Justin"))."""
        self._store["facts"][key] = value
        self._save()

    def recall(self, key: str) -> Optional[str]:
        """Retrieve a stored fact by key."""
        return self._store["facts"].get(key)

    def add_standing_instruction(self, instruction: str) -> None:
        """Add a persistent standing instruction."""
        if instruction not in self._store["standing_instructions"]:
            self._store["standing_instructions"].append(instruction)
            self._save()

    @property
    def standing_instructions(self) -> list[str]:
        return list(self._store["standing_instructions"])

    def remember_contact(self, name: str, relationship: str) -> None:
        """Store a contact relationship (e.g. "John is my business partner")."""
        self._store["contacts"][name] = relationship
        self._save()

    # ---- system prompt ----

    def build_system_prompt(self) -> str:
        """Build the system prompt injecting all long-term memory."""
        facts = self._store["facts"]
        instructions = self._store["standing_instructions"]
        contacts = self._store["contacts"]

        facts_section = ""
        if facts:
            lines = "\n".join(f"- {k}: {v}" for k, v in facts.items())
            facts_section = f"About the user:\n{lines}\n\n"

        instructions_section = ""
        if instructions:
            lines = "\n".join(f"- {i}" for i in instructions)
            instructions_section = f"Standing instructions:\n{lines}\n\n"

        contacts_section = ""
        if contacts:
            lines = "\n".join(f"- {name}: {rel}" for name, rel in contacts.items())
            contacts_section = f"Known contacts:\n{lines}\n\n"

        return _SYSTEM_PROMPT_TEMPLATE.format(
            facts_section=facts_section,
            instructions_section=instructions_section,
            contacts_section=contacts_section,
        )

    # ---- persistence ----

    def _load(self) -> dict[str, Any]:
        if self._path.exists():
            try:
                with open(self._path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {k: (list(v) if isinstance(v, list) else dict(v))
                for k, v in _EMPTY_STORE.items()}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._store, f, indent=2)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/voice/test_memory.py -v
```
Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/openjarvis/voice/memory.py tests/voice/test_memory.py
git commit -m "feat(voice): add session and long-term voice memory"
```

---

## Task 9: Email Tools

**Files:**
- Create: `src/openjarvis/voice/tools/__init__.py`
- Create: `src/openjarvis/voice/tools/email_tools.py`
- Create: `tests/voice/test_email_tools.py`

These tools call the Gmail REST API directly using `httpx` (already a project dependency). Multi-account support: each account has its own OAuth token stored in the credentials vault.

- [ ] **Step 1: Create tools/__init__.py**

```python
"""Voice assistant action tools."""
```

- [ ] **Step 2: Write the failing test**

Create `tests/voice/test_email_tools.py`:

```python
"""Tests for email tools with mocked httpx."""
import pytest
import respx
import httpx

from openjarvis.voice.tools.email_tools import list_emails, format_email_summary


def test_format_email_summary():
    msg = {
        "id": "abc123",
        "snippet": "Let's meet Thursday at 2pm",
        "payload": {
            "headers": [
                {"name": "From", "value": "Sarah <sarah@example.com>"},
                {"name": "Subject", "value": "Meeting Thursday"},
                {"name": "Date", "value": "Tue, 10 Jun 2026 09:00:00 +0000"},
            ]
        },
    }
    summary = format_email_summary(msg)
    assert "Sarah" in summary
    assert "Meeting Thursday" in summary
    assert "Thursday at 2pm" in summary


@respx.mock
def test_list_emails_returns_summaries():
    # Mock Gmail messages.list
    respx.get("https://gmail.googleapis.com/gmail/v1/users/me/messages").mock(
        return_value=httpx.Response(200, json={
            "messages": [{"id": "msg1", "threadId": "t1"}]
        })
    )
    # Mock Gmail messages.get
    respx.get("https://gmail.googleapis.com/gmail/v1/users/me/messages/msg1").mock(
        return_value=httpx.Response(200, json={
            "id": "msg1",
            "snippet": "Project update attached",
            "payload": {
                "headers": [
                    {"name": "From", "value": "Bob <bob@example.com>"},
                    {"name": "Subject", "value": "Project update"},
                    {"name": "Date", "value": "Tue, 10 Jun 2026 10:00:00 +0000"},
                ]
            },
        })
    )
    results = list_emails(token="fake_token", max_results=5)
    assert len(results) == 1
    assert "Project update" in results[0]
```

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/voice/test_email_tools.py -v
```
Expected: `ImportError`

- [ ] **Step 4: Implement email_tools.py**

```python
"""Gmail action tools for the voice assistant."""
from __future__ import annotations

import base64
import email as email_lib
from email.mime.text import MIMEText
from typing import Optional

import httpx

_GMAIL_BASE = "https://gmail.googleapis.com/gmail/v1/users/me"


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _get_header(payload: dict, name: str) -> str:
    for h in payload.get("headers", []):
        if h["name"].lower() == name.lower():
            return h["value"]
    return ""


def format_email_summary(message: dict) -> str:
    """Convert a raw Gmail message dict to a readable summary string."""
    payload = message.get("payload", {})
    sender = _get_header(payload, "From")
    subject = _get_header(payload, "Subject")
    snippet = message.get("snippet", "")
    return f"From {sender} — Subject: {subject} — {snippet}"


def list_emails(
    token: str,
    *,
    query: str = "is:unread",
    max_results: int = 5,
) -> list[str]:
    """Return a list of email summary strings for the given query."""
    resp = httpx.get(
        f"{_GMAIL_BASE}/messages",
        headers=_headers(token),
        params={"q": query, "maxResults": max_results},
    )
    resp.raise_for_status()
    messages = resp.json().get("messages", [])
    summaries = []
    for m in messages:
        detail = httpx.get(
            f"{_GMAIL_BASE}/messages/{m['id']}",
            headers=_headers(token),
            params={"format": "metadata", "metadataHeaders": ["From", "Subject", "Date"]},
        )
        detail.raise_for_status()
        summaries.append(format_email_summary(detail.json()))
    return summaries


def get_email_body(token: str, message_id: str) -> str:
    """Fetch and decode the plain-text body of an email."""
    resp = httpx.get(
        f"{_GMAIL_BASE}/messages/{message_id}",
        headers=_headers(token),
        params={"format": "full"},
    )
    resp.raise_for_status()
    payload = resp.json().get("payload", {})
    return _extract_body(payload)


def _extract_body(payload: dict) -> str:
    """Recursively extract plain text body from a Gmail payload."""
    mime_type = payload.get("mimeType", "")
    if mime_type == "text/plain":
        data = payload.get("body", {}).get("data", "")
        return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")
    for part in payload.get("parts", []):
        body = _extract_body(part)
        if body:
            return body
    return ""


def send_email(
    token: str,
    *,
    to: str,
    subject: str,
    body: str,
    cc: Optional[str] = None,
    bcc: Optional[str] = None,
) -> str:
    """Send an email via Gmail. Returns the sent message ID."""
    msg = MIMEText(body)
    msg["to"] = to
    msg["subject"] = subject
    if cc:
        msg["cc"] = cc
    if bcc:
        msg["bcc"] = bcc
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    resp = httpx.post(
        f"{_GMAIL_BASE}/messages/send",
        headers=_headers(token),
        json={"raw": raw},
    )
    resp.raise_for_status()
    return resp.json()["id"]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/voice/test_email_tools.py -v
```
Expected: all 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/openjarvis/voice/tools/ tests/voice/test_email_tools.py
git commit -m "feat(voice): add Gmail email tools (list, read, send)"
```

---

## Task 10: Calendar Tools

**Files:**
- Create: `src/openjarvis/voice/tools/calendar_tools.py`
- Create: `tests/voice/test_calendar_tools.py`

- [ ] **Step 1: Write the failing test**

Create `tests/voice/test_calendar_tools.py`:

```python
"""Tests for calendar tools with mocked httpx."""
import respx
import httpx
import pytest

from openjarvis.voice.tools.calendar_tools import list_events, format_event_summary


def test_format_event_summary():
    event = {
        "summary": "Team standup",
        "start": {"dateTime": "2026-06-11T09:00:00-04:00"},
        "end": {"dateTime": "2026-06-11T09:30:00-04:00"},
        "location": "Zoom",
    }
    summary = format_event_summary(event)
    assert "Team standup" in summary
    assert "09:00" in summary


@respx.mock
def test_list_events_returns_summaries():
    respx.get("https://www.googleapis.com/calendar/v3/calendars/primary/events").mock(
        return_value=httpx.Response(200, json={
            "items": [
                {
                    "summary": "Dentist",
                    "start": {"dateTime": "2026-06-12T14:00:00-04:00"},
                    "end": {"dateTime": "2026-06-12T15:00:00-04:00"},
                }
            ]
        })
    )
    results = list_events(token="fake_token", calendar_id="primary")
    assert len(results) == 1
    assert "Dentist" in results[0]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/voice/test_calendar_tools.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement calendar_tools.py**

```python
"""Google Calendar action tools for the voice assistant."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import httpx

_GCAL_BASE = "https://www.googleapis.com/calendar/v3"


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def format_event_summary(event: dict) -> str:
    title = event.get("summary", "Untitled event")
    start = event.get("start", {})
    dt_str = start.get("dateTime") or start.get("date", "")
    location = event.get("location", "")
    try:
        dt = datetime.fromisoformat(dt_str)
        time_str = dt.strftime("%A %b %d at %I:%M %p")
    except ValueError:
        time_str = dt_str
    parts = [f"{title} — {time_str}"]
    if location:
        parts.append(f"at {location}")
    return " ".join(parts)


def list_events(
    token: str,
    *,
    calendar_id: str = "primary",
    time_min: Optional[str] = None,
    time_max: Optional[str] = None,
    max_results: int = 10,
) -> list[str]:
    """Return event summary strings for the given calendar and time range."""
    params: dict = {
        "maxResults": max_results,
        "orderBy": "startTime",
        "singleEvents": "true",
    }
    if time_min:
        params["timeMin"] = time_min
    if time_max:
        params["timeMax"] = time_max
    if not time_min:
        params["timeMin"] = datetime.now(timezone.utc).isoformat()

    resp = httpx.get(
        f"{_GCAL_BASE}/calendars/{calendar_id}/events",
        headers=_headers(token),
        params=params,
    )
    resp.raise_for_status()
    return [format_event_summary(e) for e in resp.json().get("items", [])]


def create_event(
    token: str,
    *,
    title: str,
    start_datetime: str,
    end_datetime: str,
    description: str = "",
    location: str = "",
    calendar_id: str = "primary",
) -> str:
    """Create a calendar event. Returns the new event ID."""
    body = {
        "summary": title,
        "start": {"dateTime": start_datetime},
        "end": {"dateTime": end_datetime},
    }
    if description:
        body["description"] = description
    if location:
        body["location"] = location
    resp = httpx.post(
        f"{_GCAL_BASE}/calendars/{calendar_id}/events",
        headers=_headers(token),
        json=body,
    )
    resp.raise_for_status()
    return resp.json()["id"]


def update_event(
    token: str,
    *,
    event_id: str,
    calendar_id: str = "primary",
    **fields,
) -> None:
    """Patch an existing event with the provided fields."""
    resp = httpx.patch(
        f"{_GCAL_BASE}/calendars/{calendar_id}/events/{event_id}",
        headers=_headers(token),
        json=fields,
    )
    resp.raise_for_status()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/voice/test_calendar_tools.py -v
```
Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/openjarvis/voice/tools/calendar_tools.py tests/voice/test_calendar_tools.py
git commit -m "feat(voice): add Google Calendar tools (list, create, update)"
```

---

## Task 11: Tasks Tools

**Files:**
- Create: `src/openjarvis/voice/tools/tasks_tools.py`
- Create: `tests/voice/test_tasks_tools.py`

- [ ] **Step 1: Write the failing test**

Create `tests/voice/test_tasks_tools.py`:

```python
"""Tests for Google Tasks tools with mocked httpx."""
import respx
import httpx

from openjarvis.voice.tools.tasks_tools import list_tasks, format_task_summary


def test_format_task_summary_with_due():
    task = {"title": "Buy groceries", "due": "2026-06-12T00:00:00.000Z", "status": "needsAction"}
    summary = format_task_summary(task)
    assert "Buy groceries" in summary
    assert "Jun 12" in summary


def test_format_task_summary_no_due():
    task = {"title": "Call dentist", "status": "needsAction"}
    summary = format_task_summary(task)
    assert "Call dentist" in summary


@respx.mock
def test_list_tasks_returns_summaries():
    respx.get("https://tasks.googleapis.com/tasks/v1/users/@me/lists").mock(
        return_value=httpx.Response(200, json={"items": [{"id": "list1", "title": "My Tasks"}]})
    )
    respx.get("https://tasks.googleapis.com/tasks/v1/lists/list1/tasks").mock(
        return_value=httpx.Response(200, json={
            "items": [{"title": "Walk the dog", "status": "needsAction"}]
        })
    )
    results = list_tasks(token="fake_token")
    assert len(results) == 1
    assert "Walk the dog" in results[0]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/voice/test_tasks_tools.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement tasks_tools.py**

```python
"""Google Tasks action tools for the voice assistant."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import httpx

_TASKS_BASE = "https://tasks.googleapis.com/tasks/v1"


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def format_task_summary(task: dict) -> str:
    title = task.get("title", "Untitled task")
    due = task.get("due", "")
    if due:
        try:
            dt = datetime.fromisoformat(due.replace("Z", "+00:00"))
            due_str = dt.strftime("due %b %d")
        except ValueError:
            due_str = due
        return f"{title} ({due_str})"
    return title


def _get_task_lists(token: str) -> list[dict]:
    resp = httpx.get(f"{_TASKS_BASE}/users/@me/lists", headers=_headers(token))
    resp.raise_for_status()
    return resp.json().get("items", [])


def list_tasks(
    token: str,
    *,
    task_list_id: Optional[str] = None,
    show_completed: bool = False,
) -> list[str]:
    """Return task summary strings from all task lists (or a specific one)."""
    if task_list_id:
        lists = [{"id": task_list_id}]
    else:
        lists = _get_task_lists(token)

    summaries: list[str] = []
    for tl in lists:
        params = {"showCompleted": str(show_completed).lower(), "showHidden": "false"}
        resp = httpx.get(
            f"{_TASKS_BASE}/lists/{tl['id']}/tasks",
            headers=_headers(token),
            params=params,
        )
        resp.raise_for_status()
        for task in resp.json().get("items", []):
            if task.get("status") != "completed":
                summaries.append(format_task_summary(task))
    return summaries


def add_task(
    token: str,
    *,
    title: str,
    notes: str = "",
    due: Optional[str] = None,
    task_list_id: Optional[str] = None,
) -> str:
    """Add a task. Returns the new task ID."""
    if not task_list_id:
        lists = _get_task_lists(token)
        task_list_id = lists[0]["id"] if lists else "@default"
    body: dict = {"title": title}
    if notes:
        body["notes"] = notes
    if due:
        body["due"] = due
    resp = httpx.post(
        f"{_TASKS_BASE}/lists/{task_list_id}/tasks",
        headers=_headers(token),
        json=body,
    )
    resp.raise_for_status()
    return resp.json()["id"]


def complete_task(token: str, *, task_id: str, task_list_id: Optional[str] = None) -> None:
    """Mark a task as completed."""
    if not task_list_id:
        lists = _get_task_lists(token)
        task_list_id = lists[0]["id"] if lists else "@default"
    resp = httpx.patch(
        f"{_TASKS_BASE}/lists/{task_list_id}/tasks/{task_id}",
        headers=_headers(token),
        json={"status": "completed"},
    )
    resp.raise_for_status()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/voice/test_tasks_tools.py -v
```
Expected: all 3 tests PASS.

- [ ] **Step 5: Run all voice tests together**

```bash
pytest tests/voice/ -v
```
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/openjarvis/voice/tools/tasks_tools.py tests/voice/test_tasks_tools.py
git commit -m "feat(voice): add Google Tasks tools (list, add, complete)"
```

---

## Task 12: Voice Agent

**Files:**
- Create: `src/openjarvis/voice/agent.py`

The VoiceAgent wires together the router, memory, confirmation, and tools into a single `respond()` method.

- [ ] **Step 1: Create agent.py**

```python
"""Voice assistant agent: routes requests, manages memory, handles confirmation."""
from __future__ import annotations

import logging
from typing import Optional

from openjarvis.voice.confirmation import ActionType, classify_action, format_confirmation
from openjarvis.voice.memory import VoiceMemory
from openjarvis.voice.router import Route, classify_route

logger = logging.getLogger(__name__)

_TOOL_DESCRIPTIONS = {
    "list_emails": "List recent emails. Args: query (str), max_results (int)",
    "get_email_body": "Get full body of an email. Args: message_id (str)",
    "send_email": "Send an email. Args: to (str), subject (str), body (str)",
    "list_events": "List calendar events. Args: time_min (ISO str), time_max (ISO str)",
    "create_calendar_event": "Create a calendar event. Args: title, start_datetime (ISO), end_datetime (ISO), location",
    "list_tasks": "List open tasks.",
    "add_task": "Add a task. Args: title (str), notes (str), due (ISO date str)",
    "complete_task": "Mark a task complete. Args: task_id (str)",
}


class VoiceAgent:
    """Processes voice input: routes to local/cloud LLM, injects memory, handles confirmation.

    This class does NOT call the LLM directly — it prepares the prompt and
    delegates to the engine. For the initial implementation, it uses a simple
    function-calling loop powered by the existing NativeReActAgent.
    """

    def __init__(
        self,
        *,
        memory: VoiceMemory,
        local_engine,
        local_model: str,
        cloud_engine,
        cloud_model: str,
        gmail_tokens: dict[str, str],   # label → OAuth token
        calendar_token: str,
        tasks_token: str,
        confirmation_timeout_s: float = 15.0,
    ) -> None:
        self._memory = memory
        self._local_engine = local_engine
        self._local_model = local_model
        self._cloud_engine = cloud_engine
        self._cloud_model = cloud_model
        self._gmail_tokens = gmail_tokens
        self._calendar_token = calendar_token
        self._tasks_token = tasks_token
        self._confirmation_timeout_s = confirmation_timeout_s

    def respond(
        self,
        user_input: str,
        *,
        speak_confirmation: callable,
        listen_for_response: callable,
    ) -> str:
        """Process a voice utterance and return the spoken response string.

        Args:
            user_input: Transcribed text from the user.
            speak_confirmation: Callable(text) — speaks a confirmation prompt aloud.
            listen_for_response: Callable() → str — records and transcribes the
                user's yes/no response after a confirmation prompt.
        """
        self._memory.add_turn("user", user_input)

        route = classify_route(user_input)
        engine = self._cloud_engine if route == Route.CLOUD else self._local_engine
        model = self._cloud_model if route == Route.CLOUD else self._local_model

        logger.debug("Routing '%s' → %s (%s)", user_input[:60], route.value, model)

        system_prompt = self._memory.build_system_prompt()
        history = self._memory.session_history[:-1]  # exclude the turn we just added

        # Build messages for the engine
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_input})

        # Generate response
        result = engine.chat(messages, model=model)
        response_text = result.content if hasattr(result, "content") else str(result)

        # Check if the response contains a tool call requiring confirmation
        tool_name, tool_args = self._parse_tool_call(response_text)
        if tool_name:
            action_type = classify_action(tool_name)
            if action_type != ActionType.READ:
                confirmation_prompt = format_confirmation(action_type, **tool_args)
                speak_confirmation(confirmation_prompt)
                user_reply = listen_for_response()
                if not self._is_affirmative(user_reply):
                    self._memory.add_turn("assistant", "Cancelled.")
                    return "Got it, cancelled."
                # Execute the confirmed action
                response_text = self._execute_tool(tool_name, tool_args)

        self._memory.add_turn("assistant", response_text)

        # Persist anything Jarvis learned (e.g. "remember that John is my partner")
        self._maybe_persist_fact(user_input, response_text)

        return response_text

    def _parse_tool_call(self, text: str) -> tuple[str, dict]:
        """Extract tool name and args from an LLM response. Returns ('', {}) if no tool call."""
        # Simple heuristic: look for TOOL: prefix in the response.
        # The system prompt instructs the LLM to use this format.
        import re
        match = re.search(r"TOOL:\s*(\w+)\s*\(([^)]*)\)", text)
        if not match:
            return "", {}
        tool_name = match.group(1)
        raw_args = match.group(2)
        args = {}
        for pair in raw_args.split(","):
            if "=" in pair:
                k, _, v = pair.partition("=")
                args[k.strip()] = v.strip().strip('"\'')
        return tool_name, args

    def _execute_tool(self, tool_name: str, args: dict) -> str:
        """Execute a confirmed tool action and return a spoken result."""
        from openjarvis.voice.tools import email_tools, calendar_tools, tasks_tools

        primary_token = next(iter(self._gmail_tokens.values()), "")

        dispatch = {
            "list_emails": lambda: "\n".join(
                email_tools.list_emails(primary_token, **args)
            ),
            "send_email": lambda: (
                email_tools.send_email(primary_token, **args),
                "Email sent."
            )[-1],
            "list_events": lambda: "\n".join(
                calendar_tools.list_events(self._calendar_token, **args)
            ),
            "create_calendar_event": lambda: (
                calendar_tools.create_event(self._calendar_token, **args),
                "Event created."
            )[-1],
            "list_tasks": lambda: "\n".join(
                tasks_tools.list_tasks(self._tasks_token)
            ),
            "add_task": lambda: (
                tasks_tools.add_task(self._tasks_token, **args),
                "Task added."
            )[-1],
            "complete_task": lambda: (
                tasks_tools.complete_task(self._tasks_token, **args),
                "Task marked complete."
            )[-1],
        }

        fn = dispatch.get(tool_name)
        if fn:
            return fn()
        return "I'm not sure how to do that yet."

    @staticmethod
    def _is_affirmative(text: str) -> bool:
        affirmatives = {"yes", "yeah", "yep", "go ahead", "send it", "do it", "sure", "ok", "okay"}
        return any(word in text.lower() for word in affirmatives)

    def _maybe_persist_fact(self, user_input: str, response: str) -> None:
        """If the user said 'remember that...', persist it to long-term memory."""
        import re
        match = re.search(r"remember that (.+)", user_input, re.IGNORECASE)
        if match:
            fact = match.group(1).strip()
            # Use a timestamp-based key to avoid collisions
            import time
            key = f"user_fact_{int(time.time())}"
            self._memory.remember(key, fact)
```

- [ ] **Step 2: Verify the module imports**

```bash
python -c "from openjarvis.voice.agent import VoiceAgent; print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/openjarvis/voice/agent.py
git commit -m "feat(voice): add VoiceAgent with routing, memory, confirmation"
```

---

## Task 13: Voice Loop

**Files:**
- Create: `src/openjarvis/voice/loop.py`

The loop ties everything together: wake word → capture → STT → agent → TTS → repeat.

- [ ] **Step 1: Create loop.py**

```python
"""Main voice assistant loop: wake word → STT → agent → TTS → repeat."""
from __future__ import annotations

import logging
import signal
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_CHIME = "\a"  # terminal bell as a fallback; replaced with audio in production


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

                # Play chime to signal Jarvis is listening
                self._tts.speak("Mm?")

                # Record utterance
                text = self._record_and_transcribe()
                if not text.strip():
                    continue

                logger.info("User said: %s", text)

                # Get response from agent (may trigger confirmation flow)
                response = self._agent.respond(
                    text,
                    speak_confirmation=self._tts.speak,
                    listen_for_response=self._record_and_transcribe,
                )

                # Speak the response
                self._tts.speak(response)

        finally:
            self._detector.stop()
            logger.info("Voice loop stopped")

    def stop(self) -> None:
        """Signal the loop to stop after the current turn."""
        self._stop_event.set()
```

- [ ] **Step 2: Verify the module imports**

```bash
python -c "from openjarvis.voice.loop import VoiceLoop; print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/openjarvis/voice/loop.py
git commit -m "feat(voice): add main VoiceLoop orchestrator"
```

---

## Task 14: CLI — `jarvis voice` Command Group

**Files:**
- Create: `src/openjarvis/cli/voice_cmd.py`
- Modify: `src/openjarvis/cli/__init__.py`

- [ ] **Step 1: Create voice_cmd.py**

```python
"""``jarvis voice`` command group — train, setup, and start the voice assistant."""
from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console

from openjarvis.core.config import load_config

console = Console()

_MODEL_PATH = Path.home() / ".openjarvis" / "models" / "hey_jarvis.onnx"


@click.group("voice")
def voice() -> None:
    """Always-on voice assistant — say 'Hey Jarvis' to start."""


@voice.command("train")
@click.option("--samples", default=20, show_default=True,
              help="Number of TTS samples per voice (more = better accuracy, longer training)")
def voice_train(samples: int) -> None:
    """Generate synthetic training data and train the 'Hey Jarvis' wake word model.

    Requires macOS (uses the built-in `say` command). Takes ~10-20 minutes.
    Only needs to be run once.
    """
    from openjarvis.voice.train_wake_word import run_training

    if _MODEL_PATH.exists():
        click.confirm(
            f"Model already exists at {_MODEL_PATH}. Re-train?",
            abort=True,
        )

    console.print("[bold]Training 'Hey Jarvis' wake word model...[/bold]")
    model_path = run_training(n_per_voice=samples)
    console.print(f"[green]Done! Model saved to {model_path}[/green]")
    console.print("Run [bold]jarvis voice setup[/bold] next to connect your accounts.")


@voice.command("setup")
def voice_setup() -> None:
    """Connect Gmail, Calendar, and Tasks accounts via OAuth.

    Run this once before starting the voice assistant for the first time.
    """
    console.print("[bold]Voice Assistant Setup[/bold]\n")
    _setup_gmail()
    _setup_calendar()
    _setup_tasks()
    console.print("\n[green]Setup complete! Run [bold]jarvis voice start[/bold] to begin.[/green]")


def _setup_gmail() -> None:
    console.print("[bold]Gmail accounts[/bold]")
    console.print("You can connect multiple Gmail accounts. Press Enter with no input when done.\n")
    cfg = load_config()
    accounts = []
    idx = 1
    while True:
        label = click.prompt(f"Account {idx} label (e.g. 'work email', 'personal')", default="")
        if not label:
            break
        console.print(f"Opening browser for OAuth — sign in to your {label} Gmail account...")
        token = _run_google_oauth(
            scopes=[
                "https://www.googleapis.com/auth/gmail.readonly",
                "https://www.googleapis.com/auth/gmail.send",
            ],
            label=f"gmail_{label.replace(' ', '_')}",
        )
        if token:
            accounts.append({"label": label, "token_path": token})
            console.print(f"[green]  ✓ {label} connected[/green]")
        idx += 1
    console.print(f"Connected {len(accounts)} Gmail account(s).")


def _setup_calendar() -> None:
    console.print("\n[bold]Google Calendar[/bold]")
    console.print("Opening browser for OAuth...")
    _run_google_oauth(
        scopes=["https://www.googleapis.com/auth/calendar"],
        label="gcalendar",
    )
    console.print("[green]  ✓ Calendar connected[/green]")


def _setup_tasks() -> None:
    console.print("\n[bold]Google Tasks[/bold]")
    console.print("Opening browser for OAuth...")
    _run_google_oauth(
        scopes=["https://www.googleapis.com/auth/tasks"],
        label="gtasks",
    )
    console.print("[green]  ✓ Tasks connected[/green]")


def _run_google_oauth(*, scopes: list[str], label: str) -> str | None:
    """Run the OAuth flow and save the token. Returns the token file path."""
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore[import]
        from openjarvis.core.config import DEFAULT_CONFIG_DIR

        client_secrets = DEFAULT_CONFIG_DIR / "google_client_secret.json"
        if not client_secrets.exists():
            console.print(
                f"[red]Missing {client_secrets}[/red]\n"
                "Download your OAuth 2.0 client credentials from Google Cloud Console\n"
                "(APIs & Services → Credentials → Download JSON) and save to that path."
            )
            return None

        flow = InstalledAppFlow.from_client_secrets_file(str(client_secrets), scopes=scopes)
        creds = flow.run_local_server(port=0)
        token_path = DEFAULT_CONFIG_DIR / f"{label}_token.json"
        with open(token_path, "w") as f:
            f.write(creds.to_json())
        return str(token_path)
    except Exception as exc:
        console.print(f"[red]OAuth failed: {exc}[/red]")
        return None


@voice.command("start")
@click.option("--device", default="", help="Microphone device name fragment (e.g. 'Logitech')")
@click.option("--stt-model", default="large-v3", show_default=True, help="Faster-Whisper model size")
@click.option("--local-model", default="qwen2.5:14b", show_default=True, help="Local Ollama model")
def voice_start(device: str, stt_model: str, local_model: str) -> None:
    """Start the always-on voice assistant."""
    if not _MODEL_PATH.exists():
        console.print(
            "[red]Wake word model not found.[/red] "
            "Run [bold]jarvis voice train[/bold] first."
        )
        sys.exit(1)

    from openjarvis.voice.capture import find_device_index
    from openjarvis.engine.ollama import OllamaEngine

    mic_device_idx = None
    if device:
        mic_device_idx = find_device_index(device)
        if mic_device_idx is None:
            console.print(f"[yellow]Warning: no device matching '{device}' found, using default[/yellow]")

    cfg = load_config()
    tokens = _load_tokens(cfg)

    local_engine = OllamaEngine()
    cloud_engine = _get_cloud_engine(cfg)

    from openjarvis.voice.loop import VoiceLoop

    loop = VoiceLoop(
        wake_word_model=_MODEL_PATH,
        stt_model=stt_model,
        mic_device=mic_device_idx,
        local_engine=local_engine,
        local_model=local_model,
        cloud_engine=cloud_engine,
        cloud_model=cfg.voice_assistant.cloud_model,
        gmail_tokens=tokens["gmail"],
        calendar_token=tokens.get("gcalendar", ""),
        tasks_token=tokens.get("gtasks", ""),
    )
    loop.run()


def _load_tokens(cfg) -> dict:
    """Load OAuth tokens from disk. Returns {service: token_string} dict."""
    import json
    from openjarvis.core.config import DEFAULT_CONFIG_DIR

    tokens: dict = {"gmail": {}}

    # Gmail accounts
    for token_file in DEFAULT_CONFIG_DIR.glob("gmail_*_token.json"):
        label = token_file.stem.replace("gmail_", "").replace("_token", "")
        try:
            data = json.loads(token_file.read_text())
            tokens["gmail"][label] = data.get("token", "")
        except Exception:
            pass

    # Calendar and Tasks
    for service in ("gcalendar", "gtasks"):
        token_file = DEFAULT_CONFIG_DIR / f"{service}_token.json"
        if token_file.exists():
            try:
                data = json.loads(token_file.read_text())
                tokens[service] = data.get("token", "")
            except Exception:
                pass

    return tokens


def _get_cloud_engine(cfg):
    """Return the configured cloud engine (Anthropic Claude)."""
    try:
        from openjarvis.engine.anthropic import AnthropicEngine
        return AnthropicEngine()
    except Exception:
        from openjarvis.engine.ollama import OllamaEngine
        return OllamaEngine()
```

- [ ] **Step 2: Register `voice` in CLI __init__.py**

Open `src/openjarvis/cli/__init__.py`. Add the import after the other imports (before line 44 where the `@click.group` starts):

```python
from openjarvis.cli.voice_cmd import voice
```

Add the command registration after the other `cli.add_command` lines (around line 145):

```python
cli.add_command(voice, "voice")
```

- [ ] **Step 3: Verify the command is registered**

```bash
jarvis voice --help
```
Expected output includes:
```
Usage: jarvis voice [OPTIONS] COMMAND [ARGS]...

  Always-on voice assistant — say 'Hey Jarvis' to start.

Commands:
  start  Start the always-on voice assistant.
  setup  Connect Gmail, Calendar, and Tasks accounts via OAuth.
  train  Generate synthetic training data and train the wake word model.
```

- [ ] **Step 4: Commit**

```bash
git add src/openjarvis/cli/voice_cmd.py src/openjarvis/cli/__init__.py
git commit -m "feat(voice): add jarvis voice CLI group (train, setup, start)"
```

---

## Task 15: Run All Tests and Final Verification

- [ ] **Step 1: Run the full voice test suite**

```bash
pytest tests/voice/ -v
```
Expected: all tests PASS. Zero failures.

- [ ] **Step 2: Verify CLI help tree**

```bash
jarvis voice --help
jarvis voice train --help
jarvis voice setup --help
jarvis voice start --help
```
Expected: each command shows its options without error.

- [ ] **Step 3: Verify imports don't break the main CLI**

```bash
jarvis --help
```
Expected: normal jarvis help output, no import errors.

- [ ] **Step 4: Verify wake word training script runs (dry run)**

```bash
python -c "
from openjarvis.voice.train_wake_word import generate_samples
from pathlib import Path
import tempfile
with tempfile.TemporaryDirectory() as d:
    generate_samples(Path(d), n_per_voice=1)
    import os
    files = list(Path(d).glob('*.wav'))
    print(f'{len(files)} sample(s) generated: ok')
"
```
Expected: `N sample(s) generated: ok` (one per voice × one phrase)

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "feat(voice): complete voice assistant implementation"
```

---

## First-Use Instructions (After Implementation)

Once all tasks are complete, here is the sequence to get the voice assistant running:

```bash
# 1. Install dependencies
uv pip install -e ".[voice]"

# 2. Train the wake word model (~20 minutes, Mac only)
jarvis voice train

# 3. Set up Google accounts (follow prompts)
#    Requires: google_client_secret.json in ~/.openjarvis/
jarvis voice setup

# 4. Start Jarvis (plug in Logitech webcam microphone first)
jarvis voice start --device Logitech

# Say "Hey Jarvis" from across the room.
```
