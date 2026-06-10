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

VOICES = [
    "Alex", "Samantha", "Victoria", "Tom", "Fred",
    "Karen", "Moira", "Tessa", "Daniel", "Rishi",
]

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
    skipped_voices: list[str] = []
    for voice in VOICES:
        voice_failed = False
        for phrase in PHRASES:
            if voice_failed:
                break
            for i in range(n_per_voice):
                aiff_path = output_dir / f"sample_{count:04d}.aiff"
                wav_path = output_dir / f"sample_{count:04d}.wav"
                try:
                    subprocess.run(
                        ["say", "-v", voice, "-r", str(140 + (i % 40) - 20), phrase,
                         "-o", str(aiff_path)],
                        check=True, capture_output=True,
                    )
                    subprocess.run(
                        ["afconvert", "-f", "WAVE", "-d", "LEI16@16000",
                         str(aiff_path), str(wav_path)],
                        check=True, capture_output=True,
                    )
                    aiff_path.unlink(missing_ok=True)
                    count += 1
                except subprocess.CalledProcessError:
                    # Clean up any partial files left behind
                    aiff_path.unlink(missing_ok=True)
                    wav_path.unlink(missing_ok=True)
                    if voice not in skipped_voices:
                        skipped_voices.append(voice)
                    voice_failed = True
                    break

    if skipped_voices:
        print(f"Skipped voices (not installed or failed): {', '.join(skipped_voices)}")
    print(f"Generated {count} samples in {output_dir}")


def train(positive_dir: Path, output_dir: Path) -> Path:
    """Train openwakeword model on positive samples."""
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
