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
