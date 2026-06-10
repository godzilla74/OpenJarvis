"""Route voice requests between local Ollama model and cloud Claude Sonnet."""
from __future__ import annotations

import re
from enum import Enum


class Route(Enum):
    LOCAL = "local"
    CLOUD = "cloud"


_CLOUD_SIGNALS = [
    r"\bproper (reply|response|email)\b",
    r"\bdetailed (reply|response|email|summary)\b",
    r"\bdraft\b.{0,40}\b(contract|proposal|negotiat|formal|important)\b",
    r"\b(then|and then|after that|finally)\b.{0,60}\b(reply|send|draft|write)\b",
    r"\bmulti.step\b",
    r"\bsummariz.{0,10} (and|then) (reply|respond|draft)\b",
]

_CLOUD_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _CLOUD_SIGNALS]

_COMPLEX_WORD_COUNT = 30


def classify_route(text: str) -> Route:
    """Return LOCAL for simple requests, CLOUD for complex ones."""
    for pattern in _CLOUD_PATTERNS:
        if pattern.search(text):
            return Route.CLOUD
    if len(text.split()) > _COMPLEX_WORD_COUNT:
        return Route.CLOUD
    return Route.LOCAL
