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

    @property
    def session_history(self) -> list[dict[str, str]]:
        return list(self._session)

    def add_turn(self, role: str, content: str) -> None:
        """Append a conversation turn to the in-memory session."""
        self._session.append({"role": role, "content": content})

    def clear_session(self) -> None:
        self._session.clear()

    def remember(self, key: str, value: str) -> None:
        """Store a fact persistently."""
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
        """Store a contact relationship."""
        self._store["contacts"][name] = relationship
        self._save()

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
