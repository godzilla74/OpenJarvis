# Voice Assistant Design — "Hey Jarvis"

**Date:** 2026-06-10  
**Status:** Approved  

---

## Overview

An always-on voice assistant that runs on a dedicated Mac Mini (Apple Silicon). The user walks into the room, says "Hey Jarvis", and has a natural spoken dialogue. Jarvis can answer questions, read and draft emails, manage calendar events and tasks, and execute actions — always confirming before doing anything irreversible.

---

## Architecture

```
[Logitech USB Mic]
        │
        ▼
Wake Word Detection (Porcupine, local, ~1% CPU)
        │  "Hey Jarvis" detected
        ▼
Audio Capture (sounddevice, silence detection)
        │
        ▼
STT — Faster-Whisper large-v3 (local, Apple Silicon)
        │
        ▼
VoiceAgent
  ├── Router: local MLX model (primary) → Claude Sonnet (complex fallback)
  ├── Long-term Memory (~/.openjarvis/memory/)
  ├── Gmail connector (multiple accounts)
  ├── Google Calendar connector
  └── Google Tasks connector
        │
        ▼
Confirmation Wrapper (smart mode)
        │
        ▼
TTS — Kokoro local (primary) → OpenAI TTS (fallback for long responses)
        │
        ▼
[Speakers] → loops back to wake word listener
```

---

## Components

### 1. `src/openjarvis/voice/` (new package)

| File | Responsibility |
|---|---|
| `wake_word.py` | Porcupine listener; fires callback on "Hey Jarvis"; runs continuously at ~1% CPU |
| `capture.py` | Records from USB mic via `sounddevice`; ends utterance on 1.5s silence; plays chime on wake |
| `loop.py` | Main orchestrator: wake → capture → STT → agent → confirmation → TTS → loop |
| `confirmation.py` | Classifies action type; applies brief or detailed confirmation logic |
| `router.py` | Scores task complexity; routes to local MLX or Claude Sonnet |

### 2. `src/openjarvis/cli/voice_cmd.py` (new)

Adds `jarvis voice` command. Flags:
- `--device` — select audio input device by name (defaults to first USB audio device)
- `--wake-word` — override wake phrase (default: "hey jarvis")
- `--tts` — choose TTS backend: `kokoro` (default) or `openai`

### 3. Memory (extend existing)

Two layers persisted to `~/.openjarvis/memory/`:

**Working memory** — current session dialogue context. Allows follow-up references ("make that Friday instead", "reply to the second one").

**Long-term memory** — JSON store, updated automatically:
- User name and preferences
- Known contacts and relationships ("John is my business partner")
- Standing instructions ("always BCC me on sent emails")
- Past decisions and preferences learned over time

Retrieval: semantic search via the local LLM. No vector database required at this scale.

### 4. Connectors (existing, configured for voice)

**Gmail (multiple accounts)**
- Each account OAuth token stored in the existing credentials vault
- Labelled during a one-time `jarvis voice setup` step: user says the label for each account ("work email", "personal email", etc.)
- Jarvis checks all accounts for reads; confirms which account for sends

**Google Calendar**
- Read events, create events, reschedule, cancel
- Natural language date parsing ("next Tuesday", "end of the month")

**Google Tasks**
- Read and add tasks
- Mark complete by voice
- Optional: surface overdue items proactively

### 5. LLM Routing

| Task type | Model |
|---|---|
| Read / summarize email | Local MLX (Qwen2.5-14B or Llama3.1-8B) |
| List calendar events | Local MLX |
| Add task / create event | Local MLX |
| Draft email reply | Local MLX if reply is ≤3 sentences; Claude Sonnet if thread is long, tone is sensitive, or user says "write a proper reply" |
| Multi-step planning | Claude Sonnet |
| General Q&A | Local MLX |

Routing is automatic based on a lightweight complexity classifier built into `router.py`.

---

## Confirmation Flow

| Action | Behavior |
|---|---|
| Read / query | No confirmation — answer immediately |
| Create task or calendar event | Brief — *"I'll add a dentist appointment Thursday at 2pm. Go ahead?"* |
| Draft + send email | Full readback — reads subject, recipient, and full body, then *"Want me to send that?"* |
| Reply to email | Full readback — reads subject, recipient, and full body |
| Modify or reschedule | Brief summary of what's changing |
| Delete | Not available in v1 |

**Confirmation vocabulary Jarvis understands:**
- Confirm: "yes", "go ahead", "send it", "do it", "sure", "yep"
- Cancel: "no", "cancel", "stop", "never mind", "don't"
- Revise: "change the...", "actually...", "make it..." → Jarvis revises and reads back again

**Timeout rule:** If no response within 15 seconds after asking for confirmation, Jarvis cancels the action and says so.

---

## Voice Pipeline Details

**Wake word:** Picovoice Porcupine (free tier). Has "Jarvis" as a built-in keyword — zero training needed. Runs entirely on-device; API key required for initialization only, no network calls during operation.

**STT:** Faster-Whisper `large-v3` — already in codebase. Runs on Apple Silicon via Core ML. Target transcription latency: ~1–2s for a typical utterance.

**TTS:** Kokoro local — already in codebase. Falls back to OpenAI TTS for responses longer than ~200 words where voice quality matters more.

**End-to-end latency target:** Wake word fires → Jarvis starts speaking in under 4 seconds for typical queries.

**Always-listening design:** While Jarvis is speaking a response, the wake word listener is already armed in the background. Saying "Hey Jarvis" mid-response interrupts and restarts.

---

## New Dependencies

| Package | Purpose |
|---|---|
| `pvporcupine` | Wake word detection (Picovoice Porcupine) |
| `sounddevice` | Audio capture from USB mic |
| `numpy` | Audio buffer handling (likely already present) |

MLX and Faster-Whisper are already optional extras in `pyproject.toml`.

---

## Out of Scope (v1)

- Deleting emails, events, or tasks
- Smart home / computer control
- Proactive interruptions (Jarvis speaking unprompted)
- Multi-user voice profiles
- Custom wake word training

---

## Success Criteria

1. "Hey Jarvis" reliably wakes the assistant from across the room
2. Jarvis understands natural speech and responds within 4 seconds
3. Email reads, calendar queries, and task additions work end-to-end
4. No action is taken on email or calendar without a spoken confirmation
5. Jarvis remembers standing preferences across sessions
