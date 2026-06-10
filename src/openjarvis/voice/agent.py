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
    """Processes voice input: routes to local/cloud LLM, injects memory, handles confirmation."""

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
        """Process a voice utterance and return the spoken response string."""
        self._memory.add_turn("user", user_input)

        route = classify_route(user_input)
        engine = self._cloud_engine if route == Route.CLOUD else self._local_engine
        model = self._cloud_model if route == Route.CLOUD else self._local_model

        logger.debug("Routing '%s' → %s (%s)", user_input[:60], route.value, model)

        system_prompt = self._memory.build_system_prompt()
        history = self._memory.session_history[:-1]

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_input})

        result = engine.generate(messages, model=model)
        response_text = result["content"]

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
                response_text = self._execute_tool(tool_name, tool_args)

        self._memory.add_turn("assistant", response_text)
        self._maybe_persist_fact(user_input, response_text)

        return response_text

    def _parse_tool_call(self, text: str) -> tuple[str, dict]:
        """Extract tool name and args from LLM response. Returns ('', {}) if none."""
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
        """If the user said 'remember that...', persist to long-term memory."""
        import re
        import time
        match = re.search(r"remember that (.+)", user_input, re.IGNORECASE)
        if match:
            fact = match.group(1).strip()
            key = f"user_fact_{int(time.time())}"
            self._memory.remember(key, fact)
