"""Gmail action tools for the voice assistant."""
from __future__ import annotations

import base64
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
