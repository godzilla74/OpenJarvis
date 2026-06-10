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
