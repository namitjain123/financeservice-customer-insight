"""Single place where an API client is constructed.

Every tool imports from here, so switching provider means editing config/settings.py
once rather than hunting base URLs through four tool modules.
"""

from __future__ import annotations

from openai import AsyncOpenAI, OpenAI

from config import settings


def get_async_client() -> AsyncOpenAI:
    """Async client, for the concurrent extraction and labelling steps."""
    return AsyncOpenAI(api_key=settings.require_api_key(), base_url=settings.BASE_URL)


def get_client() -> OpenAI:
    """Sync client, for the one-shot embedding call."""
    return OpenAI(api_key=settings.require_api_key(), base_url=settings.BASE_URL)
