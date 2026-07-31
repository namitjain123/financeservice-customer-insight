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


def get_embed_client() -> OpenAI:
    """Sync client for embeddings - a separate provider from chat.

    Gemini's chat and embedding quotas are tracked independently, but the
    embedding one is a hard daily cap that has been unreliable to predict a
    reset time for. Embeddings are the only thing this client is used for;
    chat/topic-extraction stays on get_async_client() and is unaffected.
    """
    return OpenAI(api_key=settings.require_embed_api_key(), base_url=settings.EMBED_BASE_URL)
