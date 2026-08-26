"""
utils/gemini.py
Shared Google GenAI (google-genai) client factory.

Centralizes client creation so every module uses the same configured client
and API key. Uses the new `google-genai` SDK (the legacy `google-generativeai`
package is deprecated).
"""
from __future__ import annotations

import logging
from functools import lru_cache

from google import genai

from config import cfg

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_client() -> "genai.Client":
    """Return a cached google-genai Client configured with the Gemini API key."""
    if not cfg.GEMINI_API_KEY:
        logger.warning(
            "Creating Gemini client without an API key; LLM calls will fail "
            "until GEMINI_API_KEY is set."
        )
    return genai.Client(api_key=cfg.GEMINI_API_KEY)
