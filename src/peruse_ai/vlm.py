"""
peruse_ai.vlm
~~~~~~~~~~~~~
Local VLM adapter — factory + prompt builder for Ollama, LM Studio, and OpenAI-compatible backends.
"""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from peruse_ai.config import PeruseConfig, VLMBackend

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT = """\
You are Peruse-AI, an expert web exploration agent. You are given:
1. A screenshot of the current browser viewport.
2. A simplified DOM tree of interactive elements on the page.
3. A high-level task from the user.

Your job is to decide the SINGLE best next action to take toward completing the task.

Rules:
- Respond ONLY with a valid JSON object (no markdown fences).
- Always include "thought" (your reasoning) and "action" (what to do).
- Supported actions:
    {"action": "click", "element_id": <int>}
    {"action": "type", "element_id": <int>, "text": "<value>"}
    {"action": "scroll", "direction": "up"|"down"}
    {"action": "navigate", "url": "<url>"}
    {"action": "wait", "seconds": <int>}
    {"action": "done", "summary": "<final summary of what was accomplished>"}
- Use "done" only when the task is fully complete or clearly impossible.
"""

INSIGHT_SYSTEM_PROMPT = """\
You are a data analyst. Analyze the provided screenshot(s) of a web application and produce a concise Markdown report summarizing all visible data, charts, tables, and key metrics. Highlight trends, anomalies, and notable figures.
"""

UX_REVIEW_SYSTEM_PROMPT = """\
You are a senior UX/UI designer. Critique the provided screenshot(s) of a web application. Evaluate:
- Visual hierarchy and layout
- Color contrast and accessibility (WCAG)
- Button/target sizes (touch-friendly?)
- Information density and readability
- Consistency and modern design patterns
Output a Markdown report with specific, actionable suggestions.
"""

# ---------------------------------------------------------------------------
# VLM Factory
# ---------------------------------------------------------------------------


def create_vlm(config: PeruseConfig) -> BaseChatModel:
    """Create a LangChain-compatible chat model for the configured VLM backend.

    Returns:
        A BaseChatModel instance ready for .invoke() calls.

    Raises:
        ValueError: If the backend is not supported.
        ConnectionError: If the backend is unreachable.
    """
    if config.vlm_backend == VLMBackend.OLLAMA:
        return _create_ollama_vlm(config)
    elif config.vlm_backend == VLMBackend.LMSTUDIO:
        return _create_lmstudio_vlm(config)
    elif config.vlm_backend == VLMBackend.OPENAI_COMPAT:
        return _create_openai_compat_vlm(config)
    else:
        raise ValueError(f"Unsupported VLM backend: {config.vlm_backend}")


def _create_ollama_vlm(config: PeruseConfig) -> BaseChatModel:
    """Create an Ollama-backed VLM via LangChain's ChatOllama."""
    from langchain_ollama import ChatOllama

    logger.info("Initializing Ollama VLM: model=%s, base_url=%s", config.vlm_model, config.vlm_base_url)
    return ChatOllama(
        model=config.vlm_model,
        base_url=config.get_ollama_base_url(),
        temperature=config.vlm_temperature,
        timeout=config.vlm_timeout,
    )


def _create_lmstudio_vlm(config: PeruseConfig) -> BaseChatModel:
    """Create an LM Studio-backed VLM via OpenAI-compatible endpoint."""
    from langchain_openai import ChatOpenAI

    base_url = config.get_lmstudio_base_url()
    logger.info("Initializing LM Studio VLM: model=%s, base_url=%s", config.vlm_model, base_url)
    return ChatOpenAI(
        model=config.vlm_model,
        base_url=base_url,
        api_key=config.vlm_api_key or "lm-studio",  # LM Studio doesn't require a real key
        temperature=config.vlm_temperature,
        timeout=config.vlm_timeout,
    )


def _create_openai_compat_vlm(config: PeruseConfig) -> BaseChatModel:
    """Create an OpenAI-compatible VLM (generic local endpoint)."""
    from langchain_openai import ChatOpenAI

    logger.info(
        "Initializing OpenAI-compatible VLM: model=%s, base_url=%s",
        config.vlm_model,
        config.vlm_base_url,
    )
    return ChatOpenAI(
        model=config.vlm_model,
        base_url=config.vlm_base_url,
        api_key=config.vlm_api_key or "not-needed",
        temperature=config.vlm_temperature,
        timeout=config.vlm_timeout,
    )


# ---------------------------------------------------------------------------
# Prompt Builders
# ---------------------------------------------------------------------------


def encode_image_b64(image_bytes: bytes) -> str:
    """Encode raw image bytes to a base64 string."""
    return base64.b64encode(image_bytes).decode("utf-8")


def build_vision_prompt(
    screenshot_b64: str,
    dom_text: str,
    task: str,
    step_history: list[dict] | None = None,
) -> list:
    """Build a multi-modal prompt with screenshot + DOM for the agent loop.

    Args:
        screenshot_b64: Base64-encoded screenshot image.
        dom_text: Simplified DOM text with indexed interactive elements.
        task: The user's high-level goal.
        step_history: Optional list of previous step summaries.

    Returns:
        A list of LangChain message objects ready for model.invoke().
    """
    messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)]

    # Build the human message content blocks
    content_blocks = []

    # History context
    if step_history:
        history_text = "\n".join(
            f"Step {i + 1}: {step.get('thought', '')} → {step.get('action', '')}"
            for i, step in enumerate(step_history[-10:])  # last 10 steps
        )
        content_blocks.append({"type": "text", "text": f"## Previous Steps\n{history_text}\n"})

    # Task
    content_blocks.append({"type": "text", "text": f"## Task\n{task}\n"})

    # DOM
    content_blocks.append({"type": "text", "text": f"## Interactive DOM Elements\n```\n{dom_text}\n```\n"})

    # Screenshot
    content_blocks.append(
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
        }
    )

    content_blocks.append(
        {"type": "text", "text": "Based on the screenshot and DOM above, what is the best next action? Respond with JSON only."}
    )

    messages.append(HumanMessage(content=content_blocks))
    return messages


def build_analysis_prompt(
    screenshots_b64: list[str],
    system_prompt: str,
    context: str = "",
) -> list:
    """Build a multi-image analysis prompt for insight/UX generation.

    Args:
        screenshots_b64: List of base64-encoded screenshot images.
        system_prompt: The system prompt defining the analysis role.
        context: Optional additional context about the session.

    Returns:
        A list of LangChain message objects.
    """
    messages = [SystemMessage(content=system_prompt)]

    content_blocks = []
    if context:
        content_blocks.append({"type": "text", "text": context})

    for i, img_b64 in enumerate(screenshots_b64):
        content_blocks.append({"type": "text", "text": f"### Screenshot {i + 1}"})
        content_blocks.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
            }
        )

    content_blocks.append({"type": "text", "text": "Produce your Markdown report now."})
    messages.append(HumanMessage(content=content_blocks))
    return messages


async def check_vlm_connection(config: PeruseConfig) -> dict:
    """Check if the configured VLM backend is reachable and responding.

    Returns:
        A dict with keys: 'status' ('ok' | 'error'), 'backend', 'model', 'message'.
    """
    try:
        vlm = create_vlm(config)
        response = await vlm.ainvoke("Say 'hello' in one word.")
        return {
            "status": "ok",
            "backend": config.vlm_backend.value,
            "model": config.vlm_model,
            "message": f"VLM responded: {response.content[:100]}",
        }
    except Exception as e:
        return {
            "status": "error",
            "backend": config.vlm_backend.value,
            "model": config.vlm_model,
            "message": f"Connection failed: {e}",
        }
