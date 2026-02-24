"""
peruse_ai.outputs
~~~~~~~~~~~~~~~~~
Multi-output generators that post-process an AgentResult into Markdown reports.
"""

from __future__ import annotations

import io
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from peruse_ai.vlm import (
    INSIGHT_SYSTEM_PROMPT,
    UX_REVIEW_SYSTEM_PROMPT,
    build_analysis_prompt,
    encode_image_b64,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from peruse_ai.agent import AgentResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Insights
# ---------------------------------------------------------------------------


async def generate_data_insights(result: AgentResult, vlm: BaseChatModel) -> str:
    """Generate a Markdown report summarizing visible data from the session.

    Uses the VLM to analyze screenshots and extract data insights.

    Args:
        result: The completed AgentResult.
        vlm: The VLM instance for analysis.

    Returns:
        A Markdown string with the data insights report.
    """
    if not result.screenshots:
        return "# Data Insights\n\nNo screenshots were captured during the session.\n"

    # Use a sample of unique screenshots — dedup removes near-identical frames
    screenshots = _sample_screenshots(result.screenshots, max_count=10)
    screenshots_b64 = [encode_image_b64(s) for s in screenshots]

    context = (
        f"Session summary: Explored {result.url}\n"
        f"Task: {result.task}\n"
        f"Total steps: {len(result.steps)}\n"
        f"Agent summary: {result.final_summary}\n"
    )

    messages = build_analysis_prompt(
        screenshots_b64=screenshots_b64,
        system_prompt=INSIGHT_SYSTEM_PROMPT,
        context=context,
    )

    logger.info("Generating data insights from %d screenshots...", len(screenshots))
    try:
        response = await vlm.ainvoke(messages)
        report = response.content.strip()
    except Exception as e:
        logger.error("Failed to generate data insights: %s", e)
        report = ""

    if not report:
        report = "> ⚠️ **Warning:** The VLM returned an empty response or encountered an error. This typically happens when the prompt length (number of images) exceeds the model's configured context window limit (`num_ctx`). Try increasing `vlm_num_ctx` in the configuration.\n"

    # Wrap with header
    header = (
        f"# 📊 Data Insights Report\n\n"
        f"**URL:** {result.url}  \n"
        f"**Task:** {result.task}  \n"
        f"**Generated:** {_timestamp()}  \n\n---\n\n"
    )
    return header + report


# ---------------------------------------------------------------------------
# UX/UI Review
# ---------------------------------------------------------------------------


async def generate_ux_review(result: AgentResult, vlm: BaseChatModel) -> str:
    """Generate a Markdown UX/UI critique from session screenshots.

    Args:
        result: The completed AgentResult.
        vlm: The VLM instance for analysis.

    Returns:
        A Markdown string with the UX/UI review report.
    """
    if not result.screenshots:
        return "# UX/UI Review\n\nNo screenshots were captured during the session.\n"

    screenshots = _sample_screenshots(result.screenshots, max_count=8)
    screenshots_b64 = [encode_image_b64(s) for s in screenshots]

    context = (
        f"Web application: {result.url}\n"
        f"The following screenshots were taken during automated exploration.\n"
        f"Total pages/states visited: {len(result.steps)}\n"
    )

    messages = build_analysis_prompt(
        screenshots_b64=screenshots_b64,
        system_prompt=UX_REVIEW_SYSTEM_PROMPT,
        context=context,
    )

    logger.info("Generating UX review from %d screenshots...", len(screenshots))
    try:
        response = await vlm.ainvoke(messages)
        report = response.content.strip()
    except Exception as e:
        logger.error("Failed to generate UX review: %s", e)
        report = ""

    if not report:
        report = "> ⚠️ **Warning:** The VLM returned an empty response or encountered an error. This typically happens when the prompt length (number of images) exceeds the model's configured context window limit (`num_ctx`). Try increasing `vlm_num_ctx` in the configuration.\n"

    header = (
        f"# 🎨 UX/UI Review Report\n\n"
        f"**URL:** {result.url}  \n"
        f"**Generated:** {_timestamp()}  \n\n---\n\n"
    )
    return header + report


# ---------------------------------------------------------------------------
# Bug Report
# ---------------------------------------------------------------------------


def generate_bug_report(result: AgentResult) -> str:
    """Generate a Markdown bug report from captured errors — no VLM needed.

    Args:
        result: The completed AgentResult.

    Returns:
        A Markdown string with the bug report.
    """
    console_logs = result.all_console_logs
    network_errors = result.all_network_errors

    lines = [
        f"# 🐛 Bug Report\n",
        f"**URL:** {result.url}  ",
        f"**Task:** {result.task}  ",
        f"**Steps Executed:** {len(result.steps)}  ",
        f"**Agent Completed:** {'Yes' if result.completed else 'No'}  ",
        f"**Generated:** {_timestamp()}  \n",
        "---\n",
    ]

    # Console errors
    lines.append("## Console Errors & Warnings\n")
    if console_logs:
        lines.append(f"**{len(console_logs)} issue(s) detected:**\n")
        for i, log in enumerate(console_logs, 1):
            lines.append(f"{i}. `{log}`")
        lines.append("")
    else:
        lines.append("✅ No console errors or warnings detected.\n")

    # Network errors
    lines.append("## Network Errors (4xx/5xx)\n")
    if network_errors:
        lines.append(f"**{len(network_errors)} failed request(s):**\n")
        lines.append("| # | Status | URL |")
        lines.append("|---|--------|-----|")
        for i, err in enumerate(network_errors, 1):
            lines.append(f"| {i} | `{err['status']} {err.get('status_text', '')}` | `{err['url'][:100]}` |")
        lines.append("")
    else:
        lines.append("✅ No network errors detected.\n")

    # Reproduction steps
    lines.append("## Reproduction Steps\n")
    if result.steps:
        lines.append("The agent performed the following actions:\n")
        for step in result.steps:
            action = step.parsed_action
            action_type = action.get("action", "unknown")
            desc = _describe_action(action)
            lines.append(f"{step.step_number}. **{action_type}**: {desc}")
        lines.append("")
    else:
        lines.append("No steps were recorded.\n")

    # Agent errors
    if result.error:
        lines.append("## Agent Errors\n")
        lines.append(f"⚠️ The agent encountered an error: `{result.error}`\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------


async def save_outputs(
    result: AgentResult,
    output_dir: Path,
    vlm: BaseChatModel | None = None,
    generate_insights: bool = True,
    generate_ux: bool = True,
    generate_bugs: bool = True,
) -> dict[str, Path]:
    """Generate and save all reports + screenshots to the output directory.

    Args:
        result: The completed AgentResult.
        output_dir: Directory to write files to (created if it doesn't exist).
        vlm: VLM instance (required for insights and UX reports).
        generate_insights: Whether to generate the data insights report.
        generate_ux: Whether to generate the UX review report.
        generate_bugs: Whether to generate the bug report.

    Returns:
        A dict mapping report type to file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, Path] = {}

    # Shared filename components
    model_name = vlm.__class__.__name__ if vlm else "no_vlm"
    ts = _timestamp_slug()

    # Save screenshots
    screenshots_dir = output_dir / "screenshots"
    screenshots_dir.mkdir(exist_ok=True)
    for i, step in enumerate(result.steps):
        img_path = screenshots_dir / f"step_{i + 1:03d}.png"
        img_path.write_bytes(step.perception.screenshot)
    logger.info("Saved %d screenshots to %s", len(result.steps), screenshots_dir)

    # Bug report (no VLM needed)
    if generate_bugs:
        bug_report = generate_bug_report(result)
        bug_path = output_dir / f"bug_report_{model_name}_{ts}.md"
        bug_path.write_text(bug_report, encoding="utf-8")
        saved["bugs"] = bug_path
        logger.info("Bug report saved to %s", bug_path)

    # VLM-powered reports
    if vlm:
        if generate_insights:
            insights = await generate_data_insights(result, vlm)
            insights_path = output_dir / f"data_insights_{model_name}_{ts}.md"
            insights_path.write_text(insights, encoding="utf-8")
            saved["insights"] = insights_path
            logger.info("Data insights saved to %s", insights_path)

        if generate_ux:
            ux_review = await generate_ux_review(result, vlm)
            ux_path = output_dir / f"ux_review_{model_name}_{ts}.md"
            ux_path.write_text(ux_review, encoding="utf-8")
            saved["ux"] = ux_path
            logger.info("UX review saved to %s", ux_path)

    return saved


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_screenshots(screenshots: list[bytes], max_count: int = 5) -> list[bytes]:
    """Deduplicate near-identical screenshots, then sample evenly.

    Uses perceptual hashing to drop consecutive frames that look the same
    (e.g. from scroll-only or stuck-in-loop steps). Then samples evenly
    from the unique set.
    """
    if not screenshots:
        return []

    # --- Step 1: Deduplicate consecutive near-identical screenshots ---
    unique: list[bytes] = [screenshots[0]]
    prev_hash = _image_hash(screenshots[0])
    for shot in screenshots[1:]:
        h = _image_hash(shot)
        if h != prev_hash:
            unique.append(shot)
            prev_hash = h

    logger.info(
        "Screenshot dedup: %d total → %d unique", len(screenshots), len(unique)
    )

    # --- Step 2: Sample evenly from the unique set ---
    if len(unique) <= max_count:
        return unique

    step = len(unique) / max_count
    indices = [int(i * step) for i in range(max_count)]
    # Always include the last screenshot
    if indices[-1] != len(unique) - 1:
        indices[-1] = len(unique) - 1
    return [unique[i] for i in indices]


def _image_hash(image_bytes: bytes, size: int = 8) -> int:
    """Compute a simple average-hash for a screenshot (for dedup, not crypto).

    Resizes the image to ``size x size`` grayscale and compares each pixel to
    the mean to produce a 64-bit hash. Identical or near-identical images
    produce the same hash.
    """
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes)).convert("L").resize(
            (size, size), Image.LANCZOS
        )
        pixels = list(img.getdata())
        avg = sum(pixels) / len(pixels)
        return sum(1 << i for i, px in enumerate(pixels) if px >= avg)
    except Exception:
        # If hashing fails, return id so nothing gets deduped
        return id(image_bytes)


def _describe_action(action: dict) -> str:
    """Human-readable description of an action."""
    action_type = action.get("action", "")
    if action_type == "click":
        return f"Clicked element [{action.get('element_id', '?')}]"
    elif action_type == "type":
        return f"Typed '{action.get('text', '')[:40]}' into element [{action.get('element_id', '?')}]"
    elif action_type == "scroll":
        return f"Scrolled {action.get('direction', 'down')}"
    elif action_type == "navigate":
        return f"Navigated to {action.get('url', '')[:80]}"
    elif action_type == "wait":
        return f"Waited {action.get('seconds', '?')} seconds"
    elif action_type == "done":
        return action.get("summary", "Task completed")[:120]
    return str(action)


def _timestamp() -> str:
    """ISO timestamp string for report headers."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _timestamp_slug() -> str:
    """Filesystem-safe timestamp for filenames (no colons or spaces)."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
