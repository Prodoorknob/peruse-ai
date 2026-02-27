"""
peruse_ai.agent
~~~~~~~~~~~~~~~
Main agent loop: perceive → plan → act, orchestrating browser, VLM, and perception.
"""

from __future__ import annotations

import asyncio
import json
import io
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from peruse_ai.browser import BrowserManager
from peruse_ai.config import PeruseConfig
from peruse_ai.perception import ErrorMonitor, PagePerception, attach_error_monitor, perceive
from peruse_ai.vlm import build_vision_prompt, create_vlm, encode_image_b64

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AgentStep:
    """Record of a single perceive-plan-act iteration."""

    step_number: int
    perception: PagePerception
    vlm_response_raw: str
    parsed_action: dict
    thought: str = ""
    error: str | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentResult:
    """Complete result of an agent run."""

    url: str
    task: str
    steps: list[AgentStep] = field(default_factory=list)
    final_summary: str = ""
    total_time_seconds: float = 0.0
    completed: bool = False
    error: str | None = None

    @property
    def screenshots(self) -> list[bytes]:
        """All screenshots captured during the run."""
        return [step.perception.screenshot for step in self.steps]

    @property
    def all_console_logs(self) -> list[str]:
        """All console logs accumulated across the session."""
        logs = []
        seen = set()
        for step in self.steps:
            for log in step.perception.console_logs:
                if log not in seen:
                    logs.append(log)
                    seen.add(log)
        return logs

    @property
    def all_network_errors(self) -> list[dict]:
        """All network errors accumulated across the session."""
        errors = []
        seen = set()
        for step in self.steps:
            for err in step.perception.network_errors:
                key = f"{err['status']}-{err['url']}"
                if key not in seen:
                    errors.append(err)
                    seen.add(key)
        return errors


# ---------------------------------------------------------------------------
# Action executor
# ---------------------------------------------------------------------------


async def execute_action(page, action: dict, dom_elements: list[dict]) -> None:
    """Execute a parsed VLM action on the Playwright page.

    Args:
        page: The Playwright Page.
        action: Parsed action dict from VLM response.
        dom_elements: Current DOM elements from perception (for element_id lookup).
    """
    action_type = action.get("action", "")

    if action_type == "click":
        element_id = action.get("element_id", -1)
        target = _find_element(dom_elements, element_id)
        if target:
            rect = target["rect"]
            cx = rect["x"] + rect["w"] // 2
            cy = rect["y"] + rect["h"] // 2
            logger.info("Clicking element [%d] at (%d, %d): %s", element_id, cx, cy, target["text"][:40])
            await page.mouse.click(cx, cy)
            await page.wait_for_timeout(1000)
        else:
            logger.warning("Element [%d] not found in DOM.", element_id)

    elif action_type == "type":
        element_id = action.get("element_id", -1)
        text = action.get("text", "")
        target = _find_element(dom_elements, element_id)
        if target:
            rect = target["rect"]
            cx = rect["x"] + rect["w"] // 2
            cy = rect["y"] + rect["h"] // 2
            logger.info("Typing into element [%d]: '%s'", element_id, text[:40])
            await page.mouse.click(cx, cy)
            await page.wait_for_timeout(300)
            await page.keyboard.type(text, delay=50)
            await page.wait_for_timeout(500)
        else:
            logger.warning("Element [%d] not found for typing.", element_id)

    elif action_type == "scroll":
        direction = action.get("direction", "down")
        delta = -400 if direction == "up" else 400
        logger.info("Scrolling %s", direction)
        await page.mouse.wheel(0, delta)
        await page.wait_for_timeout(800)

    elif action_type == "select":
        element_id = action.get("element_id", -1)
        value = action.get("value", "")
        target = _find_element(dom_elements, element_id)
        if target and target.get("tag") == "select" and value:
            rect = target["rect"]
            logger.info("Selecting '%s' in element [%d]", value, element_id)
            try:
                # Find the <select> at the matching position and use select_option
                selects = await page.query_selector_all("select")
                for sel in selects:
                    box = await sel.bounding_box()
                    if box and abs(box["x"] - rect["x"]) < 5 and abs(box["y"] - rect["y"]) < 5:
                        selected = False
                        # Strategy 1: Match by label (visible text)
                        try:
                            await sel.select_option(label=value, timeout=3000)
                            selected = True
                        except Exception:
                            pass
                        # Strategy 2: Match by value attribute
                        if not selected:
                            try:
                                await sel.select_option(value=value, timeout=3000)
                                selected = True
                            except Exception:
                                pass
                        # Strategy 3: Partial/substring match against available options
                        if not selected:
                            try:
                                options = await sel.query_selector_all("option")
                                for opt in options:
                                    opt_text = (await opt.inner_text()).strip()
                                    opt_value = await opt.get_attribute("value") or ""
                                    if (value.lower() in opt_text.lower()
                                            or value.lower() in opt_value.lower()
                                            or opt_text.lower() in value.lower()
                                            or opt_value.lower() in value.lower()):
                                        await sel.select_option(value=opt_value, timeout=3000)
                                        selected = True
                                        break
                            except Exception:
                                pass
                        if selected:
                            logger.info("Selected option '%s' successfully", value)
                        else:
                            logger.warning("No matching option found for '%s' in element [%d]", value, element_id)
                        break
                else:
                    logger.warning("Could not locate <select> element [%d] on page.", element_id)
            except Exception as e:
                logger.warning("Failed to select option '%s': %s", value, e)
            await page.wait_for_timeout(1000)
        else:
            logger.warning("Element [%d] is not a <select> or no value provided.", element_id)

    elif action_type == "navigate":
        url = action.get("url", "")
        if url:
            logger.info("Navigating to: %s", url)
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            await page.wait_for_timeout(1000)

    elif action_type == "wait":
        seconds = min(action.get("seconds", 2), 10)  # Cap at 10 seconds
        logger.info("Waiting %d seconds", seconds)
        await page.wait_for_timeout(seconds * 1000)

    elif action_type == "done":
        logger.info("Agent signals done: %s", action.get("summary", "")[:100])

    else:
        logger.warning("Unknown action type: %s", action_type)


def _find_element(dom_elements: list[dict], element_id: int) -> Optional[dict]:
    """Find a DOM element by its indexed ID."""
    for el in dom_elements:
        if el.get("id") == element_id:
            return el
    return None


def _action_signature(action: dict) -> str:
    """Create a comparable signature from an action dict, ignoring 'thought'."""
    key_parts = [action.get("action", "")]
    if "element_id" in action:
        key_parts.append(f"el={action['element_id']}")
    if "text" in action:
        key_parts.append(f"text={action['text']}")
    if "value" in action:
        key_parts.append(f"val={action['value']}")
    if "direction" in action:
        key_parts.append(f"dir={action['direction']}")
    if "url" in action:
        key_parts.append(f"url={action['url']}")
    return "|".join(key_parts)


def _compress_screenshot(png_bytes: bytes, quality: int = 60) -> bytes:
    """Convert PNG screenshot to JPEG to reduce token usage for the VLM.

    Args:
        png_bytes: Raw PNG image bytes.
        quality: JPEG quality (1-100). Lower = smaller but lossier.

    Returns:
        JPEG image bytes.
    """
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        # Downscale to 640px wide if larger — halves tokens for the VLM
        if img.width > 640:
            ratio = 640 / img.width
            img = img.resize((640, int(img.height * ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()
    except ImportError:
        logger.debug("Pillow not installed, sending raw PNG to VLM")
        return png_bytes


# ---------------------------------------------------------------------------
# VLM response parser
# ---------------------------------------------------------------------------


def parse_vlm_response(raw_response: str) -> dict:
    """Parse the VLM's JSON response into an action dict.

    Handles common issues like markdown code fences, trailing text, mixed content,
    and natural language responses from local VLMs.

    Args:
        raw_response: The raw string content from the VLM.

    Returns:
        A parsed action dict. Falls back to scroll (not done) on parse failure
        so the agent keeps exploring instead of prematurely stopping.
    """
    text = raw_response.strip()

    # --- Strategy 1: Strip markdown code fences ---
    import re

    # Remove ```json ... ``` or ``` ... ``` blocks
    fenced = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    # --- Strategy 2: Direct JSON parse ---
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "action" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass

    # --- Strategy 3: Find JSON object with brace matching ---
    json_candidates = _extract_json_objects(text)
    for candidate in json_candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and "action" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue

    # --- Strategy 4: Simple brace extraction (first { to last }) ---
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text[start:end])
            if isinstance(parsed, dict):
                # Even if missing "action", try to recover
                if "action" not in parsed:
                    parsed.setdefault("action", "scroll")
                    parsed.setdefault("direction", "down")
                return parsed
        except json.JSONDecodeError:
            pass

    # --- Strategy 5: Keyword-based fallback from natural language ---
    text_lower = text.lower()

    # Detect if VLM is describing a select/dropdown action in natural language
    select_match = re.search(
        r"(?:select|choose|pick|change).*?(?:option|dropdown|filter|value).*?\[?(\d+)\]?", text_lower
    )
    if select_match:
        element_id = int(select_match.group(1))
        # Try to extract the value to select
        value_match = re.search(r"[\"']([^\"']+)[\"']", text)
        value = value_match.group(1) if value_match else ""
        if value:
            logger.info("Recovered select action from natural language: element_id=%d, value='%s'", element_id, value)
            return {"action": "select", "element_id": element_id, "value": value, "thought": f"(recovered from text) {text[:100]}"}

    # Detect if VLM is describing a click action in natural language
    click_match = re.search(r"(?:click|press|tap).*?(?:element|button|link)?\s*\[?(\d+)\]?", text_lower)
    if click_match:
        element_id = int(click_match.group(1))
        logger.info("Recovered click action from natural language: element_id=%d", element_id)
        return {"action": "click", "element_id": element_id, "thought": f"(recovered from text) {text[:100]}"}

    # Detect scroll intent
    if any(word in text_lower for word in ["scroll down", "scroll page", "see more", "below"]):
        logger.info("Recovered scroll action from natural language")
        return {"action": "scroll", "direction": "down", "thought": f"(recovered from text) {text[:100]}"}

    if any(word in text_lower for word in ["scroll up", "back to top", "above"]):
        logger.info("Recovered scroll-up action from natural language")
        return {"action": "scroll", "direction": "up", "thought": f"(recovered from text) {text[:100]}"}

    # Detect done/complete intent
    if any(word in text_lower for word in ["task is complete", "task is done", "finished", "nothing more"]):
        logger.info("Recovered done action from natural language")
        return {"action": "done", "summary": text[:200], "thought": text[:100]}

    # --- Fallback: Scroll down to keep exploring (NOT done!) ---
    logger.warning("Could not parse VLM response, falling back to scroll. Raw: %s", text[:300])
    return {
        "action": "scroll",
        "direction": "down",
        "thought": f"(parse failed, auto-scrolling to continue exploration) {text[:100]}",
        "_parse_failed": True,
    }


def _extract_json_objects(text: str) -> list[str]:
    """Extract potential JSON objects from text using brace matching."""
    objects = []
    i = 0
    while i < len(text):
        if text[i] == "{":
            depth = 0
            start = i
            in_string = False
            escape_next = False
            for j in range(i, len(text)):
                char = text[j]
                if escape_next:
                    escape_next = False
                    continue
                if char == "\\":
                    escape_next = True
                    continue
                if char == '"' and not escape_next:
                    in_string = not in_string
                if not in_string:
                    if char == "{":
                        depth += 1
                    elif char == "}":
                        depth -= 1
                        if depth == 0:
                            objects.append(text[start : j + 1])
                            i = j
                            break
        i += 1
    return objects


# ---------------------------------------------------------------------------
# Main Agent
# ---------------------------------------------------------------------------


class PeruseAgent:
    """The main Peruse-AI agent that orchestrates autonomous web exploration.

    Args:
        config: PeruseConfig instance with all settings.
        url: The starting URL to explore.
        task: High-level natural language task/goal.
    """

    def __init__(
        self,
        config: PeruseConfig | None = None,
        url: str = "",
        task: str = "",
    ) -> None:
        self.config = config or PeruseConfig()
        self.url = url
        self.task = task
        self._vlm = None

    async def run(self) -> AgentResult:
        """Execute the full agent loop: perceive → plan → act until done or max_steps.

        Returns:
            An AgentResult with the complete session history.
        """
        start_time = time.time()
        result = AgentResult(url=self.url, task=self.task)

        logger.info("Starting Peruse-AI agent | URL: %s | Task: %s", self.url, self.task)

        # Initialize VLM
        try:
            self._vlm = create_vlm(self.config, json_mode=True)
        except Exception as e:
            result.error = f"Failed to initialize VLM: {e}"
            logger.error(result.error)
            return result

        # Run the agent loop
        async with BrowserManager(self.config) as browser:
            page = await browser.new_page(self.url)
            monitor = attach_error_monitor(page)

            step_history: list[dict] = []
            consecutive_parse_failures = 0
            max_parse_failures = 5  # Threshold before issuing a parse-failure nudge
            recent_actions: list[str] = []  # Track action signatures for loop detection
            max_repeated_actions = 7  # Threshold for identical consecutive actions
            low_variety_window = 12  # Window size for low-variety detection
            low_variety_threshold = 2  # Max unique actions before considering it a loop
            nudge_count = 0  # Total nudges sent (shared between loop and parse nudges)
            pending_nudge: str | None = None  # Nudge to inject in the next step
            avoided_elements: set[int] = set()  # Elements flagged in nudges

            for step_num in range(1, self.config.max_steps + 1):
                logger.info("=== Step %d / %d ===", step_num, self.config.max_steps)

                try:
                    step = await self._execute_step(
                        page=page,
                        monitor=monitor,
                        step_num=step_num,
                        step_history=step_history,
                        nudge=pending_nudge,
                        avoided_elements=avoided_elements if avoided_elements else None,
                    )
                    pending_nudge = None  # Consume the nudge after it's been sent
                    result.steps.append(step)

                    # Track parse failures
                    if step.parsed_action.get("_parse_failed"):
                        consecutive_parse_failures += 1
                        logger.warning(
                            "Parse failure %d/%d",
                            consecutive_parse_failures,
                            max_parse_failures,
                        )
                        if consecutive_parse_failures >= max_parse_failures:
                            nudge_count += 1
                            if nudge_count > self.config.max_nudges:
                                result.error = (
                                    f"Stopped after {max_parse_failures} consecutive VLM parse failures "
                                    f"and {nudge_count - 1} nudge(s). "
                                    "The model may not support structured JSON output well."
                                )
                                logger.error(result.error)
                                break
                            else:
                                logger.info(
                                    "Parse failure threshold reached (nudge %d/%d). Nudging agent for proper JSON.",
                                    nudge_count, self.config.max_nudges,
                                )
                                pending_nudge = (
                                    "Your recent responses could not be parsed as valid JSON. "
                                    "Remember: your ENTIRE response must be a single JSON object with no text before "
                                    'or after it. Example: {"thought": "...", "action": "scroll", "direction": "down"}'
                                )
                                consecutive_parse_failures = 0  # Reset counter after nudge
                    else:
                        consecutive_parse_failures = 0  # Reset on successful parse

                    # --- Loop detection with nudge ---
                    action_sig = _action_signature(step.parsed_action)
                    recent_actions.append(action_sig)

                    loop_detected = False
                    stuck_sigs: set[str] = set()

                    # Check 1: Identical consecutive actions (7 in a row)
                    if len(recent_actions) >= max_repeated_actions:
                        tail = recent_actions[-max_repeated_actions:]
                        if len(set(tail)) == 1:
                            loop_detected = True
                            stuck_sigs = set(tail)

                    # Check 2: Low variety — oscillating between 2 actions over 12 steps
                    if not loop_detected and len(recent_actions) >= low_variety_window:
                        window = recent_actions[-low_variety_window:]
                        unique_in_window = set(window)
                        if len(unique_in_window) <= low_variety_threshold:
                            loop_detected = True
                            stuck_sigs = unique_in_window

                    if loop_detected:
                        nudge_count += 1
                        # Extract element IDs from stuck actions for targeted avoidance
                        for sig in stuck_sigs:
                            for part in sig.split("|"):
                                if part.startswith("el="):
                                    try:
                                        avoided_elements.add(int(part[3:]))
                                    except ValueError:
                                        pass

                        if nudge_count > self.config.max_nudges:
                            logger.warning(
                                "Loop detected after %d nudges: stuck actions %s. Stopping.",
                                nudge_count - 1, stuck_sigs,
                            )
                            result.completed = True
                            result.final_summary = (
                                f"Agent stopped: stuck in a loop with actions {stuck_sigs} "
                                f"despite {nudge_count - 1} nudge(s)."
                            )
                            break
                        else:
                            avoid_str = ", ".join(f"[{eid}]" for eid in sorted(avoided_elements))
                            logger.info(
                                "Loop detected (nudge %d/%d): stuck actions %s. Nudging agent. Avoid elements: %s",
                                nudge_count, self.config.max_nudges, stuck_sigs, avoid_str,
                            )
                            pending_nudge = (
                                "You have been repeating the same actions without making progress. "
                                f"STOP interacting with element(s) {avoid_str} — you have already tried them. "
                                "Instead, try one of these approaches:\n"
                                "- Click a DIFFERENT navigation link or tab you have NOT visited yet\n"
                                "- Scroll down to find new content below the fold\n"
                                "- Navigate to a completely different page\n"
                                "- If you have explored enough, use the 'done' action to finish\n"
                                "Do NOT repeat any action you have already tried."
                            )
                            recent_actions.clear()  # Reset to give agent a fresh window

                    # Track history for VLM context
                    step_history.append({
                        "thought": step.thought,
                        "action": json.dumps(step.parsed_action),
                    })

                    # Check if agent is done
                    if step.parsed_action.get("action") == "done":
                        result.final_summary = step.parsed_action.get("summary", "")
                        result.completed = True
                        logger.info("Agent completed task: %s", result.final_summary[:100])
                        break

                except Exception as e:
                    logger.error("Error in step %d: %s", step_num, e, exc_info=True)
                    result.error = f"Step {step_num} failed: {e}"
                    break

        result.total_time_seconds = time.time() - start_time
        logger.info(
            "Agent finished | Steps: %d | Time: %.1fs | Completed: %s",
            len(result.steps),
            result.total_time_seconds,
            result.completed,
        )
        return result

    async def _execute_step(
        self,
        page: Any,
        monitor: ErrorMonitor,
        step_num: int,
        step_history: list[dict],
        nudge: str | None = None,
        avoided_elements: set[int] | None = None,
    ) -> AgentStep:
        """Execute a single perceive → plan → act cycle.

        Args:
            page: The Playwright Page.
            monitor: The ErrorMonitor instance.
            step_num: Current step number.
            step_history: History of previous steps for VLM context.
            nudge: Optional nudge message injected when the agent appears stuck.
            avoided_elements: Element IDs the agent should not interact with.
                If the VLM targets one of these, the action is replaced with a scroll.

        Returns:
            An AgentStep record.
        """
        # 1. PERCEIVE
        perception = await perceive(page, monitor, max_dom_elements=self.config.max_dom_elements)

        # 2. PLAN — send perception to VLM
        screenshot_b64 = encode_image_b64(
            _compress_screenshot(perception.screenshot, quality=self.config.screenshot_quality)
        )
        messages = build_vision_prompt(
            screenshot_b64=screenshot_b64,
            dom_text=perception.dom_text,
            task=self.task,
            step_history=step_history,
            page_meta=perception.page_meta,
            persona=self.config.persona,
            extra_instructions=self.config.extra_instructions,
            nudge=nudge,
        )

        logger.info("Sending perception to VLM (DOM elements: %d)...", len(perception.dom_elements))
        response = await self._vlm_invoke_with_retry(messages)
        raw_response = response.content
        logger.debug("VLM response: %s", raw_response[:300])

        # 3. PARSE response
        parsed = parse_vlm_response(raw_response)
        thought = parsed.get("thought", "")

        # 4. ENFORCE — block actions on avoided elements
        if avoided_elements:
            target_id = parsed.get("element_id", -1)
            if target_id in avoided_elements and parsed.get("action") not in ("done", "scroll", "wait", "navigate"):
                logger.warning(
                    "Blocked action '%s' on avoided element [%d] — substituting scroll.",
                    parsed.get("action"), target_id,
                )
                parsed = {
                    "action": "scroll",
                    "direction": "down",
                    "thought": f"(blocked: element [{target_id}] is avoided) {thought[:100]}",
                    "_blocked": True,
                }

        # 5. ACT
        await execute_action(page, parsed, perception.dom_elements)

        return AgentStep(
            step_number=step_num,
            perception=perception,
            vlm_response_raw=raw_response,
            parsed_action=parsed,
            thought=thought,
        )

    async def _vlm_invoke_with_retry(self, messages: list) -> Any:
        """Invoke the VLM with automatic retry and cooldown on crash.

        Handles transient failures from unstable GPU backends (e.g. IPEX-LLM
        Vulkan on Intel ARC) by retrying with exponential backoff.
        """
        last_error = None
        for attempt in range(1 + self.config.vlm_retries):
            try:
                return await self._vlm.ainvoke(messages)
            except Exception as e:
                last_error = e
                if attempt < self.config.vlm_retries:
                    cooldown = self.config.vlm_cooldown * (attempt + 1)
                    logger.warning(
                        "VLM call failed (attempt %d/%d): %s. "
                        "Retrying in %.1fs...",
                        attempt + 1,
                        1 + self.config.vlm_retries,
                        str(e)[:200],
                        cooldown,
                    )
                    await asyncio.sleep(cooldown)
                else:
                    logger.error(
                        "VLM call failed after %d attempts: %s",
                        1 + self.config.vlm_retries,
                        str(e)[:200],
                    )
        raise last_error
