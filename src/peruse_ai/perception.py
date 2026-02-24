"""
peruse_ai.perception
~~~~~~~~~~~~~~~~~~~~
Dual-channel perception: DOM extraction + visual screenshot capture + error monitoring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from playwright.async_api import Page, Response

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DOM extraction JS — strips styling, returns only interactive elements
# ---------------------------------------------------------------------------

EXTRACT_DOM_JS = """
(maxElements) => {
    const INTERACTIVE_SELECTORS = [
        'a[href]', 'button', 'input', 'textarea', 'select',
        '[role="button"]', '[role="link"]', '[role="tab"]',
        '[role="menuitem"]', '[role="checkbox"]', '[role="radio"]',
        '[role="option"]', '[role="combobox"]', '[role="listbox"]',
        '[onclick]', '[tabindex]', 'summary',
    ];

    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const elements = [];
    let id = 0;

    for (const selector of INTERACTIVE_SELECTORS) {
        for (const el of document.querySelectorAll(selector)) {
            // Skip hidden elements
            const style = window.getComputedStyle(el);
            if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') {
                continue;
            }

            const rect = el.getBoundingClientRect();
            if (rect.width === 0 || rect.height === 0) continue;

            // Only include elements at least partially within the viewport
            if (rect.bottom < 0 || rect.top > vh || rect.right < 0 || rect.left > vw) {
                continue;
            }

            const tag = el.tagName.toLowerCase();
            const text = (el.textContent || '').trim().slice(0, 80);
            const type = el.getAttribute('type') || '';
            const placeholder = el.getAttribute('placeholder') || '';
            const ariaLabel = el.getAttribute('aria-label') || '';
            const href = el.getAttribute('href') || '';
            const value = el.value || '';

            // Dedup: skip if we already have this element
            const signature = `${tag}-${rect.x}-${rect.y}`;
            if (elements.some(e => e._sig === signature)) continue;

            const entry = {
                id: id++,
                tag,
                text: text || placeholder || ariaLabel || `[${tag}]`,
                type,
                href: href ? href.slice(0, 120) : undefined,
                value: value ? value.slice(0, 60) : undefined,
                rect: {
                    x: Math.round(rect.x),
                    y: Math.round(rect.y),
                    w: Math.round(rect.width),
                    h: Math.round(rect.height),
                },
                _sig: signature,
            };

            // For <select> elements, include available options so the VLM
            // knows what values can be selected
            if (tag === 'select') {
                const opts = [];
                for (const opt of el.options) {
                    opts.push(opt.textContent.trim().slice(0, 40));
                }
                entry.options = opts.slice(0, 20); // cap at 20 options
            }

            elements.push(entry);

            // Stop if we hit the cap
            if (maxElements > 0 && elements.length >= maxElements) break;
        }
        if (maxElements > 0 && elements.length >= maxElements) break;
    }

    // Remove internal _sig field
    return elements.map(({ _sig, ...rest }) => rest);
}
"""

# JS to get basic page metadata
PAGE_META_JS = """
() => ({
    title: document.title,
    url: window.location.href,
    scrollHeight: document.documentElement.scrollHeight,
    clientHeight: document.documentElement.clientHeight,
    scrollTop: window.scrollY,
})
"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PagePerception:
    """Complete perception snapshot of a page at a point in time."""

    screenshot: bytes
    dom_elements: list[dict]
    dom_text: str
    page_meta: dict
    console_logs: list[str] = field(default_factory=list)
    network_errors: list[dict] = field(default_factory=list)


@dataclass
class ErrorMonitor:
    """Accumulates console logs and network errors from a page session."""

    console_logs: list[str] = field(default_factory=list)
    network_errors: list[dict] = field(default_factory=list)

    def on_console(self, msg) -> None:
        """Handler for page console events."""
        level = msg.type  # 'log', 'warning', 'error', etc.
        text = msg.text
        if level in ("error", "warning"):
            entry = f"[{level.upper()}] {text}"
            self.console_logs.append(entry)
            logger.debug("Console %s: %s", level, text)

    def on_page_error(self, error) -> None:
        """Handler for uncaught page exceptions."""
        entry = f"[PAGE_ERROR] {error}"
        self.console_logs.append(entry)
        logger.warning("Page error: %s", error)

    def on_response(self, response: Response) -> None:
        """Handler for network responses — captures 4xx/5xx errors."""
        if response.status >= 400:
            entry = {
                "url": response.url[:200],
                "status": response.status,
                "status_text": response.status_text,
            }
            self.network_errors.append(entry)
            logger.debug("Network error: %s %s", response.status, response.url[:80])


# ---------------------------------------------------------------------------
# Core perception functions
# ---------------------------------------------------------------------------


def attach_error_monitor(page: Page) -> ErrorMonitor:
    """Attach console and network error listeners to a page.

    Call this right after creating a page, before navigation.

    Args:
        page: The Playwright Page to monitor.

    Returns:
        An ErrorMonitor instance that will accumulate errors.
    """
    monitor = ErrorMonitor()
    page.on("console", monitor.on_console)
    page.on("pageerror", monitor.on_page_error)
    page.on("response", monitor.on_response)
    logger.info("Error monitor attached to page.")
    return monitor


async def capture_screenshot(page: Page, full_page: bool = False) -> bytes:
    """Take a PNG screenshot of the current viewport.

    Args:
        page: The Playwright Page.
        full_page: If True, capture the full scrollable page.

    Returns:
        Raw PNG bytes.
    """
    return await page.screenshot(full_page=full_page, type="png")


async def extract_dom(page: Page, max_elements: int = 100) -> tuple[list[dict], str]:
    """Extract interactive DOM elements and return them as structured data + text summary.

    Only elements visible within the current viewport are included, capped at max_elements.

    Args:
        page: The Playwright Page.
        max_elements: Maximum number of DOM elements to extract (0 = unlimited).

    Returns:
        A tuple of (raw element list, formatted text representation).
    """
    elements = await page.evaluate(EXTRACT_DOM_JS, max_elements)

    # Build a human-readable text summary for the VLM
    lines = []
    for el in elements:
        parts = [f"[{el['id']}]", f"<{el['tag']}>"]
        if el.get("type"):
            parts.append(f"type={el['type']}")
        parts.append(f'"{el["text"]}"')
        if el.get("href"):
            parts.append(f"href={el['href']}")
        if el.get("value"):
            parts.append(f"value={el['value']}")
        if el.get("options"):
            opts_str = ", ".join(el["options"][:10])
            parts.append(f"options=[{opts_str}]")
        pos = el.get("rect", {})
        parts.append(f"@({pos.get('x', 0)},{pos.get('y', 0)} {pos.get('w', 0)}x{pos.get('h', 0)})")
        lines.append(" ".join(parts))

    dom_text = "\n".join(lines)
    logger.info("Extracted %d interactive DOM elements.", len(elements))
    return elements, dom_text


async def get_page_meta(page: Page) -> dict:
    """Get basic page metadata (title, URL, scroll position).

    Args:
        page: The Playwright Page.

    Returns:
        A dict with title, url, scrollHeight, clientHeight, scrollTop.
    """
    return await page.evaluate(PAGE_META_JS)


async def perceive(page: Page, monitor: ErrorMonitor, full_page: bool = False, max_dom_elements: int = 100) -> PagePerception:
    """Perform a full perception pass: screenshot + DOM + metadata + errors.

    Args:
        page: The Playwright Page.
        monitor: The ErrorMonitor attached to this page.
        full_page: If True, capture full scrollable screenshot.
        max_dom_elements: Maximum number of DOM elements to extract (0 = unlimited).

    Returns:
        A PagePerception dataclass with all captured data.
    """
    screenshot = await capture_screenshot(page, full_page=full_page)
    dom_elements, dom_text = await extract_dom(page, max_elements=max_dom_elements)
    page_meta = await get_page_meta(page)

    return PagePerception(
        screenshot=screenshot,
        dom_elements=dom_elements,
        dom_text=dom_text,
        page_meta=page_meta,
        console_logs=list(monitor.console_logs),
        network_errors=list(monitor.network_errors),
    )
