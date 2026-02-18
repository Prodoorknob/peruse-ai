"""
peruse_ai.browser
~~~~~~~~~~~~~~~~~
Thin async wrapper around Playwright's browser lifecycle.
"""

from __future__ import annotations

import logging
from types import TracebackType
from typing import Optional

from playwright.async_api import Browser, BrowserContext, Page, async_playwright

from peruse_ai.config import PeruseConfig

logger = logging.getLogger(__name__)


class BrowserManager:
    """Async context manager that launches and manages a Playwright Chromium instance.

    Usage::

        async with BrowserManager(config) as manager:
            page = await manager.new_page("https://example.com")
            # ... interact with page ...
    """

    def __init__(self, config: PeruseConfig) -> None:
        self.config = config
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None

    async def __aenter__(self) -> BrowserManager:
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.config.headless,
        )
        self._context = await self._browser.new_context(
            viewport={
                "width": self.config.viewport_width,
                "height": self.config.viewport_height,
            },
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
        )
        logger.info(
            "Browser launched (headless=%s, viewport=%dx%d)",
            self.config.headless,
            self.config.viewport_width,
            self.config.viewport_height,
        )
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        logger.info("Browser closed.")

    @property
    def context(self) -> BrowserContext:
        """Get the active browser context."""
        if self._context is None:
            raise RuntimeError("BrowserManager is not initialized. Use 'async with' syntax.")
        return self._context

    async def new_page(self, url: str, wait_until: str = "domcontentloaded") -> Page:
        """Create a new page, navigate to the URL, and wait for it to load.

        Args:
            url: The URL to navigate to.
            wait_until: Playwright wait condition ('load', 'domcontentloaded', 'networkidle').

        Returns:
            The Playwright Page object.
        """
        page = await self.context.new_page()
        logger.info("Navigating to: %s (wait_until=%s)", url, wait_until)
        await page.goto(url, wait_until=wait_until, timeout=30_000)
        return page

    async def take_screenshot(self, page: Page, full_page: bool = False) -> bytes:
        """Capture a screenshot of the current page state.

        Args:
            page: The Playwright Page to screenshot.
            full_page: If True, captures the full scrollable page.

        Returns:
            Raw PNG image bytes.
        """
        return await page.screenshot(full_page=full_page, type="png")
