"""Tests for peruse_ai.perception (unit-level, no real browser)."""

from __future__ import annotations

from peruse_ai.perception import ErrorMonitor


class MockConsoleMessage:
    """Mock for Playwright console message events."""

    def __init__(self, msg_type: str, text: str):
        self.type = msg_type
        self.text = text


class MockResponse:
    """Mock for Playwright response events."""

    def __init__(self, url: str, status: int, status_text: str = ""):
        self.url = url
        self.status = status
        self.status_text = status_text


class TestErrorMonitor:
    """Test the ErrorMonitor event handlers."""

    def test_captures_console_errors(self):
        monitor = ErrorMonitor()
        monitor.on_console(MockConsoleMessage("error", "Uncaught TypeError: x is not defined"))
        assert len(monitor.console_logs) == 1
        assert "[ERROR]" in monitor.console_logs[0]

    def test_captures_console_warnings(self):
        monitor = ErrorMonitor()
        monitor.on_console(MockConsoleMessage("warning", "Deprecation warning"))
        assert len(monitor.console_logs) == 1
        assert "[WARNING]" in monitor.console_logs[0]

    def test_ignores_console_log(self):
        monitor = ErrorMonitor()
        monitor.on_console(MockConsoleMessage("log", "Some debug info"))
        assert len(monitor.console_logs) == 0

    def test_captures_page_error(self):
        monitor = ErrorMonitor()
        monitor.on_page_error("RuntimeError: Something broke")
        assert len(monitor.console_logs) == 1
        assert "[PAGE_ERROR]" in monitor.console_logs[0]

    def test_captures_network_4xx(self):
        monitor = ErrorMonitor()
        monitor.on_response(MockResponse("https://api.example.com/data", 404, "Not Found"))
        assert len(monitor.network_errors) == 1
        assert monitor.network_errors[0]["status"] == 404

    def test_captures_network_5xx(self):
        monitor = ErrorMonitor()
        monitor.on_response(MockResponse("https://api.example.com/crash", 500, "Internal Server Error"))
        assert len(monitor.network_errors) == 1
        assert monitor.network_errors[0]["status"] == 500

    def test_ignores_successful_responses(self):
        monitor = ErrorMonitor()
        monitor.on_response(MockResponse("https://example.com/ok", 200, "OK"))
        monitor.on_response(MockResponse("https://example.com/redirect", 301, "Moved"))
        assert len(monitor.network_errors) == 0

    def test_accumulates_multiple_errors(self):
        monitor = ErrorMonitor()
        monitor.on_console(MockConsoleMessage("error", "Error 1"))
        monitor.on_console(MockConsoleMessage("error", "Error 2"))
        monitor.on_response(MockResponse("https://api.example.com/a", 404, "Not Found"))
        monitor.on_response(MockResponse("https://api.example.com/b", 500, "Error"))
        assert len(monitor.console_logs) == 2
        assert len(monitor.network_errors) == 2
