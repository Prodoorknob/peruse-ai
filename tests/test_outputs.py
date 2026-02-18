"""Tests for peruse_ai.outputs (bug report generation and file saving)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from peruse_ai.agent import AgentResult, AgentStep
from peruse_ai.outputs import generate_bug_report, save_outputs
from peruse_ai.perception import PagePerception


def _make_perception(**kwargs) -> PagePerception:
    """Helper to create a minimal PagePerception."""
    defaults = {
        "screenshot": b"\x89PNG\r\n\x1a\n" + b"\x00" * 100,  # Minimal PNG-like bytes
        "dom_elements": [],
        "dom_text": "",
        "page_meta": {"title": "Test", "url": "https://example.com"},
        "console_logs": [],
        "network_errors": [],
    }
    defaults.update(kwargs)
    return PagePerception(**defaults)


def _make_step(step_number: int, action: dict, **kwargs) -> AgentStep:
    """Helper to create a minimal AgentStep."""
    return AgentStep(
        step_number=step_number,
        perception=kwargs.get("perception", _make_perception(**kwargs)),
        vlm_response_raw="{}",
        parsed_action=action,
        thought=action.get("thought", ""),
    )


def _make_result(**kwargs) -> AgentResult:
    """Helper to create a minimal AgentResult."""
    defaults = {
        "url": "https://example.com",
        "task": "Test task",
        "steps": [],
        "completed": True,
    }
    defaults.update(kwargs)
    return AgentResult(**defaults)


class TestBugReport:
    """Test bug report generation."""

    def test_empty_session(self):
        result = _make_result()
        report = generate_bug_report(result)
        assert "# 🐛 Bug Report" in report
        assert "No console errors" in report
        assert "No network errors" in report

    def test_with_console_errors(self):
        step = _make_step(
            1,
            {"action": "click", "element_id": 0},
            console_logs=["[ERROR] Uncaught TypeError"],
        )
        result = _make_result(steps=[step])
        report = generate_bug_report(result)
        assert "Uncaught TypeError" in report
        assert "1 issue(s)" in report

    def test_with_network_errors(self):
        step = _make_step(
            1,
            {"action": "click", "element_id": 0},
            network_errors=[{"url": "https://api.example.com/data", "status": 500, "status_text": "Error"}],
        )
        result = _make_result(steps=[step])
        report = generate_bug_report(result)
        assert "500" in report
        assert "api.example.com" in report

    def test_with_reproduction_steps(self):
        steps = [
            _make_step(1, {"action": "click", "element_id": 3}),
            _make_step(2, {"action": "type", "element_id": 5, "text": "hello"}),
            _make_step(3, {"action": "done", "summary": "Finished"}),
        ]
        result = _make_result(steps=steps)
        report = generate_bug_report(result)
        assert "Clicked element [3]" in report
        assert "hello" in report
        assert "Finished" in report

    def test_agent_error_included(self):
        result = _make_result(error="VLM timeout after 120s", completed=False)
        report = generate_bug_report(result)
        assert "VLM timeout" in report
        assert "Agent Errors" in report


class TestSaveOutputs:
    """Test file saving logic."""

    @pytest.mark.asyncio
    async def test_creates_output_directory(self, tmp_path):
        output_dir = tmp_path / "reports"
        result = _make_result(
            steps=[_make_step(1, {"action": "done", "summary": "Done"})]
        )
        saved = await save_outputs(
            result, output_dir, vlm=None,
            generate_insights=False, generate_ux=False, generate_bugs=True,
        )
        assert output_dir.exists()
        assert "bugs" in saved
        assert saved["bugs"].exists()

    @pytest.mark.asyncio
    async def test_saves_screenshots(self, tmp_path):
        output_dir = tmp_path / "reports"
        result = _make_result(
            steps=[
                _make_step(1, {"action": "click", "element_id": 0}),
                _make_step(2, {"action": "done", "summary": "Done"}),
            ]
        )
        await save_outputs(
            result, output_dir, vlm=None,
            generate_insights=False, generate_ux=False, generate_bugs=True,
        )
        screenshots_dir = output_dir / "screenshots"
        assert screenshots_dir.exists()
        assert len(list(screenshots_dir.glob("*.png"))) == 2

    @pytest.mark.asyncio
    async def test_bug_report_content(self, tmp_path):
        output_dir = tmp_path / "reports"
        step = _make_step(
            1,
            {"action": "click", "element_id": 0},
            console_logs=["[ERROR] Something broke"],
        )
        result = _make_result(steps=[step])
        saved = await save_outputs(
            result, output_dir, vlm=None,
            generate_insights=False, generate_ux=False, generate_bugs=True,
        )
        content = saved["bugs"].read_text(encoding="utf-8")
        assert "Something broke" in content
