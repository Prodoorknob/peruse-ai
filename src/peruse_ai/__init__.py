"""
Peruse-AI: A local-first universal web agent.

Autonomously explores web applications using browser-use, Playwright,
and a local Vision-Language Model to produce data insights, UX reviews,
and bug reports.
"""

from __future__ import annotations

from peruse_ai.agent import AgentResult, AgentStep, PeruseAgent
from peruse_ai.config import PeruseConfig, VLMBackend
from peruse_ai.outputs import (
    generate_bug_report,
    generate_data_insights,
    generate_ux_review,
    save_outputs,
)
from peruse_ai.vlm import create_vlm

__version__ = "0.1.0"

__all__ = [
    # Core
    "PeruseAgent",
    "PeruseConfig",
    "VLMBackend",
    # Results
    "AgentResult",
    "AgentStep",
    # VLM
    "create_vlm",
    # Outputs
    "generate_data_insights",
    "generate_ux_review",
    "generate_bug_report",
    "save_outputs",
]
