"""
peruse_ai.config
~~~~~~~~~~~~~~~~
Centralized configuration using Pydantic Settings.
All settings can be loaded from env vars (PERUSE_*), .env files, or direct kwargs.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class VLMBackend(str, Enum):
    """Supported VLM backend providers."""

    OLLAMA = "ollama"
    LMSTUDIO = "lmstudio"
    OPENAI_COMPAT = "openai_compat"
    JINA = "jina"


class PeruseConfig(BaseSettings):
    """Configuration for the Peruse-AI agent.

    Settings are loaded in priority order:
    1. Direct kwargs passed to the constructor
    2. Environment variables prefixed with PERUSE_
    3. Values from a .env file
    4. Default values
    """

    model_config = SettingsConfigDict(
        env_prefix="PERUSE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- VLM Settings ---
    vlm_backend: VLMBackend = Field(
        default=VLMBackend.OLLAMA,
        description="Which VLM backend to use: 'ollama', 'lmstudio', 'openai_compat', or 'jina'.",
    )
    vlm_model: str = Field(
        default="qwen3-vl:6b",
        description="Model identifier (e.g. Ollama tag, LM Studio model name).",
    )
    vlm_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for the VLM API endpoint.",
    )
    vlm_api_key: Optional[str] = Field(
        default=None,
        description="API key (only needed for openai_compat backends that require auth).",
    )
    vlm_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for VLM responses.",
    )
    vlm_timeout: int = Field(
        default=120,
        description="Timeout in seconds for a single VLM inference call.",
    )
    vlm_num_ctx: int = Field(
        default=32768,
        description="Context window size (tokens) for local VLMs like Ollama. Smaller values speed up inference.",
    )
    vlm_retries: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Number of retry attempts if the VLM backend crashes (e.g. Vulkan/IPEX-LLM instability).",
    )
    vlm_cooldown: float = Field(
        default=3.0,
        ge=0.0,
        description="Seconds to wait before retrying after a VLM crash. Gives the GPU time to recover.",
    )

    # --- Browser Settings ---
    headless: bool = Field(
        default=True,
        description="Run the browser in headless mode.",
    )
    viewport_width: int = Field(default=1280, ge=320, description="Browser viewport width.")
    viewport_height: int = Field(default=720, ge=240, description="Browser viewport height.")

    # --- Agent Settings ---
    max_steps: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum number of perceive-plan-act iterations.",
    )
    max_dom_elements: int = Field(
        default=100,
        ge=0,
        description="Maximum number of DOM elements to send to the VLM per step (0 = unlimited). "
        "Lower values reduce prompt size and speed up inference.",
    )
    screenshot_quality: int = Field(
        default=80,
        ge=10,
        le=100,
        description="JPEG quality for saved screenshots (10-100).",
    )
    extra_instructions: str = Field(
        default="",
        description="Additional instructions appended to the agent system prompt. "
        "Use this to add domain-specific guidance without altering the base prompt's JSON format rules.",
    )
    persona: str = Field(
        default="",
        description="Agent persona prepended to the system prompt. "
        "Example: 'an extremely experienced AD for a prestigious american sports focused university'. "
        "When set, the prompt begins with 'You are [persona]. ' before the base web browsing agent prompt.",
    )
    max_nudges: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Maximum number of nudge messages before the agent hard-stops. "
        "When the agent repeats the same action or has consecutive parse failures, "
        "it receives a nudge to try something different instead of immediately stopping.",
    )

    # --- Output Settings ---
    output_dir: Path = Field(
        default=Path("./peruse_output"),
        description="Directory where reports and screenshots are saved.",
    )

    @field_validator("vlm_base_url")
    @classmethod
    def _normalize_base_url(cls, v: str) -> str:
        """Strip trailing slashes from base URL."""
        return v.rstrip("/")

    @field_validator("output_dir", mode="before")
    @classmethod
    def _coerce_output_dir(cls, v: str | Path) -> Path:
        return Path(v)

    def get_ollama_base_url(self) -> str:
        """Return the Ollama-specific base URL (default port 11434)."""
        return self.vlm_base_url

    def get_lmstudio_base_url(self) -> str:
        """Return the LM Studio-specific base URL (default port 1234)."""
        if self.vlm_base_url == "http://localhost:11434":
            return "http://localhost:1234/v1"
        return self.vlm_base_url

    def get_jina_base_url(self) -> str:
        """Return the Jina API specific base URL."""
        if self.vlm_base_url == "http://localhost:11434":
            return "https://api-beta-vlm.jina.ai/v1"
        return self.vlm_base_url
