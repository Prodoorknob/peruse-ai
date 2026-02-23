"""Tests for peruse_ai.config."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from peruse_ai.config import PeruseConfig, VLMBackend


class TestPeruseConfigDefaults:
    """Verify that default config values are correct."""

    def test_default_backend(self):
        config = PeruseConfig()
        assert config.vlm_backend == VLMBackend.OLLAMA

    def test_default_model(self):
        config = PeruseConfig()
        assert config.vlm_model == "qwen3-vl:6b"

    def test_default_base_url(self):
        config = PeruseConfig()
        assert config.vlm_base_url == "http://localhost:11434"

    def test_default_headless(self):
        config = PeruseConfig()
        assert config.headless is True

    def test_default_viewport(self):
        config = PeruseConfig()
        assert config.viewport_width == 1280
        assert config.viewport_height == 720

    def test_default_max_steps(self):
        config = PeruseConfig()
        assert config.max_steps == 50

    def test_default_output_dir(self):
        config = PeruseConfig()
        assert config.output_dir == Path("./peruse_output")


class TestPeruseConfigOverrides:
    """Verify that config can be overridden via kwargs."""

    def test_override_backend(self):
        config = PeruseConfig(vlm_backend=VLMBackend.LMSTUDIO)
        assert config.vlm_backend == VLMBackend.LMSTUDIO

    def test_override_model(self):
        config = PeruseConfig(vlm_model="qwen2.5-vl:72b")
        assert config.vlm_model == "qwen2.5-vl:72b"

    def test_override_headless(self):
        config = PeruseConfig(headless=False)
        assert config.headless is False

    def test_override_max_steps(self):
        config = PeruseConfig(max_steps=10)
        assert config.max_steps == 10

    def test_override_output_dir(self):
        config = PeruseConfig(output_dir="/tmp/test_output")
        assert config.output_dir == Path("/tmp/test_output")


class TestPeruseConfigValidation:
    """Verify that invalid config raises clear errors."""

    def test_invalid_temperature_too_high(self):
        with pytest.raises(Exception):
            PeruseConfig(vlm_temperature=3.0)

    def test_invalid_temperature_negative(self):
        with pytest.raises(Exception):
            PeruseConfig(vlm_temperature=-1.0)

    def test_invalid_max_steps_zero(self):
        with pytest.raises(Exception):
            PeruseConfig(max_steps=0)

    def test_invalid_max_steps_too_high(self):
        with pytest.raises(Exception):
            PeruseConfig(max_steps=1000)

    def test_base_url_trailing_slash_stripped(self):
        config = PeruseConfig(vlm_base_url="http://localhost:11434/")
        assert config.vlm_base_url == "http://localhost:11434"


class TestPeruseConfigBackendURLs:
    """Verify backend-specific URL helpers."""

    def test_ollama_base_url_default(self):
        config = PeruseConfig()
        assert config.get_ollama_base_url() == "http://localhost:11434"

    def test_lmstudio_base_url_auto_switch(self):
        config = PeruseConfig(vlm_backend=VLMBackend.LMSTUDIO)
        assert config.get_lmstudio_base_url() == "http://localhost:1234/v1"

    def test_lmstudio_base_url_custom(self):
        config = PeruseConfig(
            vlm_backend=VLMBackend.LMSTUDIO,
            vlm_base_url="http://192.168.1.100:1234/v1",
        )
        assert config.get_lmstudio_base_url() == "http://192.168.1.100:1234/v1"

    def test_jina_base_url_auto_switch(self):
        config = PeruseConfig(vlm_backend=VLMBackend.JINA)
        assert config.get_jina_base_url() == "https://api-beta-vlm.jina.ai/v1"

    def test_jina_base_url_custom(self):
        config = PeruseConfig(
            vlm_backend=VLMBackend.JINA,
            vlm_base_url="https://api.custom.jina.ai",
        )
        assert config.get_jina_base_url() == "https://api.custom.jina.ai"


class TestPeruseConfigEnvVars:
    """Verify that environment variables override defaults."""

    def test_env_var_model(self, monkeypatch):
        monkeypatch.setenv("PERUSE_VLM_MODEL", "llama3.2-vision:11b")
        config = PeruseConfig()
        assert config.vlm_model == "llama3.2-vision:11b"

    def test_env_var_headless(self, monkeypatch):
        monkeypatch.setenv("PERUSE_HEADLESS", "false")
        config = PeruseConfig()
        assert config.headless is False

    def test_env_var_max_steps(self, monkeypatch):
        monkeypatch.setenv("PERUSE_MAX_STEPS", "25")
        config = PeruseConfig()
        assert config.max_steps == 25
