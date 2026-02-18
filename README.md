# 🔍 Peruse-AI

**A local-first universal web agent** that autonomously explores web applications and produces structured reports — powered by [browser-use](https://github.com/browser-use/browser-use), [Playwright](https://playwright.dev/python/), and a local Vision-Language Model (Qwen2.5-VL via Ollama).

---

## ✨ Features

- **Autonomous Web Exploration** — Give it a URL and a goal; it figures out the rest.
- **Dual-Channel Perception** — Combines DOM extraction *and* visual screenshots for robust element detection.
- **100% Local** — Your data never leaves your machine. Runs on Ollama, LM Studio, or any OpenAI-compatible local endpoint.
- **Multi-Output Pipeline** — Generates three report types from a single session:
  - 📊 **Data Insights** — Summaries of charts, tables, and visible data.
  - 🎨 **UX/UI Review** — Contrast, layout, accessibility, and usability critique.
  - 🐛 **Bug Report** — Console errors, failed requests, and reproduction steps.
- **Beautiful CLI** — Rich terminal output with progress bars and colored logs.

---

## 🚀 Quickstart

### Prerequisites

1. **Python 3.10+**
2. **Ollama** installed and running ([install guide](https://ollama.com/download))
3. Pull the VLM model:
   ```bash
   ollama pull qwen2.5-vl:7b
   ```

### Install

```bash
pip install peruse-ai
playwright install chromium
```

### Run

```bash
# Full exploration
peruse run --url "https://example.com/dashboard" \
           --task "Explore the dashboard and summarize all visible data"

# Bug scan only
peruse scan --url "https://example.com" \
            --task "Click every link and report errors"

# Check VLM connectivity
peruse check-vlm
```

### Python API

```python
import asyncio
from peruse_ai import PeruseAgent, PeruseConfig

config = PeruseConfig(vlm_model="qwen2.5-vl:7b")
agent = PeruseAgent(
    config=config,
    url="https://example.com/dashboard",
    task="Summarize the visible data and flag any UI issues",
)
result = asyncio.run(agent.run())
print(result.insights)
```

---

## ⚙️ Configuration

All settings can be passed via constructor, environment variables (`PERUSE_*`), or a `.env` file.

| Setting | Env Var | Default | Description |
|---|---|---|---|
| `vlm_backend` | `PERUSE_VLM_BACKEND` | `"ollama"` | `"ollama"`, `"lmstudio"`, or `"openai_compat"` |
| `vlm_model` | `PERUSE_VLM_MODEL` | `"qwen2.5-vl:7b"` | Model identifier |
| `vlm_base_url` | `PERUSE_VLM_BASE_URL` | `"http://localhost:11434"` | API endpoint |
| `headless` | `PERUSE_HEADLESS` | `True` | Run browser headless |
| `max_steps` | `PERUSE_MAX_STEPS` | `50` | Max agent loop iterations |
| `output_dir` | `PERUSE_OUTPUT_DIR` | `"./peruse_output"` | Report output directory |

---

## 🛠️ Development

```bash
git clone https://github.com/rajas/peruse-ai.git
cd peruse-ai
pip install -e ".[dev]"
playwright install chromium
pytest tests/ -v
```

---

## 📄 License

MIT
