"""
peruse_ai.cli
~~~~~~~~~~~~~
Click-based CLI providing the `peruse` command with rich terminal output.
"""

from __future__ import annotations

import asyncio
import logging
import sys

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def _setup_logging(verbose: bool) -> None:
    """Configure logging with Rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )
    # Quiet noisy libraries
    logging.getLogger("playwright").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


VALID_REPORTS = {"insights", "ux", "bugs", "all"}


def _parse_reports(reports_str: str) -> tuple[bool, bool, bool]:
    """Parse the --reports option into boolean flags.

    Args:
        reports_str: Comma-separated report names (e.g. "insights,bugs").

    Returns:
        Tuple of (generate_insights, generate_ux, generate_bugs).
    """
    requested = {r.strip().lower() for r in reports_str.split(",")}
    invalid = requested - VALID_REPORTS
    if invalid:
        raise click.BadParameter(
            f"Unknown report type(s): {', '.join(sorted(invalid))}. "
            f"Valid options: {', '.join(sorted(VALID_REPORTS))}"
        )
    if "all" in requested:
        return True, True, True
    return "insights" in requested, "ux" in requested, "bugs" in requested


# ---------------------------------------------------------------------------
# CLI Group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(version="0.1.0", prog_name="peruse-ai")
def main():
    """Peruse-AI -- Autonomous web exploration agent powered by local VLMs."""
    pass


# ---------------------------------------------------------------------------
# `peruse run` — Full exploration with all outputs
# ---------------------------------------------------------------------------


@main.command()
@click.option("--url", required=True, help="Starting URL to explore.")
@click.option("--task", required=True, help="High-level goal for the agent.")
@click.option("--model", default="qwen3-vl:6b", help="VLM model name.")
@click.option("--backend", default="ollama", type=click.Choice(["ollama", "lmstudio", "openai_compat", "jina"]))
@click.option("--base-url", default=None, help="VLM API base URL (auto-detected for ollama/lmstudio).")
@click.option("--output", default="./peruse_output", help="Output directory for reports.", type=click.Path())
@click.option("--max-steps", default=50, help="Maximum agent loop iterations.", type=int)
@click.option("--headless/--no-headless", default=True, help="Run browser headless.")
@click.option("--reports", default="all", help="Comma-separated reports to generate: insights, ux, bugs, all.")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
def run(url, task, model, backend, base_url, output, max_steps, headless, reports, verbose):
    """Run a full exploration session and generate all reports."""
    _setup_logging(verbose)
    gen_insights, gen_ux, gen_bugs = _parse_reports(reports)
    asyncio.run(_run_agent(url, task, model, backend, base_url, output, max_steps, headless,
                           gen_insights, gen_ux, gen_bugs))


async def _run_agent(url, task, model, backend, base_url, output, max_steps, headless,
                     gen_insights, gen_ux, gen_bugs):
    """Internal async handler for the run command."""
    from peruse_ai.agent import PeruseAgent
    from peruse_ai.config import PeruseConfig, VLMBackend
    from peruse_ai.outputs import save_outputs
    from peruse_ai.vlm import create_vlm

    if backend == "jina" and model == "qwen3-vl:6b":
        model = "jina-vlm"

    # Build config
    config_kwargs = {
        "vlm_backend": VLMBackend(backend),
        "vlm_model": model,
        "headless": headless,
        "max_steps": max_steps,
        "output_dir": output,
    }
    if base_url:
        config_kwargs["vlm_base_url"] = base_url

    config = PeruseConfig(**config_kwargs)

    # Show run banner
    console.print(
        Panel.fit(
            f"[bold cyan] Peruse-AI[/bold cyan]\n\n"
            f"[bold]URL:[/bold] {url}\n"
            f"[bold]Task:[/bold] {task}\n"
            f"[bold]Model:[/bold] {model} ({backend})\n"
            f"[bold]Max Steps:[/bold] {max_steps}\n"
            f"[bold]Headless:[/bold] {headless}",
            title="Agent Configuration",
            border_style="cyan",
        )
    )

    # Run agent
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task_id = progress.add_task("Running agent...", total=None)

        agent = PeruseAgent(config=config, url=url, task=task)
        result = await agent.run()

        progress.update(task_id, description=f"Completed in {result.total_time_seconds:.1f}s")

    # Summary
    status_emoji = "✅" if result.completed else "⚠️"
    console.print(f"\n{status_emoji} Agent finished | Steps: {len(result.steps)} | "
                  f"Time: {result.total_time_seconds:.1f}s")

    if result.final_summary:
        console.print(f"[dim]Summary: {result.final_summary}[/dim]")

    if result.error:
        console.print(f"[red]Error: {result.error}[/red]")

    # Generate outputs
    console.print("\n[bold]Generating reports...[/bold]")
    needs_vlm = gen_insights or gen_ux
    vlm = create_vlm(config) if needs_vlm else None
    saved = await save_outputs(
        result, config.output_dir, vlm=vlm,
        generate_insights=gen_insights, generate_ux=gen_ux, generate_bugs=gen_bugs,
    )

    console.print(Panel.fit(
        "\n".join(f"  📄 {name}: {path}" for name, path in saved.items()),
        title="Generated Reports",
        border_style="green",
    ))


# ---------------------------------------------------------------------------
# `peruse scan` — Lightweight bug scan
# ---------------------------------------------------------------------------


@main.command()
@click.option("--url", required=True, help="Starting URL to scan.")
@click.option("--task", default="Navigate all links and report any errors encountered.",
              help="Scan goal.")
@click.option("--model", default="qwen3-vl:6b", help="VLM model name.")
@click.option("--backend", default="ollama", type=click.Choice(["ollama", "lmstudio", "openai_compat", "jina"]))
@click.option("--base-url", default=None, help="VLM API base URL.")
@click.option("--output", default="./peruse_output", help="Output directory.", type=click.Path())
@click.option("--max-steps", default=30, help="Maximum steps for scan.", type=int)
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
def scan(url, task, model, backend, base_url, output, max_steps, verbose):
    """Run a lightweight bug scan (bug report only, no VLM analysis)."""
    _setup_logging(verbose)
    asyncio.run(_scan_agent(url, task, model, backend, base_url, output, max_steps))


async def _scan_agent(url, task, model, backend, base_url, output, max_steps):
    """Internal async handler for the scan command."""
    from peruse_ai.agent import PeruseAgent
    from peruse_ai.config import PeruseConfig, VLMBackend
    from peruse_ai.outputs import save_outputs

    if backend == "jina" and model == "qwen3-vl:6b":
        model = "jina-vlm"

    config_kwargs = {
        "vlm_backend": VLMBackend(backend),
        "vlm_model": model,
        "headless": True,
        "max_steps": max_steps,
        "output_dir": output,
    }
    if base_url:
        config_kwargs["vlm_base_url"] = base_url

    config = PeruseConfig(**config_kwargs)

    console.print(f"[bold cyan] Scanning:[/bold cyan] {url}")
    console.print(f"[dim]Task: {task}[/dim]\n")

    agent = PeruseAgent(config=config, url=url, task=task)
    result = await agent.run()

    console.print(f"{'✅' if result.completed else '⚠️'} Scan finished | "
                  f"Steps: {len(result.steps)} | Time: {result.total_time_seconds:.1f}s")

    # Only bug report (no VLM analysis)
    saved = await save_outputs(
        result, config.output_dir, vlm=None,
        generate_insights=False, generate_ux=False, generate_bugs=True,
    )

    for name, path in saved.items():
        console.print(f"  📄 {name}: {path}")


# ---------------------------------------------------------------------------
# `peruse check-vlm` — Verify VLM connectivity
# ---------------------------------------------------------------------------


@main.command("check-vlm")
@click.option("--model", default="qwen3-vl:6b", help="VLM model name.")
@click.option("--backend", default="ollama", type=click.Choice(["ollama", "lmstudio", "openai_compat", "jina"]))
@click.option("--base-url", default=None, help="VLM API base URL.")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
def check_vlm(model, backend, base_url, verbose):
    """Check VLM backend connectivity and model availability."""
    _setup_logging(verbose)
    asyncio.run(_check_vlm(model, backend, base_url))


async def _check_vlm(model, backend, base_url):
    """Internal async handler for check-vlm."""
    from peruse_ai.config import PeruseConfig, VLMBackend
    from peruse_ai.vlm import check_vlm_connection

    if backend == "jina" and model == "qwen3-vl:6b":
        model = "jina-vlm"

    config_kwargs = {
        "vlm_backend": VLMBackend(backend),
        "vlm_model": model,
    }
    if base_url:
        config_kwargs["vlm_base_url"] = base_url

    config = PeruseConfig(**config_kwargs)

    console.print(f"[bold]Checking VLM connection...[/bold]")
    console.print(f"  Backend: {backend}")
    console.print(f"  Model: {model}")
    console.print(f"  URL: {config.vlm_base_url}\n")

    result = await check_vlm_connection(config)

    if result["status"] == "ok":
        console.print(f"[green]✅ VLM is reachable![/green]")
        console.print(f"[dim]{result['message']}[/dim]")
    else:
        console.print(f"[red]❌ VLM connection failed[/red]")
        console.print(f"[red]{result['message']}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
