"""
peruse_ai.focus_group
~~~~~~~~~~~~~~~~~~~~~
Run multiple PeruseAgent instances concurrently, each with a unique persona.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field

from peruse_ai.agent import AgentResult, PeruseAgent
from peruse_ai.config import PeruseConfig
from peruse_ai.outputs import save_outputs
from peruse_ai.vlm import create_vlm

logger = logging.getLogger(__name__)


@dataclass
class FocusGroupResult:
    """Aggregated result from a focus group run."""

    personas: list[str]
    url: str
    task: str
    results: list[AgentResult] = field(default_factory=list)
    persona_map: dict[str, AgentResult] = field(default_factory=dict)


class FocusGroup:
    """Run N PeruseAgent instances concurrently with different personas.

    Each persona gets:
    - Its own PeruseAgent with the persona set in the config
    - Its own browser instance (via PeruseAgent.run())
    - Its own VLM instance (via PeruseAgent.run())
    - Its own output sub-directory under the base output_dir

    Args:
        personas: List of persona strings.
        url: The URL all agents will explore.
        task: The task/goal all agents will pursue.
        config: Base PeruseConfig. Each persona gets a derived copy.
        generate_insights: Whether to generate data insights reports.
        generate_ux: Whether to generate UX review reports.
        generate_bugs: Whether to generate bug reports.

    Example::

        fg = FocusGroup(
            personas=["a senior UX designer", "a data analyst", "a QA engineer"],
            url="https://example.com/dashboard",
            task="Explore the dashboard and identify issues",
        )
        result = await fg.run()
        for persona, agent_result in result.persona_map.items():
            print(f"{persona}: {agent_result.final_summary}")
    """

    def __init__(
        self,
        personas: list[str],
        url: str,
        task: str,
        config: PeruseConfig | None = None,
        generate_insights: bool = True,
        generate_ux: bool = True,
        generate_bugs: bool = True,
    ) -> None:
        self.personas = personas
        self.url = url
        self.task = task
        self.config = config or PeruseConfig()
        self.generate_insights = generate_insights
        self.generate_ux = generate_ux
        self.generate_bugs = generate_bugs

    async def run(self) -> FocusGroupResult:
        """Run all persona agents concurrently and collect results.

        Returns:
            A FocusGroupResult containing all individual AgentResults.
        """
        logger.info(
            "Starting FocusGroup with %d personas | URL: %s | Task: %s",
            len(self.personas),
            self.url,
            self.task,
        )

        tasks = [self._run_persona(persona) for persona in self.personas]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        fg_result = FocusGroupResult(
            personas=list(self.personas),
            url=self.url,
            task=self.task,
        )

        for persona, result in zip(self.personas, results):
            if isinstance(result, Exception):
                logger.error("Persona '%s' failed: %s", persona, result)
                error_result = AgentResult(
                    url=self.url,
                    task=self.task,
                    error=f"Agent failed: {result}",
                )
                fg_result.results.append(error_result)
                fg_result.persona_map[persona] = error_result
            else:
                fg_result.results.append(result)
                fg_result.persona_map[persona] = result

        logger.info(
            "FocusGroup completed: %d/%d personas succeeded",
            sum(1 for r in fg_result.results if r.completed),
            len(self.personas),
        )
        return fg_result

    async def _run_persona(self, persona: str) -> AgentResult:
        """Run a single persona agent and save its outputs.

        Args:
            persona: The persona string for this agent.

        Returns:
            The AgentResult from the run.
        """
        persona_config = self._make_persona_config(persona)
        agent = PeruseAgent(config=persona_config, url=self.url, task=self.task)

        logger.info(
            "Starting persona agent: '%s' -> %s", persona, persona_config.output_dir
        )
        result = await agent.run()

        # Save outputs to persona-specific directory
        needs_vlm = self.generate_insights or self.generate_ux
        vlm = create_vlm(persona_config) if needs_vlm else None
        await save_outputs(
            result,
            persona_config.output_dir,
            vlm=vlm,
            generate_insights=self.generate_insights,
            generate_ux=self.generate_ux,
            generate_bugs=self.generate_bugs,
        )

        logger.info(
            "Persona '%s' completed: %d steps, %.1fs",
            persona,
            len(result.steps),
            result.total_time_seconds,
        )
        return result

    def _make_persona_config(self, persona: str) -> PeruseConfig:
        """Create a persona-specific config derived from the base config.

        The output_dir is set to {base_output_dir}/{persona_slug}/.

        Args:
            persona: The persona string.

        Returns:
            A new PeruseConfig with persona and output_dir set.
        """
        slug = _slugify(persona)
        persona_output = self.config.output_dir / slug
        return self.config.model_copy(
            update={
                "persona": persona,
                "output_dir": persona_output,
            }
        )


def _slugify(text: str) -> str:
    """Convert a persona string to a filesystem-safe slug.

    Examples:
        "a senior UX designer" -> "a-senior-ux-designer"
        "an extremely experienced AD" -> "an-extremely-experienced-ad"
    """
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)  # Remove non-alphanumeric chars
    text = re.sub(r"[\s_]+", "-", text)  # Replace whitespace/underscores with hyphens
    text = re.sub(r"-+", "-", text)  # Collapse multiple hyphens
    return text.strip("-")
