from __future__ import annotations

import unittest

from src.generation.hallucination import (
    HALLUCINATION_TARGET_ANCHOR,
    NARRATIVE_PHASES,
    _section_role,
    hallucination_temperature,
)
from src.service.consumer_store import CREATIVITY_LEVELS
from src.service.runtime import make_ollama_client
from src.utils.config import AppConfig


def config_for(target: float) -> AppConfig:
    config = AppConfig()
    config.generation.hallucination_target = target
    return config


class CreativityDialTests(unittest.TestCase):
    def test_each_creativity_profile_samples_at_its_own_temperature(self) -> None:
        temperatures = {
            name: hallucination_temperature(config_for(level))
            for name, level in CREATIVITY_LEVELS.items()
        }

        self.assertEqual(len(set(temperatures.values())), len(CREATIVITY_LEVELS))
        self.assertLess(temperatures["stable"], temperatures["balanced"])
        self.assertLess(temperatures["balanced"], temperatures["bold"])

    def test_the_default_target_keeps_the_historical_temperature(self) -> None:
        config = config_for(HALLUCINATION_TARGET_ANCHOR)
        expected = (
            config.generation.temperature
            + config.generation.hallucination_temperature_delta
        )

        self.assertAlmostEqual(hallucination_temperature(config), expected, places=6)

    def test_extreme_targets_stay_inside_the_sampling_bounds(self) -> None:
        for target in (0.0, 1.0, -5.0, 5.0):
            temperature = hallucination_temperature(config_for(target))
            self.assertGreaterEqual(temperature, 0.1)
            self.assertLessEqual(temperature, 1.3)


class SamplingOptionTests(unittest.TestCase):
    def test_prose_calls_carry_top_p_and_repeat_penalty(self) -> None:
        config = AppConfig()
        client = make_ollama_client(config, dry_run=True)

        options = client._chat_options(0.95, 1800)

        self.assertEqual(options["top_p"], config.ollama.top_p)
        self.assertEqual(options["repeat_penalty"], config.ollama.repeat_penalty)

    def test_options_omit_sampling_keys_when_unset(self) -> None:
        from src.llm.ollama_client import OllamaClient

        client = OllamaClient(base_url="http://localhost:11434", chat_model="m", embed_model="e")

        options = client._chat_options(0.8, 500)

        self.assertNotIn("top_p", options)
        self.assertNotIn("repeat_penalty", options)


class SectionRoleTests(unittest.TestCase):
    def test_an_active_outline_owns_the_plot_instead_of_the_fixed_phases(self) -> None:
        with_outline = _section_role(8, 17, has_outline=True)
        without_outline = _section_role(8, 17, has_outline=False)

        self.assertEqual(without_outline, NARRATIVE_PHASES[7])
        self.assertNotIn(without_outline, with_outline)
        self.assertIn("outline beat", with_outline)

    def test_the_craft_role_is_identical_across_sections(self) -> None:
        roles = {_section_role(index, 17, has_outline=True) for index in range(1, 18)}

        self.assertEqual(len(roles), 1)


if __name__ == "__main__":
    unittest.main()
