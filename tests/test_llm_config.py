"""
Unit tests for EvaluatorLLMConfig and per-evaluator LLM settings.
"""

import os
import pytest
from unittest.mock import patch

from evaluators.llm_utils import (
    EvaluatorLLMConfig,
    EvaluatorModelConfig,
    get_evaluator_llm_config,
    reset_evaluator_llm_config,
    get_model_for_evaluator,
    get_temperature_for_evaluator,
    get_max_tokens_for_evaluator,
)


class TestEvaluatorModelConfig:
    """Test EvaluatorModelConfig dataclass."""

    def test_default_values(self):
        config = EvaluatorModelConfig()
        assert config.model == "gpt-4o-mini"
        assert config.temperature == 0.0
        assert config.max_tokens == 512
        assert config.provider is None

    def test_custom_values(self):
        config = EvaluatorModelConfig(
            model="gpt-4o",
            temperature=0.0,
            max_tokens=1000,
            provider="openai",
        )
        assert config.model == "gpt-4o"
        assert config.temperature == 0.0
        assert config.max_tokens == 1000
        assert config.provider == "openai"

    def test_get_provider_explicit(self):
        config = EvaluatorModelConfig(model="gpt-4o", provider="openai")
        assert config.get_provider() == "openai"

    def test_get_provider_auto_openai(self):
        config = EvaluatorModelConfig(model="gpt-4o-mini")
        assert config.get_provider() == "openai"

    def test_get_provider_auto_anthropic(self):
        config = EvaluatorModelConfig(model="claude-3-sonnet")
        assert config.get_provider() == "anthropic"


class TestEvaluatorLLMConfig:
    """Test EvaluatorLLMConfig dataclass."""

    def test_default_models(self):
        config = EvaluatorLLMConfig()
        assert config.macro.model == "gpt-4o-mini"
        assert config.execution.model == "gpt-4o-mini"
        assert config.gdpval.model == "gpt-4o"
        assert config.bizfinbench.model == "gpt-4o-mini"
        assert config.debate.model == "gpt-4o"
        assert config.public_csv.model == "gpt-4o-mini"

    def test_all_temperatures_are_zero(self):
        """Ensure all temperatures are 0 for reproducibility."""
        config = EvaluatorLLMConfig()
        assert config.macro.temperature == 0.0
        assert config.execution.temperature == 0.0
        assert config.gdpval.temperature == 0.0
        assert config.bizfinbench.temperature == 0.0
        assert config.debate.temperature == 0.0
        assert config.public_csv.temperature == 0.0
        assert config.default.temperature == 0.0

    def test_get_config_known_evaluator(self):
        config = EvaluatorLLMConfig()
        macro_config = config.get_config("macro")
        assert macro_config.model == "gpt-4o-mini"

    def test_get_config_unknown_evaluator(self):
        config = EvaluatorLLMConfig()
        unknown_config = config.get_config("unknown_evaluator")
        assert unknown_config.model == config.default.model

    def test_get_model(self):
        config = EvaluatorLLMConfig()
        assert config.get_model("gdpval") == "gpt-4o"
        assert config.get_model("macro") == "gpt-4o-mini"

    def test_get_temperature(self):
        config = EvaluatorLLMConfig()
        assert config.get_temperature("gdpval") == 0.0
        assert config.get_temperature("macro") == 0.0

    def test_get_max_tokens(self):
        config = EvaluatorLLMConfig()
        assert config.get_max_tokens("gdpval") == 2000
        assert config.get_max_tokens("macro") == 256


class TestEvaluatorLLMConfigFromEnv:
    """Test environment variable configuration."""

    def setup_method(self):
        """Reset config before each test."""
        reset_evaluator_llm_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_evaluator_llm_config()

    def test_from_env_default_model_override(self):
        with patch.dict(os.environ, {"EVAL_LLM_DEFAULT_MODEL": "gpt-4o"}):
            config = EvaluatorLLMConfig.from_env()
            assert config.default.model == "gpt-4o"
            # Per-evaluator should also use default if not explicitly set
            assert config.macro.model == "gpt-4o"

    def test_from_env_per_evaluator_model(self):
        with patch.dict(os.environ, {"EVAL_LLM_MACRO_MODEL": "claude-3-sonnet"}):
            config = EvaluatorLLMConfig.from_env()
            assert config.macro.model == "claude-3-sonnet"
            # Others should remain default
            assert config.execution.model == "gpt-4o-mini"

    def test_from_env_per_evaluator_temperature(self):
        # Note: temperature should stay 0 for reproducibility
        # but the system allows override for testing
        with patch.dict(os.environ, {"EVAL_LLM_GDPVAL_TEMPERATURE": "0.0"}):
            config = EvaluatorLLMConfig.from_env()
            assert config.gdpval.temperature == 0.0

    def test_from_env_per_evaluator_max_tokens(self):
        with patch.dict(os.environ, {"EVAL_LLM_MACRO_MAX_TOKENS": "500"}):
            config = EvaluatorLLMConfig.from_env()
            assert config.macro.max_tokens == 500

    def test_from_env_per_evaluator_provider(self):
        with patch.dict(os.environ, {"EVAL_LLM_MACRO_PROVIDER": "anthropic"}):
            config = EvaluatorLLMConfig.from_env()
            assert config.macro.provider == "anthropic"

    def test_from_env_invalid_temperature_ignored(self):
        with patch.dict(os.environ, {"EVAL_LLM_MACRO_TEMPERATURE": "invalid"}):
            config = EvaluatorLLMConfig.from_env()
            # Should keep default
            assert config.macro.temperature == 0.0

    def test_from_env_invalid_max_tokens_ignored(self):
        with patch.dict(os.environ, {"EVAL_LLM_MACRO_MAX_TOKENS": "invalid"}):
            config = EvaluatorLLMConfig.from_env()
            # Should keep default
            assert config.macro.max_tokens == 256


class TestGlobalConfigFunctions:
    """Test global config helper functions."""

    def setup_method(self):
        reset_evaluator_llm_config()

    def teardown_method(self):
        reset_evaluator_llm_config()

    def test_get_evaluator_llm_config_cached(self):
        config1 = get_evaluator_llm_config()
        config2 = get_evaluator_llm_config()
        assert config1 is config2  # Same instance

    def test_reset_evaluator_llm_config(self):
        config1 = get_evaluator_llm_config()
        reset_evaluator_llm_config()
        config2 = get_evaluator_llm_config()
        assert config1 is not config2  # Different instance

    def test_get_model_for_evaluator(self):
        assert get_model_for_evaluator("gdpval") == "gpt-4o"
        assert get_model_for_evaluator("macro") == "gpt-4o-mini"

    def test_get_temperature_for_evaluator(self):
        assert get_temperature_for_evaluator("gdpval") == 0.0
        assert get_temperature_for_evaluator("macro") == 0.0

    def test_get_max_tokens_for_evaluator(self):
        assert get_max_tokens_for_evaluator("gdpval") == 2000
        assert get_max_tokens_for_evaluator("macro") == 256

    def test_env_override_reflected_in_helpers(self):
        with patch.dict(os.environ, {"EVAL_LLM_MACRO_MODEL": "custom-model"}):
            reset_evaluator_llm_config()
            assert get_model_for_evaluator("macro") == "custom-model"


class TestReproducibilityRequirements:
    """
    Tests to ensure LLM-as-judge reproducibility requirements are met.

    Per submission guidelines:
    - Use deterministic settings when relying on LLM judges
    - Fix relevant parameters (e.g., temperature)
    - Validate that repeated runs produce consistent scores
    """

    def test_all_evaluators_have_zero_temperature(self):
        """Critical: All evaluators must use temperature=0 for reproducibility."""
        config = EvaluatorLLMConfig()
        evaluators = ["macro", "execution", "gdpval", "bizfinbench", "debate", "public_csv"]

        for name in evaluators:
            temp = config.get_temperature(name)
            assert temp == 0.0, f"Evaluator '{name}' has temperature={temp}, must be 0.0"

    def test_default_temperature_is_zero(self):
        """Default config must also use temperature=0."""
        config = EvaluatorLLMConfig()
        assert config.default.temperature == 0.0
