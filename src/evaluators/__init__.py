"""
Evaluation components for CIO-Agent FAB++ system.

This module provides the hierarchical evaluation framework:
- MacroEvaluator: Strategic reasoning assessment
- FundamentalEvaluator: Data accuracy validation
- ExecutionEvaluator: Action quality assessment
- OptionsEvaluator: Options trading task assessment
"""

from evaluators.macro import MacroEvaluator
from evaluators.fundamental import FundamentalEvaluator
from evaluators.execution import ExecutionEvaluator
from evaluators.cost_tracker import CostTracker, LLMCallRecord
from evaluators.options import OptionsEvaluator, OptionsScore, OPTIONS_CATEGORIES

__all__ = [
    "MacroEvaluator",
    "FundamentalEvaluator",
    "ExecutionEvaluator",
    "OptionsEvaluator",
    "OptionsScore",
    "OPTIONS_CATEGORIES",
    "CostTracker",
    "LLMCallRecord",
]
