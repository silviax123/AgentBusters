#!/usr/bin/env python3
"""
Demo script: Run Purple Agent analysis and Green Agent evaluation

This demonstrates the full CIO-Agent FAB++ system with:
1. Purple Agent performing financial analysis via HTTP
2. Green Agent evaluating the Purple Agent's response
3. Adversarial debate phase

Uses NVIDIA Q3 FY2026 earnings as a sample task with real financial data.

Usage:
    python scripts/run_demo.py
    PURPLE_ENDPOINT=http://localhost:9110 python scripts/run_demo.py
"""

import asyncio
import os
import sys
from datetime import datetime

from rich.console import Console
from rich.panel import Panel

sys.path.insert(0, "src")

from cio_agent.models import (
    Task,
    TaskCategory,
    TaskDifficulty,
    TaskRubric,
    GroundTruth,
    FinancialData,
)
from cio_agent.evaluator import ComprehensiveEvaluator, EvaluationReporter
from cio_agent.a2a_client import PurpleHTTPAgentClient

PURPLE_ENDPOINT = os.environ.get("PURPLE_ENDPOINT", "http://localhost:9110")

console = Console()


def create_nvidia_task() -> Task:
    """Create NVIDIA Q3 FY2026 earnings task with real financial data.

    Sources:
    - https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-third-quarter-fiscal-2026
    - https://www.cnbc.com/2025/11/19/nvidia-nvda-earnings-report-q3-2026.html
    """
    return Task(
        question_id="NVIDIA_Q3_FY2026_demo",
        category=TaskCategory.BEAT_OR_MISS,
        question=(
            "Did NVIDIA beat or miss analyst expectations in Q3 FY2026 "
            "(quarter ended October 26, 2025)? Analyze the earnings results "
            "including revenue, EPS, data center performance, and Blackwell GPU demand."
        ),
        ticker="NVDA",
        fiscal_year=2026,
        simulation_date=datetime(2025, 11, 20),  # After Q3 results announced
        ground_truth=GroundTruth(
            macro_thesis=(
                "NVIDIA's Q3 FY2026 results demonstrate unprecedented AI compute demand. "
                "Revenue hit a record $57B (+62% YoY), significantly beating the $54.92B "
                "consensus. EPS of $1.30 beat the $1.25 estimate. Blackwell GPU sales were "
                "'off the charts' with cloud GPUs sold out. Data center revenue of $51.2B "
                "(+66% YoY) drove growth. Q4 guidance of $65B exceeded $61.66B consensus."
            ),
            key_themes=[
                "AI compute demand",
                "Blackwell GPU",
                "data center growth",
                "beat expectations",
                "cloud GPU sold out",
                "record revenue",
            ],
            financials=FinancialData(
                revenue=57_000_000_000,       # $57.0B record revenue
                net_income=31_910_000_000,    # $31.91B net income (+65% YoY)
                gross_margin=0.734,           # 73.4% GAAP gross margin
                eps=1.30,                     # $1.30 GAAP diluted EPS
                extra_fields={
                    "data_center_revenue": 51_200_000_000,
                    "gaming_revenue": 4_300_000_000,
                    "yoy_revenue_growth": 0.62,
                    "consensus_revenue": 54_920_000_000,
                    "consensus_eps": 1.25,
                }
            ),
            expected_recommendation="Beat",
            numerical_answer=57_000_000_000,
        ),
        difficulty=TaskDifficulty.MEDIUM,
        rubric=TaskRubric(
            criteria=[
                "Correctly identify beat/miss status",
                "Provide actual vs expected figures (Revenue and EPS)",
                "Analyze data center segment performance",
                "Discuss Blackwell GPU demand and AI compute trends",
            ],
            mandatory_elements=[
                "beat or miss determination",
                "revenue figures",
                "EPS figures",
            ],
        ),
        requires_code_execution=False,
    )


async def run_demo():
    """Run the full demo pipeline."""

    console.print(Panel.fit(
        "[bold blue]AgentBusters Demo[/bold blue]\n\n"
        "Purple Agent (Finance Analyst) vs Green Agent (Evaluator)\n"
        "Task: NVIDIA Q3 FY2026 Earnings Analysis\n"
        f"Purple Endpoint: {PURPLE_ENDPOINT}"
    ))

    # Create the evaluation task
    console.print("\n[cyan]1. Creating evaluation task...[/cyan]")
    task = create_nvidia_task()

    console.print(f"   Category: {task.category.value}")
    console.print(f"   Ticker: {task.ticker}")
    console.print(f"   Fiscal Year: {task.fiscal_year}")
    console.print(f"   Difficulty: {task.difficulty.value}")
    console.print(f"   Question: {task.question[:80]}...")

    # Create Purple Agent HTTP client
    console.print("\n[cyan]2. Connecting to Purple Agent...[/cyan]")
    agent = PurpleHTTPAgentClient(
        base_url=PURPLE_ENDPOINT,
        agent_id="purple-finance-agent",
        model="purple-http",
    )
    console.print(f"   Endpoint: {PURPLE_ENDPOINT}")

    # Run full evaluation
    console.print("\n[cyan]3. Running evaluation pipeline...[/cyan]")
    console.print("   - Phase 1: Task Assignment")
    console.print("   - Phase 2: Adversarial Debate")
    console.print("   - Phase 3: Scoring")

    evaluator = ComprehensiveEvaluator()
    result = await evaluator.run_full_evaluation(
        task=task,
        agent_client=agent,
        conduct_debate=True,
    )

    # Display results
    console.print("\n" + "=" * 60)
    console.print("[bold green]EVALUATION COMPLETE[/bold green]")
    console.print("=" * 60)

    # Print summary
    summary = EvaluationReporter.generate_summary(result)
    console.print(summary)

    # Print detailed scores
    console.print("\n[bold cyan]Detailed Score Breakdown:[/bold cyan]")
    console.print(f"  Macro Score: {result.role_score.macro.score:.1f}/100")
    console.print(f"    - {result.role_score.macro.feedback}")
    console.print(f"  Fundamental Score: {result.role_score.fundamental.score:.1f}/100")
    console.print(f"    - {result.role_score.fundamental.feedback}")
    console.print(f"  Execution Score: {result.role_score.execution.score:.1f}/100")
    console.print(f"    - {result.role_score.execution.feedback}")

    console.print(f"\n[bold cyan]Debate Result:[/bold cyan]")
    console.print(f"  Multiplier: {result.debate_result.debate_multiplier}x")
    console.print(f"  Conviction: {result.debate_result.conviction_level.value}")
    console.print(f"  {result.debate_result.feedback}")

    console.print(f"\n[bold cyan]Efficiency:[/bold cyan]")
    console.print(f"  Total Cost: ${result.cost_breakdown.total_cost_usd:.4f}")
    console.print(f"  Tool Calls: {len(result.tool_calls)}")
    console.print(f"  Temporal Violations: {len(result.lookahead_penalty.violations)}")

    # Final score
    console.print(Panel.fit(
        f"[bold green]Final Alpha Score: {result.alpha_score.score:.2f}[/bold green]\n\n"
        f"Formula: ({result.alpha_score.role_score:.1f} × {result.alpha_score.debate_multiplier}) / "
        f"(ln(1 + {result.alpha_score.cost_usd:.4f}) × (1 + {result.alpha_score.lookahead_penalty}))",
        title="[bold]Result[/bold]"
    ))

    return result


if __name__ == "__main__":
    asyncio.run(run_demo())
