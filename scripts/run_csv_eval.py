"""
Batch evaluation runner for CSV datasets, with optional Purple Agent calls.

Examples:
- Mock agent (default):
    python -m scripts.run_csv_eval --dataset-path /app/data/public.csv --output /tmp/summary.json

- Call Purple Agent's /analyze endpoint:
    python -m scripts.run_csv_eval --dataset-path /app/data/public.csv \
      --purple-endpoint http://fab-plus-purple-agent:8001 \
      --output /tmp/summary.json
"""

import argparse
import json
import random
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx

from cio_agent.datasets.csv_provider import CsvFinanceDatasetProvider
from cio_agent.evaluator import ComprehensiveEvaluator
from cio_agent.models import AgentResponse, FinancialData, TaskDifficulty, DebateRebuttal
from cio_agent.orchestrator import MockAgentClient
from cio_agent.task_generator import DynamicTaskGenerator


class PurpleDirectClient:
    """Minimal client hitting Purple Agent /analyze (non-A2A JSON POST)."""

    def __init__(self, endpoint: str, agent_id: str = "purple-agent-client", model: str = "purple-direct", timeout: int = 120):
        self.endpoint = endpoint.rstrip("/")
        self.agent_id = agent_id
        self.model = model
        self.timeout = timeout

    async def _post_analyze(self, question: str, ticker: str | None = None) -> dict[str, Any]:
        url = f"{self.endpoint}/analyze"
        payload = {"question": question}
        if ticker:
            payload["ticker"] = ticker
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()

    async def process_task(self, task) -> AgentResponse:
        data = await self._post_analyze(task.question, task.ticker)
        analysis = data.get("analysis") or json.dumps(data)
        recommendation = data.get("recommendation") or analysis[:500]

        return AgentResponse(
            agent_id=self.agent_id,
            task_id=task.question_id,
            analysis=analysis,
            recommendation=recommendation,
            extracted_financials=FinancialData(),
            tool_calls=[],
            code_executions=[],
        )

    async def process_challenge(self, task_id: str, challenge: str, original_response: AgentResponse) -> AgentResponse:
        # For simplicity, reuse analyze to produce a rebuttal; could be upgraded to a specific A2A method.
        data = await self._post_analyze(challenge)
        analysis = data.get("analysis") or json.dumps(data)
        recommendation = data.get("recommendation") or analysis[:500]

        return DebateRebuttal(
            agent_id=self.agent_id,
            task_id=task_id,
            defense=analysis,
            new_evidence_cited=[],
            tool_calls=[],
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-evaluate CSV finance tasks.")
    parser.add_argument("--dataset-path", required=True, help="Path to CSV dataset")
    parser.add_argument(
        "--simulation-date",
        default=None,
        help="Simulation date YYYY-MM-DD (default: previous year start)",
    )
    parser.add_argument(
        "--difficulty",
        action="append",
        choices=[d.value for d in TaskDifficulty],
        help="Optional difficulty filter; repeatable",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional limit of examples to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--no-debate", action="store_true", help="Skip debate phase")
    parser.add_argument(
        "--purple-endpoint",
        default=None,
        help="Optional Purple Agent endpoint (HTTP). If set, tasks are sent to purple /analyze.",
    )
    parser.add_argument("--output", required=True, help="Output JSON file for summary/results")
    return parser.parse_args()


def _load_templates(args: argparse.Namespace):
    provider = CsvFinanceDatasetProvider(args.dataset_path)
    templates = provider.to_templates()
    if args.difficulty:
        difficulties = {TaskDifficulty(d) for d in args.difficulty}
        templates = [t for t in templates if t.difficulty in difficulties]
    if args.limit:
        templates = templates[: args.limit]
    return provider, templates


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)

    sim_date = (
        datetime.strptime(args.simulation_date, "%Y-%m-%d")
        if args.simulation_date
        else datetime(datetime.now().year - 1, 1, 1)
    )

    provider, templates = _load_templates(args)
    generator = DynamicTaskGenerator(dataset_provider=provider)
    evaluator = ComprehensiveEvaluator()
    if args.purple_endpoint:
        agent = PurpleDirectClient(endpoint=args.purple_endpoint, agent_id="purple-agent-client")
    else:
        agent = MockAgentClient(agent_id="batch-agent", model="gpt-4o")

    results: list[dict[str, Any]] = []

    for tpl in templates:
        try:
            task = asyncio_run(generator.generate_task(tpl.template_id, sim_date))
            if not task:
                results.append({"template_id": tpl.template_id, "error": "task_generation_failed"})
                continue

            eval_result = asyncio_run(
                evaluator.run_full_evaluation(
                    task=task,
                    agent_client=agent,
                    conduct_debate=not args.no_debate,
                )
            )

            alpha = getattr(eval_result.alpha_score, "score", None)
            role = getattr(eval_result.role_score, "total", None)
            debate_obj = getattr(eval_result, "debate_result", None)
            debate = getattr(debate_obj, "debate_multiplier", None) if debate_obj else None
            cost_obj = getattr(eval_result, "cost_breakdown", None)
            cost = getattr(cost_obj, "total_cost_usd", None) if cost_obj else None

            results.append(
                {
                    "task_id": task.question_id,
                    "template_id": tpl.template_id,
                    "category": task.category.value,
                    "difficulty": task.difficulty.value,
                    "alpha_score": alpha,
                    "role_score": role,
                    "debate_multiplier": debate,
                    "cost": cost,
                    "error": None,
                }
            )
        except Exception as e:
            results.append({"template_id": tpl.template_id, "error": str(e)})

    alpha_values = [r["alpha_score"] for r in results if isinstance(r.get("alpha_score"), (int, float))]
    summary: dict[str, Any] = {
        "count": len(results),
        "alpha_mean": statistics.mean(alpha_values) if alpha_values else None,
        "alpha_median": statistics.median(alpha_values) if alpha_values else None,
        "alpha_min": min(alpha_values) if alpha_values else None,
        "alpha_max": max(alpha_values) if alpha_values else None,
        "by_difficulty": {},
    }
    diff_counts: dict[str, int] = {}
    for r in results:
        diff = r.get("difficulty")
        if diff:
            diff_counts[diff] = diff_counts.get(diff, 0) + 1
    summary["by_difficulty"] = diff_counts

    output = {"summary": summary, "results": results}
    try:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(output, indent=2))
        output_path = args.output
    except PermissionError:
        fallback = "/tmp/summary.json"
        Path(fallback).write_text(json.dumps(output, indent=2))
        output_path = fallback
        print(f"Warning: cannot write to {args.output}, wrote to {fallback} instead")

    print(json.dumps(summary, indent=2))
    print(f"Summary written to: {output_path}")


def asyncio_run(coro):
    import asyncio

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


if __name__ == "__main__":
    main()
