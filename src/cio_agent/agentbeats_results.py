"""AgentBeats Results Formatter

Formats evaluation results in AgentBeats-compliant JSON for leaderboard integration.
See: https://docs.agentbeats.dev/tutorial/
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


class AgentBeatsResultsFormatter:
    """Formats and saves evaluation results for AgentBeats leaderboard."""

    def __init__(
        self,
        scenario_id: str | None = None,
        green_agent_id: str | None = None,
        results_dir: str = "results",
    ):
        """
        Initialize the formatter.

        Args:
            scenario_id: AgentBeats scenario UUID (from scenario.toml)
            green_agent_id: Green agent's AgentBeats UUID
            results_dir: Directory to save results files
        """
        self.scenario_id = scenario_id or os.environ.get("AGENTBEATS_SCENARIO_ID", "")
        self.green_agent_id = green_agent_id or os.environ.get("AGENTBEATS_GREEN_AGENT_ID", "")
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def format_results(
        self,
        participant_id: str,
        participant_name: str,
        evaluation_results: dict[str, Any],
        by_dataset: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Format evaluation results in AgentBeats-compliant format.

        Args:
            participant_id: Participant's AgentBeats UUID
            participant_name: Human-readable participant name
            evaluation_results: Raw evaluation results from Green Agent
            by_dataset: Per-dataset breakdown

        Returns:
            AgentBeats-compliant results dictionary
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        run_id = str(uuid4())

        # Extract aggregate metrics
        avg_score = evaluation_results.get("average_score", 0.0)
        accuracy = evaluation_results.get("accuracy", 0.0)
        num_evaluated = evaluation_results.get("num_evaluated", 0)
        num_successful = evaluation_results.get("num_successful", 0)

        # Build per-dataset scores
        dataset_scores = {}
        if by_dataset:
            for ds_name, ds_data in by_dataset.items():
                dataset_scores[ds_name] = {
                    "count": ds_data.get("count", 0),
                    "mean_score": ds_data.get("mean_score", 0.0),
                    "accuracy": ds_data.get("accuracy", 0.0),
                }

        # AgentBeats result format
        result = {
            "schema_version": "1.0",
            "run_id": run_id,
            "scenario_id": self.scenario_id,
            "timestamp": timestamp,
            "green_agent": {
                "id": self.green_agent_id,
                "benchmark": evaluation_results.get("benchmark", "FAB++"),
                "version": evaluation_results.get("version", "1.0"),
            },
            "participants": {
                participant_id: {
                    "name": participant_name,
                    "score": avg_score,
                    "accuracy": accuracy,
                    "tasks_evaluated": num_evaluated,
                    "tasks_successful": num_successful,
                    "dataset_scores": dataset_scores,
                }
            },
            "metadata": {
                "config": evaluation_results.get("config_summary", {}),
                "sampling": evaluation_results.get("sampling_strategy", "stratified"),
            },
            "detailed_results": evaluation_results.get("results", []),
        }

        return result

    def save_results(
        self,
        results: dict[str, Any],
        filename: str | None = None,
    ) -> Path:
        """
        Save results to a JSON file.

        Args:
            results: AgentBeats-formatted results
            filename: Optional filename (defaults to run_id.json)

        Returns:
            Path to the saved results file
        """
        if filename is None:
            filename = f"{results['run_id']}.json"

        filepath = self.results_dir / filename

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)

        return filepath

    def save_leaderboard_entry(
        self,
        results: dict[str, Any],
    ) -> Path:
        """
        Save a compact leaderboard entry for DuckDB queries.

        Args:
            results: AgentBeats-formatted results

        Returns:
            Path to the saved leaderboard entry file
        """
        # Create leaderboard directory
        leaderboard_dir = self.results_dir / "leaderboard"
        leaderboard_dir.mkdir(parents=True, exist_ok=True)

        # Extract leaderboard entry for each participant
        entries = []
        for participant_id, participant_data in results["participants"].items():
            entry = {
                "run_id": results["run_id"],
                "scenario_id": results["scenario_id"],
                "timestamp": results["timestamp"],
                "participant_id": participant_id,
                "participant_name": participant_data["name"],
                "score": participant_data["score"],
                "accuracy": participant_data["accuracy"],
                "tasks_evaluated": participant_data["tasks_evaluated"],
                "tasks_successful": participant_data["tasks_successful"],
                # Flatten dataset scores for easier querying
                **{
                    f"{ds}_score": ds_data["mean_score"]
                    for ds, ds_data in participant_data.get("dataset_scores", {}).items()
                },
                **{
                    f"{ds}_accuracy": ds_data["accuracy"]
                    for ds, ds_data in participant_data.get("dataset_scores", {}).items()
                },
            }
            entries.append(entry)

        # Save as newline-delimited JSON for easy DuckDB loading
        filepath = leaderboard_dir / "entries.ndjson"
        with open(filepath, "a") as f:
            for entry in entries:
                f.write(json.dumps(entry, default=str) + "\n")

        return filepath


def format_and_save_results(
    participant_id: str,
    participant_name: str,
    evaluation_results: dict[str, Any],
    by_dataset: dict[str, Any] | None = None,
    scenario_id: str | None = None,
    green_agent_id: str | None = None,
    results_dir: str = "results",
) -> tuple[Path, Path]:
    """
    Convenience function to format and save results.

    Returns:
        Tuple of (full results path, leaderboard entry path)
    """
    formatter = AgentBeatsResultsFormatter(
        scenario_id=scenario_id,
        green_agent_id=green_agent_id,
        results_dir=results_dir,
    )

    results = formatter.format_results(
        participant_id=participant_id,
        participant_name=participant_name,
        evaluation_results=evaluation_results,
        by_dataset=by_dataset,
    )

    full_path = formatter.save_results(results)
    leaderboard_path = formatter.save_leaderboard_entry(results)

    return full_path, leaderboard_path
