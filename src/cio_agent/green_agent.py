"""
Green Agent Implementation for AgentBeats Platform

This is the core agent logic that orchestrates evaluation of Purple Agents.
It receives an assessment request with participant agent URLs and config,
then runs the FAB++ evaluation pipeline.
"""

import json
from typing import Any, Optional
from pydantic import BaseModel, HttpUrl, ValidationError
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message

from cio_agent.messenger import Messenger
from cio_agent.evaluator import ComprehensiveEvaluator, EvaluationReporter
from cio_agent.task_generator import DynamicTaskGenerator
from cio_agent.models import Task as FABTask, TaskCategory, TaskDifficulty, GroundTruth, FinancialData, TaskRubric


class EvalRequest(BaseModel):
    """
    Request format sent by the AgentBeats platform to green agents.
    
    The platform sends this JSON structure when initiating an assessment.
    """
    participants: dict[str, HttpUrl]  # role -> agent URL
    config: dict[str, Any]


class GreenAgent:
    """
    CIO-Agent Green Agent for FAB++ Finance Agent Benchmark.
    
    This agent evaluates Purple Agents on their financial analysis capabilities
    using the FAB++ evaluation framework.
    
    Required roles:
        - purple_agent: The finance agent being evaluated
        
    Config options:
        - ticker: Stock ticker to analyze (default: "NVDA")
        - task_category: Type of task (default: "beat_or_miss")
        - num_tasks: Number of evaluation tasks (default: 1)
        - conduct_debate: Whether to run adversarial debate (default: True)
    """
    
    # Required participant roles
    required_roles: list[str] = ["purple_agent"]
    
    # Required config keys (optional ones will have defaults)
    required_config_keys: list[str] = []

    def __init__(self, synthetic_questions: Optional[list[dict]] = None):
        """
        Initialize the Green Agent.
        
        Args:
            synthetic_questions: Optional list of synthetic questions to use
                                for evaluation. If provided, these will be used
                                instead of generating new tasks.
        """
        self.messenger = Messenger()
        self.evaluator = ComprehensiveEvaluator()
        self.task_generator = DynamicTaskGenerator()
        self.synthetic_questions = synthetic_questions or []

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        """Validate the assessment request."""
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"

        missing_config_keys = set(self.required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"

        return True, "ok"

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """
        Run the FAB++ evaluation assessment.

        Args:
            message: The incoming A2A message containing the EvalRequest
            updater: TaskUpdater for reporting progress and results
        """
        input_text = get_message_text(message)

        # Parse and validate the assessment request
        try:
            request: EvalRequest = EvalRequest.model_validate_json(input_text)
            ok, msg = self.validate_request(request)
            if not ok:
                await updater.reject(new_agent_text_message(msg))
                return
        except ValidationError as e:
            await updater.reject(new_agent_text_message(f"Invalid request: {e}"))
            return

        # Extract configuration
        purple_agent_url = str(request.participants["purple_agent"])
        ticker = request.config.get("ticker", "NVDA")
        task_category = request.config.get("task_category", "beat_or_miss")
        num_tasks = request.config.get("num_tasks", 1)
        conduct_debate = request.config.get("conduct_debate", True)

        # Report starting
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Starting FAB++ evaluation for {ticker}...")
        )

        try:
            # Generate evaluation task(s)
            await updater.update_status(
                TaskState.working,
                new_agent_text_message("Generating evaluation tasks...")
            )
            
            # Get simulation date from config or use current date
            from datetime import datetime
            simulation_date_str = request.config.get("simulation_date")
            if simulation_date_str:
                simulation_date = datetime.fromisoformat(simulation_date_str)
            else:
                simulation_date = datetime.now()
            
            # Use synthetic questions if available, otherwise generate tasks
            if self.synthetic_questions:
                tasks = self._convert_synthetic_to_tasks(num_tasks)
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"Using {len(tasks)} synthetic questions for evaluation...")
                )
            else:
                tasks = await self.task_generator.generate_task_batch(
                    count=num_tasks,
                    simulation_date=simulation_date,
                )
            
            if not tasks:
                # Create a default task if generation fails
                from datetime import datetime
                from cio_agent.models import GroundTruth, FinancialData, TaskRubric
                
                tasks = [FABTask(
                    question_id=f"fab_{ticker}_eval",
                    category=TaskCategory.BEAT_OR_MISS,
                    question=f"Did {ticker} beat or miss analyst expectations in the most recent quarter?",
                    ticker=ticker,
                    fiscal_year=2026,
                    simulation_date=datetime.now(),
                    ground_truth=GroundTruth(
                        macro_thesis="Evaluate earnings performance",
                        key_themes=["revenue", "earnings", "guidance"],
                        financials=FinancialData(),
                        expected_recommendation="Evaluate",
                    ),
                    difficulty=TaskDifficulty.MEDIUM,
                    rubric=TaskRubric(
                        criteria=["Accuracy", "Analysis depth", "Recommendation quality"],
                        mandatory_elements=["beat/miss determination"],
                    ),
                )]

            all_results = []
            
            for i, task in enumerate(tasks):
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"Evaluating task {i+1}/{len(tasks)}: {task.question_id}")
                )
                
                # Send task to Purple Agent
                task_message = json.dumps({
                    "question": task.question,
                    "ticker": task.ticker,
                    "fiscal_year": task.fiscal_year,
                    "category": task.category.value,
                })
                
                try:
                    response = await self.messenger.talk_to_agent(
                        message=task_message,
                        url=purple_agent_url,
                        new_conversation=True,
                        timeout=300,
                    )
                    
                    # Parse agent response
                    from cio_agent.models import AgentResponse, FinancialData as FD
                    agent_response = AgentResponse(
                        agent_id="purple_agent",
                        task_id=task.question_id,
                        analysis=response,
                        recommendation=self._extract_recommendation(response),
                        extracted_financials=FD(),  # Would be parsed from response
                        tool_calls=[],
                        code_executions=[],
                        execution_time_seconds=0.0,
                    )
                    
                    # Conduct debate if enabled
                    agent_rebuttal = None
                    if conduct_debate:
                        await updater.update_status(
                            TaskState.working,
                            new_agent_text_message("Conducting adversarial debate...")
                        )
                        
                        counter_arg = f"Challenge: What are the key risks to your {ticker} analysis?"
                        rebuttal_response = await self.messenger.talk_to_agent(
                            message=counter_arg,
                            url=purple_agent_url,
                            new_conversation=False,
                        )
                        
                        from cio_agent.models import DebateRebuttal
                        agent_rebuttal = DebateRebuttal(
                            agent_id="purple_agent",
                            task_id=task.question_id,
                            defense=rebuttal_response,
                        )
                    
                    # Evaluate response
                    result = await self.evaluator.evaluate_response(
                        task=task,
                        agent_response=agent_response,
                        agent_rebuttal=agent_rebuttal,
                    )
                    
                    all_results.append({
                        "task_id": task.question_id,
                        "alpha_score": result.alpha_score.score,
                        "role_score": result.role_score.total,
                        "debate_multiplier": result.debate_result.debate_multiplier,
                    })
                    
                except Exception as e:
                    all_results.append({
                        "task_id": task.question_id,
                        "error": str(e),
                        "alpha_score": 0.0,
                    })

            # Calculate aggregate metrics
            valid_results = [r for r in all_results if "error" not in r]
            avg_alpha = sum(r["alpha_score"] for r in valid_results) / len(valid_results) if valid_results else 0.0
            
            # Create assessment result
            assessment_result = {
                "benchmark": "FAB++ Finance Agent Benchmark",
                "version": "1.0.0",
                "purple_agent": purple_agent_url,
                "ticker": ticker,
                "num_tasks": len(tasks),
                "num_successful": len(valid_results),
                "average_alpha_score": round(avg_alpha, 2),
                "results": all_results,
            }
            
            # Report results as artifact
            await updater.add_artifact(
                parts=[
                    Part(root=TextPart(text=f"FAB++ Evaluation Complete\n\nAverage Alpha Score: {avg_alpha:.2f}")),
                    Part(root=DataPart(data=assessment_result)),
                ],
                name="evaluation_result",
            )

        except Exception as e:
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(f"Evaluation failed: {str(e)}")
            )
            raise

    def _extract_recommendation(self, response: str) -> str:
        """Extract recommendation from agent response."""
        response_lower = response.lower()
        if "beat" in response_lower:
            return "Beat"
        elif "miss" in response_lower:
            return "Miss"
        elif "buy" in response_lower:
            return "Buy"
        elif "sell" in response_lower:
            return "Sell"
        elif "hold" in response_lower:
            return "Hold"
        return "Unknown"

    def _convert_synthetic_to_tasks(self, num_tasks: int) -> list[FABTask]:
        """
        Convert synthetic question dicts to FABTask objects.
        
        Args:
            num_tasks: Maximum number of tasks to return
            
        Returns:
            List of FABTask objects
        """
        from datetime import datetime
        
        tasks = []
        questions_to_use = self.synthetic_questions[:num_tasks]
        
        for sq in questions_to_use:
            # Handle category enum
            category_value = sq.get("category", "Quantitative Retrieval")
            try:
                category = TaskCategory(category_value)
            except ValueError:
                category = TaskCategory.QUANTITATIVE_RETRIEVAL
            
            # Handle difficulty enum
            difficulty_value = sq.get("difficulty", "medium")
            try:
                difficulty = TaskDifficulty(difficulty_value)
            except ValueError:
                difficulty = TaskDifficulty.MEDIUM
            
            # Build ground truth with required fields
            ground_truth = GroundTruth(
                macro_thesis=str(sq.get("ground_truth_formatted", "Evaluate the analysis")),
                key_themes=sq.get("calculation_steps", []),
                expected_recommendation=str(sq.get("ground_truth_formatted", "")),
                financials=FinancialData(),
            )
            
            # Build rubric from components
            rubric_data = sq.get("rubric", {})
            rubric_components = rubric_data.get("components", [])
            rubric = TaskRubric(
                criteria=[c.get("description", "") for c in rubric_components],
                max_score=rubric_data.get("max_score", 100),
            )
            
            task = FABTask(
                question_id=sq.get("question_id", f"SYN_{len(tasks):04d}"),
                category=category,
                difficulty=difficulty,
                question=sq.get("question", ""),
                ticker=sq.get("ticker", "AAPL"),
                fiscal_year=sq.get("fiscal_year", 2024),
                simulation_date=datetime.now(),
                ground_truth=ground_truth,
                rubric=rubric,
                requires_code_execution=sq.get("requires_code_execution", False),
            )
            tasks.append(task)
        
        return tasks
