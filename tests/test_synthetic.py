"""
Tests for synthetic question generation and evaluation.

Tests the following new features:
- SyntheticQuestion to Task conversion
- evaluate-synthetic CLI command
- A2A server with synthetic questions
"""

import pytest
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from cio_agent.models import (
    Task,
    TaskCategory,
    TaskDifficulty,
    GroundTruth,
    FinancialData,
    TaskRubric,
)


# Sample synthetic question data (matching the format from generate-synthetic)
SAMPLE_SYNTHETIC_QUESTION = {
    "question_id": "SYN_QUAN_0001",
    "category": "Quantitative Retrieval",
    "difficulty": "easy",
    "question": "What was AAPL's EBITDA in fiscal year 2024?",
    "ground_truth_value": 134661000000.0,
    "ground_truth_formatted": "$134.66B",
    "ticker": "AAPL",
    "fiscal_year": 2024,
    "calculation_steps": ["Retrieved EBITDA from FY2024 income statement"],
    "rubric": {
        "components": [
            {"name": "retrieval_accuracy", "description": "Correctly extracted EBITDA value", "expected_value": "$134.66B", "weight": 0.5},
            {"name": "fiscal_year", "description": "Referenced correct fiscal year", "expected_value": "2024", "weight": 0.3},
            {"name": "units", "description": "Correctly stated units (millions/billions USD)", "expected_value": None, "weight": 0.2}
        ],
        "max_score": 100
    },
    "requires_code_execution": False
}

SAMPLE_QUALITATIVE_QUESTION = {
    "question_id": "SYN_QUAL_0002",
    "category": "Qualitative Retrieval",
    "difficulty": "medium",
    "question": "Describe AAPL's main business and products.",
    "ground_truth_value": "Apple Inc. is a technology company...",
    "ground_truth_formatted": "Apple designs, manufactures, and markets consumer electronics...",
    "ticker": "AAPL",
    "fiscal_year": 2026,
    "calculation_steps": ["Retrieved company overview from filings"],
    "rubric": {
        "components": [
            {"name": "content_accuracy", "description": "Answer captures key business description elements", "expected_value": None, "weight": 0.4}
        ],
        "max_score": 100
    }
}


class TestSyntheticQuestionConversion:
    """Tests for converting synthetic questions to Task objects."""
    
    def test_convert_quantitative_question(self):
        """Test converting a quantitative retrieval question."""
        from cio_agent.cli import TaskCategory, TaskDifficulty, GroundTruth, TaskRubric, Task
        from cio_agent.models import FinancialData
        
        sq_data = SAMPLE_SYNTHETIC_QUESTION
        
        # Handle category enum
        category_value = sq_data.get("category", "Quantitative Retrieval")
        try:
            category = TaskCategory(category_value)
        except ValueError:
            category = TaskCategory.QUANTITATIVE_RETRIEVAL
        
        # Handle difficulty enum
        difficulty_value = sq_data.get("difficulty", "medium")
        try:
            difficulty = TaskDifficulty(difficulty_value)
        except ValueError:
            difficulty = TaskDifficulty.MEDIUM
        
        # Build ground truth
        ground_truth = GroundTruth(
            macro_thesis=str(sq_data.get("ground_truth_formatted", "")),
            key_themes=sq_data.get("calculation_steps", []),
            expected_recommendation=str(sq_data.get("ground_truth_formatted", "")),
            financials=FinancialData(),
        )
        
        # Build rubric
        rubric_data = sq_data.get("rubric", {})
        rubric_components = rubric_data.get("components", [])
        rubric = TaskRubric(
            criteria=[c.get("description", "") for c in rubric_components],
            max_score=rubric_data.get("max_score", 100),
        )
        
        # Create task
        task = Task(
            question_id=sq_data.get("question_id", "SYN_UNKNOWN"),
            category=category,
            difficulty=difficulty,
            question=sq_data.get("question", ""),
            ticker=sq_data.get("ticker", "AAPL"),
            fiscal_year=sq_data.get("fiscal_year", 2024),
            simulation_date=datetime.now(),
            ground_truth=ground_truth,
            rubric=rubric,
            requires_code_execution=sq_data.get("requires_code_execution", False),
        )
        
        assert task.question_id == "SYN_QUAN_0001"
        assert task.category == TaskCategory.QUANTITATIVE_RETRIEVAL
        assert task.difficulty == TaskDifficulty.EASY
        assert task.ticker == "AAPL"
        assert task.fiscal_year == 2024
        assert "EBITDA" in task.question
        assert len(task.rubric.criteria) == 3
        
    def test_convert_qualitative_question(self):
        """Test converting a qualitative retrieval question."""
        from cio_agent.models import FinancialData
        
        sq_data = SAMPLE_QUALITATIVE_QUESTION
        
        category = TaskCategory(sq_data["category"])
        difficulty = TaskDifficulty(sq_data["difficulty"])
        
        ground_truth = GroundTruth(
            macro_thesis=str(sq_data.get("ground_truth_formatted", "")),
            key_themes=sq_data.get("calculation_steps", []),
            financials=FinancialData(),
        )
        
        rubric_data = sq_data.get("rubric", {})
        rubric = TaskRubric(
            criteria=[c.get("description", "") for c in rubric_data.get("components", [])],
            max_score=rubric_data.get("max_score", 100),
        )
        
        task = Task(
            question_id=sq_data["question_id"],
            category=category,
            difficulty=difficulty,
            question=sq_data["question"],
            ticker=sq_data["ticker"],
            fiscal_year=sq_data["fiscal_year"],
            simulation_date=datetime.now(),
            ground_truth=ground_truth,
            rubric=rubric,
        )
        
        assert task.question_id == "SYN_QUAL_0002"
        assert task.category == TaskCategory.QUALITATIVE_RETRIEVAL
        assert task.difficulty == TaskDifficulty.MEDIUM
        
    def test_invalid_category_fallback(self):
        """Test that invalid categories fall back to default."""
        sq_data = {**SAMPLE_SYNTHETIC_QUESTION, "category": "Invalid Category"}
        
        category_value = sq_data.get("category")
        try:
            category = TaskCategory(category_value)
        except ValueError:
            category = TaskCategory.QUANTITATIVE_RETRIEVAL
        
        assert category == TaskCategory.QUANTITATIVE_RETRIEVAL

    def test_invalid_difficulty_fallback(self):
        """Test that invalid difficulties fall back to default."""
        sq_data = {**SAMPLE_SYNTHETIC_QUESTION, "difficulty": "impossible"}
        
        difficulty_value = sq_data.get("difficulty")
        try:
            difficulty = TaskDifficulty(difficulty_value)
        except ValueError:
            difficulty = TaskDifficulty.MEDIUM
        
        assert difficulty == TaskDifficulty.MEDIUM


class TestSyntheticQuestionsFile:
    """Tests for loading synthetic questions from JSON files."""
    
    def test_load_synthetic_questions_json(self, tmp_path):
        """Test loading synthetic questions from a JSON file."""
        # Create a test questions file
        questions_file = tmp_path / "questions.json"
        questions_data = {
            "generated_at": "2026-01-09T00:00:00",
            "count": 2,
            "questions": [
                SAMPLE_SYNTHETIC_QUESTION,
                SAMPLE_QUALITATIVE_QUESTION
            ]
        }
        questions_file.write_text(json.dumps(questions_data))
        
        # Load and verify
        loaded = json.loads(questions_file.read_text())
        assert loaded["count"] == 2
        assert len(loaded["questions"]) == 2
        assert loaded["questions"][0]["question_id"] == "SYN_QUAN_0001"
        assert loaded["questions"][1]["question_id"] == "SYN_QUAL_0002"

    def test_empty_questions_file(self, tmp_path):
        """Test handling of empty questions file."""
        questions_file = tmp_path / "empty.json"
        questions_data = {
            "generated_at": "2026-01-09T00:00:00",
            "count": 0,
            "questions": []
        }
        questions_file.write_text(json.dumps(questions_data))
        
        loaded = json.loads(questions_file.read_text())
        assert loaded["count"] == 0
        assert len(loaded["questions"]) == 0


class TestA2AServerSyntheticQuestions:
    """Tests for A2A server with synthetic questions."""
    
    def test_load_synthetic_questions_function(self, tmp_path):
        """Test the load_synthetic_questions function."""
        from cio_agent.a2a_server import load_synthetic_questions
        
        # Create a test questions file
        questions_file = tmp_path / "questions.json"
        questions_data = {
            "questions": [
                SAMPLE_SYNTHETIC_QUESTION,
                SAMPLE_QUALITATIVE_QUESTION
            ]
        }
        questions_file.write_text(json.dumps(questions_data))
        
        # Load questions
        questions = load_synthetic_questions(str(questions_file))
        
        assert len(questions) == 2
        assert questions[0]["question_id"] == "SYN_QUAN_0001"
        
    def test_load_missing_file_raises(self):
        """Test that loading a missing file raises an error."""
        from cio_agent.a2a_server import load_synthetic_questions
        
        with pytest.raises(FileNotFoundError):
            load_synthetic_questions("/nonexistent/path.json")
            
    def test_load_empty_questions_raises(self, tmp_path):
        """Test that loading a file with no questions raises an error."""
        from cio_agent.a2a_server import load_synthetic_questions
        
        questions_file = tmp_path / "empty.json"
        questions_file.write_text('{"questions": []}')
        
        with pytest.raises(ValueError, match="No questions found"):
            load_synthetic_questions(str(questions_file))


class TestGreenAgentSyntheticConversion:
    """Tests for GreenAgent synthetic question conversion."""
    
    def test_green_agent_init_with_synthetic_questions(self):
        """Test GreenAgent initialization with synthetic questions."""
        from cio_agent.green_agent import GreenAgent
        
        questions = [SAMPLE_SYNTHETIC_QUESTION, SAMPLE_QUALITATIVE_QUESTION]
        agent = GreenAgent(synthetic_questions=questions)
        
        assert len(agent.synthetic_questions) == 2
        
    def test_green_agent_convert_questions(self):
        """Test GreenAgent._convert_synthetic_to_tasks method."""
        from cio_agent.green_agent import GreenAgent
        
        questions = [SAMPLE_SYNTHETIC_QUESTION, SAMPLE_QUALITATIVE_QUESTION]
        agent = GreenAgent(synthetic_questions=questions)
        
        tasks = agent._convert_synthetic_to_tasks(num_tasks=2)
        
        assert len(tasks) == 2
        assert tasks[0].question_id == "SYN_QUAN_0001"
        assert tasks[1].question_id == "SYN_QUAL_0002"
        assert tasks[0].ticker == "AAPL"
        
    def test_green_agent_convert_limited_tasks(self):
        """Test that num_tasks limits the number of converted tasks."""
        from cio_agent.green_agent import GreenAgent
        
        questions = [SAMPLE_SYNTHETIC_QUESTION, SAMPLE_QUALITATIVE_QUESTION]
        agent = GreenAgent(synthetic_questions=questions)
        
        tasks = agent._convert_synthetic_to_tasks(num_tasks=1)
        
        assert len(tasks) == 1
        assert tasks[0].question_id == "SYN_QUAN_0001"
