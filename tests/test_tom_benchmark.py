"""
Tests for the Theory of Mind benchmark implementation.
"""

import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import numpy as np
import tempfile
import json

from src.tom_benchmark import (
    TheoryOfMindBenchmark,
    ToMQuestion,
    ToMResult,
    ToMEvaluation,
    QuestionType,
    Difficulty,
    ClinicalPopulation
)


class TestTheoryOfMindBenchmark:
    """Test suite for TheoryOfMindBenchmark class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a temporary CSV file for testing
        self.test_data = {
            'Scenario Name': ['Test Scenario 1', 'Test Scenario 2'],
            'Scenario Text': ['A person believes X but Y is true', 'Two people have different beliefs'],
            'Question Label': ['Q1A', 'Q2A'],
            'Question Text': ['What does the person believe?', 'Who is correct?'],
            'A': ['Option A1', 'Option A2'],
            'B': ['Option B1', 'Option B2'],
            'C': ['Option C1', 'Option C2'],
            'D': ['Option D1', 'Option D2'],
            'E': ['Option E1', 'Option E2'],
            'Correct': ['A', 'B'],
            'Explanation': ['Person believes X', 'Person 2 is correct'],
            'NBD': [0.8, 0.9],
            'DOM': [0.6, 0.7],
            'PHEN': [0.7, ''],
            'ASD': [0.5, 0.4],
            'QT': ['FB', 'SOC'],
            'DIFF': ['easy', 'medium']
        }

        # Create temporary CSV file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df = pd.DataFrame(self.test_data)
        df.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()

        # Initialize benchmark with test data
        self.benchmark = TheoryOfMindBenchmark(data_path=self.temp_file.name)

    def teardown_method(self):
        """Clean up test fixtures."""
        Path(self.temp_file.name).unlink()

    def test_data_loading(self):
        """Test that CSV data is loaded correctly."""
        assert len(self.benchmark.questions) == 2

        # Check first question
        q1 = self.benchmark.questions[0]
        assert q1.scenario_name == 'Test Scenario 1'
        assert q1.correct_answer == 'A'
        assert q1.question_type == QuestionType.FB
        assert q1.difficulty == Difficulty.EASY

        # Check clinical scores
        assert q1.clinical_scores[ClinicalPopulation.NBD] == 0.8
        assert q1.clinical_scores[ClinicalPopulation.PHEN] == 0.7

    def test_question_filtering(self):
        """Test filtering questions by type and difficulty."""
        # Filter by question type
        fb_questions = self.benchmark.get_questions_by_type(QuestionType.FB)
        assert len(fb_questions) == 1
        assert fb_questions[0].question_type == QuestionType.FB

        # Filter by difficulty
        easy_questions = self.benchmark.get_questions_by_difficulty(Difficulty.EASY)
        assert len(easy_questions) == 1
        assert easy_questions[0].difficulty == Difficulty.EASY

    def test_question_formatting(self):
        """Test question formatting for model input."""
        question = self.benchmark.questions[0]

        # Test with options
        formatted = self.benchmark.format_question_for_model(question, include_options=True)
        assert 'Scenario:' in formatted
        assert 'Question:' in formatted
        assert 'A. Option A1' in formatted
        assert 'single letter' in formatted

        # Test without options
        formatted_no_opts = self.benchmark.format_question_for_model(question, include_options=False)
        assert 'A. Option A1' not in formatted_no_opts

    def test_answer_extraction(self):
        """Test extraction of answers from model responses."""
        # Test various response formats
        test_cases = [
            ("A. This is the correct answer", "A"),
            ("The answer is B", "B"),
            ("I choose option C because...", "C"),
            ("D", "D"),
            ("(E) Final answer", "E"),
            ("Looking at this carefully, A seems right", "A"),
            ("No clear answer", None)
        ]

        for response, expected in test_cases:
            extracted = self.benchmark.extract_answer_from_response(response)
            assert extracted == expected, f"Failed for response: '{response}'"

    def test_single_question_evaluation(self):
        """Test evaluation of a single question."""
        question = self.benchmark.questions[0]

        # Mock model function that always returns A
        def mock_model_a(prompt):
            return "A. This is my answer"

        result = self.benchmark.evaluate_single_question(question, mock_model_a)

        assert isinstance(result, ToMResult)
        assert result.question == question
        assert result.extracted_answer == "A"
        assert result.is_correct == True  # Correct answer is A

        # Test incorrect answer
        def mock_model_b(prompt):
            return "B. This is wrong"

        result_wrong = self.benchmark.evaluate_single_question(question, mock_model_b)
        assert result_wrong.extracted_answer == "B"
        assert result_wrong.is_correct == False

    def test_model_evaluation(self):
        """Test complete model evaluation."""
        # Mock model that alternates between A and B
        call_count = 0

        def mock_model_alternating(prompt):
            nonlocal call_count
            call_count += 1
            return "A" if call_count % 2 == 1 else "B"

        evaluation = self.benchmark.evaluate_model(
            model_response_fn=mock_model_alternating,
            model_name="Mock Model"
        )

        assert isinstance(evaluation, ToMEvaluation)
        assert evaluation.model_name == "Mock Model"
        assert len(evaluation.results) == 2
        assert evaluation.overall_accuracy == 0.5  # 1 correct out of 2

    def test_clinical_comparison_calculation(self):
        """Test calculation of clinical population comparisons."""
        # Create mock results
        q1, q2 = self.benchmark.questions
        results = [
            ToMResult(q1, "A", "A", True),  # Correct
            ToMResult(q2, "B", "B", True)   # Correct
        ]

        evaluation = self.benchmark._calculate_evaluation_metrics("Test", results)

        # Check NBD comparison (both questions have NBD data: 0.8, 0.9)
        nbd_comp = evaluation.clinical_comparison[ClinicalPopulation.NBD]
        assert nbd_comp['model_accuracy'] == 1.0  # 100% correct
        assert nbd_comp['clinical_baseline'] == 0.85  # (0.8 + 0.9) / 2
        assert nbd_comp['performance_difference'] == 0.15  # 1.0 - 0.85

    def test_error_analysis(self):
        """Test error analysis generation."""
        # Create mock evaluation with some errors
        q1, q2 = self.benchmark.questions
        results = [
            ToMResult(q1, "A", "B", False),  # Incorrect
            ToMResult(q2, "B", "B", True)    # Correct
        ]

        evaluation = ToMEvaluation(
            model_name="Test",
            results=results,
            overall_accuracy=0.5,
            accuracy_by_type={QuestionType.FB: 0.0, QuestionType.SOC: 1.0},
            accuracy_by_difficulty={Difficulty.EASY: 0.0, Difficulty.MEDIUM: 1.0},
            clinical_comparison={}
        )

        error_analysis = self.benchmark.generate_error_analysis(evaluation)

        assert error_analysis['total_errors'] == 1
        assert error_analysis['error_rate'] == 0.5
        assert QuestionType.FB.value in error_analysis['errors_by_type']
        assert error_analysis['errors_by_type'][QuestionType.FB.value]['error_rate'] == 1.0

    def test_benchmark_statistics(self):
        """Test benchmark statistics calculation."""
        stats = self.benchmark.get_benchmark_statistics()

        assert stats['total_questions'] == 2
        assert stats['unique_scenarios'] == 2
        assert QuestionType.FB.value in stats['question_type_distribution']
        assert Difficulty.EASY.value in stats['difficulty_distribution']

    def test_evaluation_save_load(self):
        """Test saving and loading evaluation results."""
        # Create mock evaluation
        results = [ToMResult(self.benchmark.questions[0], "A", "A", True)]
        evaluation = ToMEvaluation(
            model_name="Test Model",
            results=results,
            overall_accuracy=1.0,
            accuracy_by_type={QuestionType.FB: 1.0},
            accuracy_by_difficulty={Difficulty.EASY: 1.0},
            clinical_comparison={}
        )

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            self.benchmark.save_evaluation(evaluation, temp_path)

            # Verify file exists and contains data
            assert Path(temp_path).exists()

            with open(temp_path, 'r') as f:
                loaded_data = json.load(f)

            assert loaded_data['model_name'] == "Test Model"
            assert loaded_data['overall_accuracy'] == 1.0
            assert len(loaded_data['results']) == 1

        finally:
            Path(temp_path).unlink()


class TestToMDataStructures:
    """Test ToM data structures."""

    def test_tom_question_creation(self):
        """Test ToMQuestion creation and serialization."""
        question = ToMQuestion(
            scenario_name="Test",
            scenario_text="Test scenario",
            question_label="Q1",
            question_text="Test question?",
            options={'A': 'Option A', 'B': 'Option B', 'C': 'Option C', 'D': 'Option D', 'E': 'Option E'},
            correct_answer="A",
            explanation="A is correct",
            clinical_scores={ClinicalPopulation.NBD: 0.8},
            question_type=QuestionType.FB,
            difficulty=Difficulty.EASY
        )

        # Test serialization
        question_dict = question.to_dict()
        assert question_dict['scenario_name'] == "Test"
        assert question_dict['correct_answer'] == "A"
        assert question_dict['clinical_scores'][ClinicalPopulation.NBD.value] == 0.8

    def test_tom_result_creation(self):
        """Test ToMResult creation."""
        question = ToMQuestion(
            scenario_name="Test",
            scenario_text="Test scenario",
            question_label="Q1",
            question_text="Test question?",
            options={'A': 'Option A', 'B': 'Option B', 'C': 'Option C', 'D': 'Option D', 'E': 'Option E'},
            correct_answer="A",
            explanation="A is correct",
            clinical_scores={},
            question_type=QuestionType.FB,
            difficulty=Difficulty.EASY
        )

        result = ToMResult(
            question=question,
            model_response="A. This is my answer",
            extracted_answer="A",
            is_correct=True,
            confidence_score=0.9
        )

        assert result.is_correct == True
        assert result.confidence_score == 0.9
        assert result.extracted_answer == "A"

    def test_enum_values(self):
        """Test enum value integrity."""
        # Test QuestionType enum
        assert QuestionType.FB.value == "False Belief"
        assert QuestionType.SOC.value == "Social Reasoning"

        # Test Difficulty enum
        assert Difficulty.EASY.value == "easy"
        assert Difficulty.MEDIUM.value == "medium"
        assert Difficulty.HARD.value == "hard"

        # Test ClinicalPopulation enum
        assert ClinicalPopulation.NBD.value == "Neurotypical/Non-Brain Damaged"
        assert ClinicalPopulation.ASD.value == "Autism Spectrum Disorder"


if __name__ == "__main__":
    pytest.main([__file__])