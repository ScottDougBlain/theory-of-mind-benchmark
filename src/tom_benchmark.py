"""
Theory of Mind Benchmark Implementation

A comprehensive Theory of Mind evaluation suite for LLMs based on clinical psychology
research. This benchmark translates gold-standard psychological assessments into
AI safety evaluations, enabling comparison of model performance to clinical populations.

Based on 83 test scenarios covering multiple aspects of mentalizing abilities,
with performance indicators from clinical populations (NBD, DOM, PHEN, ASD).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
import json
import csv
from pathlib import Path
import re


class QuestionType(Enum):
    """Types of Theory of Mind questions."""
    FC = "False Belief - First Contact"
    SOC = "Social Reasoning"
    PI = "Pluralistic Ignorance"
    FB = "False Belief"
    INT = "Intentionality"
    EMO = "Emotion Recognition"
    SAR = "Sarcasm Detection"
    IRO = "Irony Understanding"


class Difficulty(Enum):
    """Difficulty levels for ToM questions."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ClinicalPopulation(Enum):
    """Clinical populations with comparison data."""
    NBD = "Neurotypical/Non-Brain Damaged"
    DOM = "Dementia"
    PHEN = "Phenylketonuria"
    ASD = "Autism Spectrum Disorder"


@dataclass
class ToMQuestion:
    """Represents a single Theory of Mind question."""
    scenario_name: str
    scenario_text: str
    question_label: str
    question_text: str
    options: Dict[str, str]  # A-E options
    correct_answer: str
    explanation: str
    clinical_scores: Dict[ClinicalPopulation, Optional[float]]
    question_type: QuestionType
    difficulty: Difficulty

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "scenario_name": self.scenario_name,
            "scenario_text": self.scenario_text,
            "question_label": self.question_label,
            "question_text": self.question_text,
            "options": self.options,
            "correct_answer": self.correct_answer,
            "explanation": self.explanation,
            "clinical_scores": {pop.value: score for pop, score in self.clinical_scores.items()},
            "question_type": self.question_type.value,
            "difficulty": self.difficulty.value
        }


@dataclass
class ToMResult:
    """Results for a single ToM question evaluation."""
    question: ToMQuestion
    model_response: str
    extracted_answer: Optional[str]
    is_correct: bool
    confidence_score: Optional[float] = None
    reasoning_quality: Optional[float] = None
    response_time: Optional[float] = None


@dataclass
class ToMEvaluation:
    """Complete evaluation results for a model."""
    model_name: str
    results: List[ToMResult]
    overall_accuracy: float
    accuracy_by_type: Dict[QuestionType, float]
    accuracy_by_difficulty: Dict[Difficulty, float]
    clinical_comparison: Dict[ClinicalPopulation, Dict[str, float]]
    emergence_analysis: Optional[Dict] = None


class TheoryOfMindBenchmark:
    """
    Comprehensive Theory of Mind evaluation suite for Large Language Models.

    This benchmark implements gold-standard psychological assessments to evaluate
    LLM mentalizing capabilities, with comparison to clinical population baselines.

    Features:
    - 83 test scenarios across multiple ToM domains
    - Comparison to clinical populations (NBD, DOM, PHEN, ASD)
    - Analysis of capability emergence across model scales
    - Support for multiple model APIs (OpenAI, Anthropic, Hugging Face)
    - Error analysis revealing systematic failure modes
    """

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the ToM benchmark.

        Args:
            data_path: Path to the ToM_Bench.csv file. If None, uses default location.
        """
        if data_path is None:
            data_path = Path(__file__).parent.parent / "data" / "ToM_Bench.csv"

        self.data_path = Path(data_path)
        self.questions = self._load_questions()
        self.evaluation_history: List[ToMEvaluation] = []

    def _load_questions(self) -> List[ToMQuestion]:
        """Load and parse ToM questions from CSV file."""
        questions = []

        try:
            df = pd.read_csv(self.data_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"ToM benchmark data not found at {self.data_path}")

        for _, row in df.iterrows():
            # Parse options (A-E)
            options = {
                'A': row['A'],
                'B': row['B'],
                'C': row['C'],
                'D': row['D'],
                'E': row['E']
            }

            # Parse clinical scores (handle missing values)
            clinical_scores = {}
            for pop in ClinicalPopulation:
                score_val = row.get(pop.value)
                if pd.notna(score_val) and score_val != '':
                    try:
                        clinical_scores[pop] = float(score_val)
                    except (ValueError, TypeError):
                        clinical_scores[pop] = None
                else:
                    clinical_scores[pop] = None

            # Parse question type and difficulty
            try:
                question_type = QuestionType(row['QT'])
            except (ValueError, KeyError):
                question_type = QuestionType.SOC  # Default fallback

            try:
                difficulty = Difficulty(row['DIFF'].lower())
            except (ValueError, KeyError):
                difficulty = Difficulty.MEDIUM  # Default fallback

            question = ToMQuestion(
                scenario_name=row['Scenario Name'],
                scenario_text=row['Scenario Text'],
                question_label=row['Question Label'],
                question_text=row['Question Text'],
                options=options,
                correct_answer=row['Correct'],
                explanation=row['Explanation'],
                clinical_scores=clinical_scores,
                question_type=question_type,
                difficulty=difficulty
            )

            questions.append(question)

        return questions

    def get_questions_by_type(self, question_type: QuestionType) -> List[ToMQuestion]:
        """Get all questions of a specific type."""
        return [q for q in self.questions if q.question_type == question_type]

    def get_questions_by_difficulty(self, difficulty: Difficulty) -> List[ToMQuestion]:
        """Get all questions of a specific difficulty."""
        return [q for q in self.questions if q.difficulty == difficulty]

    def format_question_for_model(self, question: ToMQuestion,
                                 include_options: bool = True,
                                 include_instructions: bool = True) -> str:
        """
        Format a question for model evaluation.

        Args:
            question: The ToM question to format
            include_options: Whether to include multiple choice options
            include_instructions: Whether to include evaluation instructions

        Returns:
            Formatted prompt string
        """
        prompt_parts = []

        if include_instructions:
            prompt_parts.append(
                "You will be presented with a scenario involving social cognition and theory of mind. "
                "Please read carefully and answer the question that follows."
            )
            prompt_parts.append("")

        # Add scenario
        prompt_parts.append("**Scenario:**")
        prompt_parts.append(question.scenario_text.strip())
        prompt_parts.append("")

        # Add question
        prompt_parts.append("**Question:**")
        prompt_parts.append(question.question_text.strip())
        prompt_parts.append("")

        # Add options if requested
        if include_options:
            prompt_parts.append("**Options:**")
            for letter, option in question.options.items():
                if option and option.strip():  # Skip empty options
                    prompt_parts.append(f"{letter}. {option.strip()}")
            prompt_parts.append("")

            if include_instructions:
                prompt_parts.append("Please provide your answer as a single letter (A, B, C, D, or E) "
                                  "followed by a brief explanation of your reasoning.")

        return "\n".join(prompt_parts)

    def extract_answer_from_response(self, response: str) -> Optional[str]:
        """
        Extract the selected answer (A-E) from model response.

        Args:
            response: The model's response text

        Returns:
            Extracted answer letter or None if not found
        """
        # Common patterns for answer extraction
        patterns = [
            r"(?:answer|choice|option|select)?\s*(?:is\s*)?:?\s*([A-E])\b",
            r"^([A-E])\.",
            r"\b([A-E])\s*[-:.]",
            r"^([A-E])\b",
            r"\(([A-E])\)",
        ]

        # Try each pattern
        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
            if matches:
                return matches[0].upper()

        # Look for letter at start of response
        lines = response.strip().split('\n')
        if lines:
            first_line = lines[0].strip()
            if len(first_line) >= 1 and first_line[0].upper() in 'ABCDE':
                return first_line[0].upper()

        return None

    def evaluate_single_question(self, question: ToMQuestion,
                                model_response_fn: Callable[[str], str],
                                **kwargs) -> ToMResult:
        """
        Evaluate a single question with a model.

        Args:
            question: The ToM question to evaluate
            model_response_fn: Function that takes prompt and returns model response
            **kwargs: Additional arguments for question formatting

        Returns:
            ToMResult with evaluation outcome
        """
        # Format question for model
        prompt = self.format_question_for_model(question, **kwargs)

        # Get model response
        response = model_response_fn(prompt)

        # Extract answer
        extracted_answer = self.extract_answer_from_response(response)

        # Check correctness
        is_correct = (extracted_answer == question.correct_answer) if extracted_answer else False

        return ToMResult(
            question=question,
            model_response=response,
            extracted_answer=extracted_answer,
            is_correct=is_correct
        )

    def evaluate_model(self, model_response_fn: Callable[[str], str],
                      model_name: str = "Unknown Model",
                      question_subset: Optional[List[ToMQuestion]] = None,
                      **kwargs) -> ToMEvaluation:
        """
        Evaluate a model on the complete ToM benchmark.

        Args:
            model_response_fn: Function that takes prompt and returns model response
            model_name: Name identifier for the model
            question_subset: Optional subset of questions to evaluate
            **kwargs: Additional arguments for question formatting

        Returns:
            Complete ToMEvaluation with results and analysis
        """
        questions_to_eval = question_subset if question_subset else self.questions

        print(f"Evaluating {model_name} on {len(questions_to_eval)} ToM questions...")

        results = []
        for i, question in enumerate(questions_to_eval, 1):
            if i % 10 == 0:
                print(f"Progress: {i}/{len(questions_to_eval)}")

            result = self.evaluate_single_question(question, model_response_fn, **kwargs)
            results.append(result)

        # Calculate metrics
        evaluation = self._calculate_evaluation_metrics(model_name, results)

        # Store in history
        self.evaluation_history.append(evaluation)

        return evaluation

    def _calculate_evaluation_metrics(self, model_name: str,
                                    results: List[ToMResult]) -> ToMEvaluation:
        """Calculate comprehensive evaluation metrics."""
        total_questions = len(results)
        correct_answers = sum(1 for r in results if r.is_correct)
        overall_accuracy = correct_answers / total_questions if total_questions > 0 else 0

        # Accuracy by question type
        accuracy_by_type = {}
        for qtype in QuestionType:
            type_results = [r for r in results if r.question.question_type == qtype]
            if type_results:
                type_correct = sum(1 for r in type_results if r.is_correct)
                accuracy_by_type[qtype] = type_correct / len(type_results)
            else:
                accuracy_by_type[qtype] = 0.0

        # Accuracy by difficulty
        accuracy_by_difficulty = {}
        for difficulty in Difficulty:
            diff_results = [r for r in results if r.question.difficulty == difficulty]
            if diff_results:
                diff_correct = sum(1 for r in diff_results if r.is_correct)
                accuracy_by_difficulty[difficulty] = diff_correct / len(diff_results)
            else:
                accuracy_by_difficulty[difficulty] = 0.0

        # Clinical population comparison
        clinical_comparison = self._calculate_clinical_comparison(results)

        return ToMEvaluation(
            model_name=model_name,
            results=results,
            overall_accuracy=overall_accuracy,
            accuracy_by_type=accuracy_by_type,
            accuracy_by_difficulty=accuracy_by_difficulty,
            clinical_comparison=clinical_comparison
        )

    def _calculate_clinical_comparison(self, results: List[ToMResult]) -> Dict[ClinicalPopulation, Dict[str, float]]:
        """Calculate comparison to clinical population baselines."""
        comparison = {}

        for population in ClinicalPopulation:
            # Get questions where this population has data
            pop_results = []
            for result in results:
                if population in result.question.clinical_scores:
                    clinical_score = result.question.clinical_scores[population]
                    if clinical_score is not None:
                        pop_results.append((result, clinical_score))

            if pop_results:
                # Calculate model accuracy on these questions
                model_correct = sum(1 for result, _ in pop_results if result.is_correct)
                model_accuracy = model_correct / len(pop_results)

                # Calculate average clinical baseline
                clinical_baseline = np.mean([score for _, score in pop_results])

                # Calculate performance difference
                performance_diff = model_accuracy - clinical_baseline

                comparison[population] = {
                    "model_accuracy": model_accuracy,
                    "clinical_baseline": clinical_baseline,
                    "performance_difference": performance_diff,
                    "questions_with_data": len(pop_results)
                }
            else:
                comparison[population] = {
                    "model_accuracy": 0.0,
                    "clinical_baseline": 0.0,
                    "performance_difference": 0.0,
                    "questions_with_data": 0
                }

        return comparison

    def generate_error_analysis(self, evaluation: ToMEvaluation) -> Dict:
        """Generate analysis of systematic failure modes."""
        incorrect_results = [r for r in evaluation.results if not r.is_correct]

        # Error patterns by question type
        errors_by_type = {}
        for qtype in QuestionType:
            type_errors = [r for r in incorrect_results if r.question.question_type == qtype]
            errors_by_type[qtype.value] = {
                "count": len(type_errors),
                "total_type_questions": len([r for r in evaluation.results if r.question.question_type == qtype]),
                "error_rate": len(type_errors) / len([r for r in evaluation.results if r.question.question_type == qtype])
                              if len([r for r in evaluation.results if r.question.question_type == qtype]) > 0 else 0
            }

        # Error patterns by difficulty
        errors_by_difficulty = {}
        for difficulty in Difficulty:
            diff_errors = [r for r in incorrect_results if r.question.difficulty == difficulty]
            errors_by_difficulty[difficulty.value] = {
                "count": len(diff_errors),
                "total_difficulty_questions": len([r for r in evaluation.results if r.question.difficulty == difficulty]),
                "error_rate": len(diff_errors) / len([r for r in evaluation.results if r.question.difficulty == difficulty])
                              if len([r for r in evaluation.results if r.question.difficulty == difficulty]) > 0 else 0
            }

        # Most challenging questions
        most_challenging = []
        for result in incorrect_results:
            most_challenging.append({
                "question_label": result.question.question_label,
                "scenario_name": result.question.scenario_name,
                "question_type": result.question.question_type.value,
                "difficulty": result.question.difficulty.value,
                "model_answer": result.extracted_answer,
                "correct_answer": result.question.correct_answer
            })

        return {
            "total_errors": len(incorrect_results),
            "error_rate": len(incorrect_results) / len(evaluation.results),
            "errors_by_type": errors_by_type,
            "errors_by_difficulty": errors_by_difficulty,
            "most_challenging_questions": most_challenging[:10]  # Top 10
        }

    def save_evaluation(self, evaluation: ToMEvaluation, output_path: str) -> None:
        """Save evaluation results to JSON file."""
        # Convert to serializable format
        eval_dict = {
            "model_name": evaluation.model_name,
            "overall_accuracy": evaluation.overall_accuracy,
            "accuracy_by_type": {qtype.value: acc for qtype, acc in evaluation.accuracy_by_type.items()},
            "accuracy_by_difficulty": {diff.value: acc for diff, acc in evaluation.accuracy_by_difficulty.items()},
            "clinical_comparison": {pop.value: comp for pop, comp in evaluation.clinical_comparison.items()},
            "results": [
                {
                    "question_label": r.question.question_label,
                    "is_correct": r.is_correct,
                    "extracted_answer": r.extracted_answer,
                    "correct_answer": r.question.correct_answer,
                    "model_response": r.model_response
                }
                for r in evaluation.results
            ]
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(eval_dict, f, indent=2)

        print(f"Evaluation saved to: {output_path}")

    def get_benchmark_statistics(self) -> Dict:
        """Get overall benchmark statistics."""
        total_questions = len(self.questions)

        # Question type distribution
        type_distribution = {}
        for qtype in QuestionType:
            count = len([q for q in self.questions if q.question_type == qtype])
            type_distribution[qtype.value] = count

        # Difficulty distribution
        difficulty_distribution = {}
        for difficulty in Difficulty:
            count = len([q for q in self.questions if q.difficulty == difficulty])
            difficulty_distribution[difficulty.value] = count

        # Clinical data availability
        clinical_data_availability = {}
        for population in ClinicalPopulation:
            count = len([q for q in self.questions
                        if population in q.clinical_scores and q.clinical_scores[population] is not None])
            clinical_data_availability[population.value] = count

        return {
            "total_questions": total_questions,
            "question_type_distribution": type_distribution,
            "difficulty_distribution": difficulty_distribution,
            "clinical_data_availability": clinical_data_availability,
            "unique_scenarios": len(set(q.scenario_name for q in self.questions))
        }