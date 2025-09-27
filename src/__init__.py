"""
Theory of Mind Benchmark for Large Language Models

A comprehensive evaluation suite for assessing mentalizing capabilities in LLMs,
based on clinical psychology research and validated against clinical population baselines.
"""

from .tom_benchmark import (
    TheoryOfMindBenchmark,
    ToMQuestion,
    ToMResult,
    ToMEvaluation,
    QuestionType,
    Difficulty,
    ClinicalPopulation
)

from .model_interfaces import (
    BaseModelInterface,
    OpenAIInterface,
    AnthropicInterface,
    HuggingFaceInterface,
    MockModelInterface,
    ModelConfig,
    create_model_interface,
    get_preset_model,
    PRESET_MODELS
)

from .visualization_dashboard import ToMVisualizationDashboard

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    # Core benchmark classes
    "TheoryOfMindBenchmark",
    "ToMQuestion",
    "ToMResult",
    "ToMEvaluation",

    # Enums
    "QuestionType",
    "Difficulty",
    "ClinicalPopulation",

    # Model interfaces
    "BaseModelInterface",
    "OpenAIInterface",
    "AnthropicInterface",
    "HuggingFaceInterface",
    "MockModelInterface",
    "ModelConfig",
    "create_model_interface",
    "get_preset_model",
    "PRESET_MODELS",

    # Visualization
    "ToMVisualizationDashboard"
]