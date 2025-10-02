"""
Example script for running the Theory of Mind benchmark on various models.

This script demonstrates how to evaluate different LLMs on the ToM benchmark
and generate comprehensive analysis reports.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from tom_benchmark import TheoryOfMindBenchmark
from model_interfaces import create_model_interface, get_preset_model, PRESET_MODELS
from visualization_dashboard import ToMVisualizationDashboard


def run_single_model_evaluation(model_name: str, provider: str = "auto"):
    """
    Run ToM benchmark on a single model.

    Args:
        model_name: Name of the model to evaluate
        provider: API provider (auto-detected if not specified)
    """
    print(f"Evaluating {model_name} on Theory of Mind Benchmark")
    print("=" * 60)

    # Initialize benchmark
    benchmark = TheoryOfMindBenchmark()

    # Get benchmark statistics
    stats = benchmark.get_benchmark_statistics()
    print(f"Loaded {stats['total_questions']} ToM questions")
    print(f"Question types: {list(stats['question_type_distribution'].keys())}")
    print(f"Difficulty levels: {list(stats['difficulty_distribution'].keys())}")
    print()

    # Create model interface
    try:
        if model_name in PRESET_MODELS:
            model_interface = get_preset_model(model_name)
        else:
            model_interface = create_model_interface(model_name, provider)

        print(f"Model interface created successfully")
        print(f"Configuration: {model_interface.config.__dict__}")
        print()

    except Exception as e:
        print(f"Error creating model interface: {e}")
        print("Make sure you have the required API keys set as environment variables:")
        print("- OPENAI_API_KEY for OpenAI models")
        print("- ANTHROPIC_API_KEY for Anthropic models")
        print("- HUGGINGFACE_API_KEY for Hugging Face models")
        return

    # Run evaluation
    try:
        evaluation = benchmark.evaluate_model(
            model_response_fn=model_interface,
            model_name=model_name
        )

        # Print results
        print_evaluation_summary(evaluation)

        # Save detailed results
        output_dir = f"results_{model_name.replace('/', '_')}"
        Path(output_dir).mkdir(exist_ok=True)

        benchmark.save_evaluation(evaluation, f"{output_dir}/evaluation.json")

        # Generate error analysis
        error_analysis = benchmark.generate_error_analysis(evaluation)
        print_error_analysis(error_analysis)

        # Create visualizations
        dashboard = ToMVisualizationDashboard()

        # Individual model plots
        dashboard.plot_error_analysis_heatmap(evaluation, f"{output_dir}/error_heatmap.png")
        dashboard.plot_response_distribution(evaluation, f"{output_dir}/response_distribution.html")

        print(f"\nResults saved to: {output_dir}/")
        print(f"Usage statistics: {model_interface.get_stats()}")

    except Exception as e:
        print(f"Error during evaluation: {e}")


def run_comparative_analysis(model_list: List[str]):
    """
    Run comparative analysis across multiple models.

    Args:
        model_list: List of model names to compare
    """
    print(f"Running comparative analysis on {len(model_list)} models")
    print("=" * 60)

    benchmark = TheoryOfMindBenchmark()
    evaluations = []

    # Evaluate each model
    for model_name in model_list:
        print(f"\nEvaluating: {model_name}")
        try:
            if model_name in PRESET_MODELS:
                model_interface = get_preset_model(model_name)
            else:
                model_interface = create_model_interface(model_name)

            evaluation = benchmark.evaluate_model(
                model_response_fn=model_interface,
                model_name=model_name
            )
            evaluations.append(evaluation)

            print(f"✓ {model_name}: {evaluation.overall_accuracy:.1%} accuracy")

        except Exception as e:
            print(f"✗ {model_name}: Error - {e}")

    if not evaluations:
        print("No successful evaluations completed.")
        return

    # Generate comparative visualizations
    print(f"\nGenerating comparative analysis...")
    dashboard = ToMVisualizationDashboard()

    # Create comprehensive report
    report_path = dashboard.create_comprehensive_report(
        evaluations,
        output_dir="comparative_analysis"
    )

    print(f"Comprehensive report generated: {report_path}")

    # Print comparative summary
    print("\nComparative Summary:")
    print("-" * 40)
    for eval in sorted(evaluations, key=lambda x: x.overall_accuracy, reverse=True):
        print(f"{eval.model_name:20s}: {eval.overall_accuracy:.1%}")


def run_demo_with_mock_models():
    """Run demonstration with mock models (no API keys required)."""
    print("Running Theory of Mind Benchmark Demo")
    print("(Using mock models - no API keys required)")
    print("=" * 60)

    mock_models = ["mock-random", "mock-correct"]
    run_comparative_analysis(mock_models)


def print_evaluation_summary(evaluation):
    """Print formatted evaluation summary."""
    print("\nEVALUATION RESULTS:")
    print("-" * 40)
    print(f"Model: {evaluation.model_name}")
    print(f"Overall Accuracy: {evaluation.overall_accuracy:.1%}")
    print()

    print("Performance by Question Type:")
    for qtype, accuracy in evaluation.accuracy_by_type.items():
        print(f"  {qtype.value:20s}: {accuracy:.1%}")
    print()

    print("Performance by Difficulty:")
    for difficulty, accuracy in evaluation.accuracy_by_difficulty.items():
        print(f"  {difficulty.value:20s}: {accuracy:.1%}")
    print()

    print("Clinical Population Comparison:")
    for population, comparison in evaluation.clinical_comparison.items():
        if comparison['questions_with_data'] > 0:
            print(f"  {population.value:20s}: {comparison['performance_difference']:+.1%} vs baseline")


def print_error_analysis(error_analysis):
    """Print formatted error analysis."""
    print("\nERROR ANALYSIS:")
    print("-" * 40)
    print(f"Total Errors: {error_analysis['total_errors']}")
    print(f"Error Rate: {error_analysis['error_rate']:.1%}")
    print()

    print("Errors by Question Type:")
    for qtype, data in error_analysis['errors_by_type'].items():
        print(f"  {qtype:20s}: {data['error_rate']:.1%} ({data['count']}/{data['total_type_questions']})")
    print()

    print("Errors by Difficulty:")
    for difficulty, data in error_analysis['errors_by_difficulty'].items():
        print(f"  {difficulty:20s}: {data['error_rate']:.1%} ({data['count']}/{data['total_difficulty_questions']})")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description="Theory of Mind Benchmark Evaluation")

    parser.add_argument("--model", type=str, help="Single model to evaluate")
    parser.add_argument("--provider", type=str, default="auto",
                       choices=["auto", "openai", "anthropic", "huggingface", "mock"],
                       help="API provider")
    parser.add_argument("--compare", nargs="+", help="List of models to compare")
    parser.add_argument("--demo", action="store_true", help="Run demo with mock models")
    parser.add_argument("--list-models", action="store_true", help="List available preset models")

    args = parser.parse_args()

    if args.list_models:
        print("Available preset models:")
        for model_name in PRESET_MODELS:
            print(f"  {model_name}")
        return

    if args.demo:
        run_demo_with_mock_models()
    elif args.model:
        run_single_model_evaluation(args.model, args.provider)
    elif args.compare:
        run_comparative_analysis(args.compare)
    else:
        print("Please specify --model, --compare, --demo, or --list-models")
        print("Examples:")
        print("  python run_benchmark.py --demo")
        print("  python run_benchmark.py --model gpt-4")
        print("  python run_benchmark.py --compare gpt-4 claude-3-sonnet")
        print("  python run_benchmark.py --list-models")


if __name__ == "__main__":
    main()