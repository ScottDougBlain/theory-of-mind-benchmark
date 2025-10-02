# Theory of Mind Benchmark for LLMs

A comprehensive Theory of Mind evaluation suite for Large Language Models based on clinical psychology research. This benchmark translates gold-standard psychological assessments into AI safety evaluations, enabling systematic evaluation of mentalizing capabilities and comparison to clinical population baselines.

## Overview

Theory of Mind (ToM) - the ability to understand that others have beliefs, desires, and intentions different from one's own - is fundamental to social cognition and crucial for AI safety. This benchmark provides systematic evaluation of LLM mentalizing capabilities using psychology-inspired scenarios.

### Features

- 83 test scenarios covering multiple ToM domains
- Clinical psychology-informed question design
- Multi-API support (OpenAI, Anthropic, Hugging Face)
- Capability emergence analysis across model scales
- Interactive visualization dashboard
- Systematic error pattern detection

## 📊 Benchmark Structure

### Question Types

- **False Belief (FB)**: Classic false-belief task variations
- **Social Reasoning (SOC)**: Complex social inference scenarios
- **Pluralistic Ignorance (PI)**: Understanding when others hide true beliefs
- **Intentionality (INT)**: Goal and intention attribution
- **Emotion Recognition (EMO)**: Affective state inference
- **Sarcasm/Irony Detection (SAR/IRO)**: Non-literal language understanding

### Clinical Psychology Inspiration

The benchmark scenarios are inspired by research on Theory of Mind assessment in clinical populations, including studies of neurotypical controls, autism spectrum individuals, and cognitive impairment. Note that direct performance comparisons require careful interpretation due to methodological differences between clinical assessments and LLM evaluations.

### Difficulty Levels

- Easy: Basic false-belief tasks, direct social inference
- Medium: Multi-step reasoning, implicit mental states
- Hard: Complex social dynamics, nested beliefs

## Quick Start

### Installation

```bash
git clone https://github.com/blai90/theory-of-mind-benchmark.git
cd theory-of-mind-benchmark
pip install -r requirements.txt
```

### Basic Usage

```python
from src.tom_benchmark import TheoryOfMindBenchmark
from src.model_interfaces import create_model_interface

# Initialize benchmark
benchmark = TheoryOfMindBenchmark()

# Create model interface (requires API key)
model = create_model_interface("gpt-4", provider="openai")

# Run evaluation
evaluation = benchmark.evaluate_model(model, model_name="GPT-4")

# View results
print(f"Overall Accuracy: {evaluation.overall_accuracy:.1%}")
print(f"Performance by Type: {evaluation.accuracy_by_type}")
```

### Command Line Interface

```bash
# Run demo with mock models (no API keys required)
python examples/run_benchmark.py --demo

# Evaluate single model
python examples/run_benchmark.py --model gpt-4

# Compare multiple models
python examples/run_benchmark.py --compare gpt-4 claude-3-sonnet llama-2-7b

# List available models
python examples/run_benchmark.py --list-models
```

## Example Results

### Model Performance Comparison

*Example results format (specific numbers depend on evaluation configuration and model versions):*

| Model | Overall | False Belief | Social Reasoning | Pluralistic Ignorance |
|-------|---------|--------------|------------------|-----------------------|
| GPT-4 | -- | -- | -- | -- |
| Claude-3-Sonnet | -- | -- | -- | -- |
| LLaMA-2-70B | -- | -- | -- | -- |

Run your own evaluations to generate performance metrics.

## Methodology

### Question Format

Each ToM scenario follows a standardized structure:

```
**Scenario:**
[Detailed social situation with mental state information]

**Question:**
[Specific query about beliefs, intentions, or social dynamics]

**Options:**
A. [Response option 1]
B. [Response option 2]
C. [Response option 3]
D. [Response option 4]
E. [Response option 5]
```

### Evaluation Metrics

- **Overall Accuracy**: Percentage of correct responses across all questions
- **Type-Specific Accuracy**: Performance breakdown by ToM question type
- **Difficulty Analysis**: Accuracy distribution across easy/medium/hard questions
- **Clinical Comparison**: Performance difference vs. clinical population baselines
- **Error Pattern Analysis**: Systematic failure mode identification

### Scoring System

Models are evaluated on:
1. **Answer Accuracy**: Correct multiple-choice selection
2. **Reasoning Quality**: Coherence of explanation (future enhancement)
3. **Response Consistency**: Reliability across similar scenarios
4. **Clinical Calibration**: Alignment with human population data

## Technical Architecture

### Core Components

#### `TheoryOfMindBenchmark`
Main benchmark class handling data loading, evaluation orchestration, and results analysis.

```python
# Load benchmark data
benchmark = TheoryOfMindBenchmark()

# Get benchmark statistics
stats = benchmark.get_benchmark_statistics()
print(f"Total questions: {stats['total_questions']}")

# Evaluate specific question subset
fb_questions = benchmark.get_questions_by_type(QuestionType.FB)
evaluation = benchmark.evaluate_model(model, question_subset=fb_questions)
```

#### `ModelInterfaces`
Standardized interfaces for different LLM providers with automatic retry, rate limiting, and error handling.

```python
from src.model_interfaces import create_model_interface, get_preset_model

# Auto-detect provider based on model name
model = create_model_interface("gpt-4")  # Automatically uses OpenAI

# Use preset configurations
model = get_preset_model("claude-3-sonnet")

# Custom configuration
config = ModelConfig(model_name="custom-model", temperature=0.1)
model = create_model_interface("custom-model", config=config)
```

#### `VisualizationDashboard`
Comprehensive visualization suite for results analysis and reporting.

```python
from src.visualization_dashboard import ToMVisualizationDashboard

dashboard = ToMVisualizationDashboard()

# Generate comprehensive report
report_path = dashboard.create_comprehensive_report(
    evaluations=[eval1, eval2, eval3],
    output_dir="analysis_results"
)
```

## Visualization Examples

### Performance Comparison
Interactive bar charts showing overall accuracy across models with breakdown by question type and difficulty.

### Clinical Population Analysis
Comparative analysis showing model performance relative to clinical baselines, highlighting areas where AI exceeds or falls short of human performance.

### Capability Emergence
Line plots tracking ToM capability development across model scales, revealing emergence thresholds and plateau points.

### Error Analysis Heatmap
2D heatmaps showing error patterns across question types and difficulty levels, identifying systematic failure modes.

## Analysis Features

### Error Pattern Detection

The benchmark automatically identifies systematic failure patterns:

```python
error_analysis = benchmark.generate_error_analysis(evaluation)

print(f"Error rate by type: {error_analysis['errors_by_type']}")
print(f"Most challenging questions: {error_analysis['most_challenging_questions']}")
```

### Clinical Comparison Analysis

Compare model performance to clinical populations:

```python
# Models exceeding neurotypical baseline
for eval in evaluations:
    nbd_comparison = eval.clinical_comparison[ClinicalPopulation.NBD]
    if nbd_comparison['performance_difference'] > 0:
        print(f"{eval.model_name} exceeds neurotypical baseline by {nbd_comparison['performance_difference']:.1%}")
```

### Capability Emergence Tracking

Track ToM emergence across model scales:

```python
model_sizes = [7, 13, 30, 65, 175]  # Billion parameters
dashboard.plot_capability_emergence(evaluations, model_sizes)
```

## Testing and Validation

### Test Suite

Comprehensive test coverage including:

```bash
# Run all tests
pytest tests/ --cov=src

# Run specific test modules
pytest tests/test_tom_benchmark.py -v
pytest tests/test_model_interfaces.py -v
pytest tests/test_visualization.py -v
```

### Mock Models for Development

Test framework functionality without API calls:

```python
from src.model_interfaces import MockModelInterface, ModelConfig

# Create mock model with specific behavior
config = ModelConfig(model_name="mock-test")
mock_model = MockModelInterface(config, response_pattern="random")

# Run evaluation
evaluation = benchmark.evaluate_model(mock_model, "Mock Model")
```

## Research Foundation

### Psychological Basis

This benchmark is grounded in decades of Theory of Mind research:

- **Baron-Cohen et al. (1985)**: Classic false-belief tasks
- **Premack & Woodruff (1978)**: Original ToM concept
- **Happé (1994)**: Advanced ToM in autism
- **Frith & Happé (1994)**: Clinical population studies

### AI Safety Relevance

Theory of Mind capabilities are crucial for AI safety because they enable:

- **Deception Detection**: Understanding when humans may not reveal true preferences
- **Intent Alignment**: Accurately modeling human goals and values
- **Social Navigation**: Appropriate behavior in complex social contexts
- **Trust Calibration**: Knowing when human judgment may be compromised

### Clinical Translation

The benchmark translates clinical assessment protocols to AI evaluation:

1. **Standardized Scenarios**: Based on validated psychological experiments
2. **Population Baselines**: Performance anchored to clinical research data
3. **Systematic Evaluation**: Consistent methodology across conditions
4. **Interpretable Results**: Meaningful comparison to human capabilities

## Configuration Options

### Model Configuration

```python
from src.model_interfaces import ModelConfig

config = ModelConfig(
    model_name="gpt-4",
    max_tokens=1000,
    temperature=0.3,  # Lower for more consistent reasoning
    top_p=1.0,
    retry_attempts=3,
    timeout=30.0
)
```

### Evaluation Settings

```python
# Evaluate subset of questions
easy_questions = benchmark.get_questions_by_difficulty(Difficulty.EASY)
evaluation = benchmark.evaluate_model(model, question_subset=easy_questions)

# Custom formatting options
evaluation = benchmark.evaluate_model(
    model,
    include_options=True,
    include_instructions=True
)
```

### Visualization Customization

```python
dashboard = ToMVisualizationDashboard(style="darkgrid")

# Custom color schemes
dashboard.model_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
dashboard.clinical_colors[ClinicalPopulation.NBD] = '#2E86AB'
```

## Contributing

We welcome contributions that enhance the benchmark's scientific rigor and practical utility:

### Research Contributions
- **New Question Types**: Additional ToM domains (e.g., emotional ToM, recursive beliefs)
- **Clinical Data**: Performance baselines from additional populations
- **Validation Studies**: Correlation with other ToM measures
- **Cross-Cultural Analysis**: Performance across different cultural contexts

### Technical Contributions
- **Model Integrations**: Support for new LLM APIs and local models
- **Evaluation Metrics**: Novel scoring approaches and analysis methods
- **Visualization Tools**: Enhanced plotting and dashboard features
- **Performance Optimization**: Faster evaluation and analysis pipelines

### Development Guidelines

```bash
# Set up development environment
git clone https://github.com/blai90/theory-of-mind-benchmark.git
cd theory-of-mind-benchmark
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run quality checks
black src/ tests/
flake8 src/ tests/
mypy src/
pytest tests/
```

## 📄 Citation

If you use this benchmark in your research, please cite:

```bibtex
@misc{theory_of_mind_benchmark_2024,
  title={Theory of Mind Benchmark for Large Language Models},
  author={Lai, Brandon},
  year={2024},
  url={https://github.com/blai90/theory-of-mind-benchmark}
}
```

## Related Work

### Academic Papers
- Baron-Cohen, S., Leslie, A. M., & Frith, U. (1985). Does the autistic child have a theory of mind?
- Premack, D., & Woodruff, G. (1978). Does the chimpanzee have a theory of mind?
- Happé, F. G. (1994). An advanced test of theory of mind

### Other ToM Benchmarks
- **ToMi**: Machine reading comprehension approach
- **Social IQa**: Commonsense reasoning about social situations
- **SWAG**: Situations with adversarial generations

### Clinical Assessments
- **Strange Stories Test**: Advanced ToM assessment
- **Reading the Mind in the Eyes**: Emotion recognition task
- **Faux Pas Test**: Social inappropriate detection

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Clinical psychology research community for foundational ToM work
- AI safety researchers highlighting the importance of social cognition
- Open source maintainers of the visualization and ML libraries used
- Beta testers and early adopters providing valuable feedback

---

## Future Directions

### Enhanced Capabilities
- **Multimodal ToM**: Integration of visual and textual social cues
- **Dynamic Assessment**: Adaptive difficulty based on performance
- **Longitudinal Tracking**: ToM development across training iterations
- **Cultural Sensitivity**: Cross-cultural validation and bias analysis

### Research Applications
- **Model Architecture Analysis**: Correlation between architecture and ToM performance
- **Training Data Effects**: Impact of social interaction data on ToM capabilities
- **Fine-tuning Studies**: Targeted improvement of ToM abilities
- **Safety Implications**: ToM failures in high-stakes scenarios

This benchmark represents a crucial step toward rigorous evaluation of AI social cognition capabilities, bridging clinical psychology research with AI safety requirements.