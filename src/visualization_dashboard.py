"""
Visualization Dashboard for Theory of Mind Benchmark Results

This module provides comprehensive visualization tools for analyzing ToM benchmark
results, including capability emergence patterns, clinical comparisons, and
error analysis visualizations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

from tom_benchmark import ToMEvaluation, QuestionType, Difficulty, ClinicalPopulation


class ToMVisualizationDashboard:
    """
    Comprehensive visualization dashboard for Theory of Mind benchmark results.

    Provides interactive and static visualizations for:
    - Overall performance analysis
    - Capability emergence across model scales
    - Clinical population comparisons
    - Error analysis and failure modes
    - Question type and difficulty breakdowns
    """

    def __init__(self, style: str = "whitegrid"):
        """
        Initialize the visualization dashboard.

        Args:
            style: Seaborn style for static plots
        """
        sns.set_style(style)
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 11

        # Color palettes
        self.model_colors = px.colors.qualitative.Set1
        self.clinical_colors = {
            ClinicalPopulation.NBD: '#2E86AB',
            ClinicalPopulation.DOM: '#A23B72',
            ClinicalPopulation.PHEN: '#F18F01',
            ClinicalPopulation.ASD: '#C73E1D'
        }

    def plot_overall_performance(self, evaluations: List[ToMEvaluation],
                               save_path: Optional[str] = None) -> go.Figure:
        """
        Create overall performance comparison across models.

        Args:
            evaluations: List of model evaluations
            save_path: Optional path to save the plot

        Returns:
            Plotly figure object
        """
        # Prepare data
        models = [eval.model_name for eval in evaluations]
        accuracies = [eval.overall_accuracy for eval in evaluations]

        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=models,
                y=accuracies,
                text=[f"{acc:.1%}" for acc in accuracies],
                textposition='auto',
                marker=dict(
                    color=self.model_colors[:len(models)],
                    line=dict(color='black', width=1)
                )
            )
        ])

        fig.update_layout(
            title="Theory of Mind Benchmark - Overall Performance",
            xaxis_title="Model",
            yaxis_title="Accuracy",
            yaxis=dict(range=[0, 1], tickformat='.0%'),
            template="plotly_white",
            font=dict(size=12),
            height=500
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_performance_by_category(self, evaluations: List[ToMEvaluation],
                                   category: str = "type",
                                   save_path: Optional[str] = None) -> go.Figure:
        """
        Create performance breakdown by question type or difficulty.

        Args:
            evaluations: List of model evaluations
            category: "type" or "difficulty"
            save_path: Optional path to save the plot

        Returns:
            Plotly figure object
        """
        # Prepare data
        data = []
        for eval in evaluations:
            if category == "type":
                for qtype, accuracy in eval.accuracy_by_type.items():
                    data.append({
                        'Model': eval.model_name,
                        'Category': qtype.value,
                        'Accuracy': accuracy
                    })
            elif category == "difficulty":
                for difficulty, accuracy in eval.accuracy_by_difficulty.items():
                    data.append({
                        'Model': eval.model_name,
                        'Category': difficulty.value,
                        'Accuracy': accuracy
                    })

        df = pd.DataFrame(data)

        # Create grouped bar chart
        fig = px.bar(
            df,
            x='Category',
            y='Accuracy',
            color='Model',
            barmode='group',
            title=f"Performance by Question {category.title()}",
            color_discrete_sequence=self.model_colors
        )

        fig.update_layout(
            yaxis=dict(range=[0, 1], tickformat='.0%'),
            template="plotly_white",
            font=dict(size=12),
            height=500
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_clinical_comparison(self, evaluations: List[ToMEvaluation],
                               save_path: Optional[str] = None) -> go.Figure:
        """
        Create comparison with clinical population baselines.

        Args:
            evaluations: List of model evaluations
            save_path: Optional path to save the plot

        Returns:
            Plotly figure object
        """
        # Create subplots for each clinical population
        populations = list(ClinicalPopulation)
        n_populations = len(populations)

        fig = make_subplots(
            rows=(n_populations + 1) // 2,
            cols=2,
            subplot_titles=[pop.value for pop in populations],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        for idx, population in enumerate(populations):
            row = idx // 2 + 1
            col = idx % 2 + 1

            # Prepare data for this population
            models = []
            model_accuracies = []
            clinical_baselines = []

            for eval in evaluations:
                if population in eval.clinical_comparison:
                    comp_data = eval.clinical_comparison[population]
                    if comp_data['questions_with_data'] > 0:
                        models.append(eval.model_name)
                        model_accuracies.append(comp_data['model_accuracy'])
                        clinical_baselines.append(comp_data['clinical_baseline'])

            if models:
                # Add model performance bars
                fig.add_trace(
                    go.Bar(
                        x=models,
                        y=model_accuracies,
                        name=f'Model Performance',
                        marker_color='lightblue',
                        showlegend=(idx == 0)
                    ),
                    row=row, col=col
                )

                # Add clinical baseline line
                fig.add_trace(
                    go.Scatter(
                        x=models,
                        y=clinical_baselines,
                        mode='lines+markers',
                        name=f'Clinical Baseline',
                        line=dict(color=self.clinical_colors[population], width=3),
                        marker=dict(size=8),
                        showlegend=(idx == 0)
                    ),
                    row=row, col=col
                )

        fig.update_layout(
            title="Model Performance vs Clinical Population Baselines",
            template="plotly_white",
            font=dict(size=10),
            height=600,
            showlegend=True,
            legend=dict(x=0.02, y=0.98)
        )

        # Update y-axes to show percentages
        for i in range(1, (n_populations + 1) // 2 + 1):
            for j in range(1, 3):
                fig.update_yaxes(tickformat='.0%', row=i, col=j)

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_capability_emergence(self, evaluations: List[ToMEvaluation],
                                model_sizes: Optional[List[float]] = None,
                                save_path: Optional[str] = None) -> go.Figure:
        """
        Plot capability emergence across model scales.

        Args:
            evaluations: List of model evaluations (ordered by model size)
            model_sizes: Optional list of model parameter counts (in billions)
            save_path: Optional path to save the plot

        Returns:
            Plotly figure object
        """
        if model_sizes is None:
            # Use model indices if sizes not provided
            model_sizes = list(range(len(evaluations)))
            x_title = "Model Index"
        else:
            x_title = "Model Size (B Parameters)"

        # Overall accuracy emergence
        overall_accuracies = [eval.overall_accuracy for eval in evaluations]

        fig = go.Figure()

        # Add overall accuracy line
        fig.add_trace(
            go.Scatter(
                x=model_sizes,
                y=overall_accuracies,
                mode='lines+markers',
                name='Overall Accuracy',
                line=dict(width=3, color='black'),
                marker=dict(size=8)
            )
        )

        # Add lines for each question type
        colors = px.colors.qualitative.Set2
        for idx, qtype in enumerate(QuestionType):
            qtype_accuracies = []
            for eval in evaluations:
                if qtype in eval.accuracy_by_type:
                    qtype_accuracies.append(eval.accuracy_by_type[qtype])
                else:
                    qtype_accuracies.append(0)

            fig.add_trace(
                go.Scatter(
                    x=model_sizes,
                    y=qtype_accuracies,
                    mode='lines+markers',
                    name=qtype.value,
                    line=dict(width=2, color=colors[idx % len(colors)]),
                    marker=dict(size=6)
                )
            )

        fig.update_layout(
            title="Theory of Mind Capability Emergence Across Model Scales",
            xaxis_title=x_title,
            yaxis_title="Accuracy",
            yaxis=dict(range=[0, 1], tickformat='.0%'),
            template="plotly_white",
            font=dict(size=12),
            height=500,
            legend=dict(x=0.02, y=0.98)
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_error_analysis_heatmap(self, evaluation: ToMEvaluation,
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Create error analysis heatmap showing failure patterns.

        Args:
            evaluation: Single model evaluation
            save_path: Optional path to save the plot

        Returns:
            Matplotlib figure object
        """
        # Create error matrix: question type vs difficulty
        error_matrix = np.zeros((len(QuestionType), len(Difficulty)))
        total_matrix = np.zeros((len(QuestionType), len(Difficulty)))

        qtype_to_idx = {qtype: idx for idx, qtype in enumerate(QuestionType)}
        diff_to_idx = {diff: idx for idx, diff in enumerate(Difficulty)}

        # Count errors and totals
        for result in evaluation.results:
            qtype_idx = qtype_to_idx[result.question.question_type]
            diff_idx = diff_to_idx[result.question.difficulty]

            total_matrix[qtype_idx, diff_idx] += 1
            if not result.is_correct:
                error_matrix[qtype_idx, diff_idx] += 1

        # Calculate error rates
        with np.errstate(divide='ignore', invalid='ignore'):
            error_rate_matrix = np.divide(error_matrix, total_matrix)
            error_rate_matrix = np.nan_to_num(error_rate_matrix)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            error_rate_matrix,
            annot=True,
            fmt='.2f',
            cmap='Reds',
            xticklabels=[diff.value for diff in Difficulty],
            yticklabels=[qtype.value for qtype in QuestionType],
            cbar_kws={'label': 'Error Rate'},
            ax=ax
        )

        ax.set_title(f'Error Rate Heatmap - {evaluation.model_name}')
        ax.set_xlabel('Difficulty')
        ax.set_ylabel('Question Type')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_response_distribution(self, evaluation: ToMEvaluation,
                                 save_path: Optional[str] = None) -> go.Figure:
        """
        Plot distribution of model responses (A-E choices).

        Args:
            evaluation: Single model evaluation
            save_path: Optional path to save the plot

        Returns:
            Plotly figure object
        """
        # Count response distribution
        response_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'No Answer': 0}

        for result in evaluation.results:
            if result.extracted_answer in response_counts:
                response_counts[result.extracted_answer] += 1
            else:
                response_counts['No Answer'] += 1

        # Expected uniform distribution
        total_responses = len(evaluation.results)
        expected_per_choice = total_responses / 5  # Assuming 5 choices A-E

        choices = list(response_counts.keys())
        actual_counts = list(response_counts.values())
        expected_counts = [expected_per_choice] * len(choices)

        fig = go.Figure(data=[
            go.Bar(
                x=choices,
                y=actual_counts,
                name='Actual',
                marker_color='lightblue'
            ),
            go.Bar(
                x=choices[:5],  # Only A-E for expected
                y=expected_counts[:5],
                name='Expected (Uniform)',
                marker_color='red',
                opacity=0.6
            )
        ])

        fig.update_layout(
            title=f'Response Distribution - {evaluation.model_name}',
            xaxis_title='Response Choice',
            yaxis_title='Count',
            barmode='group',
            template="plotly_white",
            font=dict(size=12),
            height=400
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def create_comprehensive_report(self, evaluations: List[ToMEvaluation],
                                  output_dir: str = "tom_analysis_report",
                                  model_sizes: Optional[List[float]] = None) -> str:
        """
        Generate comprehensive HTML report with all visualizations.

        Args:
            evaluations: List of model evaluations
            output_dir: Directory to save report and plots
            model_sizes: Optional model sizes for emergence analysis

        Returns:
            Path to generated HTML report
        """
        import os
        from pathlib import Path

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Generate all plots
        plots = {}

        # Overall performance
        fig = self.plot_overall_performance(evaluations)
        plot_path = output_path / "overall_performance.html"
        fig.write_html(plot_path)
        plots['overall'] = plot_path.name

        # Performance by type
        fig = self.plot_performance_by_category(evaluations, "type")
        plot_path = output_path / "performance_by_type.html"
        fig.write_html(plot_path)
        plots['by_type'] = plot_path.name

        # Performance by difficulty
        fig = self.plot_performance_by_category(evaluations, "difficulty")
        plot_path = output_path / "performance_by_difficulty.html"
        fig.write_html(plot_path)
        plots['by_difficulty'] = plot_path.name

        # Clinical comparison
        fig = self.plot_clinical_comparison(evaluations)
        plot_path = output_path / "clinical_comparison.html"
        fig.write_html(plot_path)
        plots['clinical'] = plot_path.name

        # Capability emergence
        if len(evaluations) > 1:
            fig = self.plot_capability_emergence(evaluations, model_sizes)
            plot_path = output_path / "capability_emergence.html"
            fig.write_html(plot_path)
            plots['emergence'] = plot_path.name

        # Individual model analyses
        for eval in evaluations:
            # Error heatmap
            fig = self.plot_error_analysis_heatmap(eval)
            plot_path = output_path / f"error_heatmap_{eval.model_name.replace('/', '_')}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            # Response distribution
            fig = self.plot_response_distribution(eval)
            plot_path = output_path / f"response_dist_{eval.model_name.replace('/', '_')}.html"
            fig.write_html(plot_path)

        # Generate HTML report
        html_content = self._generate_html_report(evaluations, plots, output_dir)

        report_path = output_path / "tom_benchmark_report.html"
        with open(report_path, 'w') as f:
            f.write(html_content)

        return str(report_path)

    def _generate_html_report(self, evaluations: List[ToMEvaluation],
                            plots: Dict[str, str], output_dir: str) -> str:
        """Generate HTML report content."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Theory of Mind Benchmark Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; margin-bottom: 40px; }}
                .section {{ margin: 30px 0; }}
                .plot-container {{ text-align: center; margin: 20px 0; }}
                .metrics-table {{ border-collapse: collapse; width: 100%; }}
                .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
                .metrics-table th {{ background-color: #f2f2f2; }}
                .summary-box {{ background-color: #f9f9f9; padding: 20px; border-radius: 5px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Theory of Mind Benchmark Analysis Report</h1>
                <p>Comprehensive evaluation of {len(evaluations)} model(s) on ToM capabilities</p>
            </div>

            <div class="section">
                <h2>Executive Summary</h2>
                <div class="summary-box">
                    <p><strong>Models Evaluated:</strong> {', '.join([e.model_name for e in evaluations])}</p>
                    <p><strong>Best Overall Performance:</strong> {max(evaluations, key=lambda x: x.overall_accuracy).model_name} ({max(evaluations, key=lambda x: x.overall_accuracy).overall_accuracy:.1%})</p>
                    <p><strong>Total Questions:</strong> {len(evaluations[0].results) if evaluations else 0}</p>
                </div>
            </div>

            <div class="section">
                <h2>Overall Performance Comparison</h2>
                <div class="plot-container">
                    <iframe src="{plots.get('overall', '')}" width="100%" height="520" frameborder="0"></iframe>
                </div>
            </div>

            <div class="section">
                <h2>Performance by Question Type</h2>
                <div class="plot-container">
                    <iframe src="{plots.get('by_type', '')}" width="100%" height="520" frameborder="0"></iframe>
                </div>
            </div>

            <div class="section">
                <h2>Performance by Difficulty</h2>
                <div class="plot-container">
                    <iframe src="{plots.get('by_difficulty', '')}" width="100%" height="520" frameborder="0"></iframe>
                </div>
            </div>

            <div class="section">
                <h2>Clinical Population Comparison</h2>
                <div class="plot-container">
                    <iframe src="{plots.get('clinical', '')}" width="100%" height="620" frameborder="0"></iframe>
                </div>
            </div>

            {'<div class="section"><h2>Capability Emergence</h2><div class="plot-container"><iframe src="' + plots.get('emergence', '') + '" width="100%" height="520" frameborder="0"></iframe></div></div>' if 'emergence' in plots else ''}

            <div class="section">
                <h2>Detailed Metrics</h2>
                {self._generate_metrics_table(evaluations)}
            </div>

            <div class="section">
                <h2>Generated Files</h2>
                <p>All plots and data files have been saved to: <code>{output_dir}</code></p>
                <ul>
                    <li>Interactive plots: *.html files</li>
                    <li>Static plots: *.png files</li>
                    <li>Raw data: JSON evaluation files</li>
                </ul>
            </div>

        </body>
        </html>
        """
        return html

    def _generate_metrics_table(self, evaluations: List[ToMEvaluation]) -> str:
        """Generate HTML table of detailed metrics."""
        html = '<table class="metrics-table"><tr><th>Model</th><th>Overall Accuracy</th>'

        # Add question type columns
        for qtype in QuestionType:
            html += f'<th>{qtype.value}</th>'

        # Add difficulty columns
        for difficulty in Difficulty:
            html += f'<th>{difficulty.value}</th>'

        html += '</tr>'

        # Add data rows
        for eval in evaluations:
            html += f'<tr><td>{eval.model_name}</td><td>{eval.overall_accuracy:.1%}</td>'

            # Question type accuracies
            for qtype in QuestionType:
                acc = eval.accuracy_by_type.get(qtype, 0)
                html += f'<td>{acc:.1%}</td>'

            # Difficulty accuracies
            for difficulty in Difficulty:
                acc = eval.accuracy_by_difficulty.get(difficulty, 0)
                html += f'<td>{acc:.1%}</td>'

            html += '</tr>'

        html += '</table>'
        return html