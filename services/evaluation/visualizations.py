"""
Visualization utilities for evaluation results.

Generates charts, plots, and HTML reports for translation evaluation metrics.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def create_metrics_comparison_chart(
    results: Dict[str, float],
    output_path: Path,
    title: str = "Metrics Comparison",
):
    """
    Create a bar chart comparing different metrics.
    
    Args:
        results: Dictionary mapping metric name to score
        output_path: Path to save the chart
        title: Chart title
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics = list(results.keys())
        scores = list(results.values())
        
        bars = ax.bar(metrics, scores, color=sns.color_palette("husl", len(metrics)))
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=10)
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(scores) * 1.2 if scores else 1)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved metrics comparison chart to {output_path}")
    
    except Exception as e:
        logger.error(f"Error creating metrics comparison chart: {e}")


def create_score_distribution(
    scores: List[float],
    metric_name: str,
    output_path: Path,
    title: Optional[str] = None,
):
    """
    Create a distribution plot (histogram + box plot) for a metric.
    
    Args:
        scores: List of scores
        metric_name: Name of the metric
        output_path: Path to save the plot
        title: Optional title
    """
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), 
                                        gridspec_kw={'height_ratios': [3, 1]})
        
        # Histogram
        ax1.hist(scores, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.axvline(np.mean(scores), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(scores):.2f}')
        ax1.axvline(np.median(scores), color='green', linestyle='--', 
                    label=f'Median: {np.median(scores):.2f}')
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.legend()
        ax1.set_title(title or f'{metric_name} Distribution', 
                      fontsize=13, fontweight='bold')
        
        # Box plot
        ax2.boxplot(scores, vert=False, widths=0.6)
        ax2.set_xlabel(f'{metric_name} Score', fontsize=11)
        ax2.set_yticks([])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved {metric_name} distribution to {output_path}")
    
    except Exception as e:
        logger.error(f"Error creating score distribution: {e}")


def create_per_sample_heatmap(
    sample_scores: pd.DataFrame,
    output_path: Path,
    title: str = "Per-Sample Metric Scores",
):
    """
    Create a heatmap showing metric scores for each sample.
    
    Args:
        sample_scores: DataFrame with samples as rows and metrics as columns
        output_path: Path to save the heatmap
        title: Chart title
    """
    try:
        fig, ax = plt.subplots(figsize=(12, max(8, len(sample_scores) * 0.3)))
        
        # Normalize scores to 0-1 for better visualization
        normalized = sample_scores.copy()
        for col in sample_scores.columns:
            col_min = sample_scores[col].min()
            col_max = sample_scores[col].max()
            if col_max > col_min:
                normalized[col] = (sample_scores[col] - col_min) / (col_max - col_min)
        
        sns.heatmap(normalized, annot=sample_scores, fmt='.2f', 
                   cmap='RdYlGn', cbar_kws={'label': 'Normalized Score'},
                   linewidths=0.5, ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel('Sample Index', fontsize=11)
        ax.set_xlabel('Metric', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved per-sample heatmap to {output_path}")
    
    except Exception as e:
        logger.error(f"Error creating heatmap: {e}")


def create_metrics_table_image(
    summary: Dict,
    output_path: Path,
):
    """
    Create a formatted table image with metric scores.
    
    Args:
        summary: Summary dictionary with aggregate scores
        output_path: Path to save the table image
    """
    try:
        # Prepare data
        data = []
        for metric, value in summary.get('aggregate_scores', {}).items():
            stats = summary.get('score_statistics', {}).get(metric, {})
            data.append([
                metric.upper(),
                f"{value:.3f}",
                f"{stats.get('mean', 0):.3f}",
                f"{stats.get('std', 0):.3f}",
                f"{stats.get('min', 0):.3f}",
                f"{stats.get('max', 0):.3f}",
            ])
        
        columns = ['Metric', 'Corpus', 'Mean', 'Std', 'Min', 'Max']
        
        fig, ax = plt.subplots(figsize=(10, max(4, len(data) * 0.4)))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=data, colLabels=columns, 
                        cellLoc='center', loc='center',
                        colWidths=[0.15, 0.17, 0.17, 0.17, 0.17, 0.17])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(data) + 1):
            for j in range(len(columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved metrics table to {output_path}")
    
    except Exception as e:
        logger.error(f"Error creating metrics table: {e}")


def generate_html_report(
    summary: Dict,
    visualization_dir: Path,
    output_path: Path,
    title: str = "Translation Evaluation Report",
):
    """
    Generate an interactive HTML report.
    
    Args:
        summary: Summary dictionary with all results
        visualization_dir: Directory containing visualization images
        output_path: Path to save HTML report
        title: Report title
    """
    try:
        from jinja2 import Template
        from datetime import datetime
        
        # HTML template
        template_str = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background-color: #2196F3;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .section {
            background-color: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1, h2 {
            margin-top: 0;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .metric-card {
            background-color: #e3f2fd;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }
        .metric-name {
            font-size: 14px;
            color: #666;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 28px;
            font-weight: bold;
            color: #1976D2;
        }
        .visualization {
            text-align: center;
            margin: 20px 0;
        }
        .visualization img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #4CAF50;
            color: white;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .footer {
            text-align: center;
            color: #666;
            font-size: 12px;
            margin-top: 30px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ title }}</h1>
        <p>Run ID: {{ run_id }} | Generated: {{ timestamp }}</p>
        <p>Translation Type: {{ translation_type }} | Language Pair: {{ language_pair }}</p>
        <p>Total Samples: {{ total_samples }}</p>
    </div>
    
    <div class="section">
        <h2>Aggregate Scores</h2>
        <div class="metrics-grid">
            {% for metric, score in aggregate_scores.items() %}
            <div class="metric-card">
                <div class="metric-name">{{ metric.upper() }}</div>
                <div class="metric-value">{{ "%.2f"|format(score) }}</div>
            </div>
            {% endfor %}
        </div>
    </div>
    
    <div class="section">
        <h2>Visualizations</h2>
        
        {% if vis_files.comparison %}
        <div class="visualization">
            <h3>Metrics Comparison</h3>
            <img src="{{ vis_files.comparison }}" alt="Metrics Comparison">
        </div>
        {% endif %}
        
        {% if vis_files.table %}
        <div class="visualization">
            <h3>Metrics Summary Table</h3>
            <img src="{{ vis_files.table }}" alt="Metrics Table">
        </div>
        {% endif %}
        
        {% for metric, path in vis_files.distributions.items() %}
        <div class="visualization">
            <h3>{{ metric.upper() }} Distribution</h3>
            <img src="{{ path }}" alt="{{ metric }} Distribution">
        </div>
        {% endfor %}
        
        {% if vis_files.heatmap %}
        <div class="visualization">
            <h3>Per-Sample Scores</h3>
            <img src="{{ vis_files.heatmap }}" alt="Per-Sample Heatmap">
        </div>
        {% endif %}
    </div>
    
    <div class="section">
        <h2>Score Statistics</h2>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Mean</th>
                    <th>Std Dev</th>
                    <th>Min</th>
                    <th>Max</th>
                </tr>
            </thead>
            <tbody>
                {% for metric, stats in score_statistics.items() %}
                <tr>
                    <td>{{ metric.upper() }}</td>
                    <td>{{ "%.3f"|format(stats.mean) }}</td>
                    <td>{{ "%.3f"|format(stats.std) }}</td>
                    <td>{{ "%.3f"|format(stats.min) }}</td>
                    <td>{{ "%.3f"|format(stats.max) }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
    
    <div class="footer">
        <p>Generated by Translation Evaluation System</p>
    </div>
</body>
</html>
"""
        
        # Collect visualization files
        vis_files = {
            'comparison': None,
            'table': None,
            'distributions': {},
            'heatmap': None,
        }
        
        for f in visualization_dir.glob('*.png'):
            rel_path = f.name
            if 'comparison' in f.name:
                vis_files['comparison'] = rel_path
            elif 'table' in f.name:
                vis_files['table'] = rel_path
            elif 'distribution' in f.name:
                metric = f.stem.split('_')[0]
                vis_files['distributions'][metric] = rel_path
            elif 'heatmap' in f.name:
                vis_files['heatmap'] = rel_path
        
        # Render template
        template = Template(template_str)
        html = template.render(
            title=title,
            run_id=summary.get('run_id', 'unknown'),
            timestamp=summary.get('timestamp', datetime.now().isoformat()),
            translation_type=summary.get('translation_type', 'unknown'),
            language_pair=summary.get('language_pair', 'unknown'),
            total_samples=summary.get('total_samples', 0),
            aggregate_scores=summary.get('aggregate_scores', {}),
            score_statistics=summary.get('score_statistics', {}),
            vis_files=vis_files,
        )
        
        output_path.write_text(html, encoding='utf-8')
        logger.info(f"Saved HTML report to {output_path}")
    
    except Exception as e:
        logger.error(f"Error generating HTML report: {e}")


def generate_all_visualizations(
    summary: Dict,
    detailed_results: pd.DataFrame,
    output_dir: Path,
):
    """
    Generate all visualizations for evaluation results.
    
    Args:
        summary: Summary dictionary
        detailed_results: DataFrame with per-sample results
        output_dir: Directory to save visualizations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Metrics comparison
    if summary.get('aggregate_scores'):
        create_metrics_comparison_chart(
            summary['aggregate_scores'],
            output_dir / 'metrics_comparison.png',
        )
    
    # 2. Metrics table
    create_metrics_table_image(summary, output_dir / 'metrics_table.png')
    
    # 3. Score distributions
    for metric in summary.get('metrics_computed', []):
        if f'{metric}_score' in detailed_results.columns:
            scores = detailed_results[f'{metric}_score'].dropna().tolist()
            if scores:
                create_score_distribution(
                    scores,
                    metric.upper(),
                    output_dir / f'{metric}_distribution.png',
                )
    
    # 4. Per-sample heatmap (limit to first 50 samples for readability)
    score_columns = [col for col in detailed_results.columns if col.endswith('_score')]
    if score_columns:
        sample_scores = detailed_results[score_columns].head(50).copy()
        sample_scores.columns = [col.replace('_score', '').upper() for col in sample_scores.columns]
        if not sample_scores.empty:
            create_per_sample_heatmap(
                sample_scores,
                output_dir / 'per_sample_heatmap.png',
            )
    
    logger.info(f"All visualizations saved to {output_dir}")
