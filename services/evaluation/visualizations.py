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

# Quality thresholds for COMET/BLASER-based classification
QUALITY_THRESHOLDS = {
    'excellent': 0.75,
    'good': 0.60,
    'fair': 0.40,
    'poor': 0.0,
}

# Metric characteristics for normalization and interpretation
METRIC_INFO = {
    'bleu': {
        'name': 'BLEU',
        'scale': (0, 100),
        'higher_is_better': True,
        'description': 'N-gram overlap with reference',
        'good_range': (20, 40),
        'excellent_threshold': 50,
    },
    'chrf': {
        'name': 'chrF',
        'scale': (0, 100),
        'higher_is_better': True,
        'description': 'Character n-gram F-score',
        'good_range': (40, 60),
        'excellent_threshold': 70,
    },
    'comet': {
        'name': 'COMET',
        'scale': (0, 1),
        'higher_is_better': True,
        'description': 'Neural semantic quality',
        'good_range': (0.6, 0.8),
        'excellent_threshold': 0.8,
    },
    'mcd': {
        'name': 'MCD',
        'scale': (0, 20),
        'higher_is_better': False,
        'description': 'Mel-cepstral distortion (dB)',
        'good_range': (4, 6),
        'excellent_threshold': 4,
    },
    'blaser': {
        'name': 'BLASER',
        'scale': (0, 5),
        'higher_is_better': True,
        'description': 'Speech translation quality',
        'good_range': (3.5, 4.5),
        'excellent_threshold': 4.5,
    },
}


def categorize_quality(score: float, metric: str = 'comet') -> str:
    """
    Categorize translation quality based on score.

    Args:
        score: Metric score
        metric: Metric name (for threshold adjustment)

    Returns:
        Quality category: 'excellent', 'good', 'fair', or 'poor'
    """
    if metric == 'comet' or metric == 'blaser':
        if score >= QUALITY_THRESHOLDS['excellent']:
            return 'excellent'
        elif score >= QUALITY_THRESHOLDS['good']:
            return 'good'
        elif score >= QUALITY_THRESHOLDS['fair']:
            return 'fair'
        else:
            return 'poor'
    elif metric == 'mcd':
        # For MCD, lower is better
        if score <= 4:
            return 'excellent'
        elif score <= 6:
            return 'good'
        elif score <= 8:
            return 'fair'
        else:
            return 'poor'
    else:
        # BLEU/chrF
        if score >= 70:
            return 'excellent'
        elif score >= 50:
            return 'good'
        elif score >= 30:
            return 'fair'
        else:
            return 'poor'


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

        {% if vis_files.quality_dashboard %}
        <div class="visualization">
            <h3>Quality Dashboard</h3>
            <img src="{{ vis_files.quality_dashboard }}" alt="Quality Dashboard">
            <p style="color: #666; font-size: 14px; margin-top: 10px;">
                Comprehensive quality analysis showing distribution, trends, and categorization of translation quality.
            </p>
        </div>
        {% endif %}

        {% if vis_files.normalized_metrics %}
        <div class="visualization">
            <h3>Normalized Metrics Comparison</h3>
            <img src="{{ vis_files.normalized_metrics }}" alt="Normalized Metrics">
            <p style="color: #666; font-size: 14px; margin-top: 10px;">
                All metrics normalized to 0-1 scale with expected quality ranges indicated.
            </p>
        </div>
        {% endif %}

        {% if vis_files.comparison %}
        <div class="visualization">
            <h3>Metrics Comparison (Original Scale)</h3>
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
            'quality_dashboard': None,
            'normalized_metrics': None,
            'comparison': None,
            'table': None,
            'distributions': {},
            'heatmap': None,
        }

        for f in visualization_dir.glob('*.png'):
            rel_path = f.name
            if 'quality_dashboard' in f.name:
                vis_files['quality_dashboard'] = rel_path
            elif 'normalized_metrics' in f.name:
                vis_files['normalized_metrics'] = rel_path
            elif 'comparison' in f.name:
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


def create_quality_dashboard(
    df: pd.DataFrame,
    primary_metric: str,
    output_path: Path,
):
    """
    Create a comprehensive quality dashboard with categorization and insights.

    Args:
        df: DataFrame with per-sample results
        primary_metric: Primary metric for quality categorization
        output_path: Path to save the dashboard
    """
    try:
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        metric_col = f'{primary_metric}_score'
        if metric_col not in df.columns:
            logger.warning(f"Metric {metric_col} not found in data")
            return

        # Categorize samples
        df['quality_category'] = df[metric_col].apply(
            lambda x: categorize_quality(x, primary_metric)
        )

        # 1. Quality Distribution (Pie Chart)
        ax1 = fig.add_subplot(gs[0, 0])
        quality_counts = df['quality_category'].value_counts()
        colors = {'excellent': '#4CAF50', 'good': '#8BC34A',
                  'fair': '#FFC107', 'poor': '#F44336'}
        ax1.pie(quality_counts.values, labels=quality_counts.index, autopct='%1.1f%%',
                colors=[colors.get(cat, '#999') for cat in quality_counts.index],
                startangle=90)
        ax1.set_title('Quality Distribution', fontweight='bold', fontsize=12)

        # 2. Score Distribution (Histogram with quality zones)
        ax2 = fig.add_subplot(gs[0, 1:])
        scores = df[metric_col].dropna()
        ax2.hist(scores, bins=min(30, len(scores)), color='skyblue', edgecolor='black', alpha=0.7)

        # Add quality zone shading
        if primary_metric in ['comet', 'blaser']:
            ax2.axvspan(0, QUALITY_THRESHOLDS['fair'], alpha=0.1, color='red', label='Poor')
            ax2.axvspan(QUALITY_THRESHOLDS['fair'], QUALITY_THRESHOLDS['good'],
                       alpha=0.1, color='orange', label='Fair')
            ax2.axvspan(QUALITY_THRESHOLDS['good'], QUALITY_THRESHOLDS['excellent'],
                       alpha=0.1, color='lightgreen', label='Good')
            ax2.axvspan(QUALITY_THRESHOLDS['excellent'], scores.max(),
                       alpha=0.1, color='green', label='Excellent')

        ax2.axvline(scores.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {scores.mean():.3f}')
        ax2.set_xlabel(f'{METRIC_INFO.get(primary_metric, {}).get("name", primary_metric.upper())} Score', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.set_title(f'{METRIC_INFO.get(primary_metric, {}).get("name", primary_metric.upper())} Distribution',
                     fontweight='bold', fontsize=12)
        ax2.legend(fontsize=8)

        # 3. Statistics Table
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.axis('off')
        stats_data = [
            ['Total Samples', f"{len(df)}"],
            ['Mean Score', f"{scores.mean():.3f}"],
            ['Median Score', f"{scores.median():.3f}"],
            ['Std Dev', f"{scores.std():.3f}"],
            ['Min Score', f"{scores.min():.3f}"],
            ['Max Score', f"{scores.max():.3f}"],
            ['Excellent', f"{quality_counts.get('excellent', 0)} ({quality_counts.get('excellent', 0)/len(df)*100:.1f}%)"],
            ['Good', f"{quality_counts.get('good', 0)} ({quality_counts.get('good', 0)/len(df)*100:.1f}%)"],
            ['Fair', f"{quality_counts.get('fair', 0)} ({quality_counts.get('fair', 0)/len(df)*100:.1f}%)"],
            ['Poor', f"{quality_counts.get('poor', 0)} ({quality_counts.get('poor', 0)/len(df)*100:.1f}%)"],
        ]
        table = ax3.table(cellText=stats_data, cellLoc='left', loc='center',
                         colWidths=[0.5, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        ax3.set_title('Statistics', fontweight='bold', fontsize=12, pad=20)

        # 4. All Metrics Comparison (if available)
        ax4 = fig.add_subplot(gs[1, 1:])
        metric_cols = [col for col in df.columns if col.endswith('_score')]
        if metric_cols:
            metric_data = []
            metric_names = []
            for col in metric_cols:
                metric_name = col.replace('_score', '')
                if metric_name in METRIC_INFO:
                    scores_norm = df[col].dropna()
                    if METRIC_INFO[metric_name]['higher_is_better']:
                        scale_max = METRIC_INFO[metric_name]['scale'][1]
                        normalized = scores_norm / scale_max
                    else:
                        scale_max = METRIC_INFO[metric_name]['scale'][1]
                        normalized = 1 - (scores_norm / scale_max)

                    metric_data.append(normalized.tolist())
                    metric_names.append(METRIC_INFO[metric_name]['name'])

            if metric_data:
                bp = ax4.boxplot(metric_data, labels=metric_names, patch_artist=True)
                for patch, color in zip(bp['boxes'], sns.color_palette("husl", len(metric_data))):
                    patch.set_facecolor(color)
                ax4.set_ylabel('Normalized Score (0-1)', fontsize=11)
                ax4.set_title('All Metrics Comparison (Normalized)', fontweight='bold', fontsize=12)
                ax4.tick_params(axis='x', rotation=45)

        # 5. Score Trend (if uuid or index available)
        ax5 = fig.add_subplot(gs[2, :])
        sample_indices = range(len(df))
        ax5.plot(sample_indices, df[metric_col], marker='o', linestyle='-',
                linewidth=1, markersize=4, alpha=0.7)
        ax5.axhline(scores.mean(), color='red', linestyle='--', linewidth=1.5,
                   label=f'Mean: {scores.mean():.3f}', alpha=0.7)

        # Color-code points by quality
        for idx, (score, quality) in enumerate(zip(df[metric_col], df['quality_category'])):
            ax5.scatter(idx, score, c=colors.get(quality, '#999'),
                       s=50, alpha=0.6, edgecolors='black', linewidth=0.5)

        ax5.set_xlabel('Sample Index', fontsize=11)
        ax5.set_ylabel(f'{METRIC_INFO.get(primary_metric, {}).get("name", primary_metric.upper())} Score', fontsize=11)
        ax5.set_title('Score Trend Across Samples', fontweight='bold', fontsize=12)
        ax5.legend(fontsize=9)
        ax5.grid(alpha=0.3)

        plt.suptitle(f'Translation Quality Dashboard - {METRIC_INFO.get(primary_metric, {}).get("name", primary_metric.upper())}',
                    fontsize=16, fontweight='bold', y=0.98)

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved quality dashboard to {output_path}")

    except Exception as e:
        logger.error(f"Error creating quality dashboard: {e}", exc_info=True)


def create_normalized_metrics_comparison(
    summary: Dict,
    output_path: Path,
):
    """
    Create a normalized comparison chart showing all metrics on a 0-1 scale
    with context about what "good" means for each metric.

    Args:
        summary: Summary dictionary with aggregate scores
        output_path: Path to save the chart
    """
    try:
        fig, ax = plt.subplots(figsize=(12, 7))

        metrics_data = []
        for metric, score in summary.get('aggregate_scores', {}).items():
            if metric in METRIC_INFO:
                info = METRIC_INFO[metric]

                # Normalize score to 0-1
                if info['higher_is_better']:
                    normalized = score / info['scale'][1]
                else:
                    # For MCD, invert and normalize
                    normalized = 1 - (score / info['scale'][1])

                # Normalize good range
                if info['higher_is_better']:
                    good_min = info['good_range'][0] / info['scale'][1]
                    good_max = info['good_range'][1] / info['scale'][1]
                else:
                    good_min = 1 - (info['good_range'][1] / info['scale'][1])
                    good_max = 1 - (info['good_range'][0] / info['scale'][1])

                metrics_data.append({
                    'metric': info['name'],
                    'normalized_score': normalized,
                    'original_score': score,
                    'good_min': good_min,
                    'good_max': good_max,
                    'description': info['description'],
                })

        if not metrics_data:
            return

        # Sort by normalized score
        metrics_data = sorted(metrics_data, key=lambda x: x['normalized_score'], reverse=True)

        y_pos = np.arange(len(metrics_data))
        normalized_scores = [m['normalized_score'] for m in metrics_data]
        metric_names = [m['metric'] for m in metrics_data]

        # Create bars
        bars = ax.barh(y_pos, normalized_scores, height=0.6, color='skyblue',
                      edgecolor='black', linewidth=1.5)

        # Color bars based on quality
        for i, (bar, data) in enumerate(zip(bars, metrics_data)):
            if data['good_min'] <= data['normalized_score'] <= data['good_max']:
                bar.set_color('#4CAF50')  # Good - green
            elif data['normalized_score'] >= data['good_max']:
                bar.set_color('#2E7D32')  # Excellent - dark green
            elif data['normalized_score'] >= 0.5:
                bar.set_color('#FFC107')  # Fair - yellow
            else:
                bar.set_color('#F44336')  # Poor - red

        # Add "good range" shading
        for i, data in enumerate(metrics_data):
            ax.barh(i, data['good_max'] - data['good_min'], left=data['good_min'],
                   height=0.6, color='lightgreen', alpha=0.3, edgecolor='green',
                   linewidth=1, linestyle='--')

        # Add value labels
        for i, data in enumerate(metrics_data):
            ax.text(data['normalized_score'] + 0.02, i,
                   f"{data['original_score']:.2f}",
                   va='center', fontsize=10, fontweight='bold')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(metric_names, fontsize=11)
        ax.set_xlabel('Normalized Score (0 = Worst, 1 = Best)', fontsize=12)
        ax.set_xlim(0, 1.15)
        ax.set_title('Metrics Comparison (Normalized with Quality Ranges)',
                    fontsize=14, fontweight='bold')

        # Add legend
        from matplotlib.patches import Rectangle
        legend_elements = [
            Rectangle((0, 0), 1, 1, fc='#2E7D32', label='Excellent'),
            Rectangle((0, 0), 1, 1, fc='#4CAF50', label='Good'),
            Rectangle((0, 0), 1, 1, fc='#FFC107', label='Fair'),
            Rectangle((0, 0), 1, 1, fc='#F44336', label='Poor'),
            Rectangle((0, 0), 1, 1, fc='lightgreen', alpha=0.3,
                     edgecolor='green', linestyle='--', label='Expected Range'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

        # Add grid
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved normalized metrics comparison to {output_path}")

    except Exception as e:
        logger.error(f"Error creating normalized comparison: {e}", exc_info=True)


def generate_all_visualizations(
    summary: Dict,
    detailed_results: pd.DataFrame,
    output_dir: Path,
    primary_metric: str = None,
):
    """
    Generate all visualizations for evaluation results.

    Args:
        summary: Summary dictionary
        detailed_results: DataFrame with per-sample results
        output_dir: Directory to save visualizations
        primary_metric: Primary metric for quality categorization (auto-detected if None)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect primary metric if not specified
    if primary_metric is None:
        translation_type = summary.get('translation_type')
        if translation_type == 'audio_to_audio':
            primary_metric = 'blaser' if 'blaser' in summary.get('metrics_computed', []) else 'comet'
        else:
            primary_metric = 'comet'

    # 1. Quality Dashboard (NEW)
    if f'{primary_metric}_score' in detailed_results.columns:
        create_quality_dashboard(
            detailed_results,
            primary_metric,
            output_dir / 'quality_dashboard.png',
        )

    # 2. Normalized Metrics Comparison (NEW)
    if summary.get('aggregate_scores'):
        create_normalized_metrics_comparison(
            summary,
            output_dir / 'normalized_metrics.png',
        )

    # 3. Original Metrics comparison
    if summary.get('aggregate_scores'):
        create_metrics_comparison_chart(
            summary['aggregate_scores'],
            output_dir / 'metrics_comparison.png',
        )

    # 4. Metrics table
    create_metrics_table_image(summary, output_dir / 'metrics_table.png')

    # 5. Score distributions
    for metric in summary.get('metrics_computed', []):
        if f'{metric}_score' in detailed_results.columns:
            scores = detailed_results[f'{metric}_score'].dropna().tolist()
            if scores:
                create_score_distribution(
                    scores,
                    metric.upper(),
                    output_dir / f'{metric}_distribution.png',
                )

    # 6. Per-sample heatmap (limit to first 50 samples for readability)
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
