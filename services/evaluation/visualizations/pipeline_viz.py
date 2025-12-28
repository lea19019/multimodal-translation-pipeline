#!/usr/bin/env python3
"""
Pipeline Visualization Generator

Creates cross-pipeline comparison visualizations including heatmaps,
rankings charts, and radar plots.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import sys

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from math import pi

# Add scripts directory for reusing existing viz functions
scripts_dir = Path(__file__).parent.parent / 'scripts'
sys.path.insert(0, str(scripts_dir))

from visualizations import (
    normalize_metric_value, 
    METRIC_INFO, 
    create_metrics_comparison_chart as create_viz_metrics_comparison,
    create_language_metric_heatmap
)

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")


def create_pipeline_language_heatmap(
    df: pd.DataFrame,
    metric: str,
    output_path: Path,
    title: str = None
):
    """
    Create heatmap showing pipeline performance across languages for a specific metric.

    Args:
        df: DataFrame with columns: pipeline_name, language, {metric}
        metric: Metric column name (e.g., 'bleu', 'chrf', 'mcd')
        output_path: Path to save heatmap
        title: Optional custom title
    """
    output_path = Path(output_path)

    if metric not in df.columns:
        logger.warning(f"Metric {metric} not found in DataFrame")
        return

    # Pivot data: pipelines as rows, languages as columns
    pivot_df = df.pivot(index='pipeline_name', columns='language', values=metric)

    # Sort pipelines by average score across languages
    if METRIC_INFO.get(metric, {}).get('higher_is_better', True):
        pivot_df = pivot_df.loc[pivot_df.mean(axis=1).sort_values(ascending=False).index]
    else:
        # For MCD (lower is better), sort ascending
        pivot_df = pivot_df.loc[pivot_df.mean(axis=1).sort_values(ascending=True).index]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Determine colormap based on metric direction
    if METRIC_INFO.get(metric, {}).get('higher_is_better', True):
        cmap = 'RdYlGn'  # Green is good
    else:
        cmap = 'RdYlGn_r'  # Green is good (reversed for lower-is-better)

    # Create heatmap
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt='.2f',
        cmap=cmap,
        cbar_kws={'label': f'{metric.upper()} Score'},
        linewidths=0.5,
        ax=ax
    )

    # Title
    if title is None:
        metric_name = METRIC_INFO.get(metric, {}).get('name', metric.upper())
        title = f'Pipeline Performance by Language - {metric_name}'

    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Language', fontsize=12, fontweight='bold')
    plt.ylabel('Pipeline', fontsize=12, fontweight='bold')

    # Rotate x-axis labels
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Saved heatmap: {output_path.name}")


def create_overall_rankings_chart(
    rankings: List[Dict],
    output_path: Path,
    title: str = "Overall Pipeline Rankings"
):
    """
    Create horizontal bar chart showing overall pipeline rankings.

    Args:
        rankings: List of dicts with 'pipeline_name' and 'avg_score'
        output_path: Path to save chart
        title: Chart title
    """
    output_path = Path(output_path)

    if not rankings:
        logger.warning("No rankings data provided")
        return

    # Extract data
    pipeline_names = [r['pipeline_name'] for r in rankings]
    scores = [r['avg_score'] for r in rankings]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(6, len(rankings) * 0.5)))

    # Color bars by performance tier
    colors = []
    for score in scores:
        if score >= 0.75:
            colors.append('#2ecc71')  # Green - excellent
        elif score >= 0.60:
            colors.append('#3498db')  # Blue - good
        elif score >= 0.40:
            colors.append('#f39c12')  # Orange - fair
        else:
            colors.append('#e74c3c')  # Red - poor

    # Create horizontal bar chart
    y_pos = np.arange(len(pipeline_names))
    bars = ax.barh(y_pos, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    # Add score labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(score + 0.01, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}',
                va='center', fontsize=10, fontweight='bold')

    # Customize
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pipeline_names, fontsize=10)
    ax.set_xlabel('Overall Score (normalized)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, 1.0)
    ax.grid(axis='x', alpha=0.3)

    # Add legend for color tiers
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Excellent (≥0.75)'),
        Patch(facecolor='#3498db', label='Good (≥0.60)'),
        Patch(facecolor='#f39c12', label='Fair (≥0.40)'),
        Patch(facecolor='#e74c3c', label='Poor (<0.40)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Saved rankings chart: {output_path.name}")


def create_radar_chart(
    df: pd.DataFrame,
    pipeline_ids: List[str],
    output_path: Path,
    title: str = "Pipeline Comparison - Radar Chart"
):
    """
    Create radar chart comparing multiple pipelines across metrics.

    Args:
        df: Comparison DataFrame
        pipeline_ids: List of pipeline IDs to include (max 4-5 for clarity)
        output_path: Path to save chart
        title: Chart title
    """
    output_path = Path(output_path)

    # Select pipelines
    selected_df = df[df['pipeline_id'].isin(pipeline_ids)]

    if len(selected_df) == 0:
        logger.warning("No pipelines selected for radar chart")
        return

    # Metrics to include (skip NMT-only metrics for non-NMT pipelines)
    metrics = ['bleu', 'chrf', 'comet', 'mcd', 'blaser']

    # Aggregate scores across languages for each pipeline
    pipeline_scores = {}
    for pipeline_id in pipeline_ids:
        pipeline_data = selected_df[selected_df['pipeline_id'] == pipeline_id]
        if len(pipeline_data) == 0:
            continue

        pipeline_name = pipeline_data.iloc[0]['pipeline_name']
        scores = []

        for metric in metrics:
            if metric in pipeline_data.columns:
                # Get mean score across languages, normalized to 0-1
                metric_values = pipeline_data[metric].dropna()
                if len(metric_values) > 0:
                    avg_value = metric_values.mean()

                    # Normalize
                    if metric == 'bleu' or metric == 'chrf':
                        normalized = avg_value / 100.0
                    elif metric == 'comet':
                        normalized = avg_value  # Already 0-1
                    elif metric == 'mcd':
                        # Lower is better, invert (assume max 15)
                        normalized = max(0, 1 - (avg_value / 15.0))
                    elif metric == 'blaser':
                        normalized = avg_value / 5.0
                    else:
                        normalized = avg_value

                    scores.append(normalized)
                else:
                    scores.append(0)
            else:
                scores.append(0)

        pipeline_scores[pipeline_name] = scores

    if not pipeline_scores:
        logger.warning("No valid data for radar chart")
        return

    # Create radar chart
    num_metrics = len(metrics)
    angles = [n / float(num_metrics) * 2 * pi for n in range(num_metrics)]
    angles += angles[:1]  # Complete the circle

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # Plot each pipeline
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    for i, (pipeline_name, scores) in enumerate(pipeline_scores.items()):
        values = scores + scores[:1]  # Complete the circle
        ax.plot(angles, values, 'o-', linewidth=2, label=pipeline_name,
                color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.15, color=colors[i % len(colors)])

    # Customize
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.upper() for m in metrics], fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True)

    plt.title(title, fontsize=14, fontweight='bold', pad=30)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Saved radar chart: {output_path.name}")


def create_per_language_comparison(
    df: pd.DataFrame,
    language: str,
    output_path: Path
):
    """
    Create comparison chart for a specific language across all pipelines.

    Args:
        df: Comparison DataFrame
        language: Language name
        output_path: Path to save chart
    """
    output_path = Path(output_path)

    # Filter for this language
    lang_df = df[df['language'] == language].copy()

    if len(lang_df) == 0:
        logger.warning(f"No data for language: {language}")
        return

    # Sort by BLEU score (or first available metric)
    sort_by = None
    for metric in ['bleu', 'comet', 'chrf', 'blaser']:
        if metric in lang_df.columns and lang_df[metric].notna().any():
            sort_by = metric
            break
    
    if sort_by:
        # Higher is better for these metrics
        lang_df = lang_df.sort_values(sort_by, ascending=False)
    else:
        # Try MCD (lower is better)
        if 'mcd' in lang_df.columns and lang_df['mcd'].notna().any():
            lang_df = lang_df.sort_values('mcd', ascending=True)

    # Create figure with subplots for each metric
    metrics = []
    for m in ['bleu', 'chrf', 'comet', 'mcd', 'blaser']:
        if m in lang_df.columns and lang_df[m].notna().any():
            metrics.append(m)

    if not metrics:
        logger.warning(f"No metrics available for language: {language}")
        return

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 8))

    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        # Get scores
        scores = lang_df[metric].values
        pipeline_names = lang_df['pipeline_name'].values

        # Color by performance
        colors = []
        for score in scores:
            if pd.isna(score):
                colors.append('#cccccc')
            else:
                if metric == 'mcd':
                    # Lower is better
                    if score <= 4:
                        colors.append('#2ecc71')
                    elif score <= 6:
                        colors.append('#3498db')
                    elif score <= 8:
                        colors.append('#f39c12')
                    else:
                        colors.append('#e74c3c')
                else:
                    # Higher is better
                    normalized = score / 100.0 if metric in ['bleu', 'chrf'] else score
                    if normalized >= 0.75:
                        colors.append('#2ecc71')
                    elif normalized >= 0.60:
                        colors.append('#3498db')
                    elif normalized >= 0.40:
                        colors.append('#f39c12')
                    else:
                        colors.append('#e74c3c')

        # Create horizontal bar chart
        y_pos = np.arange(len(pipeline_names))
        bars = ax.barh(y_pos, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

        # Add score labels
        for bar, score in zip(bars, scores):
            if not pd.isna(score):
                ax.text(score + (max(scores) * 0.02), bar.get_y() + bar.get_height()/2,
                        f'{score:.1f}',
                        va='center', fontsize=9, fontweight='bold')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(pipeline_names, fontsize=8)
        ax.set_xlabel('Score', fontsize=10, fontweight='bold')
        metric_name = METRIC_INFO.get(metric, {}).get('name', metric.upper())
        ax.set_title(metric_name, fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

    plt.suptitle(f'Pipeline Comparison - {language.capitalize()}',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Saved per-language comparison: {output_path.name}")


def create_comparison_table(
    df: pd.DataFrame,
    output_path: Path,
    language: str = None
):
    """
    Create a clean table visualization comparing pipelines across metrics.

    Args:
        df: Comparison DataFrame
        output_path: Path to save table image
        language: Optional language filter (if None, shows all)
    """
    output_path = Path(output_path)

    # Filter by language if specified
    if language:
        table_df = df[df['language'] == language].copy()
        title = f'Pipeline Comparison Table - {language.capitalize()}'
    else:
        table_df = df.copy()
        title = 'Pipeline Comparison Table - All Languages'

    if len(table_df) == 0:
        logger.warning(f"No data for comparison table{' (language: ' + language + ')' if language else ''}")
        return

    # Select and order columns for display
    display_cols = ['pipeline_name', 'language', 'n_samples']
    metrics = []
    for m in ['bleu', 'chrf', 'comet', 'mcd', 'blaser']:
        if m in table_df.columns and table_df[m].notna().any():
            display_cols.append(m)
            metrics.append(m)

    table_data = table_df[display_cols].copy()

    # Round metric values
    for m in metrics:
        table_data[m] = table_data[m].round(2)

    # Rename columns for display
    column_names = {
        'pipeline_name': 'Pipeline',
        'language': 'Language',
        'n_samples': 'N',
        'bleu': 'BLEU',
        'chrf': 'chrF',
        'comet': 'COMET',
        'mcd': 'MCD',
        'blaser': 'BLASER'
    }
    table_data = table_data.rename(columns=column_names)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, max(4, len(table_data) * 0.4 + 2)))
    ax.axis('tight')
    ax.axis('off')

    # Create table
    table = ax.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Color header
    for i in range(len(table_data.columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')

    # Color cells based on performance
    for i in range(len(table_data)):
        # Alternate row colors
        row_color = '#f0f0f0' if i % 2 == 0 else 'white'
        
        for j in range(len(table_data.columns)):
            cell = table[(i + 1, j)]
            col_name = table_data.columns[j]
            
            # Color metric cells
            if col_name in ['BLEU', 'chrF', 'COMET', 'BLASER']:
                try:
                    value = float(table_data.iloc[i, j])
                    if col_name == 'BLEU' or col_name == 'chrF':
                        normalized = value / 100.0
                    else:
                        normalized = value / 5.0 if col_name == 'BLASER' else value
                    
                    if normalized >= 0.75:
                        cell.set_facecolor('#90EE90')  # Light green
                    elif normalized >= 0.60:
                        cell.set_facecolor('#ADD8E6')  # Light blue
                    elif normalized >= 0.40:
                        cell.set_facecolor('#FFE4B5')  # Light orange
                    else:
                        cell.set_facecolor('#FFB6C1')  # Light red
                except (ValueError, TypeError):
                    cell.set_facecolor(row_color)
            elif col_name == 'MCD':
                try:
                    value = float(table_data.iloc[i, j])
                    # Lower is better for MCD
                    if value <= 4:
                        cell.set_facecolor('#90EE90')  # Light green
                    elif value <= 6:
                        cell.set_facecolor('#ADD8E6')  # Light blue
                    elif value <= 8:
                        cell.set_facecolor('#FFE4B5')  # Light orange
                    else:
                        cell.set_facecolor('#FFB6C1')  # Light red
                except (ValueError, TypeError):
                    cell.set_facecolor(row_color)
            else:
                cell.set_facecolor(row_color)

    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Saved comparison table: {output_path.name}")


def create_all_visualizations(
    comparison_df: pd.DataFrame,
    rankings: Dict,
    best_info: Dict,
    output_dir: Path
):
    """
    Generate all cross-pipeline visualizations.

    Args:
        comparison_df: Comparison DataFrame
        rankings: Rankings dict from comparator
        best_info: Best pipelines dict from comparator
        output_dir: Directory to save visualizations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating all visualizations in: {output_dir}")

    # 1. Comparison table (NEW - easier to read than heatmaps)
    logger.info("\n📊 Creating comparison tables...")
    
    # Overall comparison table
    output_path = output_dir / 'comparison_table_all.png'
    create_comparison_table(comparison_df, output_path)
    
    # Per-language comparison tables
    lang_dir = output_dir / 'per_language'
    lang_dir.mkdir(exist_ok=True)
    
    for language in comparison_df['language'].unique():
        output_path = lang_dir / f'{language}_comparison_table.png'
        create_comparison_table(comparison_df, output_path, language=language)

    # 2. Heatmaps for each metric (keep for multi-language overview)
    logger.info("\n📊 Creating pipeline × language heatmaps...")
    for metric in ['bleu', 'chrf', 'comet', 'mcd', 'blaser']:
        if metric in comparison_df.columns and comparison_df[metric].notna().any():
            output_path = output_dir / f'{metric}_heatmap_pipeline_x_language.png'
            create_pipeline_language_heatmap(comparison_df, metric, output_path)

    # 3. Overall rankings chart
    logger.info("\n📊 Creating overall rankings chart...")
    if 'overall' in rankings:
        output_path = output_dir / 'overall_rankings.png'
        create_overall_rankings_chart(rankings['overall'], output_path)

    # 4. Radar chart (top 4 pipelines) - only if multiple pipelines
    logger.info("\n📊 Creating radar chart...")
    if 'overall' in rankings and len(rankings['overall']) >= 2:
        top_pipelines = [r['pipeline_id'] for r in rankings['overall'][:min(4, len(rankings['overall']))]]
        output_path = output_dir / 'pipeline_radar_comparison.png'
        create_radar_chart(comparison_df, top_pipelines, output_path)
    else:
        logger.info("Skipping radar chart (need at least 2 pipelines)")

    # 5. Per-language bar charts - generate even for single pipeline
    logger.info("\n📊 Creating per-language bar charts...")
    
    for language in comparison_df['language'].unique():
        output_path = lang_dir / f'{language}_pipeline_comparison.png'
        create_per_language_comparison(comparison_df, language, output_path)
    
    # 6. Individual pipeline metric visualizations (like in evaluation.py)
    logger.info("\n📊 Creating individual pipeline metric charts...")
    for pipeline in comparison_df['pipeline_id'].unique():
        pipeline_df = comparison_df[comparison_df['pipeline_id'] == pipeline]
        
        for _, row in pipeline_df.iterrows():
            language = row['language']
            pipeline_name = row['pipeline_name']
            
            # Create metrics dict for this pipeline/language combo
            metrics_dict = {}
            for metric in ['bleu', 'chrf', 'comet', 'mcd', 'blaser']:
                if metric in row and pd.notna(row[metric]):
                    metrics_dict[metric] = row[metric]
            
            if metrics_dict:
                output_path = lang_dir / f'{pipeline}_{language}_metrics.png'
                title = f"{pipeline_name} - {language.capitalize()}"
                create_viz_metrics_comparison(metrics_dict, output_path, title=title)

    logger.info(f"\n✓ All visualizations saved to: {output_dir}")


if __name__ == '__main__':
    # Test visualization functions
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Test pipeline visualizations")
    parser.add_argument('--comparisons-dir', type=Path, required=True,
                        help='Path to comparisons directory')
    args = parser.parse_args()

    if not args.comparisons_dir.exists():
        print(f"Error: Comparisons directory not found: {args.comparisons_dir}")
        exit(1)

    # Load comparison data
    csv_path = args.comparisons_dir / 'cross_pipeline_stats.csv'
    rankings_path = args.comparisons_dir / 'pipeline_rankings_overall.json'
    best_path = args.comparisons_dir / 'best_pipelines.json'

    if not csv_path.exists():
        print(f"Error: Comparison CSV not found: {csv_path}")
        exit(1)

    comparison_df = pd.read_csv(csv_path)
    print(f"Loaded comparison data: {len(comparison_df)} rows")

    rankings = {}
    if rankings_path.exists():
        with open(rankings_path, 'r') as f:
            rankings = json.load(f)

    best_info = {}
    if best_path.exists():
        with open(best_path, 'r') as f:
            best_info = json.load(f)

    # Generate visualizations
    viz_dir = args.comparisons_dir.parent / 'visualizations'
    create_all_visualizations(comparison_df, rankings, best_info, viz_dir)

    print(f"\n✓ Visualizations saved to: {viz_dir}")
