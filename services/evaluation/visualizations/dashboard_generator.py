#!/usr/bin/env python3
"""
Dashboard Generator

Generates interactive HTML dashboard for pipeline comparison results.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)


class DashboardGenerator:
    """Generates interactive HTML dashboard for pipeline comparison results."""

    def __init__(self, results_dir: Path):
        """
        Initialize dashboard generator.

        Args:
            results_dir: Path to results directory
        """
        self.results_dir = Path(results_dir)
        self.comparisons_dir = self.results_dir / 'comparisons'
        self.visualizations_dir = self.results_dir / 'visualizations'
        self.manifest_path = self.results_dir / 'manifest.json'

    def generate_dashboard(self, output_path: Path = None) -> Path:
        """
        Generate complete HTML dashboard.

        Args:
            output_path: Path to save dashboard (default: results_dir/dashboard.html)

        Returns:
            Path to generated dashboard
        """
        if output_path is None:
            output_path = self.results_dir / 'dashboard.html'

        logger.info(f"Generating dashboard: {output_path}")

        # Load data
        manifest = self._load_manifest()
        comparison_df = self._load_comparison_data()
        rankings = self._load_rankings()
        best_info = self._load_best_info()

        # Generate HTML
        html = self._generate_html(manifest, comparison_df, rankings, best_info)

        # Save
        with open(output_path, 'w') as f:
            f.write(html)

        logger.info(f"✓ Dashboard saved: {output_path}")
        return output_path

    def _load_manifest(self) -> Dict:
        """Load manifest file."""
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r') as f:
                return json.load(f)
        return {}

    def _load_comparison_data(self) -> pd.DataFrame:
        """Load comparison CSV data."""
        csv_path = self.comparisons_dir / 'cross_pipeline_stats.csv'
        if csv_path.exists():
            return pd.read_csv(csv_path)
        return pd.DataFrame()

    def _load_rankings(self) -> Dict:
        """Load rankings data."""
        rankings_path = self.comparisons_dir / 'pipeline_rankings_overall.json'
        if rankings_path.exists():
            with open(rankings_path, 'r') as f:
                return json.load(f)
        return {}

    def _load_best_info(self) -> Dict:
        """Load best pipelines data."""
        best_path = self.comparisons_dir / 'best_pipelines.json'
        if best_path.exists():
            with open(best_path, 'r') as f:
                return json.load(f)
        return {}

    def _generate_html(self, manifest: Dict, df: pd.DataFrame, rankings: Dict, best_info: Dict) -> str:
        """Generate complete HTML document."""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pipeline Comparison Results</title>
    <style>
{self._get_css()}
    </style>
</head>
<body>
    <div class="container">
        {self._generate_header(manifest)}
        {self._generate_executive_summary(best_info, rankings)}
        {self._generate_overall_rankings_section(rankings)}
        {self._generate_metric_sections(df)}
        {self._generate_language_sections(df)}
        {self._generate_visualizations_section()}
        {self._generate_data_downloads_section()}
        {self._generate_footer()}
    </div>

    <script>
{self._get_javascript()}
    </script>
</body>
</html>
"""
        return html

    def _get_css(self) -> str:
        """Generate CSS styles."""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }

        .section {
            background: white;
            padding: 30px;
            margin-bottom: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .section h2 {
            color: #667eea;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }

        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }

        .summary-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 8px;
            text-align: center;
        }

        .summary-card h3 {
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 10px;
        }

        .summary-card .value {
            font-size: 2em;
            font-weight: bold;
        }

        .best-pipeline {
            background: #2ecc71;
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }

        .best-pipeline h3 {
            font-size: 1.3em;
            margin-bottom: 10px;
        }

        .best-pipeline .pipeline-name {
            font-size: 1.8em;
            font-weight: bold;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }

        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }

        th {
            background: #667eea;
            color: white;
            font-weight: bold;
        }

        tr:hover {
            background: #f5f5f5;
        }

        .rank-badge {
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
        }

        .rank-1 { background: #ffd700; color: #333; }
        .rank-2 { background: #c0c0c0; color: #333; }
        .rank-3 { background: #cd7f32; color: white; }
        .rank-other { background: #e0e0e0; color: #666; }

        .score-good { color: #2ecc71; font-weight: bold; }
        .score-fair { color: #f39c12; font-weight: bold; }
        .score-poor { color: #e74c3c; font-weight: bold; }

        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }

        .tab-button {
            padding: 10px 20px;
            border: none;
            background: #e0e0e0;
            cursor: pointer;
            border-radius: 5px;
            font-size: 1em;
            transition: all 0.3s;
        }

        .tab-button:hover {
            background: #d0d0d0;
        }

        .tab-button.active {
            background: #667eea;
            color: white;
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
        }

        .viz-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 30px;
            margin: 20px 0;
        }

        .viz-item {
            text-align: center;
        }

        .viz-item img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .viz-item h4 {
            margin: 15px 0 5px 0;
            color: #667eea;
        }

        .download-links {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }

        .download-link {
            padding: 12px 24px;
            background: #667eea;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            transition: all 0.3s;
        }

        .download-link:hover {
            background: #5568d3;
            transform: translateY(-2px);
        }

        .footer {
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }
        """

    def _get_javascript(self) -> str:
        """Generate JavaScript code."""
        return """
        function showTab(tabName) {
            // Hide all tab contents
            const contents = document.querySelectorAll('.tab-content');
            contents.forEach(content => {
                content.classList.remove('active');
            });

            // Remove active class from all buttons
            const buttons = document.querySelectorAll('.tab-button');
            buttons.forEach(button => {
                button.classList.remove('active');
            });

            // Show selected tab content
            const selectedContent = document.getElementById(tabName);
            if (selectedContent) {
                selectedContent.classList.add('active');
            }

            // Add active class to clicked button
            const selectedButton = document.querySelector(`[onclick="showTab('${tabName}')"]`);
            if (selectedButton) {
                selectedButton.classList.add('active');
            }
        }

        // Show first tab by default
        window.onload = function() {
            const firstTab = document.querySelector('.tab-button');
            if (firstTab) {
                firstTab.click();
            }
        };
        """

    def _generate_header(self, manifest: Dict) -> str:
        """Generate header section."""
        execution_id = manifest.get('execution_id', 'Unknown')
        timestamp = manifest.get('timestamp', datetime.now().isoformat())

        return f"""
        <div class="header">
            <h1>Pipeline Comparison Results</h1>
            <p>Execution ID: {execution_id}</p>
            <p>Generated: {timestamp}</p>
        </div>
        """

    def _generate_executive_summary(self, best_info: Dict, rankings: Dict) -> str:
        """Generate executive summary section."""
        overall_best = best_info.get('overall_best', {})
        best_name = overall_best.get('pipeline_name', 'N/A')
        best_score = overall_best.get('score', 0)

        num_pipelines = len(rankings.get('overall', []))
        num_languages = len(best_info.get('by_language', {}))

        return f"""
        <div class="section">
            <h2>Executive Summary</h2>

            <div class="best-pipeline">
                <h3>🏆 Best Overall Pipeline</h3>
                <div class="pipeline-name">{best_name}</div>
                <p style="font-size: 1.2em; margin-top: 10px;">Overall Score: {best_score:.3f}</p>
            </div>

            <div class="summary-grid">
                <div class="summary-card">
                    <h3>Total Pipelines</h3>
                    <div class="value">{num_pipelines}</div>
                </div>
                <div class="summary-card">
                    <h3>Languages Evaluated</h3>
                    <div class="value">{num_languages}</div>
                </div>
                <div class="summary-card">
                    <h3>Total Evaluations</h3>
                    <div class="value">{num_pipelines * num_languages}</div>
                </div>
            </div>
        </div>
        """

    def _generate_overall_rankings_section(self, rankings: Dict) -> str:
        """Generate overall rankings section."""
        overall = rankings.get('overall', [])

        if not overall:
            return ""

        rows = ""
        for item in overall:
            rank = item['rank']
            pipeline_name = item['pipeline_name']
            score = item['avg_score']

            rank_class = f"rank-{rank}" if rank <= 3 else "rank-other"
            score_class = "score-good" if score >= 0.70 else ("score-fair" if score >= 0.50 else "score-poor")

            rows += f"""
            <tr>
                <td><span class="rank-badge {rank_class}">#{rank}</span></td>
                <td>{pipeline_name}</td>
                <td class="{score_class}">{score:.4f}</td>
            </tr>
            """

        return f"""
        <div class="section">
            <h2>Overall Rankings</h2>
            <p>Pipelines ranked by average performance across all languages and metrics.</p>

            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Pipeline</th>
                        <th>Average Score</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
        """

    def _generate_metric_sections(self, df: pd.DataFrame) -> str:
        """Generate metric-specific sections."""
        if df.empty:
            return ""

        metrics = []
        for m in ['bleu', 'chrf', 'comet', 'mcd', 'blaser']:
            if m in df.columns and df[m].notna().any():
                metrics.append(m)

        if not metrics:
            return ""

        # Create tabs
        tabs_html = '<div class="tabs">'
        for i, metric in enumerate(metrics):
            metric_name = metric.upper()
            tabs_html += f'<button class="tab-button" onclick="showTab(\'metric-{metric}\')">{metric_name}</button>'
        tabs_html += '</div>'

        # Create tab contents
        contents_html = ""
        for i, metric in enumerate(metrics):
            table_rows = ""
            for _, row in df.iterrows():
                if pd.notna(row[metric]):
                    # Determine score class based on metric value and type
                    metric_value = row[metric]
                    if metric in ['bleu', 'chrf', 'comet', 'blaser']:  # Higher is better
                        if metric == 'blaser':  # 0-5 scale
                            score_class = "score-good" if metric_value >= 4.0 else ("score-fair" if metric_value >= 3.0 else "score-poor")
                        elif metric == 'comet':  # 0-1 scale
                            score_class = "score-good" if metric_value >= 0.60 else ("score-fair" if metric_value >= 0.40 else "score-poor")
                        else:  # BLEU, chrF (0-100 scale)
                            score_class = "score-good" if metric_value >= 60 else ("score-fair" if metric_value >= 40 else "score-poor")
                    else:  # MCD - lower is better
                        score_class = "score-good" if metric_value <= 12 else ("score-fair" if metric_value <= 15 else "score-poor")
                    
                    table_rows += f"""
                    <tr>
                        <td>{row['pipeline_name']}</td>
                        <td>{row['language'].capitalize()}</td>
                        <td class="{score_class}">{row[metric]:.2f}</td>
                    </tr>
                    """

            contents_html += f"""
            <div id="metric-{metric}" class="tab-content">
                <table>
                    <thead>
                        <tr>
                            <th>Pipeline</th>
                            <th>Language</th>
                            <th>{metric.upper()} Score</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows}
                    </tbody>
                </table>
            </div>
            """

        return f"""
        <div class="section">
            <h2>Performance by Metric</h2>
            {tabs_html}
            {contents_html}
        </div>
        """

    def _generate_language_sections(self, df: pd.DataFrame) -> str:
        """Generate language-specific sections."""
        if df.empty:
            return ""

        languages = df['language'].unique()

        # Create tabs
        tabs_html = '<div class="tabs">'
        for lang in languages:
            tabs_html += f'<button class="tab-button" onclick="showTab(\'lang-{lang}\')">{lang.capitalize()}</button>'
        tabs_html += '</div>'

        # Create tab contents
        contents_html = ""
        for lang in languages:
            # Sort by BLEU if available, otherwise COMET, otherwise first available metric
            sort_metric = 'bleu' if 'bleu' in df.columns else ('comet' if 'comet' in df.columns else df.select_dtypes(include=['float64', 'float32']).columns[0])
            lang_df = df[df['language'] == lang].sort_values(sort_metric, ascending=False)

            # Get available metrics from the dataframe
            available_metrics = [col for col in ['bleu', 'chrf', 'comet', 'mcd', 'blaser'] if col in df.columns]
            
            # Build table header
            header_cols = '<th>Rank</th><th>Pipeline</th>'
            for metric in available_metrics:
                header_cols += f'<th>{metric.upper()}</th>'
            
            table_rows = ""
            for rank, (_, row) in enumerate(lang_df.iterrows(), 1):
                rank_class = f"rank-{rank}" if rank <= 3 else "rank-other"
                
                # Build metric columns
                metric_cols = ""
                for metric in available_metrics:
                    if pd.notna(row[metric]):
                        # Determine color based on metric type
                        value = row[metric]
                        if metric in ['bleu', 'chrf', 'comet', 'blaser']:  # Higher is better
                            if metric == 'blaser':
                                score_class = "score-good" if value >= 4.0 else ("score-fair" if value >= 3.0 else "score-poor")
                            elif metric == 'comet':
                                score_class = "score-good" if value >= 0.60 else ("score-fair" if value >= 0.40 else "score-poor")
                            else:  # BLEU, chrF
                                score_class = "score-good" if value >= 60 else ("score-fair" if value >= 40 else "score-poor")
                        else:  # MCD - lower is better
                            score_class = "score-good" if value <= 12 else ("score-fair" if value <= 15 else "score-poor")
                        
                        metric_cols += f'<td class="{score_class}">{value:.2f}</td>'
                    else:
                        metric_cols += '<td>N/A</td>'

                table_rows += f"""
                <tr>
                    <td><span class="rank-badge {rank_class}">#{rank}</span></td>
                    <td>{row['pipeline_name']}</td>
                    {metric_cols}
                </tr>
                """

            contents_html += f"""
            <div id="lang-{lang}" class="tab-content">
                <table>
                    <thead>
                        <tr>
                            {header_cols}
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows}
                    </tbody>
                </table>
            </div>
            """

        return f"""
        <div class="section">
            <h2>Performance by Language</h2>
            {tabs_html}
            {contents_html}
        </div>
        """

    def _generate_visualizations_section(self) -> str:
        """Generate visualizations section."""
        viz_items = []

        # Overall rankings
        overall_viz = self.visualizations_dir / 'overall_rankings.png'
        if overall_viz.exists():
            viz_items.append(('Overall Rankings', 'visualizations/overall_rankings.png'))

        # Radar chart
        radar_viz = self.visualizations_dir / 'pipeline_radar_comparison.png'
        if radar_viz.exists():
            viz_items.append(('Radar Comparison', 'visualizations/pipeline_radar_comparison.png'))

        # Metric heatmaps
        for metric in ['bleu', 'chrf', 'comet', 'mcd', 'blaser']:
            heatmap = self.visualizations_dir / f'{metric}_heatmap_pipeline_x_language.png'
            if heatmap.exists():
                viz_items.append((f'{metric.upper()} Heatmap', f'visualizations/{metric}_heatmap_pipeline_x_language.png'))

        if not viz_items:
            return ""

        viz_html = '<div class="viz-grid">'
        for title, path in viz_items:
            viz_html += f"""
            <div class="viz-item">
                <h4>{title}</h4>
                <img src="{path}" alt="{title}">
            </div>
            """
        viz_html += '</div>'

        return f"""
        <div class="section">
            <h2>Visualizations</h2>
            {viz_html}
        </div>
        """

    def _generate_data_downloads_section(self) -> str:
        """Generate data downloads section."""
        return f"""
        <div class="section">
            <h2>Download Data</h2>
            <div class="download-links">
                <a href="comparisons/cross_pipeline_stats.csv" class="download-link" download>📊 Comparison Table (CSV)</a>
                <a href="comparisons/all_pipelines_summary.json" class="download-link" download>📋 Full Results (JSON)</a>
                <a href="comparisons/pipeline_rankings_overall.json" class="download-link" download>🏆 Rankings (JSON)</a>
                <a href="comparisons/best_pipelines.json" class="download-link" download>⭐ Best Pipelines (JSON)</a>
            </div>
        </div>
        """

    def _generate_footer(self) -> str:
        """Generate footer."""
        return f"""
        <div class="footer">
            <p>Generated with Pipeline Comparison System</p>
            <p>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """


if __name__ == '__main__':
    # Test dashboard generator
    import argparse

    parser = argparse.ArgumentParser(description="Generate dashboard")
    parser.add_argument('--results-dir', type=Path, required=True,
                        help='Path to results directory')
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Error: Results directory not found: {args.results_dir}")
        exit(1)

    generator = DashboardGenerator(args.results_dir)
    dashboard_path = generator.generate_dashboard()

    print(f"\n✓ Dashboard generated: {dashboard_path}")
