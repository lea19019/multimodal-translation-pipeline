#!/usr/bin/env python3
"""
Test Visualization Improvements

Tests the new visualization functions on existing results without
running a full evaluation pipeline.
"""

import sys
from pathlib import Path
import pandas as pd
import json

# Add visualizations directory to path
sys.path.insert(0, str(Path(__file__).parent / 'visualizations'))

from pipeline_viz import create_all_visualizations

def main():
    # Use existing results from final_complete_test
    results_dir = Path(__file__).parent / 'results' / 'final_complete_test'
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return 1
    
    print("\n" + "="*70)
    print("TESTING VISUALIZATION IMPROVEMENTS")
    print("="*70)
    print(f"Using existing results from: {results_dir}")
    print("="*70 + "\n")
    
    # Load comparison data
    comparisons_dir = results_dir / 'comparisons'
    csv_path = comparisons_dir / 'cross_pipeline_stats.csv'
    
    if not csv_path.exists():
        print(f"Error: Comparison CSV not found: {csv_path}")
        return 1
    
    comparison_df = pd.read_csv(csv_path)
    print(f"✓ Loaded comparison data: {len(comparison_df)} rows")
    print(f"  Pipelines: {comparison_df['pipeline_name'].unique().tolist()}")
    print(f"  Languages: {comparison_df['language'].unique().tolist()}")
    
    # Load rankings
    rankings_path = comparisons_dir / 'pipeline_rankings_overall.json'
    rankings = {}
    if rankings_path.exists():
        with open(rankings_path, 'r') as f:
            rankings = json.load(f)
        print(f"✓ Loaded rankings data")
    
    # Load best info
    best_path = comparisons_dir / 'best_pipelines.json'
    best_info = {}
    if best_path.exists():
        with open(best_path, 'r') as f:
            best_info = json.load(f)
        print(f"✓ Loaded best pipelines data")
    
    # Generate visualizations
    print(f"\n{'='*70}")
    print("GENERATING VISUALIZATIONS")
    print("="*70 + "\n")
    
    viz_dir = results_dir / 'visualizations_updated'
    viz_dir.mkdir(exist_ok=True)
    
    create_all_visualizations(
        comparison_df=comparison_df,
        rankings=rankings,
        best_info=best_info,
        output_dir=viz_dir
    )
    
    # List generated files
    print(f"\n{'='*70}")
    print("VISUALIZATIONS GENERATED")
    print("="*70 + "\n")
    
    print("Overall visualizations:")
    for f in sorted(viz_dir.glob('*.png')):
        size_kb = f.stat().st_size / 1024
        print(f"  ✓ {f.name} ({size_kb:.1f} KB)")
    
    per_lang_dir = viz_dir / 'per_language'
    if per_lang_dir.exists():
        per_lang_files = list(per_lang_dir.glob('*.png'))
        if per_lang_files:
            print(f"\nPer-language visualizations ({len(per_lang_files)} files):")
            for f in sorted(per_lang_files):
                size_kb = f.stat().st_size / 1024
                print(f"  ✓ {f.name} ({size_kb:.1f} KB)")
    
    print(f"\n{'='*70}")
    print("✓ VISUALIZATION TEST COMPLETE!")
    print("="*70)
    print(f"\nNew visualizations saved to:")
    print(f"  {viz_dir}")
    print(f"\nYou can compare with original visualizations at:")
    print(f"  {results_dir / 'visualizations'}")
    print(f"\n{'='*70}\n")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
