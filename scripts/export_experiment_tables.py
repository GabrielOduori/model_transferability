#!/usr/bin/env python
"""
Quick script to export tables from existing experiment JSON files.

Usage:
    python scripts/export_experiment_tables.py results/experiment_20251231_122146/results_20251231_122146.json
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.reporting.results_export import export_all_tables

if __name__ == '__main__':
    if len(sys.argv) < 2:
        # Find most recent experiment
        results_dir = Path(__file__).parent.parent / 'results'
        experiment_dirs = sorted(results_dir.glob('experiment_*'))

        if experiment_dirs:
            latest_exp = experiment_dirs[-1]
            json_files = list(latest_exp.glob('results_*.json'))

            if json_files:
                json_path = json_files[0]
                print(f"Using latest experiment: {json_path}")
                export_all_tables(str(json_path))
            else:
                print(f"No results JSON found in {latest_exp}")
        else:
            print("No experiment directories found in results/")
            print("\nUsage: python scripts/export_experiment_tables.py <path_to_results.json>")
    else:
        json_path = sys.argv[1]
        export_all_tables(json_path)
