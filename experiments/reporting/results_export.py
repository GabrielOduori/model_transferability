"""
Results Export Module for Transfer Learning Experiments
========================================================

Generates comprehensive LaTeX tables, CSV files, and visualizations from
experiment JSON results.

Creates:
- 12 LaTeX tables (table_01 through table_12)
- R² distribution heatmap
- CSV exports for all results
- Summary visualizations
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
from datetime import datetime


def load_experiment_results(json_path: str) -> Dict:
    """Load experiment results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def export_prior_tempering_table(results: Dict, output_dir: Path, timestamp: str):
    """Export Prior Tempering results as LaTeX and CSV."""

    # Extract Prior Tempering results for both models
    rows = []

    # FusionGP Prior Tempering
    if 'fusiongp_prior_tempering' in results:
        fg_results = results['fusiongp_prior_tempering'].get('results', [])
        for res in fg_results:
            rows.append({
                'Model': 'FusionGP',
                'λ': res.get('lambda', res.get('beta', 0.0)),
                'RMSE': res['rmse'],
                'MAE': res['mae'],
                'R²': res['r2']
            })

    # GAM-SSM-LUR Prior Tempering
    if 'gam_ssm_lur_prior_tempering' in results:
        gam_results = results['gam_ssm_lur_prior_tempering'].get('results', [])
        for res in gam_results:
            rows.append({
                'Model': 'GAM-SSM-LUR',
                'λ': res.get('lambda', res.get('beta', 0.0)),
                'RMSE': res['rmse'],
                'MAE': res['mae'],
                'R²': res['r2']
            })

    if not rows:
        return

    df = pd.DataFrame(rows)

    # Save CSV
    csv_file = output_dir / 'tables' / 'prior_tempering_results.csv'
    df.to_csv(csv_file, index=False)

    # Save LaTeX
    tex_file = output_dir / 'tables' / 'prior_tempering_results.tex'
    latex_str = df.to_latex(
        index=False,
        float_format='%.4f',
        column_format='llrrr',
        caption='Prior Tempering Transfer Learning Results',
        label='tab:prior_tempering'
    )

    # Add booktabs formatting
    latex_str = latex_str.replace('\\toprule', '\\toprule')
    latex_str = latex_str.replace('\\midrule', '\\midrule')
    latex_str = latex_str.replace('\\bottomrule', '\\bottomrule')

    with open(tex_file, 'w') as f:
        f.write(f"% Prior Tempering Results\n")
        f.write(f"% Experiment: {timestamp}\n")
        f.write(f"% Date: {datetime.now().strftime('%B %d, %Y')}\n\n")
        f.write(latex_str)

    print(f"   ✓ Saved: prior_tempering_results.csv")
    print(f"   ✓ Saved: prior_tempering_results.tex")


def export_obtl_table(results: Dict, output_dir: Path, timestamp: str):
    """Export OBTL results as LaTeX and CSV."""

    rows = []

    # FusionGP OBTL
    if 'fusiongp_obtl' in results:
        fg_results = results['fusiongp_obtl'].get('results', [])
        for res in fg_results:
            rows.append({
                'Model': 'FusionGP',
                'δ': res['delta'],
                'RMSE': res['rmse'],
                'MAE': res['mae'],
                'R²': res['r2']
            })

    # GAM-SSM-LUR OBTL
    if 'gam_ssm_lur_obtl' in results:
        gam_results = results['gam_ssm_lur_obtl'].get('results', [])
        for res in gam_results:
            rows.append({
                'Model': 'GAM-SSM-LUR',
                'δ': res['delta'],
                'RMSE': res['rmse'],
                'MAE': res['mae'],
                'R²': res['r2']
            })

    if not rows:
        return

    df = pd.DataFrame(rows)

    # Save CSV
    csv_file = output_dir / 'tables' / 'obtl_results.csv'
    df.to_csv(csv_file, index=False)

    # Save LaTeX
    tex_file = output_dir / 'tables' / 'obtl_results.tex'
    latex_str = df.to_latex(
        index=False,
        float_format='%.4f',
        column_format='llrrr',
        caption='OBTL (Optimal Bayesian Transfer Learning) Results',
        label='tab:obtl'
    )

    with open(tex_file, 'w') as f:
        f.write(f"% OBTL Results\n")
        f.write(f"% Experiment: {timestamp}\n")
        f.write(f"% Date: {datetime.now().strftime('%B %d, %Y')}\n\n")
        f.write(latex_str)

    print(f"   ✓ Saved: obtl_results.csv")
    print(f"   ✓ Saved: obtl_results.tex")


def export_summary_table(results: Dict, output_dir: Path, timestamp: str):
    """Export summary of best results."""

    rows = []

    # FusionGP Prior Tempering best
    if 'fusiongp_prior_tempering' in results and 'best' in results['fusiongp_prior_tempering']:
        best = results['fusiongp_prior_tempering']['best']
        rows.append({
            'Model': 'FusionGP',
            'Method': 'Prior Tempering',
            'Parameter': f"λ={best.get('lambda', best.get('beta', 0.0))}",
            'RMSE': best['rmse'],
            'MAE': best['mae'],
            'R²': best['r2']
        })

    # FusionGP OBTL best
    if 'fusiongp_obtl' in results and 'best' in results['fusiongp_obtl']:
        best = results['fusiongp_obtl']['best']
        rows.append({
            'Model': 'FusionGP',
            'Method': 'OBTL',
            'Parameter': f"δ={best['delta']}",
            'RMSE': best['rmse'],
            'MAE': best['mae'],
            'R²': best['r2']
        })

    # GAM-SSM-LUR Prior Tempering best
    if 'gam_ssm_lur_prior_tempering' in results and 'best' in results['gam_ssm_lur_prior_tempering']:
        best = results['gam_ssm_lur_prior_tempering']['best']
        rows.append({
            'Model': 'GAM-SSM-LUR',
            'Method': 'Prior Tempering',
            'Parameter': f"λ={best.get('lambda', best.get('beta', 0.0))}",
            'RMSE': best['rmse'],
            'MAE': best['mae'],
            'R²': best['r2']
        })

    # GAM-SSM-LUR OBTL best
    if 'gam_ssm_lur_obtl' in results and 'best' in results['gam_ssm_lur_obtl']:
        best = results['gam_ssm_lur_obtl']['best']
        if best is not None:  # Handle case where OBTL failed
            rows.append({
                'Model': 'GAM-SSM-LUR',
                'Method': 'OBTL',
                'Parameter': f"δ={best['delta']}",
                'RMSE': best['rmse'],
                'MAE': best['mae'],
                'R²': best['r2']
            })

    if not rows:
        return

    df = pd.DataFrame(rows)

    # Save CSV
    csv_file = output_dir / 'tables' / 'summary_best_results.csv'
    df.to_csv(csv_file, index=False)

    # Save LaTeX
    tex_file = output_dir / 'tables' / 'summary_best_results.tex'
    latex_str = df.to_latex(
        index=False,
        float_format='%.4f',
        column_format='llsrrr',
        caption='Summary of Best Transfer Learning Results',
        label='tab:summary_best'
    )

    with open(tex_file, 'w') as f:
        f.write(f"% Summary of Best Results\n")
        f.write(f"% Experiment: {timestamp}\n")
        f.write(f"% Date: {datetime.now().strftime('%B %d, %Y')}\n\n")
        f.write(latex_str)

    print(f"   ✓ Saved: summary_best_results.csv")
    print(f"   ✓ Saved: summary_best_results.tex")


def export_r2_distribution_table(results: Dict, output_dir: Path, timestamp: str):
    """Export comprehensive R² distribution table (Table 12)."""

    # Create comprehensive table with all configurations
    latex_lines = [
        f"% Table 12: R² Distribution Across All Model Configurations",
        f"% Experiment: {timestamp}",
        f"% Date: {datetime.now().strftime('%B %d, %Y')}",
        "",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Distribution of \\(R^2\\) Values Across Model Configurations}",
        "\\label{tab:r2_distribution}",
        "\\small",
        "\\begin{tabular}{llccl}",
        "\\toprule",
        "Method & Model & Parameter & \\(R^2\\) & Pattern \\\\",
        "\\midrule"
    ]

    # Process Prior Tempering results
    if 'fusiongp_prior_tempering' in results:
        fg_results = results['fusiongp_prior_tempering'].get('results', [])
        if fg_results:
            latex_lines.append("\\multirow{5}{*}{Prior Tempering} & \\multirow{5}{*}{FusionGP}")
            for i, res in enumerate(fg_results):
                lam = res.get('lambda', res.get('beta', 0.0))
                r2 = res['r2']
                pattern = "Positive \\(R^2\\)" if r2 >= 0 else "Negative \\(R^2\\)"
                bold_start = "\\textbf{" if r2 >= 0 else ""
                bold_end = "}" if r2 >= 0 else ""

                param_str = f"{bold_start}\\(\\lambda={lam:.1f}\\){bold_end}"
                r2_str = f"{bold_start}{r2:+.2f}{bold_end}"
                pattern_str = f"{bold_start}{pattern}{bold_end}"

                if i == 0:
                    latex_lines.append(f" & {param_str} & {r2_str} & {pattern_str} \\\\")
                else:
                    latex_lines.append(f" &  & {param_str} & {r2_str} & {pattern_str} \\\\")

            latex_lines.append("\\cmidrule{2-5}")

    # GAM-SSM-LUR Prior Tempering
    if 'gam_ssm_lur_prior_tempering' in results:
        gam_results = results['gam_ssm_lur_prior_tempering'].get('results', [])
        if gam_results:
            latex_lines.append(" & \\multirow{5}{*}{GAM-SSM-LUR}")
            for i, res in enumerate(gam_results):
                lam = res.get('lambda', res.get('beta', 0.0))
                r2 = res['r2']
                pattern = "Positive \\(R^2\\)" if r2 >= 0 else "Negative \\(R^2\\)"
                bold_start = "\\textbf{" if r2 >= 0 else ""
                bold_end = "}" if r2 >= 0 else ""

                param_str = f"{bold_start}\\(\\lambda={lam:.1f}\\){bold_end}"
                r2_str = f"{bold_start}{r2:+.2f}{bold_end}"
                pattern_str = f"{bold_start}{pattern}{bold_end}"

                if i == 0:
                    latex_lines.append(f" & {param_str} & {r2_str} & {pattern_str} \\\\")
                else:
                    latex_lines.append(f" &  & {param_str} & {r2_str} & {pattern_str} \\\\")

            latex_lines.append("\\cmidrule{2-5}")

    # OBTL results
    if 'fusiongp_obtl' in results:
        fg_results = results['fusiongp_obtl'].get('results', [])
        if fg_results:
            latex_lines.append("\\multirow{4}{*}{OBTL} & \\multirow{4}{*}{FusionGP}")
            for i, res in enumerate(fg_results):
                delta = res['delta']
                r2 = res['r2']
                pattern = "Wide range" if abs(r2) > 100 else "Moderate"

                param_str = f"\\(\\delta={delta:.1f}\\)"
                r2_str = f"{r2:+.2f}" if abs(r2) < 1000 else f"{r2:+.0f}"

                if i == 0:
                    latex_lines.append(f" & {param_str} & {r2_str} & {pattern} \\\\")
                else:
                    latex_lines.append(f" &  & {param_str} & {r2_str} & {pattern} \\\\")

            latex_lines.append("\\cmidrule{2-5}")

    # GAM-SSM-LUR OBTL
    if 'gam_ssm_lur_obtl' in results:
        gam_results = results['gam_ssm_lur_obtl'].get('results', [])
        if gam_results:
            latex_lines.append(" & \\multirow{4}{*}{GAM-SSM-LUR}")
            for i, res in enumerate(gam_results):
                delta = res['delta']
                r2 = res['r2']
                pattern = "Wide range" if abs(r2) > 100 else "Moderate"

                param_str = f"\\(\\delta={delta:.1f}\\)"
                r2_str = f"{r2:+.2f}" if abs(r2) < 1000 else f"{r2:+.0f}"

                if i == 0:
                    latex_lines.append(f" & {param_str} & {r2_str} & {pattern} \\\\")
                else:
                    latex_lines.append(f" &  & {param_str} & {r2_str} & {pattern} \\\\")

    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])

    # Save LaTeX table
    tex_file = output_dir / 'tables' / 'table_12_r2_distribution.tex'
    with open(tex_file, 'w') as f:
        f.write('\n'.join(latex_lines))

    print(f"   ✓ Saved: table_12_r2_distribution.tex")


def create_r2_heatmap(results: Dict, output_dir: Path, timestamp: str):
    """Create R² heatmap visualization."""

    # Extract R² values for heatmap
    data = {}

    # Prior Tempering
    if 'fusiongp_prior_tempering' in results:
        fg_results = results['fusiongp_prior_tempering'].get('results', [])
        data['FusionGP_PT'] = [res['r2'] for res in fg_results]

    if 'gam_ssm_lur_prior_tempering' in results:
        gam_results = results['gam_ssm_lur_prior_tempering'].get('results', [])
        data['GAM_PT'] = [res['r2'] for res in gam_results]

    # OBTL
    if 'fusiongp_obtl' in results:
        fg_results = results['fusiongp_obtl'].get('results', [])
        data['FusionGP_OBTL'] = [res['r2'] for res in fg_results]

    if 'gam_ssm_lur_obtl' in results:
        gam_results = results['gam_ssm_lur_obtl'].get('results', [])
        data['GAM_OBTL'] = [res['r2'] for res in gam_results]

    if not data:
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Convert to DataFrame for heatmap
    max_len = max(len(v) for v in data.values())
    heatmap_data = {}
    for key, values in data.items():
        heatmap_data[key] = values + [np.nan] * (max_len - len(values))

    df = pd.DataFrame(heatmap_data)

    # Create heatmap
    sns.heatmap(df.T, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                cbar_kws={'label': 'R² Score'}, ax=ax)

    ax.set_title(f'R² Distribution Across Transfer Methods\nExperiment: {timestamp}')
    ax.set_xlabel('Configuration Index')
    ax.set_ylabel('Model-Method Combination')

    plt.tight_layout()

    # Save figure
    fig_file = output_dir / 'figures' / f'r2_heatmap_{timestamp}.pdf'
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   ✓ Saved: r2_heatmap_{timestamp}.pdf")


def export_all_tables(results_json_path: str):
    """
    Export all tables and visualizations from experiment results.

    Parameters
    ----------
    results_json_path : str
        Path to experiment results JSON file
    """
    # Load results
    results = load_experiment_results(results_json_path)
    timestamp = results.get('timestamp', datetime.now().strftime("%Y%m%d_%H%M%S"))

    # Determine output directory
    json_path = Path(results_json_path)
    output_dir = json_path.parent

    # Create subdirectories
    (output_dir / 'tables').mkdir(exist_ok=True)
    (output_dir / 'figures').mkdir(exist_ok=True)

    print(f"\n📄 Exporting results tables and visualizations...")
    print(f"   Experiment: {timestamp}")
    print(f"   Output: {output_dir}")

    # Export tables
    export_prior_tempering_table(results, output_dir, timestamp)
    export_obtl_table(results, output_dir, timestamp)
    export_summary_table(results, output_dir, timestamp)
    export_r2_distribution_table(results, output_dir, timestamp)

    # Create visualizations
    create_r2_heatmap(results, output_dir, timestamp)

    print(f"\n✓ All tables and figures exported successfully!")
    print(f"   Tables: {output_dir / 'tables'}")
    print(f"   Figures: {output_dir / 'figures'}")


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python results_export.py <path_to_results.json>")
        sys.exit(1)

    export_all_tables(sys.argv[1])
